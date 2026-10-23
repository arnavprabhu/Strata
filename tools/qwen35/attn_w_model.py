"""CPU model of k_attn_gqa (one row alone) against k_attn_gqa_w (rows sharing the tiles): the tile / mask / 2^E /
online-softmax structure in float32, the mma as an exact sum rounded to float32 (the same function in both).
Scores are given (the Q.K mmas are column-independent: a row's scores do not depend on the other rows).
Checks that every (row, head, chunk) partial (acc, m, l) is bit-identical between the two."""
import numpy as np, math, sys

f32 = np.float32
TL = 256


def fma(a, b, c):
    return f32(np.float64(a) * np.float64(b) + np.float64(c))


def butterfly_sum(parts):  # 32 lane partial sums, xor 16, 8, 4, 2, 1
    p = list(parts)
    for o in (16, 8, 4, 2, 1):
        p = [f32(p[i] + p[i ^ o]) for i in range(32)]
    return p[0]


def lane_sum(vals):  # k = lane, lane + 32, ... added in order per lane, then the butterfly
    parts = []
    for lane in range(32):
        a = f32(0)
        for k in range(lane, len(vals), 32):
            a = f32(a + vals[k])
        parts.append(a)
    return butterfly_sum(parts)


def lane_max(vals, n):
    m = f32(-np.inf)
    for k in range(n):
        m = max(m, vals[k])
    return m


def mma_acc(e, codes, wts):  # e += sum_k codes[k] * wts[k] (one mma: exact products, FP32 result)
    return f32(np.float64(e) + sum(np.float64(c) * np.float64(w) for c, w in zip(codes, wts)))


def hilo(x):
    h = np.float16(x)
    l = np.float16(f32(x) - f32(h))
    return f32(h), f32(l)


def v_phase(codes_tile, wts, svs_scaled, n_cells, inv, e_dims):
    """e[d] = sum over 16-cell groups k0 < n_cells of mma(lo) then mma(hi): codes [cell][d], weights [cell] times the scale"""
    e = [f32(0)] * e_dims
    for k0 in range(0, n_cells, 16):
        hi = []
        lo = []
        for k in range(k0, k0 + 16):
            b = f32(f32(wts[k]) * f32(svs_scaled[k]))
            h, l = hilo(b)
            hi.append(h)
            lo.append(l)
        for d in range(e_dims):
            col = [codes_tile[k][d] for k in range(k0, k0 + 16)]
            e[d] = mma_acc(e[d], col, lo)
            e[d] = mma_acc(e[d], col, hi)
    return e


def old_row(S, codes, sc, npos, p0, cc, dims):
    """k_attn_gqa for one row and one head: S [cell] scores, codes [cell][dims] int8, sc [cell] V scale (one group)."""
    pend = min(p0 + cc, npos)
    m, l = f32(-np.inf), f32(0)
    acc = [f32(0)] * dims
    for tp in range(p0, pend, TL):
        n = min(TL, pend - tp)
        n16 = (n + 15) & ~15
        sv = [f32(sc[tp + j]) if j < n else f32(0) for j in range(TL)]
        mx = lane_max([S[tp + k] for k in range(n)], n)
        mn = max(m, mx)
        e = [f32(math.exp(float(S[tp + k] - mn))) if k < n else f32(0) for k in range(n16)]
        su = lane_sum(e)
        mv = max(sv)
        ex = math.frexp(float(mv))[1]
        svs = [f32(v * f32(2.0) ** (14 - ex)) for v in sv]
        inv = f32(2.0) ** (ex - 14)
        codes_t = [codes[tp + k] if k < n else [0] * dims for k in range(n16)]
        ev = v_phase(codes_t, e + [f32(0)] * 16, svs, n, inv, dims)
        al = f32(math.exp(float(m - mn))) if m != -np.inf else f32(0)
        m = mn
        l = f32(f32(l * al) + su)
        acc = [fma(acc[d], al, f32(ev[d] * inv)) for d in range(dims)]
    return acc, m, l


def new_rows(Ss, codes, sc, npos_rows, p0, cc, dims):
    """k_attn_gqa_w for the rows (list), one head: the shared loads, per-row masks."""
    R = len(npos_rows)
    npos_max = npos_rows[-1]
    pend_max = min(p0 + cc, npos_max)
    m = [f32(-np.inf)] * R
    l = [f32(0)] * R
    acc = [[f32(0)] * dims for _ in range(R)]
    for tp in range(p0, pend_max, TL):
        nrw = [max(0, min(TL, min(p0 + cc, npos_rows[r]) - tp)) for r in range(R)]
        n = max(nrw)
        nz = TL if n > TL // 2 else TL // 2
        sv = [f32(sc[tp + j]) if j < n else f32(0) for j in range(TL)]
        # cells of the tile: real data for all of them (up to n; the clamped loads past n are never weighted)
        codes_t = [codes[tp + min(k, n - 1)] for k in range(nz)]
        for r in range(R):
            if nrw[r] <= 0:
                continue
            mx = lane_max([Ss[r][tp + k] for k in range(nrw[r])], nrw[r])
            mn = max(m[r], mx)
            e = [f32(math.exp(float(Ss[r][tp + k] - mn))) if k < nrw[r] else f32(0) for k in range(nz)]
            su = lane_sum(e)
            mv = max(sv) if nrw[r] == n else max(sv[j] if j < nrw[r] else f32(0) for j in range(TL))
            ex = math.frexp(float(mv))[1]
            p2 = f32(2.0) ** (14 - ex)
            inv = f32(2.0) ** (ex - 14)
            svs = [f32(f32(sv[k]) * p2) for k in range(nz)]
            # V: groups k0 < n (it < 8 or n > 128), a row skips the groups at or past its own cells
            ngr = (nz // 16) * 16
            e_dims = [f32(0)] * dims
            for k0 in range(0, nz, 16):
                if nrw[r] <= k0:
                    continue
                hi, lo = [], []
                for k in range(k0, k0 + 16):
                    b = f32(f32(e[k]) * f32(svs[k]))
                    h, lq = hilo(b)
                    hi.append(h)
                    lo.append(lq)
                for d in range(dims):
                    col = [codes_t[k][d] for k in range(k0, k0 + 16)]
                    e_dims[d] = mma_acc(e_dims[d], col, lo)
                    e_dims[d] = mma_acc(e_dims[d], col, hi)
            al = f32(math.exp(float(m[r] - mn))) if m[r] != -np.inf else f32(0)
            m[r] = mn
            l[r] = f32(f32(l[r] * al) + su)
            acc[r] = [fma(acc[r][d], al, f32(e_dims[d] * inv)) for d in range(dims)]
    return acc, m, l


def main():
    rng = np.random.default_rng(1)
    dims = 4
    bad = 0
    cases = 0
    for trial in range(60):
        cc = int(rng.choice([256, 512, 1024]))
        R = int(rng.integers(2, 5))
        pos = int(rng.integers(0, 3000))
        # make the rows straddle tile / chunk ends in some trials
        if trial % 3 == 0:
            pos = (int(rng.integers(1, 10)) * cc) - int(rng.integers(0, R + 1))
        if trial % 3 == 1:
            pos = (int(rng.integers(1, 10)) * 256) - int(rng.integers(0, R + 1))
        npos_rows = [pos + r + 1 for r in range(R)]
        ncell = npos_rows[-1] + 40
        codes = rng.integers(-127, 128, size=(ncell, dims)).astype(np.int64).tolist()
        sc = (np.abs(rng.normal(size=ncell)) * 0.01).astype(f32)
        # outlier scales in the cells only the last rows see
        for r in range(R):
            if rng.random() < 0.6:
                sc[npos_rows[R - 1] - 1 - int(rng.integers(0, R))] *= 1000
        Ss = [(rng.normal(size=ncell) * 3).astype(f32) for _ in range(R)]
        nchunks = (npos_rows[-1] + cc - 1) // cc
        for c in range(nchunks):
            p0 = c * cc
            new_acc, new_m, new_l = new_rows(Ss, codes, sc, npos_rows, p0, cc, dims)
            for r in range(R):
                if p0 >= npos_rows[r]:
                    continue
                oa, om, ol = old_row(Ss[r], codes, sc, npos_rows[r], p0, cc, dims)
                cases += 1
                ok = (np.array(oa, dtype=f32).tobytes() == np.array(new_acc[r], dtype=f32).tobytes()
                      and f32(om) == f32(new_m[r]) and f32(ol) == f32(new_l[r]))
                if not ok:
                    bad += 1
                    print("MISMATCH trial", trial, "chunk", c, "row", r, "npos", npos_rows, "cc", cc, oa, new_acc[r], om, new_m[r], ol, new_l[r])
    print("checked", cases, "(row, chunk) partials; mismatches:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
