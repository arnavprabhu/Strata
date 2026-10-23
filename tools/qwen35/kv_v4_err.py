# tools/qwen35/kv_v4_err.py - error of the V-cache quantization schemes considered for --kv k8v4 (synthetic V, CPU only)
# CPU error analysis of V-cache quantization schemes (HD = 256). Synthetic V, see main().
import numpy as np, sys
rng = np.random.default_rng(1)
HD = 256

def h16(x): return x.astype(np.float16).astype(np.float64)

def hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n: H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)
H256 = hadamard(256); H32 = hadamard(32)

def q_sym(x, G, levels):          # symmetric, scale = absmax / levels, codes in [-levels, levels]
    y = x.reshape(x.shape[0], -1, G)
    am = np.abs(y).max(-1, keepdims=True)
    s = h16(np.where(am > 0, am / levels, 1.0))
    c = np.clip(np.rint(y / s), -levels, levels)
    return (c * s).reshape(x.shape)

def q_q40(x, G):                  # ggml q4_0: d = signed max / -8, q = clamp(trunc(x/d + 8.5), 0, 15)
    y = x.reshape(x.shape[0], -1, G)
    i = np.abs(y).argmax(-1)[..., None]
    mv = np.take_along_axis(y, i, -1)
    d = h16(np.where(mv != 0, mv / -8.0, 1.0))
    q = np.clip(np.floor(y / d + 8.5), 0, 15)
    return ((q - 8) * d).reshape(x.shape)

def q_asym(x, G, bits=4):         # min/max, 2^bits levels, FP16 scale + FP16 min
    y = x.reshape(x.shape[0], -1, G)
    lo, hi = y.min(-1, keepdims=True), y.max(-1, keepdims=True)
    L = 2 ** bits - 1
    s = h16(np.where(hi > lo, (hi - lo) / L, 1.0)); lo = h16(lo)
    c = np.clip(np.rint((y - lo) / s), 0, L)
    return (c * s + lo).reshape(x.shape)

def rot256(f): return lambda x: f(x @ H256) @ H256.T
def rot32(f):
    def g(x):
        y = x.reshape(x.shape[0], -1, 32) @ H32
        z = f(y.reshape(x.shape))
        return (z.reshape(x.shape[0], -1, 32) @ H32.T).reshape(x.shape)
    return g

# (name, fn, bytes per value incl. scales)
S = [
 ("int8  sym g64 (current)",      lambda x: q_sym(x, 64, 127), 1 + 2 / 64),
 ("int4  sym 7   g256 (per head)", lambda x: q_sym(x, 256, 7), 0.5 + 2 / 256),
 ("int4  sym 7   g64",             lambda x: q_sym(x, 64, 7), 0.5 + 2 / 64),
 ("int4  sym 7   g32",             lambda x: q_sym(x, 32, 7), 0.5 + 2 / 32),
 ("int4  sym 7   g16",             lambda x: q_sym(x, 16, 7), 0.5 + 2 / 16),
 ("int4  q4_0    g32",             lambda x: q_q40(x, 32), 0.5 + 2 / 32),
 ("int4  q4_0    g64",             lambda x: q_q40(x, 64), 0.5 + 2 / 64),
 ("int4  asym    g32 (scale+min)", lambda x: q_asym(x, 32), 0.5 + 4 / 32),
 ("int4  asym    g64 (scale+min)", lambda x: q_asym(x, 64), 0.5 + 4 / 64),
 ("had256+q4_0   g32",             rot256(lambda x: q_q40(x, 32)), 0.5 + 2 / 32),
 ("had256+sym7   g32",             rot256(lambda x: q_sym(x, 32, 7)), 0.5 + 2 / 32),
 ("had256+sym7   g64",             rot256(lambda x: q_sym(x, 64, 7)), 0.5 + 2 / 64),
 ("had32 +sym7   g32",            rot32(lambda x: q_sym(x, 32, 7)), 0.5 + 2 / 32),
 ("int5? n/a",                     None, 0),
]
S = [s for s in S if s[1] is not None]

def make_v(kind, n):
    if kind == "gauss":   v = rng.standard_normal((n, HD))
    elif kind == "chan":  # per-channel scales lognormal(0.6), fixed across cells, plus 4 outlier channels x8
        sc = np.exp(0.6 * rng.standard_normal(HD)); sc[rng.choice(HD, 4, replace=False)] *= 8
        v = rng.standard_normal((n, HD)) * sc
    elif kind == "heavy": v = rng.standard_t(4, (n, HD))
    elif kind == "chan+tok": # channel scales and per-token magnitude variation (lognormal 0.5) and sparse spikes
        sc = np.exp(0.6 * rng.standard_normal(HD)); sc[rng.choice(HD, 4, replace=False)] *= 8
        v = rng.standard_normal((n, HD)) * sc * np.exp(0.5 * rng.standard_normal((n, 1)))
        m = rng.random((n, HD)) < 0.01
        v = v + m * rng.standard_normal((n, HD)) * 6 * sc.mean()
    return v

def attn(P, V):  return P @ V
def main():
    n = 4096
    print(f"n_cells={n}, HD={HD}. rel_v = ||vhat-v||/||v|| (RMS over cells); out_x = relative L2 error of the attention output")
    print("  for softmax weights over all 4096 cells: diffuse (logit std 1), mid (3), peaked (6), needle (one cell, logit +12 over N(0,1))")
    for kind in ["gauss", "chan", "heavy", "chan+tok"]:
        V = make_v(kind, n)
        print(f"\n=== V distribution: {kind} ===")
        print(f"{'scheme':32s} {'B/val':>6s} {'rel_v':>8s} {'diffuse':>8s} {'mid':>8s} {'peaked':>8s} {'needle':>8s}")
        for name, f, bpv in S:
            Vq = f(V)
            relv = np.sqrt(((Vq - V) ** 2).sum() / (V ** 2).sum())
            outs = []
            for mode in ["diffuse", "mid", "peaked", "needle"]:
                errs = []
                for r in range(24):
                    if mode == "needle":
                        lg = rng.standard_normal(n); lg[rng.integers(n)] += 12
                    else:
                        lg = rng.standard_normal(n) * {"diffuse": 1, "mid": 3, "peaked": 6}[mode]
                    p = np.exp(lg - lg.max()); p /= p.sum()
                    o, oq = p @ V, p @ Vq
                    errs.append(np.linalg.norm(oq - o) / np.linalg.norm(o))
                outs.append(np.sqrt(np.mean(np.square(errs))))
            print(f"{name:32s} {bpv:6.3f} {relv:8.4f} " + " ".join(f"{e:8.4f}" for e in outs))
main()
