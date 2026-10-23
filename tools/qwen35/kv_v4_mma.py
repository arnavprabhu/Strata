# tools/qwen35/kv_v4_mma.py - emulation of k_attn_gqa's tensor-core P.V for the 4-bit V store (signed scales, FP16 hi+lo weights) vs exact
# Emulates k_attn_gqa's MV P.V for the V4 store (signed scales, weights p*d*2^(14-ex) as FP16 hi+lo, FP32 sums) vs exact.
import numpy as np
rng = np.random.default_rng(3)
f16 = lambda x: np.asarray(x, np.float32).astype(np.float16)
worst = 0; tot = []
for trial in range(2000):
    n = 256
    d = (rng.standard_normal(n) * np.exp(rng.standard_normal() * 1.5)).astype(np.float32)           # signed scales
    d = f16(d).astype(np.float32); d[d == 0] = 1
    d *= np.exp(rng.standard_normal(n) * 1.0).astype(np.float32); d = f16(d).astype(np.float32)       # cell-to-cell spread
    codes = rng.integers(-8, 8, size=(n, 32)).astype(np.float32)
    spread = rng.choice([1, 3, 6, 12])
    p = np.exp((rng.standard_normal(n) * spread) - 0).astype(np.float32); p /= p.max()                  # tile weights, max 1 (exp(s - m))
    exact = (p.astype(np.float64) * d.astype(np.float64)) @ codes.astype(np.float64)
    mv = np.abs(d).max(); ex = np.frexp(mv)[1]
    svs = (d * np.float32(2.0) ** (14 - ex)).astype(np.float32)
    w = (p * svs).astype(np.float32)
    hi = f16(w); lo = f16(w - hi.astype(np.float32))
    got = (hi.astype(np.float32) @ codes + lo.astype(np.float32) @ codes) * np.float32(2.0) ** (ex - 14)
    rel = np.linalg.norm(got - exact) / max(np.linalg.norm(exact), 1e-30)
    tot.append(rel); worst = max(worst, rel)
    assert np.all(np.abs(w) <= 16384 + 1)
print("MMA emulation vs exact: median rel err %.2e, worst %.2e over 2000 random tiles (weights spread up to e^+-12)" % (np.median(tot), worst))
