// src/qwen35/q36_engine.cu - strata-q36: the engine path for Qwen3.6-35B-A3B (GGUF architecture qwen35moe),
// NEW-MODELS-PLAN.md T1.  A driver of its own beside Flash-Next's (generate/verify/prefill stay untouched, so the
// Flash-Next identity gate cannot move), built from Strata's kernels:
//
//   * weights straight from the GGUF: dense Q8_0 / K-quant matrices in their blocks (native MMVQ), the 1-D tensors and
//     the F32 projections (ssm_alpha/beta, the router, the shared-expert gate) as f32, the routed experts repacked at
//     load into native expert blobs [gate | up | down] per (layer, expert) for `native_expert_grouped`;
//   * the plain pre-norm residual layer driver: x += mixer(rms(x) * attn_norm); x += moe(rms(x) * post_attention_norm);
//   * GDN: fused conv + L2 and the fused step with the SiLU-gated norm (per token, the state carried);
//   * attention without an indexer: dense, 16 query heads over 2 KV heads, split-K over 256-cell chunks with a
//     log-sum-exp merge, causal for a batch of queries (the prompt) and for one (decode);
//   * MoE: the native router (256 / top-8), the experts grouped per expert on the device (one weight read per expert
//     for the whole batch), the shared expert with its sigmoid gate, the native combine;
//   * the prompt read in chunks of --chunk tokens through the same driver (batched matmuls, ncols <= 8 per launch);
//   * decode one token at a time from a captured graph (positions on the device), or with --no-graph eagerly.
//
// Modes: --serve (serve/server.py's engine protocol: READY / GEN / T / DONE / STOP / PP), --tokens IDS [--n N]
// [--force IDS] [--dump-full-logits F] (tools/qwen35/kl.py's reference interface), --bench-pp N / --bench-tg N.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qwen35/engine.hpp"
#include "q36_kernels.cuh"
#include "q36_sampling.h"
#if defined(STRATA_Q36_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-q36: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;   // attention split-K chunk
constexpr int kConvTile = 16;     // tokens per block of the prompt conv (k_conv_seq tile mode)

// ------------------------------------------------------------------------------------------------ small kernels
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

/// y[r] = rms(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/// x[r] += y[r]; h[r] = rms(x[r]) * w (one block per row of n): the residual add and the next norm in one launch.
/// out[r * ld_out + i] = rms(x[r * ldx + ..]) * w (one block per row of n): a norm into a slice of a wider row.
__global__ void k_rms_ld(const float* x, int ldx, const float* w, float* out, int ld_out, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = out + (size_t) blockIdx.x * ld_out;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
__global__ void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps) {
    __shared__ float sh[kThreads];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) hr[i] = xr[i] * r * w[i];
}
/// The MoE block's end for one token row (block per row): x += sum_k w_k part_k (in pick order, FMA after the first
/// product, as native_moe_combine) + sigmoid(sgate) * shared; then, with `wn`, h = rms(x) * wn (the next layer's
/// attention norm).  Replaces combine + gate + add + rms.
__global__ void k_moe_out(float* x, const float* parts, const int32_t* prow, const float* wts, int K_,
                          const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                          float eps) {
    __shared__ float sh[kThreads];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if (threadIdx.x < K_)   // pick k's row: t * K + k, or its sorted row (`prow`, the prompt's MMQ order)
        pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        s += v * v;
    }
    if (wn == nullptr) return;
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[(size_t) t * n + i] = xr[i] * r * wn[i];
}
/// GDN alpha and beta for T <= 8 rows in one launch, their epilogues applied: rows [0, HV) of the output are
/// softplus(alpha . x + dt) * ssm_a, rows [HV, 2 HV) sigmoid(beta . x).  Warp per row (grid (ceil(2 HV / 8), T)).
__global__ void k_gdn_ab(const float* Wa, const float* Wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int n_in, int HV) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= 2 * HV) return;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane != 0) return;
    if (isb) beta[(size_t) blockIdx.y * HV + r] = sigm(acc);
    else alpha[(size_t) blockIdx.y * HV + r] = softplus(acc + dt[r]) * ssm_a[r];
}
__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in a multiple of 128: one warp per row, float4 loads, 8 rows per
/// block (grid (ceil(n_out / 8), T)).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
/// GDN gate / beta in place for T rows of HV: alpha -> softplus(alpha + dt) * ssm_a, beta -> sigmoid(beta).
__global__ void k_gdn_gb(float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HV * T) return;
    const int h = i % HV;
    alpha[i] = softplus(alpha[i] + dt[h]) * ssm_a[h];
    beta[i] = sigm(beta[i]);
}
/// sh[t] *= sigmoid(gate[t]) (the shared expert's scalar gate), rows of n.
__global__ void k_gate_rows(float* sh, const float* gate, int n, int T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * T) return;
    sh[i] *= sigm(gate[i / n]);
}

/// Batch rows (several sessions decoded in one window, one token each): row t reads and writes the cache of session
/// slot[t] at cell[t]; its rotary position is cell[t] + delta[t].  Every field null: one session, the rows are its
/// consecutive cells *pos0 + t (the prompt, a verify window, a decode step).
struct RowSel {
    const int32_t* cell = nullptr;   // device [T]
    const int32_t* slot = nullptr;   // device [T]
    const int32_t* delta = nullptr;  // device [T]
    float* const* kc = nullptr;      // device [n_sessions]: this layer's K / V base of each session
    float* const* vc = nullptr;
};

/// The KV cache's int8 mode (--kv int8 / STRATA_Q36_KV=int8): kKVG values share one FP16 scale (the group's absmax /
/// 127), so a cache cell is 1 byte per value plus 2 bytes per 64 (with FP32 it is 4 bytes per value).  HD needs to be a
/// multiple of kKVG (256 and 128 are; the host falls back to FP32 otherwise).  An int8 store is ONE allocation, its
/// base pointer passed (and kept in the sessions' tables) like a FP32 store's: [kvh][ctx][HD] int8 codes, then
/// [kvh][ctx][HD / kKVG] FP16 scales, so the session / batch-row plumbing (RowSel, select, the graphs) is unchanged.
constexpr int kKVG = 64;
__host__ __device__ inline size_t kv8_codes(int HKV, int ctx, int HD) { return (size_t) HKV * ctx * HD; }
template <class P> __host__ __device__ inline auto kv8_scales(P* base, int HKV, int ctx, int HD) {
    using H = std::conditional_t<std::is_const_v<P>, const __half, __half>;
    using B = std::conditional_t<std::is_const_v<P>, const int8_t, int8_t>;
    return reinterpret_cast<H*>(reinterpret_cast<B*>(base) + kv8_codes(HKV, ctx, HD));
}

/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache, the gate kept.
/// Rotary position of a cell: the cell itself (text) plus *rdelta (what the pictures before a text run did not use
/// of their cells, see mrope), or, with `mtab` (a request's prompt cells, device int32 [cell][3] = t, h, w), the
/// interleaved M-RoPE position (llama.cpp's IMROPE with sections 11 / 11 / 10: pair p takes stream p % 3).
/// grid (H + HKV, T), block 256 (HD <= 256).  KI8 / VI8 (--kv): the K / V cells as kKVG-value groups of int8 codes and
/// one FP16 scale (see kKVG for the store's layout); <false, false> is the FP32 kernel as before.  The stores are
/// independent: --kv k8v32 runs KI8 with a FP32 V.
template <bool KI8, bool VI8>
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, float* kc, float* vc, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, const int32_t* rdelta,
                            const int32_t* mtab, RowSel rs) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    __shared__ float red[(KI8 || VI8) ? kThreads / 32 : 1];   // the warps' partial maxima (int8 stores)
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = rs.cell ? rs.cell[t] : *pos0 + t;
    const int dl = rs.delta ? rs.delta[t] : (rdelta ? *rdelta : 0);
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = threadIdx.x;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        const int rp = mtab ? __ldg(mtab + (size_t) pos * 3 + p % 3) : pos + dl;
        const float theta = (float) rp * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q[((size_t) t * H + head) * HD + i] = v;
        gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
    } else {
        const int kh = head - H;
        if constexpr (KI8) {
            const int NG = HD / kKVG;
            int8_t* kp = ((int8_t*) kc) + ((size_t) kh * ctx + pos) * HD;
            __half* ksp = kv8_scales(kc, HKV, ctx, HD) + ((size_t) kh * ctx + pos) * NG;
            const int g = i / kKVG, w0 = g * 2;   // kKVG = 2 warps
            float am = fabsf(v);
            for (int m = 16; m > 0; m >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m, 32));
            if ((i & 31) == 0) red[i >> 5] = am;
            __syncthreads();
            const float gk = fmaxf(red[w0], red[w0 + 1]);
            const float sk = gk > 0.f ? gk / 127.0f : 1.0f;
            kp[i] = (int8_t) __float2int_rn(fminf(fmaxf(v / sk, -127.0f), 127.0f));
            if ((i & (kKVG - 1)) == 0) ksp[g] = __float2half(sk);
            __syncthreads();   // red read out before the V store's round overwrites it
        } else {
            kc[((size_t) kh * ctx + pos) * HD + i] = v;
        }
        if constexpr (VI8) {
            const int NG = HD / kKVG;
            int8_t* vp = ((int8_t*) vc) + ((size_t) kh * ctx + pos) * HD;
            __half* vsp = kv8_scales(vc, HKV, ctx, HD) + ((size_t) kh * ctx + pos) * NG;
            const int g = i / kKVG, w0 = g * 2;
            const float vi = vx[((size_t) t * HKV + kh) * HD + i];
            float am = fabsf(vi);
            for (int m = 16; m > 0; m >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, m, 32));
            if ((i & 31) == 0) red[i >> 5] = am;
            __syncthreads();
            const float gv = fmaxf(red[w0], red[w0 + 1]);
            const float sv = gv > 0.f ? gv / 127.0f : 1.0f;
            vp[i] = (int8_t) __float2int_rn(fminf(fmaxf(vi / sv, -127.0f), 127.0f));
            if ((i & (kKVG - 1)) == 0) vsp[g] = __float2half(sv);
        } else {
            vc[((size_t) kh * ctx + pos) * HD + i] = vx[((size_t) t * HKV + kh) * HD + i];
        }
    }
}

/// Split-K attention, one block per (chunk of 256 cells, query row): query row qi = t * H + h (t from t0) sees the
/// cells [0, *pos0 + t].  Writes the chunk's partial accumulator, max and sum; chunks past the query's cells write
/// nothing (the merge stops before them).  grid (n_chunk, n_rows), block 256, HD == 256.  KI8 / VI8 (--kv): the K / V
/// cells dequantized with their group's FP16 scale on the fly (see kKVG for the store's layout).  The int8 reads are
/// vectorized (one uint2 per 8 K codes, one scale per kKVG group) and the V weights are pre-multiplied into shared
/// memory once per block.  <false, false> is the FP32 kernel as before.
template <bool KI8, bool VI8>
__global__ void k_attn_part(const float* q, const float* kc, const float* vc, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    __shared__ float wv[VI8 ? HD / kKVG : 1][VI8 ? kChunkCells : 1];   // VI8: sc[k] * the V group's scale, once per block
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
    if (p < npos) {
        float a = 0.f;
        if constexpr (KI8) {
            const int8_t* kp = ((const int8_t*) kc) + ((size_t) kvh * ctx + p) * HD;
            const __half* ksp = kv8_scales(kc, HKV, ctx, HD) + ((size_t) kvh * ctx + p) * (HD / kKVG);
#pragma unroll 4
            for (int i0 = 0; i0 < HD; i0 += 8) {   // one uint2 (8 int8 codes) and one scale per kKVG group
                const uint2 raw = *reinterpret_cast<const uint2*>(kp + i0);
                const int8_t* c8 = reinterpret_cast<const int8_t*>(&raw);
                const float sg = __half2float(ksp[i0 / kKVG]);
#pragma unroll
                for (int x = 0; x < 8; ++x) a = fmaf(qs[i0 + x] * sg, (float) c8[x], a);
            }
        } else {
            const float* kp = kc + ((size_t) kvh * ctx + p) * HD;
            for (int i = 0; i < HD; ++i) a += qs[i] * kp[i];
        }
        s = a * scale;
    }
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = min(kChunkCells, npos - p0);
    float acc = 0.f;
    if constexpr (VI8) {
        const int8_t* vp = ((const int8_t*) vc) + ((size_t) kvh * ctx + p0) * HD;
        const __half* vsp = kv8_scales(vc, HKV, ctx, HD) + ((size_t) kvh * ctx + p0) * (HD / kKVG);
        __syncthreads();   // sc[] is complete; the weight table takes its per-cell values now
        for (int x = j; x < (HD / kKVG) * n; x += kThreads) {
            const int g = x / n, k = x - g * n;
            wv[g][k] = sc[k] * __half2float(vsp[(size_t) k * (HD / kKVG) + g]);
        }
        __syncthreads();
        const float* wr = wv[j / kKVG];
        const int8_t* vj = vp + j;
        for (int k = 0; k < n; ++k) acc = fmaf(wr[k], (float) vj[(size_t) k * HD], acc);
    } else {
        const float* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
        for (int k = 0; k < n; ++k) acc += sc[k] * vp[(size_t) k * HD];
    }
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// The chunks merged, times sigmoid(gate): grid (n_rows), block HD.
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const float* gate,
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out, RowSel rs) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    const int nc = (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    out[gi] = num / den * sigm(gate[gi]);
}

/// 4 int8 codes (one 32-bit word) as 4 floats, exactly: each byte + 128 in the low mantissa bits of 2^23, minus
/// 2^23 + 128 (a byte permute and a float add per value, in place of the conversion unit's quarter-rate I2F)
__device__ __forceinline__ void i8x4_f32(uint32_t w, float* f) {
    const uint32_t x = w ^ 0x80808080u;
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = __uint_as_float(__byte_perm(x, 0x4B000000u, 0x7440u | i)) - 8388736.0f;
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define STRATA_Q36_GQA_MMA 1
/// 4 int8 codes as two exact FP16 pairs (bytes 0, 1 and bytes 2, 3): each byte + 128 in the mantissa of 1024, minus 1152
__device__ __forceinline__ void i8x4_h2(uint32_t w, uint32_t& lo, uint32_t& hi) {
    const uint32_t x = w ^ 0x80808080u;
    const __half2 off = __half2half2(__ushort_as_half((unsigned short) 0xE480u));   // -1152
    const uint32_t l = __byte_perm(x, 0x64646464u, 0x5140u), h = __byte_perm(x, 0x64646464u, 0x7362u);
    const __half2 a = __hadd2(*reinterpret_cast<const __half2*>(&l), off), c = __hadd2(*reinterpret_cast<const __half2*>(&h), off);
    lo = *reinterpret_cast<const uint32_t*>(&a);
    hi = *reinterpret_cast<const uint32_t*>(&c);
}
/// Two int8 codes (bytes 0 and 2 of r, the other bytes ignored) as an exact FP16 pair: each byte + 128 in the mantissa
/// of 1024, minus 1152
__device__ __forceinline__ uint32_t i8x2_h2(uint32_t r) {
    const uint32_t x = (r & 0x00FF00FFu) ^ 0x64806480u;
    const __half2 v = __hadd2(*reinterpret_cast<const __half2*>(&x), __half2half2(__ushort_as_half((unsigned short) 0xE480u)));
    return *reinterpret_cast<const uint32_t*>(&v);   // (0xE480: -1152)
}
/// x as an FP16 pair hi + lo (hi its FP16 rounding, lo the FP16 rounding of the rest: 22 bits)
__device__ __forceinline__ void f2_h2x2(float a, float b, uint32_t& hi, uint32_t& lo) {
    const __half2 h = __floats2half2_rn(a, b);
    const __half2 l = __floats2half2_rn(a - __low2float(h), b - __high2float(h));
    hi = *reinterpret_cast<const uint32_t*>(&h);
    lo = *reinterpret_cast<const uint32_t*>(&l);
}
/// c += A (16 x 16 FP16, row) * B (16 x 8 FP16, col), FP32
__device__ __forceinline__ void mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#else
#define STRATA_Q36_GQA_MMA 0
#endif
/// STRATA_Q36_ATTN_GQA=1: the split-K decode attention as one block per (query token, KV head, chunk of cc cells):
/// the block reads each K / V cell of its chunk once and computes the kGqaG query heads that share the KV head (the
/// block per query row of k_attn_part reads the chunk once per head, a thread per cell walking its 256 values).  The
/// chunk walks in tiles of kGqaTile cells: the tile's scores (8 heads x its cells) in shared memory, an online softmax
/// per head (warp h keeps head h's running max and sum, the sums so far rescaled), then the tile's P . V.  Writes
/// k_attn_part's partials (sum, max, exp-sum) per (row, chunk) for chunks of cc cells (a kGqaTile multiple);
/// k_attn_merge_cc merges them.  grid (nt, HKV, n_chunks): the blocks of one chunk are adjacent, so a window's rows
/// find the chunk in L2.  HD == 256, H / HKV == kGqaG.
///   int8 K / V on sm_80+ (most of the arithmetic on the tensor cores; FP32 sums of exact products):
///   * Q . K: S^T (16 cells x 8 heads) = K (16 cells x 16 values, the codes exact in FP16) * Q^T per mma, each query
///     value as an FP16 pair hi + lo (f2_h2x2) in two mmas, one 64-value scale group per 4 k steps (its FP32 sum times
///     the cell's scale).  The k order inside a group (both operands): lane (g, tq) = (lane / 4, lane % 4) of k step
///     ks takes the values 16 tq + 4 ks + 0, 1 (its k 2 tq, 2 tq + 1) and + 2, 3 (its k 2 tq + 8, + 9), so a lane's
///     codes of a group are one 16-byte load.
///   * P . V: out^T (16 values x 8 heads) = V^T (16 values x 16 cells, the codes) * (P x the cell's V scale)^T per
///     mma, the weights (P x V scale) as FP16 pairs hi + lo times 2^E (E per tile and scale group, from the group's
///     largest V scale alone: that scale times 2^E in [2^13, 2^14), so the dominant weights are normal FP16; weights
///     far below the running max, or of cells with a much smaller V scale, may still be FP16 subnormal or round to 0,
///     which the exact FP32 exp-sum makes negligible; 2^-E after).  Warp w owns the values 32 w .. + 31 (one scale
///     group) for every cell of the tile; lane (g, tq) loads the 4 values 32 w + 4 g .. + 3 of the cells 4 tq .. + 3
///     (its k 2 tq, 2 tq + 1, 2 tq + 8, 2 tq + 9), two mma tiles (rows: the values 32 w + 4 g + 2 t + 0 / + 1 of tile
///     t).
///   FP32 K / V (and int8 before sm_80): CUDA-core FP32 arithmetic.  Q . K: a warp per cell, each lane 8 of its
///   values (lane * 4 .. + 3 and 128 + lane * 4 .. + 3: two float4 / two words of int8 codes and their scales), the
///   heads' 8 partial sums added across the warp in one transposed butterfly (9 shuffles for the 8 sums), kGqaCpw
///   cells at a time per warp.  P . V: 64 threads per cell, 4 values each (one float4 / one word of codes), 4 cells
///   side by side, all heads' sums in registers, the 4 cell groups added in shared memory at the end.
constexpr int kGqaG = 8, kGqaTile = 256, kGqaCpw = 2;
template <bool KI8, bool VI8>
__global__ void __launch_bounds__(kThreads, 2) k_attn_gqa(const float* q, const float* kc, const float* vc, const int32_t* pos0,
                                                       int t0, int H, int HKV, int ctx, float scale, int n_chunk, int cc,
                                                       float* pacc, float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256, G = kGqaG, TL = kGqaTile, NG = HD / kKVG;
    [[maybe_unused]] constexpr bool MK = STRATA_Q36_GQA_MMA && KI8, MV = STRATA_Q36_GQA_MMA && VI8;   // Q . K / P . V on the tensor cores
    constexpr int LS = TL + 16;   // MV: a head's row of weights (padded: the B loads' 8 heads in distinct banks)
    // the tile's scores, then its weights exp(s - m): MV head-major [G][LS], else cell-major [TL][G]
    __shared__ __align__(16) float spb[MV ? G * LS : TL * G];
    __shared__ __align__(16) float qs[G * HD];                           // the queries (MK: their B fragments)
    __shared__ __align__(16) float red[MV ? 1 : 2][G][MV ? 4 : HD];      // !MV: the cell groups' sums added
    __shared__ __align__(16) float svs[MV ? NG : 1][MV ? TL : 4];       // MV: the cells' V scales times 2^E
    __shared__ float smx[kThreads / 32][NG];                             // MV: each warp's largest V scale per group
    __shared__ float mh[G], lh[G], ah[G];
    auto S = [&](int k, int h) -> float& { return MV ? spb[h * LS + k] : spb[k * G + h]; };
    const int tr = blockIdx.x, kvh = blockIdx.y, c = blockIdx.z;
    const int t = t0 + tr;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = rs.kc[rs.slot[t]];
        vc = rs.vc[rs.slot[t]];
    }
    const int p0 = c * cc;
    if (p0 >= npos) return;
    const int pend = min(p0 + cc, npos);
    const int j = threadIdx.x, lane = j & 31, w = j >> 5, g = lane >> 2, tq = lane & 3;
    const int qa = lane * 4, qb = 128 + lane * 4;   // !MK: a lane's values of a cell
    const float* qh = q + ((size_t) t * H + kvh * G) * HD;
#if STRATA_Q36_GQA_MMA
    if constexpr (MK) {   // qs: [group][ks][lane] = {hi(0, 1), hi(2, 3), lo(0, 1), lo(2, 3)} of head g
        uint4* qf = reinterpret_cast<uint4*>(qs);
        for (int x = j; x < NG * 4 * 32; x += kThreads) {
            const int l = x & 31, ks = (x >> 5) & 3, gb = x >> 7;
            const float4 v = *reinterpret_cast<const float4*>(qh + (size_t) (l >> 2) * HD + gb * kKVG + 16 * (l & 3) + 4 * ks);
            uint4 f;
            f2_h2x2(v.x, v.y, f.x, f.z);
            f2_h2x2(v.z, v.w, f.y, f.w);
            qf[x] = f;
        }
    } else
#endif
    {
        for (int x = j; x < G * HD / 4; x += kThreads) reinterpret_cast<float4*>(qs)[x] = reinterpret_cast<const float4*>(qh)[x];
    }
    if (j < G) {
        mh[j] = -INFINITY;
        lh[j] = 0.f;
    }
    __syncthreads();
    const int vg = j >> 6, d0 = (j & 63) * 4;   // !MV P . V: this thread's cell group and its 4 values
    float acc[MV ? 4 : G][4];                   // MV: [tile t * 2 + head 2 tq / 2 tq + 1][..], see the end
#pragma unroll
    for (int h = 0; h < (MV ? 4 : G); ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.f;
    const size_t cb = (size_t) kvh * ctx;   // the KV head's first cell
    for (int tp = p0; tp < pend; tp += TL) {
        const int n = min(TL, pend - tp), n16 = (n + 15) & ~15;
        float sv[NG];   // MV: thread j's cell's V scales
        if constexpr (MV) {
            const uint2 r = j < n ? *reinterpret_cast<const uint2*>(kv8_scales(vc, HKV, ctx, HD) + (cb + tp + j) * NG) : make_uint2(0, 0);
            const __half* h4 = reinterpret_cast<const __half*>(&r);
#pragma unroll
            for (int gb = 0; gb < NG; ++gb) sv[gb] = __half2float(h4[gb]);
        }
#if STRATA_Q36_GQA_MMA
        if constexpr (MK) {   // warp w: the 16-cell rows w, w + 8 of the tile
            const uint4* qf = reinterpret_cast<const uint4*>(qs);
            for (int m0 = w * 16; m0 < n; m0 += 16 * (kThreads / 32)) {
                const int r0 = m0 + g, r1 = m0 + g + 8;
                const size_t p0c = cb + tp + min(r0, n - 1), p1c = cb + tp + min(r1, n - 1);   // (past the tile: not stored)
                const int8_t* k0p = ((const int8_t*) kc) + p0c * HD + 16 * tq;
                const int8_t* k1p = ((const int8_t*) kc) + p1c * HD + 16 * tq;
                const __half* ksp = kv8_scales(kc, HKV, ctx, HD);
                uint4 c0[NG], c1[NG];
                float s0[NG], s1[NG];
#pragma unroll
                for (int gb = 0; gb < NG; ++gb) {
                    c0[gb] = *reinterpret_cast<const uint4*>(k0p + gb * kKVG);
                    c1[gb] = *reinterpret_cast<const uint4*>(k1p + gb * kKVG);
                    s0[gb] = __half2float(ksp[p0c * NG + gb]);
                    s1[gb] = __half2float(ksp[p1c * NG + gb]);
                }
                float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int gb = 0; gb < NG; ++gb) {
                    float e[4] = {0.f, 0.f, 0.f, 0.f};
                    const uint32_t w0[4] = {c0[gb].x, c0[gb].y, c0[gb].z, c0[gb].w};
                    const uint32_t w1[4] = {c1[gb].x, c1[gb].y, c1[gb].z, c1[gb].w};
#pragma unroll
                    for (int ks = 0; ks < 4; ++ks) {
                        // codes 0, 1 / 2, 3 of the word: k 2 tq, 2 tq + 1 / 2 tq + 8, + 9 (rows r0: a0, a2; r1: a1, a3)
                        uint32_t a[4];
                        i8x4_h2(w0[ks], a[0], a[2]);
                        i8x4_h2(w1[ks], a[1], a[3]);
                        const uint4 bq = qf[(gb * 4 + ks) * 32 + lane];
                        mma_f16(e, a, bq.z, bq.w);   // lo, then hi
                        mma_f16(e, a, bq.x, bq.y);
                    }
                    d[0] = fmaf(e[0], s0[gb], d[0]), d[1] = fmaf(e[1], s0[gb], d[1]);
                    d[2] = fmaf(e[2], s1[gb], d[2]), d[3] = fmaf(e[3], s1[gb], d[3]);
                }
                if (r0 < n) S(r0, 2 * tq) = d[0] * scale, S(r0, 2 * tq + 1) = d[1] * scale;
                if (r1 < n) S(r1, 2 * tq) = d[2] * scale, S(r1, 2 * tq + 1) = d[3] * scale;
            }
        } else
#endif
        // a warp's cells kGqaCpw at a time (k, k + 8, ...: each query value read from shared memory serves them all)
        for (int k0 = w; k0 < n; k0 += kGqaCpw * (kThreads / 32)) {
            float kv[kGqaCpw][8], sa[kGqaCpw], sb[kGqaCpw];
#pragma unroll
            for (int u = 0; u < kGqaCpw; ++u) {
                const size_t p = cb + tp + min(k0 + u * (kThreads / 32), n - 1);   // (a cell past the tile: computed, not stored)
                if constexpr (KI8) {
                    i8x4_f32(*reinterpret_cast<const uint32_t*>(((const int8_t*) kc) + p * HD + qa), kv[u]);
                    i8x4_f32(*reinterpret_cast<const uint32_t*>(((const int8_t*) kc) + p * HD + qb), kv[u] + 4);
                    const __half* ks = kv8_scales(kc, HKV, ctx, HD) + p * NG;
                    sa[u] = __half2float(ks[qa / kKVG]);
                    sb[u] = __half2float(ks[qb / kKVG]);
                } else {
                    const float4 ra = *reinterpret_cast<const float4*>(kc + p * HD + qa);
                    const float4 rb = *reinterpret_cast<const float4*>(kc + p * HD + qb);
                    kv[u][0] = ra.x, kv[u][1] = ra.y, kv[u][2] = ra.z, kv[u][3] = ra.w;
                    kv[u][4] = rb.x, kv[u][5] = rb.y, kv[u][6] = rb.z, kv[u][7] = rb.w;
                    sa[u] = sb[u] = 1.f;
                }
            }
            float s[kGqaCpw][G];
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 xa = *reinterpret_cast<const float4*>(qs + h * HD + qa);
                const float4 xb = *reinterpret_cast<const float4*>(qs + h * HD + qb);
#pragma unroll
                for (int u = 0; u < kGqaCpw; ++u) {
                    float a = xa.x * kv[u][0];
                    a = fmaf(xa.y, kv[u][1], a), a = fmaf(xa.z, kv[u][2], a), a = fmaf(xa.w, kv[u][3], a);
                    float b = xb.x * kv[u][4];
                    b = fmaf(xb.y, kv[u][5], b), b = fmaf(xb.z, kv[u][6], b), b = fmaf(xb.w, kv[u][7], b);
                    s[u][h] = KI8 ? fmaf(a, sa[u], b * sb[u]) : a + b;
                }
            }
            // the transposed butterfly: lane bits 4, 3, 2 pick the head a lane keeps, bits 1, 0 add the rest
            const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
#pragma unroll
            for (int u = 0; u < kGqaCpw; ++u) {
                float* su = s[u];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float snd = b4 ? su[i] : su[i + 4], kp = b4 ? su[i + 4] : su[i];
                    su[i] = kp + __shfl_xor_sync(0xffffffffu, snd, 16, 32);
                }
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const float snd = b3 ? su[i] : su[i + 2], kp = b3 ? su[i + 2] : su[i];
                    su[i] = kp + __shfl_xor_sync(0xffffffffu, snd, 8, 32);
                }
                {
                    const float snd = b2 ? su[0] : su[1], kp = b2 ? su[1] : su[0];
                    su[0] = kp + __shfl_xor_sync(0xffffffffu, snd, 4, 32);
                }
                su[0] += __shfl_xor_sync(0xffffffffu, su[0], 2, 32);
                su[0] += __shfl_xor_sync(0xffffffffu, su[0], 1, 32);
                const int k = k0 + u * (kThreads / 32);
                if ((lane & 3) == 0 && k < n) S(k, lane >> 2) = su[0] * scale;
            }
        }
        if constexpr (MV) {   // each warp's largest V scale per group
#pragma unroll
            for (int gb = 0; gb < NG; ++gb) {
                float mv = sv[gb];
                for (int o = 16; o > 0; o >>= 1) mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, o, 32));
                if (lane == 0) smx[w][gb] = mv;
            }
        }
        __syncthreads();
        {   // the online softmax: warp w for head w (MV: the weights past the tile up to n16 zero)
            float mx = -INFINITY;
            for (int k = lane; k < n; k += 32) mx = fmaxf(mx, S(k, w));
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o, 32));
            const float mo = mh[w], mn = fmaxf(mo, mx);
            float su = 0.f;
            for (int k = lane; k < (MV ? n16 : n); k += 32) {
                const float e = k < n ? expf(S(k, w) - mn) : 0.f;
                S(k, w) = e;
                su += e;
            }
            for (int o = 16; o > 0; o >>= 1) su += __shfl_xor_sync(0xffffffffu, su, o, 32);
            if constexpr (MV) {   // thread j's cell's V scales times its group's 2^E
#pragma unroll
                for (int gb = 0; gb < NG; ++gb) {
                    float mv = smx[0][gb];
                    for (int x = 1; x < kThreads / 32; ++x) mv = fmaxf(mv, smx[x][gb]);
                    int ex;
                    frexpf(mv, &ex);   // mv in [2^(ex - 1), 2^ex)
                    svs[gb][j] = ldexpf(sv[gb], 14 - ex);
                }
            }
            __syncthreads();   // every warp has read mh[] for its head before lane 0 moves it
            if (lane == 0) {
                const float al = expf(mo - mn);   // 0 on the first tile (mo = -inf)
                ah[w] = al;
                mh[w] = mn;
                lh[w] = lh[w] * al + su;
            }
        }
        __syncthreads();
#if STRATA_Q36_GQA_MMA
        if constexpr (MV) {
            const int gw = w >> 1;   // the warp's scale group
            float mv = smx[0][gw];
            for (int x = 1; x < kThreads / 32; ++x) mv = fmaxf(mv, smx[x][gw]);
            int ex;
            frexpf(mv, &ex);
            const float inv = ldexpf(1.f, ex - 14);
            float e[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            const int8_t* vb = ((const int8_t*) vc) + (cb + tp) * HD + 32 * w + 4 * g;
            for (int k0 = 0; k0 < n; k0 += 16) {
                const int kq = k0 + 4 * tq;   // the lane's cells kq .. + 3
                uint32_t x[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) x[i] = kq + i < n ? *reinterpret_cast<const uint32_t*>(vb + (size_t) (kq + i) * HD) : 0u;
                const float4 pw = *reinterpret_cast<const float4*>(&spb[g * LS + kq]);
                const float4 ps = *reinterpret_cast<const float4*>(&svs[gw][kq]);
                uint32_t bh0, bl0, bh1, bl1;
                f2_h2x2(pw.x * ps.x, pw.y * ps.y, bh0, bl0);
                f2_h2x2(pw.z * ps.z, pw.w * ps.w, bh1, bl1);
#pragma unroll
                for (int tt = 0; tt < 2; ++tt) {   // value 2 tt (rows g) and 2 tt + 1 (rows g + 8) of the lane's 4
                    const uint32_t a[4] = {i8x2_h2(__byte_perm(x[0], x[1], 0x0400u + 0x0202u * tt)),
                                           i8x2_h2(__byte_perm(x[0], x[1], 0x0501u + 0x0202u * tt)),
                                           i8x2_h2(__byte_perm(x[2], x[3], 0x0400u + 0x0202u * tt)),
                                           i8x2_h2(__byte_perm(x[2], x[3], 0x0501u + 0x0202u * tt))};
                    mma_f16(e[tt], a, bl0, bl1);   // lo, then hi
                    mma_f16(e[tt], a, bh0, bh1);
                }
            }
            const float a0 = ah[2 * tq], a1 = ah[2 * tq + 1];
#pragma unroll
            for (int tt = 0; tt < 2; ++tt) {   // acc[2 tt]: head 2 tq (rows g, g + 8), acc[2 tt + 1]: head 2 tq + 1
                acc[2 * tt][0] = fmaf(acc[2 * tt][0], a0, e[tt][0] * inv);
                acc[2 * tt][1] = fmaf(acc[2 * tt][1], a0, e[tt][2] * inv);
                acc[2 * tt + 1][0] = fmaf(acc[2 * tt + 1][0], a1, e[tt][1] * inv);
                acc[2 * tt + 1][1] = fmaf(acc[2 * tt + 1][1], a1, e[tt][3] * inv);
            }
        } else
#endif
        {
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float al = ah[h];
                acc[h][0] *= al, acc[h][1] *= al, acc[h][2] *= al, acc[h][3] *= al;
            }
#pragma unroll 4
            for (int k = vg; k < n; k += 4) {
                const size_t p = cb + tp + k;
                float v[4];
                if constexpr (VI8) {
                    i8x4_f32(*reinterpret_cast<const uint32_t*>(((const int8_t*) vc) + p * HD + d0), v);
                    const float sg = __half2float(kv8_scales(vc, HKV, ctx, HD)[p * NG + d0 / kKVG]);
                    v[0] *= sg, v[1] *= sg, v[2] *= sg, v[3] *= sg;
                } else {
                    const float4 r = *reinterpret_cast<const float4*>(vc + p * HD + d0);
                    v[0] = r.x, v[1] = r.y, v[2] = r.z, v[3] = r.w;
                }
                const float4 pa = *reinterpret_cast<const float4*>(&spb[k * G]);
                const float4 pb = *reinterpret_cast<const float4*>(&spb[k * G + 4]);
                const float ph[G] = {pa.x, pa.y, pa.z, pa.w, pb.x, pb.y, pb.z, pb.w};
#pragma unroll
                for (int h = 0; h < G; ++h)
#pragma unroll
                    for (int x = 0; x < 4; ++x) acc[h][x] = fmaf(ph[h], v[x], acc[h][x]);
            }
        }
        __syncthreads();   // the weights read out before the next tile's scores
    }
    const size_t rb = (size_t) tr * H + kvh * G;   // the block's first query row (k_attn_part's row = (t - t0) * H + h)
    if constexpr (MV) {   // lane (g, tq): the values 32 w + 4 g .. + 3 of the heads 2 tq, 2 tq + 1
        const int dv = 32 * w + 4 * g;
        *reinterpret_cast<float4*>(pacc + ((rb + 2 * tq) * n_chunk + c) * HD + dv) = make_float4(acc[0][0], acc[0][1], acc[2][0], acc[2][1]);
        *reinterpret_cast<float4*>(pacc + ((rb + 2 * tq + 1) * n_chunk + c) * HD + dv) = make_float4(acc[1][0], acc[1][1], acc[3][0], acc[3][1]);
    } else {   // the 4 cell groups added: 2 + 3 into 0 + 1, then 1 into 0
        if (vg >= 2)
#pragma unroll
            for (int h = 0; h < G; ++h) *reinterpret_cast<float4*>(&red[vg - 2][h][d0]) = make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
        __syncthreads();
        if (vg < 2)
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 r = *reinterpret_cast<const float4*>(&red[vg][h][d0]);
                acc[h][0] += r.x, acc[h][1] += r.y, acc[h][2] += r.z, acc[h][3] += r.w;
            }
        __syncthreads();
        if (vg == 1)
#pragma unroll
            for (int h = 0; h < G; ++h) *reinterpret_cast<float4*>(&red[0][h][d0]) = make_float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
        __syncthreads();
        if (vg == 0)
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 r = *reinterpret_cast<const float4*>(&red[0][h][d0]);
                const size_t o = (rb + h) * n_chunk + c;
                *reinterpret_cast<float4*>(pacc + o * HD + d0) =
                    make_float4(acc[h][0] + r.x, acc[h][1] + r.y, acc[h][2] + r.z, acc[h][3] + r.w);
            }
    }
    if (j < G) {
        const size_t o = (rb + j) * n_chunk + c;
        pm[o] = mh[j];
        pl[o] = lh[j];
    }
}
/// k_attn_merge for chunks of cc cells (k_attn_gqa's partials): grid (n_rows), block HD.
__global__ void k_attn_merge_cc(const float* pacc, const float* pm, const float* pl, const float* gate,
                                const int32_t* pos0, int t0, int H, int n_chunk, int cc, float* out, RowSel rs) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    const int nc = (npos + cc - 1) / cc;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    out[gi] = num / den * sigm(gate[gi]);
}

/// The routed entries (T tokens x K picks, entry e = t * K + k) grouped per expert, on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order.  One block.
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// k_group for the hybrid mode: only the entries whose expert has a VRAM slot (slot_of[x] >= 0) are grouped, each
/// group's blob at blobs + slot * blob; the others are the CPU pool's (their rows come from k_merge_cpu).  One block.
__global__ void k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) slot_of[x] * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}
/// The prompt attention's FP16 operands: cells [0, P) of each KV head's K / V (cache rows ctx apart) into compact
/// [kvh][P][HD] halves.  grid1(HKV * P * HD).  KI8 / VI8 (--kv): the codes dequantized with their group's FP16 scale
/// (see kKVG); <false, false> is the FP32 kernel as before.
template <bool KI8, bool VI8>
__global__ void k_kv_f16(const float* kc, const float* vc, __half* k16, __half* v16, int ctx, int P, int HD, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per, rest = i % per;
    const size_t src = (size_t) kvh * ctx * HD + rest;
    if constexpr (KI8 || VI8) {
        const int64_t p = rest / HD, d = rest % HD;
        const size_t si = (size_t) kvh * ctx * (HD / kKVG) + (size_t) p * (HD / kKVG) + d / kKVG;
        if constexpr (KI8) k16[i] = __float2half(__half2float(kv8_scales(kc, HKV, ctx, HD)[si]) * (float) ((const int8_t*) kc)[src]);
        else k16[i] = __float2half(kc[src]);
        if constexpr (VI8) v16[i] = __float2half(__half2float(kv8_scales(vc, HKV, ctx, HD)[si]) * (float) ((const int8_t*) vc)[src]);
        else v16[i] = __float2half(vc[src]);
    } else {
        k16[i] = __float2half(kc[src]);
        v16[i] = __float2half(vc[src]);
    }
}
/// The prompt's causal softmax: row (h, t) of S (ld floats, P cells) -> softmax(S * scale) over the cells
/// [0, pos0 + t], 0 past them, as FP16 probabilities (S16, the same row offsets, ld halves).  grid (T, H), block 256.
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}
/// att *= sigmoid(gate) (the attention output gate), n values.
__global__ void k_gate_mul(float* att, const float* gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= sigm(gate[i]);
}
// ---- the prompt attention over the context in blocks of cells (Engine::kv16_blk: an int8 K / V store at a long
// context): the FP16 K / V hold one block, so the softmax runs online over the blocks (flash attention's running max
// and sum, here between cuBLAS GEMMs)
/// k_kv_f16 for cells [c0, c0 + n) of each KV head into compact [kvh][n][HD] halves (the same rounding; KI8 / VI8: the
/// codes dequantized with their group's FP16 scale, see kKVG).  grid1(HKV * n * HD).
template <bool KI8, bool VI8>
__global__ void k_kv_f16_blk(const float* kc, const float* vc, __half* k16, __half* v16, int ctx, int c0, int n, int HD,
                             int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * n * HD) return;
    const int64_t per = (int64_t) n * HD;
    const int64_t kvh = i / per, rest = i % per;
    const int64_t p = c0 + rest / HD, d = rest % HD;
    const size_t src = ((size_t) kvh * ctx + p) * HD + d;
    const size_t si = ((size_t) kvh * ctx + p) * (HD / kKVG) + d / kKVG;
    if constexpr (KI8) k16[i] = __float2half(__half2float(kv8_scales(kc, HKV, ctx, HD)[si]) * (float) ((const int8_t*) kc)[src]);
    else k16[i] = __float2half(kc[src]);
    if constexpr (VI8) v16[i] = __float2half(__half2float(kv8_scales(vc, HKV, ctx, HD)[si]) * (float) ((const int8_t*) vc)[src]);
    else v16[i] = __float2half(vc[src]);
}
/// One block of cells [c0, c0 + n) of the online causal softmax: row (h, t) of S (ld floats) -> FP16 exp(S * scale - M)
/// over the cells <= pos0 + t, 0 past them, M the running max of the row over this block and the ones before (am); the
/// running sum (al) and the output so far (O row t * H + h, HD floats: P V of the blocks before) rescaled to M.  The
/// first block starts them (its P V GEMM writes O with beta 0).  grid (T, H), block 256.
__global__ void k_attn_softmax16_blk(const float* S, __half* S16, int ld, int T, int n, int c0, int pos0, float scale,
                                     float* am, float* al, float* O, int H, int HD, int first) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int nv = max(0, min(n, pos0 + t + 1 - c0));
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < nv; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    const size_t r = (size_t) t * H + h;
    const float m_old = first ? -INFINITY : am[r], l_old = first ? 0.f : al[r];
    const float m_new = fmaxf(m_old, mx);   // finite: the first block holds cell 0, which every row sees
    const float corr = first ? 0.f : expf(m_old - m_new);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) {
        const __half v = __float2half(p < nv ? expf(row[p] * scale - m_new) : 0.f);
        out[p] = v;
        sum += __half2float(v);
    }
    sum = block_sum(sum, sh);
    if (!first && corr != 1.f)
        for (int i = threadIdx.x; i < HD; i += blockDim.x) O[r * HD + i] *= corr;
    if (threadIdx.x == 0) {
        am[r] = m_new;
        al[r] = l_old * corr + sum;
    }
}
/// att *= sigmoid(gate) / l (the online softmax's sum, one per row of hd values), n values.
__global__ void k_gate_mul_l(float* att, const float* gate, const float* l, int hd, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= sigm(gate[i]) / l[i / hd];
}

/// Picture rows over the embeddings of a chunk: x[r] = img[map[r]] for the rows with map[r] >= 0 (the rows of a picture's
/// <|image_pad|> cells).  grid (T), block 256.
__global__ void k_put_rows(float* x, const float* img, const int32_t* map, int n) {
    const int r = blockIdx.x;
    const int m = map[r];
    if (m < 0) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[(size_t) r * n + i] = img[(size_t) m * n + i];
}

/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r
/// holds entry dst[r] (= t * K + k, the row of the per-slot outputs) of token src[r].  The order inside an expert is the
/// atomics' (no output depends on it: every row's products are its own).  cnt / cur: n_expert ints of scratch.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;   // entry e's sorted row: where its down product lands
}

/// The greedy pick of one logits row: out = argmax (lowest id on ties).  Stage 1: `gridDim.x` blocks each write the
/// best of their stride into (pv, pi); stage 2 (one block, n = the stage-1 block count) reads those.
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    // row blockIdx.y: its logits (stage 1, n = the vocabulary) and its kArgBlocks partials
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * 128;
    pi += (size_t) blockIdx.y * 128;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = bi[0];
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}
constexpr int kArgBlocks = 128;   // (the row stride of k_argmax's partials)

unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// out[t][o] = W[o] . x[t] in FP32 for a prompt chunk's small F32 projections (the router, the GDN alpha / beta,
/// the shared expert's gate: n_out <= 256): a (16 TM) x 64 output tile per block, K in steps of 16 through shared
/// memory, TM x 4 outputs per thread (TM = 4 for the router, 1 for the 32-wide alpha / beta: more blocks).  hipBLAS
/// picked 8 x 8 / 16 x 16 tiles for these shapes (~10% of a prompt).  K % 4 == 0.
template <int TM>
__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K, int n_out, int T, int ldo, int kz) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K, kb + kz);   // split-K: this block's K range, its own partial
    out += (size_t) blockIdx.z * T * ldo;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;   // this thread's loads: row lr of each tile, k lk .. lk + 3
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K + k0 + lk);
            As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K + k0 + lk);
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}

/// out[i] = the sum of the split-K partials part[z][i] in order (deterministic), n values each.
__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}

/// x f32 -> FP16 bits (the prompt GEMMs' activations).
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// The GDN conv for T tokens in ONE launch (the prompt): one block per 128-channel head, thread = channel, the
/// 3-entry history carried in registers across the tokens and written back at the end; SiLU, then the L2 norm of the
/// q/k heads (block < qk_heads) per token - fused_gdn_conv_l2's arithmetic, token after token.  History [channel][3].
__global__ void __launch_bounds__(128) k_conv_seq(float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T, bool write,
                                                  float* const* hist_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                  const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    __shared__ float part[4];
    if (hist_tab) {   // batch rows, grid y = group: group g's history is session slot_rows[g]'s; its rows are
                      // grow0[g] .. + gcnt[g] - 1 (one row at row g when there are no group arrays)
        const int g = blockIdx.y, r0 = grow0 ? grow0[g] : g;
        hist = hist_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        qkv += (size_t) r0 * C;
        h += (size_t) r0 * C;
    }
    const int c = blockIdx.x * 128 + threadIdx.x;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    // TILE mode (a prompt chunk: grid y > 1, no batch tables): the tiles of kConvTile tokens run side by side.  A tile's history
    // is the three inputs before its first token (exactly the values the sequential loop carries); tile 0 reads the old history,
    // and, as the only writer, stores the new one: the last three of (old history, inputs).  The arithmetic per token is the same.
    int t_first = 0, t_end = T;
    const bool tiled = hist_tab == nullptr && gridDim.y > 1;
    if (tiled) {
        t_first = (int) blockIdx.y * kConvTile;
        t_end = min(T, t_first + kConvTile);
        if (t_first == 0 && write) {
            const float o0 = v0, o1 = v1, o2 = v2;
            const float old3[3] = {o0, o1, o2};
            for (int k = 0; k < 3; ++k) {
                const int idx = T - 3 + k;
                hist[c * 3 + k] = idx >= 0 ? qkv[(size_t) idx * C + c] : old3[idx + 3];
            }
        }
        if (t_first > 0) {
            v0 = qkv[(size_t) (t_first - 3) * C + c];
            v1 = qkv[(size_t) (t_first - 2) * C + c];
            v2 = qkv[(size_t) (t_first - 1) * C + c];
        }
        if (t_first >= T) return;
    }
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const bool qk = (int) blockIdx.x < qk_heads;
    for (int t = t_first; t < t_end; ++t) {
        const float x = qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (qk) {
            float sq = y * y;
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            const float ss = part[0] + part[1] + part[2] + part[3];
            __syncthreads();
            y *= rsqrtf(ss + eps);
        }
        h[(size_t) t * C + c] = y;
    }
    if (!write || tiled) return;   // a verify window: the history is commit()'s to move; tiles: tile 0 stored it
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

/// The gated delta rule + SiLU-gated norm for T tokens in ONE launch (the prompt): fused_gdn_step_norm's kernel
/// (block = one V head, 128 columns x 4 row groups, state [row][head][col]) with its 32 state rows per thread kept in
/// registers across the tokens.  h (T, C) = q | k | v, gate / beta (T, HV), z / y (T, ZV).
__global__ void __launch_bounds__(512) k_gdn_seq(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write,
                                                 float* const* state_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                 const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    constexpr int S = 128, RG = 4, RPG = 32;
    if (state_tab) {   // batch rows, grid y = group (see k_conv_seq)
        const int g = blockIdx.y, r = grow0 ? grow0[g] : g;
        state = state_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        h += (size_t) r * C;
        gate += (size_t) r * h_v;
        beta += (size_t) r * h_v;
        z += (size_t) r * h_v * S;
        if (y) y += (size_t) r * h_v * S;
    }
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % h_k;
    const int ZV = h_v * S;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            y[(size_t) t * ZV + head * S + col] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
        }
        __syncthreads();   // sk / sq / the reductions are rewritten by the next token
    }
    if (!write) return;   // a verify window: the state is commit()'s to write
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

/// The prompt's gated delta rule with more parallelism: a state column's update needs only that column (kv, delta and
/// the output o all reduce over the rows), so a V head splits into kGdnColGroups blocks of 128 / kGdnColGroups columns,
/// each with 16 row groups of 8 rows (512 threads, 8 state values per thread in registers).  Writes the raw outputs
/// o (T, HV * 128); k_gdn_norm applies the SiLU-gated RMS norm.  grid (HV, kGdnColGroups), block (32, 16).
constexpr int kGdnColGroups = 4;
__global__ void __launch_bounds__(512) k_gdn_seq_cols(float* state, const float* h, int C, const float* gate,
                                                      const float* beta, float* o_raw, int h_k, int h_v, int T) {
    constexpr int S = 128, NC = S / kGdnColGroups, RG = 16, RPG = S / RG;
    // two barriers per token (was three): sk / sq are double-buffered, and the output sum of token t is done after token t + 1's first
    // barrier, off the chain of the recurrence (red_o is read between that barrier and the next one, written after it)
    __shared__ float sk[2][S], sq[2][S];
    __shared__ float red_kv[RG][NC];
    __shared__ float red_o[RG][NC];
    const int head = blockIdx.x, cl = threadIdx.x, rg = threadIdx.y, tid = rg * NC + cl;
    const int col = blockIdx.y * NC + cl;
    const int qh = head % h_k;
    float sv[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) sv[r] = base[r * row_stride];
    // the next token's inputs are loaded while this one is computed (the loads were the head of every iteration)
    float nsk = 0.0f, nsq = 0.0f, nvv = 0.0f, ngate = 0.0f, nbeta = 0.0f;
    auto load = [&](int t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { nsk = ht[h_k * S + qh * S + tid]; nsq = ht[qh * S + tid]; }
        nvv = ht[2 * h_k * S + head * S + col];
        ngate = gate[(size_t) t * h_v + head];
        nbeta = beta[(size_t) t * h_v + head];
    };
    if (T > 0) load(0);
    for (int t = 0; t < T; ++t) {
        const int par = t & 1;
        if (tid < S) { sk[par][tid] = nsk; sq[par][tid] = nsq; }
        const float vcur = nvv, gcur = ngate, bcur = nbeta;
        if (t + 1 < T) load(t + 1);
        __syncthreads();
        if (t > 0 && rg == 0) {   // token t - 1's output
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t) (t - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
        }
        const float g = __expf(gcur);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(sv[r], sk[par][rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (vcur - g * kv_col) * bcur;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = fmaf(g, sv[r], sk[par][rg * RPG + r] * delta);
            o = fmaf(sv[r], sq[par][rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
    }
    __syncthreads();
    if (T > 0 && rg == 0) {   // the last token's output
        float oc = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
        o_raw[(size_t) (T - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = sv[r];
}
/// y = rms(o) * gamma * SiLU(z) per (token, V head): grid (HV, T), block 128.
__global__ void __launch_bounds__(128) k_gdn_norm(const float* o_raw, const float* z, const float* gamma, float eps,
                                                  float* y, int h_v) {
    __shared__ float part[4];
    const size_t i = ((size_t) blockIdx.y * h_v + blockIdx.x) * 128 + threadIdx.x;
    const float oc = o_raw[i];
    float sq = oc * oc;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrtf(ss / 128.0f + eps);
    const float zz = z[i];
    y[i] = oc * scale * gamma[threadIdx.x] * (zz / (1.0f + __expf(-zz)));
}

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool attn = false;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    // GDN
    Mat qkv, zg, ssm_out;
    const float *alpha = nullptr, *beta = nullptr, *dt = nullptr, *ssm_a = nullptr, *conv = nullptr, *ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo;
    const float *q_norm = nullptr, *k_norm = nullptr;
    // MoE
    const float* router = nullptr;   // [n_expert][N] f32
    float* router_ext = nullptr;     // the same with the shared expert's gate row appended: [n_expert + 1][N]
    Mat sh_gate, sh_up, sh_down;
    const float* sh_gate_inp = nullptr;   // [N]
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;   // n_expert blobs, `stride` bytes apart (L.bytes rounded up to whole blocks of every
    size_t stride = 0;          // part's type, so MMQ can step over them by blocks), then kMmqTail zero bytes
                                // (hybrid mode: n_slots slots, see place_experts)
    // the GGUF's expert tensors (the mapped file stays open): place_experts builds the blobs from them
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    size_t src_gsz = 0, src_dsz = 0;
    uint8_t* hblobs = nullptr;  // hybrid mode: every expert's blob in host memory, `stride` apart
    int n_slots = 0;            // hybrid mode: the experts resident in VRAM
};

struct Model {
    strata::core::ModelGeometry g;
    std::shared_ptr<strata::GgufModel> file;   // kept mapped: the experts are placed after the engine's buffers
    bool hybrid = false;                       // not every expert fits in VRAM: slots + the CPU pool
    // the packed Q8_0 decode copies (gguf weights, packed copy, bytes): given back when that lets every expert fit
    std::vector<std::tuple<const uint8_t*, uint8_t*, size_t>> packed;
    // the GGUF-native MTP block (blk.<n_layers>, loaded with --mtp): eh_proj over [enorm(e) | hnorm(h)], then an
    // attention + MoE layer, shared_head_norm, the main head
    bool has_mtp = false;
    Layer mtp;
    Mat eh_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_shnorm = nullptr;
    int64_t top_k = 8;
    float eps = 1e-6f, rope_base = 1e7f;
    int n_rot = 64;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
};

constexpr size_t kMmqTail = 4096;
size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}   // zeroed bytes after the last expert (MMQ reads a little past a matrix)

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// IEEE binary16 -> f32 by hand. __ushort_as_half is a __device__ intrinsic and
// upload_f32 runs on the host, so a CUDA toolkit older than 12.5 refuses the
// call; widen the bits here instead, the way bf16_to_f32 does.
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) {
            u = sign;                                               // +-0
        } else {
            uint32_t m = mantissa, shift = 0;                        // subnormal: normalize it
            while ((m & 0x400) == 0) {
                m <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) {
        u = sign | 0x7f800000u | (mantissa << 13);                   // inf / nan
    } else {
        u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// A tensor as f32 on the device (F32 as it is, BF16 / F16 widened).
float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)   // F16
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 8 && h.size() % 32 == 0) {
        // Q8_0 (the Q8_0 file stores ssm_alpha / ssm_beta so): blocks of an fp16 scale and 32 int8, widened here
        for (size_t b = 0; b < h.size() / 32; ++b) {
            const uint8_t* q = src + b * 34;
            const float d = f16_to_f32((uint16_t) (q[0] | (q[1] << 8)));
            for (int j = 0; j < 32; ++j) h[b * 32 + j] = d * (float) (int8_t) q[2 + j];
        }
    } else {
        std::fprintf(stderr, "strata-q36: %s is %s, expected F32, BF16, F16 or Q8_0\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

bool load(const std::string& path, Model& m, std::string& err, bool want_mtp) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    const strata::GgufModel& gm = *m.file;
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::Qwen35Moe) {
        err = "strata-q36 runs Qwen3.6-35B-A3B (qwen35moe) files only";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const auto& g = m.g;
    if (g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 || g.n_expert > 512 || m.top_k > 15) {
        err = "a geometry this driver's kernels do not take (head 256, state 128, conv 4, <= 512 experts, top-k <= 15)";
        return false;
    }
    const std::string p = "qwen35moe.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    uint64_t bytes = 0;
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-q36: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "strata-q36: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes += nb;
        w.d = d;
        // STRATA_Q8_PACKED (on by default here): the lossless qs / d plane copy the Q8_0 decode kernels read instead,
        // bitwise the same outputs
        if (w.type == 8 && K::native_q8_0_packed_enabled() && K::native_q8_0_packed_eligible(w.n_in, w.n_out)) {
            std::vector<uint8_t> packed(nb);
            K::native_q8_0_pack_host(gm.shard(s).tensor_data(*t), packed.data(), w.n_in, w.n_out);
            uint8_t* pd = nullptr;
            CK(cudaMalloc(&pd, nb));
            CK(cudaMemcpy(pd, packed.data(), nb, cudaMemcpyHostToDevice));
            K::native_q8_0_packed_register(d, pd, w.n_in, w.n_out);
            m.packed.emplace_back(d, pd, (size_t) nb);
            bytes += nb;
        }
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) g.n_layers);
    // the MTP block whenever the file has one, with or without drafts (--mtp): the VRAM it takes is then the same
    // either way, so the hybrid mode's slot tables - and with them the greedy text - do not depend on --mtp
    (void) want_mtp;
    m.has_mtp = g.n_mtp_layers > 0;
    for (int64_t l = 0; l < g.n_layers + (m.has_mtp ? 1 : 0); ++l) {
        const bool is_mtp = l == g.n_layers;
        Layer& y = is_mtp ? m.mtp : m.layers[(size_t) l];
        y.attn = is_mtp || strata::core::is_qsa_layer(g, l);
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.post_norm = f32(L(l, "post_attention_norm.weight"));
        if (!y.attn) {
            y.qkv = mat(L(l, "attn_qkv.weight"));
            y.zg = mat(L(l, "attn_gate.weight"));
            y.ssm_out = mat(L(l, "ssm_out.weight"));
            y.alpha = f32(L(l, "ssm_alpha.weight"));
            y.beta = f32(L(l, "ssm_beta.weight"));
            y.dt = f32(L(l, "ssm_dt.bias"));
            y.ssm_a = f32(L(l, "ssm_a"));
            y.conv = f32(L(l, "ssm_conv1d.weight"));
            y.ssm_norm = f32(L(l, "ssm_norm.weight"));
        } else {
            y.wq = mat(L(l, "attn_q.weight"));
            y.wk = mat(L(l, "attn_k.weight"));
            y.wv = mat(L(l, "attn_v.weight"));
            y.wo = mat(L(l, "attn_output.weight"));
            y.q_norm = f32(L(l, "attn_q_norm.weight"));
            y.k_norm = f32(L(l, "attn_k_norm.weight"));
        }
        y.router = f32(L(l, "ffn_gate_inp.weight"));
        y.sh_gate = mat(L(l, "ffn_gate_shexp.weight"));
        y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
        y.sh_down = mat(L(l, "ffn_down_shexp.weight"));
        y.sh_gate_inp = f32(L(l, "ffn_gate_inp_shexp.weight"));
        CK(cudaMalloc(&y.router_ext, (size_t) (g.n_expert + 1) * g.n_embd * 4));
        CK(cudaMemcpy(y.router_ext, y.router, (size_t) g.n_expert * g.n_embd * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(y.router_ext + (size_t) g.n_expert * g.n_embd, y.sh_gate_inp, (size_t) g.n_embd * 4,
                      cudaMemcpyDeviceToDevice));
        bytes += (size_t) (g.n_expert + 1) * g.n_embd * 4;
        // the routed experts: [gate | up | down] per expert, the native expert layout
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type) {
            err = L(l, "ffn_gate_exps / ffn_up_exps differ in type");
            return false;
        }
        if (!K::native_expert_supported((int) tg->type, (int) td->type, g.n_embd, g.n_ff)) {
            err = L(l, "experts are ") + tg->type_name() + "/" + td->type_name() + ", which the native expert kernels " +
                  "do not take (build with -DSTRATA_Q6K_EXPERTS=ON for Q6_K)";
            return false;
        }
        y.L = K::native_expert_layout((int) tg->type, (int) td->type, g.n_embd, g.n_ff);
        const size_t gsz = (size_t) g.n_ff * y.L.gu_row, dsz = (size_t) g.n_embd * y.L.d_row;
        // a stride in whole blocks of both parts' types (MMQ steps over the experts by blocks), 16-byte aligned
        const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
        y.stride = (y.L.bytes + al - 1) / al * al;
        y.src_g = gm.shard(sg).tensor_data(*tg);
        y.src_u = gm.shard(su).tensor_data(*tu);
        y.src_d = gm.shard(sd).tensor_data(*td);
        y.src_gsz = gsz;
        y.src_dsz = dsz;
    }
    if (m.has_mtp) {
        const int64_t l = g.n_layers;
        m.eh_proj = mat(L(l, "nextn.eh_proj.weight"));
        m.mtp_enorm = f32(L(l, "nextn.enorm.weight"));
        m.mtp_hnorm = f32(L(l, "nextn.hnorm.weight"));
        m.mtp_shnorm = f32(L(l, "nextn.shared_head_norm.weight"));
    }
    m.out_norm = f32("output_norm.weight");
    m.head = mat("output.weight");
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        if (std::getenv("STRATA_Q36_EMBD_HOST") != nullptr && std::atoi(std::getenv("STRATA_Q36_EMBD_HOST")) != 0) {
            // the table stays in pinned host memory: a token reads one row (2 KB) of it over PCIe, and the card keeps its
            // 515 MiB for expert slots at a long context.  The device pointer is the mapped one (UVA: also the host's).
            uint8_t* h = nullptr;
            CK(cudaHostAlloc((void**) &h, nb, cudaHostAllocMapped));
            std::memcpy(h, gm.shard(s).tensor_data(*t), nb);
            CK(cudaHostGetDevicePointer((void**) &d, h, 0));
            std::fprintf(stderr, "strata-q36: the token embedding table (%.0f MiB) stays in host memory "
                         "(STRATA_Q36_EMBD_HOST=1)\n", (double) nb / (1 << 20));
        } else {
            CK(cudaMalloc(&d, nb));
            CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
            bytes += nb;
        }
        m.embd = d;
    }
    std::fprintf(stderr, "strata-q36: %s: %zu tensors (layout checked), %.2f GiB on the GPU in %.1f s (the routed "
                 "experts are placed after the engine's buffers)\n", path.c_str(), checked, (double) bytes / (1 << 30),
                 ms_since(t0) / 1000.0);
    return true;
}

/// Expert e of layer L as its native blob [gate | up | down] (L.L.bytes) at dst (host memory).
void expert_blob(const Layer& L, int64_t e, uint8_t* dst) {
    std::memcpy(dst, L.src_g + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.up_off, L.src_u + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.down_off, L.src_d + (size_t) e * L.src_dsz, L.src_dsz);
}

// ------------------------------------------------------------------------------------------------ the engine
/// The packed Q8_0 decode copies given back (the decode reads the GGUF layout then, bitwise the same outputs): their bytes.
size_t give_back_packed(Model& m) {
    size_t bytes = 0;
    CK(cudaDeviceSynchronize());
    for (const auto& pk : m.packed) {
        K::native_q8_0_packed_unregister(std::get<0>(pk));
        CK(cudaFree(std::get<1>(pk)));
        bytes += std::get<2>(pk);
    }
    m.packed.clear();
    return bytes;
}

float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

/// A picture in a prompt: its first cell, its rows, and the hash the conversation cache compares it by.
struct ImgKey {
    int64_t cell = 0, n = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& o) const { return cell == o.cell && n == o.n && hash == o.hash; }
};

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 32768, tmax = 512;
    int64_t pos = 0;   // cells holding a processed token
    // activations [tmax][...]
    float *x, *h, *y, *qkv, *hc, *z, *alpha, *beta, *gdn_y, *qg, *kx, *vx, *q, *gate, *att, *logits_r, *eparts, *moe_out,
        *sg, *su, *shh, *sy, *sgate, *ewts, *logits;
    float* hmain = nullptr;   // [tmax][N] the main model's final normed hidden of each row (the MTP seeds)
    float* hlast = nullptr;   // [N] that of the last processed position (zeros before the first): the current session's
    void *q8, *xq, *escr;
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick;
    int32_t* hpin = nullptr;   // pinned staging: [0] decode token, [1] position, [2..9] picks, [10] max rows, [16..] tokens
    unsigned long long* grp_ptr;
    float *pacc, *pm, *pl, *amv;
    int32_t* ami;
    int n_chunk = 0, att_rows = 0;   // attention partials: att_rows query rows x n_chunk chunks
    std::vector<float*> conv, S, kc, vc;   // per layer; the MTP block's K / V at index n_layers: the CURRENT session's (select)
    // verify windows (T <= kVerify): the GDN inputs of every row per layer, kept for the commit
    static constexpr int kVerify = 8;
    std::vector<float*> v_qkv, v_hc, v_alpha, v_beta;
    // the MTP block (when loaded)
    float *mtp_cat = nullptr, *hmtp = nullptr, *logits_m = nullptr;
    int32_t *tok_m = nullptr, *pos0_m = nullptr, *pick_m = nullptr;
    // graphs: [0] the decode step, [T] a verify window of T rows, mtp1 the MTP draft step
    // (the graphs hold the current session's state pointers as kernel arguments: one set per session)
    struct GSet {
        cudaGraphExec_t g1 = nullptr, gv[kVerify + 1] = {}, gm1 = nullptr;
    };
    std::vector<GSet> gsets;
    GSet& gs() { return gsets[(size_t) cur]; }
    strata::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;
    float* att_s = nullptr;            // the prompt attention's scores / probabilities, kAttScoreFloats floats
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;          // the split-K partials of the F32 projections: [kSplitK][tmax][<= 257]
    size_t sk_cap = 0;                 // sk_part's floats
    __half* att_p16 = nullptr;         // the probabilities in FP16 (kAttScoreFloats halves)
    __half *k16 = nullptr, *v16 = nullptr, *q16 = nullptr;   // the prompt attention's FP16 operands
    size_t kAttScoreFloats = (size_t) 64 << 20;   // 256 MB (16 M floats when the experts' fit is close)
    // the FP16 K / V above hold kv16_blk cells instead of the whole context (0: the context; an int8 K / V store (--kv) at
    // a context above 16384 cells, STRATA_Q36_KV16_BLOCK=cells sets it, 0 off): a longer prompt's attention then runs
    // block by block
    int kv16_blk = 0;
    float *att_m = nullptr, *att_l = nullptr;   // [tmax][n_head] the blocked attention's running max / sum
    static int kv16_block_cells(bool kv_i8, int ctx) {
        const char* kb = std::getenv("STRATA_Q36_KV16_BLOCK");
        int blk = kb ? std::atoi(kb) : (kv_i8 ? 16384 : 0);
        if (blk > 0) blk = std::max(256, std::min(1 << 20, blk) / 8 * 8);   // (a block's scores row fits the 64 MB buffer)
        return blk > 0 && blk < ctx ? blk : 0;
    }
    // the packed Q8_0 decode copies may be given back before the sessions when those would not fit beside them (no
    // --expert-slots / --expert-vram, STRATA_Q36_KEEP_PACKED unset: set by main), as place_experts gives them back later
    bool packed_droppable = false;
    // the prompt's MMQ experts
    bool use_mmq = false;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *ident = nullptr, *rinv = nullptr;
    void *mq_x = nullptr, *mq_hq = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    __half* x16 = nullptr;
    bool use_graph = true;
    int mtp_d = 0;   // MTP drafts per speculative step (0: none; greedy requests only)
    // --kv int8 / STRATA_Q36_KV=int8: the K / V stores as int8 codes with one FP16 scale per kKVG values (one allocation
    // per store, see kKVG; the pointers in kc / vc and the sessions' tables are that allocation's base).  kv_k8 / kv_v8
    // are independent (--kv k8v32 keeps a FP32 V next to the int8 K); decode and prompt attention dequantize on the fly.
    bool kv_k8 = false, kv_v8 = false;
    /// One K / V store's allocation in floats (dalloc's unit): FP32 cells, or int8 codes + FP16 scales.
    size_t kv_store_floats(bool i8) const {
        const size_t cells = (size_t) m->g.n_head_kv * ctx * m->g.head_dim;
        return i8 ? (cells + cells / kKVG * 2 + 3) / 4 : cells;
    }
    /// the four K / V store combinations (--kv): f(kv_ic<K8>{}, kv_ic<V8>{}) runs with the kernel's template arguments
    template <bool B> struct kv_ic { static constexpr bool value = B; };
    template <class F> void kv4(F&& f) const {
        if (kv_k8 && kv_v8) f(kv_ic<true>{}, kv_ic<true>{});
        else if (kv_k8) f(kv_ic<true>{}, kv_ic<false>{});
        else if (kv_v8) f(kv_ic<false>{}, kv_ic<true>{});
        else f(kv_ic<false>{}, kv_ic<false>{});
    }

    // ---- sessions: the conversation states.  One for a single-request engine; --batch N keeps N, each with its own
    // recurrent state and KV cache, so N requests run at once: their rows share one decode window (forward_rows).  The
    // members conv / S / kc / vc / pos / hlast above are the CURRENT session's (select() swaps them).
    struct Sess {
        std::vector<float*> conv, S, kc, vc;
        float* hlast = nullptr;
        int64_t pos = 0;
        int32_t rdelta = 0;                 // rotary offset of the text after the last picture (see k_attn_prep)
        std::vector<int32_t> live;          // the tokens whose state it holds (cell i = live[i])
        std::vector<ImgKey> imgs;           // the pictures among them
        // STRATA_Q36_TURN_CKPT=1: the recurrent state after live[0, ck_n) (a prompt's last <|im_start|>), in pinned host
        // memory; valid while live starts with those tokens (its attention cells below ck_n are then still theirs)
        int64_t ck_n = 0;
        int32_t ck_rdelta = 0;
        void* ck_host = nullptr;
        bool active = false;                // a batch slot is decoding in it
        uint64_t used = 0;                  // when it was last used (the LRU pick)
    };
    std::vector<Sess> sess;
    int n_sess = 1, cur = 0;
    uint64_t use_clock = 0;
    // device tables for batch rows: per state index (layer, then the MTP block) the session pointers
    float **d_conv_tab = nullptr, **d_S_tab = nullptr, **d_kc_tab = nullptr, **d_vc_tab = nullptr;
    int32_t *d_rcell = nullptr, *d_rslot = nullptr, *d_rdelta = nullptr;   // [kVerify] the rows' cell / session / rotary offset
    // the GDN layers run a slot's rows in sequence: per group (a slot) its session, its first row and its row count
    int32_t *d_gslot = nullptr, *d_grow0 = nullptr, *d_gcnt = nullptr;
    int rows_G = 0;                                                        // groups of the rows being run
    int32_t* hrow = nullptr;                                                // pinned staging for the rows
    bool rows_mode = false;                                                 // layer(): the rows are sessions' (not consecutive cells)
    bool bmtp = false;                                                      // --batch-mtp: batch windows verify MTP proposals
    int64_t rows_pmax = 0;                                                  // (eager) the highest cell of the rows
    // pictures (--vision): the request's rows replace the embeddings of its <|image_pad|> cells; M-RoPE
    bool vision = false;
    int32_t* d_mtab = nullptr;          // [ctx][3] t, h, w of the current request's prompt cells (vision)
    const int32_t* mtab_now = nullptr;  // set for a forward over prompt cells that need it, else null
    int32_t* d_delta = nullptr;         // device scalar: the current session's rotary offset (decode graphs read it)
    float* d_img = nullptr;             // the request's picture rows [n][N]
    size_t img_cap = 0;                 // floats
    int32_t* d_map = nullptr;           // [tmax] a chunk's row -> picture row (or -1)
    int32_t* h_map = nullptr;           // pinned
    bool map_on = false;                // this forward's embeddings take picture rows
    std::vector<int32_t> img_map;       // the request's cell -> picture row (-1: text)
    int64_t img_prompt_cells = 0;       // the request's prompt length: cells below it use the table

    // ---- the hybrid mode (Model::hybrid): per state index (the layers, then the MTP block) the slot tables; a
    // batch of up to hybrid_t rows runs its VRAM-resident experts on the GPU and the others on the CPU pool, side by
    // side; a bigger prompt chunk streams each layer's experts into a staging buffer for MMQ (double-buffered by
    // default: one layer streams while the other computes; STRATA_Q36_SINGLE_STAGE=1, or an int8 K / V store (--kv) where
    // the pair does not fit, one buffer)
    struct Hy {
        std::vector<int> slot_of, exp_in_slot;
        std::vector<double> score;   // routing counts, halved at each rebalance
        int32_t* slot_of_d = nullptr;
#if defined(STRATA_Q36_CPU_POOL)
        strata::kernels::cpu::NativeFmt fmt;
#endif
    };
    std::vector<Hy> hy;
    int hybrid_t = 32;
    bool main_hybrid = false;   // some main layer is hybrid (else only the MTP block is: the decode graph stays)
    bool mtp_graph = true;      // the MTP draft step may be a graph (not when its layer is hybrid)
    /// state index li's layer runs hybrid (its experts not all in VRAM)
    bool lhy(size_t li) const {
        if (!m->hybrid) return false;
        const Layer& L = li < m->layers.size() ? m->layers[li] : m->mtp;
        return L.n_slots < (int) m->g.n_expert;
    }
#if defined(STRATA_Q36_CPU_POOL)
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
#endif
    float *h_h = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    uint8_t* acts = nullptr;   // per token: its quantized activation (kNativeActBytes)
    cudaEvent_t ev_route = nullptr;
    uint8_t* stage[2] = {};
    int n_stage = 2;   // staging buffers: 2 double-buffered, or 1 (STRATA_Q36_SINGLE_STAGE=1, or --kv int8 short of VRAM)
    cudaStream_t cst = nullptr;
    cudaEvent_t ev_copy[2] = {}, ev_used[2] = {};
    std::vector<int> stream_seq;   // the state indices a streamed forward visits, in order
    int stream_pos = 0;
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0;   // entries computed on the CPU / GPU, slot replacements
    double hy_cpu_ms = 0;
    // decode / verify rows: their routing is counted once the rows are kept (commit), so a verify window's rejected
    // drafts never move the slots; the slots follow the counts at every kRebalanceEvery-th position exactly (a
    // window never steps over one), so plain and speculative decoding see the same slot tables at every position
    static constexpr int kRebalanceEvery = 32;
    bool rows_deferred = false;          // set while a decode step or a verify window runs
    std::vector<std::vector<int32_t>> row_ids;   // [main layer][row * top_k + k] of the last such call
    void count_rows(int n) {
        const int Kp = (int) m->top_k;
        for (size_t li = 0; li < row_ids.size() && li < hy.size(); ++li)
            if (!row_ids[li].empty())
                for (int e = 0; e < n * Kp; ++e) hy[li].score[(size_t) row_ids[li][(size_t) e]] += 1.0;
    }
    void after_rows(int n) {   // a decode step / verify window kept n rows: count them, rebalance at the boundary
        if (!m->hybrid) return;
        count_rows(n);
        if (pos % kRebalanceEvery == 0) {
            CK(cudaStreamSynchronize(st));
            rebalance(8);
        }
    }

    void alloc_session(Sess& ss) const {
        const auto& g = m->g;
        const size_t C = (size_t) g.ssm_conv_channels;
        for (const Layer& L : m->layers) {
            if (L.attn) {
                ss.conv.push_back(nullptr);
                ss.S.push_back(nullptr);
                ss.kc.push_back(dalloc(kv_store_floats(kv_k8)));
                ss.vc.push_back(dalloc(kv_store_floats(kv_v8)));
            } else {
                ss.conv.push_back(dalloc(C * (g.ssm_d_conv - 1)));
                ss.S.push_back(dalloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size));
                ss.kc.push_back(nullptr);
                ss.vc.push_back(nullptr);
            }
        }
        if (m->has_mtp) {   // the MTP block's own K / V (an attention layer at index n_layers)
            ss.conv.push_back(nullptr);
            ss.S.push_back(nullptr);
            ss.kc.push_back(dalloc(kv_store_floats(kv_k8)));
            ss.vc.push_back(dalloc(kv_store_floats(kv_v8)));
        }
        ss.hlast = dalloc((size_t) g.n_embd);
    }

    /// The current session: `i`'s state, cache and position become the engine's (conv / S / kc / vc / pos / hlast).
    void select(int i) {
        if (i == cur) return;
        sess[(size_t) cur].pos = pos;
        cur = i;
        load_cur();
    }
    void load_cur() {
        Sess& ss = sess[(size_t) cur];
        conv = ss.conv; S = ss.S; kc = ss.kc; vc = ss.vc;
        hlast = ss.hlast;
        pos = ss.pos;
        ss.used = ++use_clock;
        set_delta(ss.rdelta);
    }
    /// The rotary offset the decode graphs read (a device scalar), for the current session.
    void set_delta(int32_t d) {
        sess[(size_t) cur].rdelta = d;
        if (!d_delta) return;
        hpin[60] = d;
        CK(cudaMemcpyAsync(d_delta, hpin + 60, 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));   // (the pinned word is reused)
    }

    void init(Model& model, int ctx_, int tmax_, int n_sessions = 1, bool vision_ = false) {
        m = &model;
        ctx = ctx_;
        n_sess = std::max(1, std::min(n_sessions, kVerify));
        vision = vision_;
        // the KV stores (--kv / STRATA_Q36_KV, resolved by the command line): int8 needs head_dim to be a kKVG multiple
        // (and the split-K decode's HD == 256 holds either way)
        if ((kv_k8 || kv_v8) && (int) m->g.head_dim % kKVG != 0) {
            std::fprintf(stderr, "strata-q36: head_dim %d is not a multiple of %d: the KV cache stays FP32\n",
                         (int) m->g.head_dim, kKVG);
            kv_k8 = kv_v8 = false;
        }
        tmax = std::max(tmax_, kVerify);
        const auto& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        const size_t C = (size_t) g.ssm_conv_channels;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N); hmain = dalloc(T * N);
        qkv = dalloc(T * C); hc = dalloc(T * C); z = dalloc(T * g.ssm_value_dim);
        alpha = dalloc(T * g.ssm_v_heads); beta = dalloc(T * g.ssm_v_heads); gdn_y = dalloc(T * g.ssm_value_dim);
        qg = dalloc(T * 2 * g.n_head * g.head_dim); kx = dalloc(T * g.n_head_kv * g.head_dim);
        vx = dalloc(T * g.n_head_kv * g.head_dim); q = dalloc(T * g.n_head * g.head_dim);
        gate = dalloc(T * g.n_head * g.head_dim); att = dalloc(T * g.n_head * g.head_dim);
        logits_r = dalloc(T * (g.n_expert + 1));
        eparts = dalloc(T * Kp * N); moe_out = dalloc(T * N);
        sg = dalloc(T * g.n_ff_shexp); su = dalloc(T * g.n_ff_shexp); shh = dalloc(T * g.n_ff_shexp); sy = dalloc(T * N);
        sgate = dalloc(T); ewts = dalloc(T * Kp); logits = dalloc((size_t) kVerify * g.n_vocab);
        const int max_in = (int) std::max<int64_t>({2 * g.n_embd, g.ssm_value_dim, g.n_head * g.head_dim, g.n_ff_shexp});
        CK(cudaMalloc(&q8, K::native_q8_1_bytes(max_in, 1) * T));
        CK(cudaMalloc(&xq, K::native_q8_1_bytes((int) N, 1) * T));
        x16 = dalloc_t<__half>(T * (size_t) max_in);
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cuBLAS handle failed\n");
            std::exit(1);
        }
        const bool kv_i8 = kv_k8 || kv_v8;   // an int8 K / V store (--kv): the long-context buffers below
        att_s = dalloc(kAttScoreFloats);
        // f32mm's split-K partials: splits <= 128 / (gx * gy) with gy >= T / (16 * tm), so splits * T <= 2048 * tm (tm <= 4)
        // as well as kSplitK * T; --kv int8 sizes them by that bound (8.0 MiB instead of 32.1 at a 2048 chunk)
        sk_cap = (kv_i8 ? std::min<size_t>((size_t) kSplitK * T, 2048 * 4) : (size_t) kSplitK * T) * (g.n_expert + 1);
        sk_part = dalloc(sk_cap);
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        kv16_blk = kv16_block_cells(kv_i8, ctx);
        const size_t kv16_cells = kv16_blk > 0 ? (size_t) kv16_blk : (size_t) ctx;
        k16 = dalloc_t<__half>((size_t) g.n_head_kv * kv16_cells * g.head_dim);
        v16 = dalloc_t<__half>((size_t) g.n_head_kv * kv16_cells * g.head_dim);
        if (kv16_blk > 0) {
            att_m = dalloc(T * g.n_head);
            att_l = dalloc(T * g.n_head);
        }
        q16 = dalloc_t<__half>(T * g.n_head * g.head_dim);
        use_mmq = MMQ::built() && std::getenv("STRATA_Q36_NO_MMQ") == nullptr;
        const char* singlestage = std::getenv("STRATA_Q36_SINGLE_STAGE");
        n_stage = (singlestage && std::atoi(singlestage) > 0) ? 1 : 2;   // one staging buffer instead of two: halves the prompt staging VRAM
        auto mmq_ok = [&](const Layer& L) {
            return MMQ::supported(L.L.gu_type) && MMQ::supported(L.L.d_type) && MMQ::fits(L.L.gu_type, 2 * g.n_ff) &&
                   MMQ::fits(L.L.d_type, g.n_embd);
        };
        for (const Layer& L : m->layers) use_mmq = use_mmq && mmq_ok(L);
        if (m->has_mtp) use_mmq = use_mmq && mmq_ok(m->mtp);
        if (use_mmq) {
            const int64_t ne = (int64_t) T * Kp;
            mctx = std::make_unique<MMQ::Context>();
            rcnt = dalloc_t<int32_t>((size_t) g.n_expert);
            rcur = dalloc_t<int32_t>((size_t) g.n_expert);
            rbounds = dalloc_t<int32_t>((size_t) g.n_expert + 1);
            rmax = dalloc_t<int32_t>(1);
            ident = dalloc_t<int32_t>((size_t) ne);
            rinv = dalloc_t<int32_t>((size_t) ne);
            CK(cudaDeviceSynchronize());   // ident's cudaMemset (legacy stream) before the iota on the non-blocking stream
            MMQ::iota(ident, ne, st);
            CK(cudaMalloc(&mq_x, MMQ::q8_bytes(ne, g.n_embd)));
            CK(cudaMalloc(&mq_hq, MMQ::q8_bytes(ne, g.n_ff)));
            if (kv_i8 && 3 * FF <= N) {
                // --kv int8: gate+up ([ne][2 FF]) and the SwiGLU ([ne][FF]) in the expert outputs' rows (eparts, [ne][N]):
                // all three are on the one stream, the down product writes eparts after the SwiGLU's quantize read mq_h,
                // and nothing reads eparts between the combine and the next layer's down (96 MiB less at a 2048 chunk)
                mq_gu = eparts;
                mq_h = eparts + (size_t) ne * 2 * FF;
            } else {
                mq_gu = dalloc((size_t) ne * 2 * FF);
                mq_h = dalloc((size_t) ne * FF);
            }
        }
        std::fprintf(stderr, "strata-q36: prompt experts on %s\n", use_mmq ? "MMQ (int8)" : "the native grouped kernels");
        {
            std::string gerr;
            int64_t big = 0;
            auto take = [&](const Layer& L) {
                for (const Mat* w : {&L.qkv, &L.zg, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo, &L.sh_gate, &L.sh_up, &L.sh_down})
                    big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
            };
            for (const Layer& L : m->layers) take(L);
            if (m->has_mtp) {
                take(m->mtp);
                big = std::max<int64_t>(big, (int64_t) m->eh_proj.n_in * m->eh_proj.n_out);
            }
            if (!gemm.init(st, big, gerr)) {
                std::fprintf(stderr, "strata-q36: prompt GEMM: %s\n", gerr.c_str());
                std::exit(1);
            }
        }
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) (T * Kp), (int64_t) FF)));
        ids = dalloc_t<int32_t>(T * Kp);
        grp_ptr = dalloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        grp_start = dalloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        n_groups = dalloc_t<int32_t>(1);
        ent_dst = dalloc_t<int32_t>(T * Kp);
        ent_tok = dalloc_t<int32_t>(T * Kp);
        tok = dalloc_t<int32_t>(T);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(kVerify);
        amv = dalloc((size_t) kArgBlocks * kVerify);
        ami = dalloc_t<int32_t>((size_t) kArgBlocks * kVerify);
        CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
        n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
        att_rows = (int) (g.n_head * std::min<int64_t>((int64_t) T, 64));   // 64 query tokens per attention batch
        // (--kv int8: kGemmT, the most rows the split-K path ever runs - a longer batch takes the GEMM path -: the
        // partials are 4 KB per cell at 64, 0.78 GiB at 204800 cells, 1/8 of that at kGemmT)
        if (kv_i8) att_rows = (int) (g.n_head * std::min<int64_t>((int64_t) T, kGemmT));
        pacc = dalloc((size_t) att_rows * n_chunk * g.head_dim);
        pm = dalloc((size_t) att_rows * n_chunk);
        pl = dalloc((size_t) att_rows * n_chunk);
        for (const Layer& L : m->layers) {   // the verify windows' GDN inputs (one set, whichever session runs)
            if (L.attn) {
                for (auto* v : {&v_qkv, &v_hc, &v_alpha, &v_beta}) v->push_back(nullptr);
            } else {
                v_qkv.push_back(dalloc((size_t) kVerify * C));
                v_hc.push_back(dalloc((size_t) kVerify * C));
                v_alpha.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
                v_beta.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
            }
        }
        if (m->has_mtp) {
            mtp_cat = dalloc(T * 2 * N);
            hmtp = dalloc(T * N);
            logits_m = dalloc((size_t) g.n_vocab);
            tok_m = dalloc_t<int32_t>(T);
            pos0_m = dalloc_t<int32_t>(1);
            pick_m = dalloc_t<int32_t>(1);
        }
        sess.resize((size_t) n_sess);
        gsets.resize((size_t) n_sess);
        {   // the sessions' caches: said before they fail (a slot's K / V at the context, fp32, for the attention layers and
            // the MTP block, and its recurrent state)
            size_t n_attn = m->has_mtp ? 1 : 0, n_gdn = 0;
            for (const Layer& L : m->layers) (L.attn ? n_attn : n_gdn) += 1;
            const size_t per = (kv_store_floats(kv_k8) + kv_store_floats(kv_v8)) * 4 * n_attn +
                               (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4 * n_gdn;
            size_t fr0 = 0, tot0 = 0;
            CK(cudaMemGetInfo(&fr0, &tot0));
            // the sessions would not fit beside the packed Q8_0 decode copies (the old order failed here: --kv int8 at
            // 131072+ cells on 8 GB) and place_experts may give them back (MMQ, no --expert-slots / --expert-vram,
            // STRATA_Q36_KEEP_PACKED unset): give them back first.  Only under --kv int8 (an FP32 store keeps the old
            // order and its failure).  Every config that fit before allocates as before: one session with no margin (what
            // follows it here is a few KB; a session that fits with less than 64 MiB to spare still gives them back in
            // place_experts, for the staging buffers), several with the 1 GiB the check below asks for.
            const size_t sess_need = per * (size_t) n_sess + (vision ? (size_t) ctx * 12 : 0) +
                                     (n_sess > 1 ? ((size_t) 1 << 30) : 0);
            if (sess_need > fr0 && (kv_k8 || kv_v8) && packed_droppable && use_mmq && !model.packed.empty()) {
                const size_t pb = give_back_packed(model);
                const size_t fr_was = fr0;
                CK(cudaMemGetInfo(&fr0, &tot0));
                std::fprintf(stderr, "strata-q36: the sessions' K / V and state (%.2f GiB at a context of %d) do not fit in the "
                             "%.2f GiB free: the packed Q8_0 decode copies (%.2f GiB) given back before them, %.2f GiB free "
                             "now (STRATA_Q36_KEEP_PACKED=1 keeps them)\n", (double) per * n_sess / (1 << 30), ctx,
                             (double) fr_was / (1 << 30), (double) pb / (1 << 30), (double) fr0 / (1 << 30));
            }
            if (n_sess > 1)
                std::fprintf(stderr, "strata-q36: %d sessions (--batch): %.2f GiB of VRAM each at a context of %d (%.2f GiB in "
                             "all), %.2f GiB free before them\n", n_sess, (double) per / (1 << 30), ctx,
                             (double) per * n_sess / (1 << 30), (double) fr0 / (1 << 30));
            if (n_sess > 1 && per * (size_t) n_sess + ((size_t) 1 << 30) > fr0) {
                std::fprintf(stderr, "strata-q36: not enough VRAM for %d slots at a context of %d (%.2f GiB each, %.2f GiB "
                             "free): use fewer slots (\"parallel\" / --batch) or a shorter --max-context\n", n_sess, ctx,
                             (double) per / (1 << 30), (double) fr0 / (1 << 30));
                std::exit(1);
            }
        }
        for (Sess& ss : sess) alloc_session(ss);
        if (n_sess > 1) {   // the batch rows' tables: per state index the session pointers, on the device
            const size_t ns = sess[0].conv.size();
            std::vector<float*> hc(ns * (size_t) n_sess), hs(hc), hk(hc), hv(hc);
            for (size_t li = 0; li < ns; ++li)
                for (int s_ = 0; s_ < n_sess; ++s_) {
                    hc[li * n_sess + s_] = sess[(size_t) s_].conv[li];
                    hs[li * n_sess + s_] = sess[(size_t) s_].S[li];
                    hk[li * n_sess + s_] = sess[(size_t) s_].kc[li];
                    hv[li * n_sess + s_] = sess[(size_t) s_].vc[li];
                }
            const std::pair<float***, std::vector<float*>*> tabs[4] = {{&d_conv_tab, &hc}, {&d_S_tab, &hs},
                                                                          {&d_kc_tab, &hk}, {&d_vc_tab, &hv}};
            for (const auto& tb : tabs) {
                CK(cudaMalloc((void**) tb.first, tb.second->size() * sizeof(float*)));
                CK(cudaMemcpy((void*) *tb.first, tb.second->data(), tb.second->size() * sizeof(float*),
                              cudaMemcpyHostToDevice));
            }
            d_rcell = dalloc_t<int32_t>(kVerify);
            d_rslot = dalloc_t<int32_t>(kVerify);
            d_rdelta = dalloc_t<int32_t>(kVerify);
            d_gslot = dalloc_t<int32_t>(kVerify);
            d_grow0 = dalloc_t<int32_t>(kVerify);
            d_gcnt = dalloc_t<int32_t>(kVerify);
            CK(cudaHostAlloc((void**) &hrow, 128 * sizeof(int32_t), cudaHostAllocDefault));
        }
        if (vision) {
            d_mtab = dalloc_t<int32_t>((size_t) ctx * 3);
            d_map = dalloc_t<int32_t>(T);
            CK(cudaHostAlloc((void**) &h_map, T * sizeof(int32_t), cudaHostAllocDefault));
        }
        d_delta = dalloc_t<int32_t>(1);
        cur = 0;
        load_cur();
        CK(cudaDeviceSynchronize());   // every legacy-stream cudaMemset of the buffers above is done
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-q36: context %d cells, prompt chunk %d%s, %.2f GiB of VRAM free\n", ctx, tmax,
                     m->has_mtp ? ", MTP block loaded" : "", (double) fr / (1 << 30));
        if (kv16_blk > 0)
            std::fprintf(stderr, "strata-q36: the prompt attention over the context in blocks of %d cells: its FP16 K / V "
                         "%.3f GiB instead of %.3f GiB (STRATA_Q36_KV16_BLOCK=0: the whole context); split-K partials for "
                         "%d rows, %.3f GiB\n", kv16_blk,
                         (double) (2 * g.n_head_kv * g.head_dim * 2) * kv16_blk / (1 << 30),
                         (double) (2 * g.n_head_kv * g.head_dim * 2) * ctx / (1 << 30), att_rows / (int) g.n_head,
                         (double) att_rows * n_chunk * (g.head_dim + 2) * 4 / (1 << 30));
    }

    /// The recurrent state and the cache to a fresh sequence.
    void reset() {
        const auto& g = m->g;
        for (size_t l = 0; l < conv.size(); ++l) {
            if (conv[l]) CK(cudaMemsetAsync(conv[l], 0, (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4, st));
            if (S[l]) CK(cudaMemsetAsync(S[l], 0, (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4, st));
        }
        CK(cudaMemsetAsync(hlast, 0, (size_t) g.n_embd * 4, st));
        CK(cudaStreamSynchronize(st));
        pos = 0;
        sess[(size_t) cur].pos = 0;
        set_delta(0);
    }

    /// y [T][n_out] = W x [T][n_in] on the native MMVQ, 8 columns per launch.
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// y [T][n_out] = W [n_out][n_in] (f32) x [T][n_in]: a decode batch on the warp-per-row kernel, a prompt chunk on
    /// cuBLAS's FP32 GEMM (the router and the GDN alpha / beta stay in FP32, as llama.cpp keeps them).
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        static const bool blas_f32 = std::getenv("STRATA_Q36_BLAS_F32") != nullptr;   // the hipBLAS / cuBLAS SGEMM (A/B)
        if (!blas_f32 && n_in % 4 == 0) {
            // K split so that about 128 blocks run (the partials summed in a fixed order: deterministic); TM = 4 rows per
            // thread for the 256-wide router, 1 for the narrow ones (alpha / beta 32, the gate 1)
            const int tm = n_out > 64 ? 4 : 1;
            const int gx = (n_out + 63) / 64, gy = (T + 16 * tm - 1) / (16 * tm);
            const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
            const int kz = (n_in / want + 15) / 16 * 16;
            const int splits = (n_in + kz - 1) / kz;
            // (--kv int8 sizes sk_part by a bound, see init: outside it, never with this model's shapes, cuBLAS below)
            if (splits <= 1 || (size_t) splits * T * n_out <= sk_cap) {
                float* dst = splits > 1 ? sk_part : out;
                if (tm == 4) k_sgemm_nt<4><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
                else k_sgemm_nt<1><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
                if (splits > 1)
                    k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
                return;
            }
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, xin, n_in, &zero, out, n_out) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cublasSgemm failed\n");
            std::exit(1);
        }
    }

    /// A batch above kGemmT tokens (a prompt chunk) runs its projections as GEMMs (prefill::Gemm::native: the GGUF
    /// blocks dequantized to FP16, the activations FP16); a decode batch on the native MMVQ (q8_1 activations).
    static constexpr int kGemmT = 8;
    void prep(const float* xin, int n, int T, void* q) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xin, x16, (int64_t) n * T);
        else quant(xin, n, T, q);
    }
    void apply(const Mat& w, const void* q, float* out, int T) {
        if (T > kGemmT) gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        else mmq(w, q, out, T);
    }
    /// x [T][n] -> its q8_1 image (T contiguous columns: one vector of T * n), for every matrix that reads x.
    void quant(const float* xin, int n, int T, void* q) { K::native_quantize_q8_1(xin, q, n * T, 1, st); }
    // ---- decode kernels of this engine (q36_kernels.cuh), each the old kernel's bits (STRATA_Q36_VERIFY=1 compares)
    bool fast_k = std::getenv("STRATA_Q36_OLD_KERNELS") == nullptr;
    bool verify_k = std::getenv("STRATA_Q36_VERIFY") != nullptr;
    // STRATA_Q36_ATTN_GQA=1: the split-K decode attention as k_attn_gqa, chunks of gqa_cc cells (a kGqaTile multiple,
    // STRATA_Q36_ATTN_GQA_CHUNK; default 1024); 0: k_attn_part as before
    int gqa_cc = [] {
        const char* e = std::getenv("STRATA_Q36_ATTN_GQA");
        if (e == nullptr || std::atoi(e) == 0) return 0;
        const char* c = std::getenv("STRATA_Q36_ATTN_GQA_CHUNK");
        const int v = c != nullptr ? std::atoi(c) : 1024;
        return std::max(1, std::min(16, v / kGqaTile)) * kGqaTile;
    }();
    float* vbuf = nullptr;                       // verify scratch
    long long v_calls = 0, v_bad = 0;
    std::map<std::string, long long> v_by;   // mismatches per check
    void verify_cmp(const char* what, const float* a, const float* b, size_t n, const std::string& tag = "") {
        CK(cudaStreamSynchronize(st));
        std::vector<float> ha(n), hb(n);
        CK(cudaMemcpy(ha.data(), a, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hb.data(), b, n * 4, cudaMemcpyDeviceToHost));
        ++v_calls;
        if (std::memcmp(ha.data(), hb.data(), n * 4) != 0) {
            ++v_bad;
            size_t i = 0;
            while (i < n && std::memcmp(&ha[i], &hb[i], 4) == 0) ++i;
            if (++v_by[what] <= 3)
                std::fprintf(stderr, "strata-q36 VERIFY MISMATCH %s%s: first at %zu of %zu: new %g old %g (bits %08x %08x)\n", what,
                             tag.c_str(), i, n, (double) ha[i], (double) hb[i], *(unsigned*) &ha[i], *(unsigned*) &hb[i]);
        }
    }
    /// A Q8_0 matrix that is not on the packed decode kernel (the shared expert, attention k / v: fewer than 2048
    /// rows) for T <= 8 columns: one warp per row (q36k::k_q8_rows) in place of the 4-warp block of
    /// native_small_mmvq_kernel; the same bits.
    bool q8_small(const Mat& w, const void* q, float* out, int T) {
        if (!fast_k || w.type != 8 || T < 1 || T > 8 || w.n_in % 32 != 0) return false;
        if (K::native_q8_0_packed_eligible(w.n_in, w.n_out) && K::native_q8_0_packed_enabled()) return false;
        const int nb = w.n_in / 32;
        if (nb > 64) return false;
        const void* wp = w.d;
        const half* dpl = nullptr;
        const int NJ = nb <= 32 ? 1 : 2;
#define Q36_Q8(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (NJ == 1) q36k::launch_q8_rows<NCV, 1, false>(wp, dpl, q, out, nb, w.n_out, st);                  \
        else q36k::launch_q8_rows<NCV, 2, false>(wp, dpl, q, out, nb, w.n_out, st);                          \
        break;
        switch (T) { Q36_Q8(1) Q36_Q8(2) Q36_Q8(3) Q36_Q8(4) Q36_Q8(5) Q36_Q8(6) Q36_Q8(7) Q36_Q8(8) }
#undef Q36_Q8
        if (verify_k) {
            if (!vbuf) CK(cudaMalloc((void**) &vbuf, (size_t) 8 * 16384 * 4));
            const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
            for (int c0 = 0; c0 < T; c0 += 8) {
                const int nc = std::min(8, T - c0);
                K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, vbuf + (size_t) c0 * w.n_out, w.n_in, w.n_out,
                               nc, st);
            }
            verify_cmp("q8_small", out, vbuf, (size_t) T * w.n_out,
                       " n_in " + std::to_string(w.n_in) + " n_out " + std::to_string(w.n_out) + " T " + std::to_string(T));
        }
        return true;
    }
    // ---- the q8_1 images produced by the norm kernels themselves (fast_k, T <= kGemmT): xq_ready = the layer's input h is
    // already quantized in xq (k_moe_out_small of the layer before), xq_post = the MoE input h is (k_add_rms_q)
    bool xq_ready = false, xq_post = false;
    void* vq = nullptr;
    void verify_q8(const char* what, const float* hsrc, int T, const std::string& tag) {
        const int N = (int) m->g.n_embd;
        const size_t qb = K::native_q8_1_bytes(N, 1);
        if (!vq) CK(cudaMalloc(&vq, qb * 8));
        K::native_quantize_q8_1(hsrc, vq, N * T, 1, st);
        const long long before = v_bad;
        verify_cmp(what, (const float*) xq, (const float*) vq, qb * (size_t) T / 4, tag);
        if (v_bad != before && std::getenv("STRATA_Q36_VDEBUG") && v_by[what] <= 2) {   // a mismatching block, on the host
            std::vector<uint8_t> a(qb * (size_t) T), b(qb * (size_t) T);
            std::vector<float> hv((size_t) N * T);
            CK(cudaMemcpy(a.data(), xq, a.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), vq, b.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hv.data(), hsrc, hv.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t o = 0; o < a.size(); o += 36)
                if (std::memcmp(&a[o], &b[o], 36) != 0) {
                    const size_t blk = o / 36;
                    float v[32], t[32];
                    for (int l = 0; l < 32; ++l) v[l] = t[l] = hv[blk * 32 + (size_t) l];
                    for (int off = 16; off > 0; off >>= 1) {   // the xor butterfly, lane by lane
                        float u[32];
                        for (int l = 0; l < 32; ++l) u[l] = t[l] + t[l ^ off];
                        std::memcpy(t, u, sizeof(u));
                    }
                    unsigned ds_a, ds_b;
                    std::memcpy(&ds_a, &a[o], 4);
                    std::memcpy(&ds_b, &b[o], 4);
                    std::fprintf(stderr, "VDEBUG block %zu: ds new %08x old %08x, qs bytes %s, cpu butterfly sum %.9g (half %04x)\n",
                                 blk, ds_a, ds_b, std::memcmp(&a[o + 4], &b[o + 4], 32) == 0 ? "equal" : "DIFFER", (double) t[0],
                                 (unsigned) __half_as_ushort(__float2half(t[0])));
                    break;
                }
        }
    }
    void prep_in(int T) {
        if (xq_ready && T <= kGemmT) {
            xq_ready = false;
            return;
        }
        xq_ready = false;
        prep(h, (int) m->g.n_embd, T, xq);
    }
    void add_rms_post(const Layer& L, int T) {
        const int N = (int) m->g.n_embd;
        xq_post = false;
        if (fast_k && T <= kGemmT) {
            if (verify_k) {
                if (!vx_) {
                    CK(cudaMalloc((void**) &vx_, (size_t) 8 * 2048 * 4));
                    CK(cudaMalloc((void**) &vh_, (size_t) 8 * 2048 * 4));
                }
                CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
                k_add_rms<<<T, kThreads, 0, st>>>(vx_, y, L.post_norm, vh_, N, m->eps);
            }
            q36k::k_add_rms_q<<<T, 256, 0, st>>>(x, y, L.post_norm, h, (q36k::QB*) xq, N, m->eps);
            xq_post = true;
            if (verify_k) {
                verify_cmp("add_rms x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                verify_cmp("add_rms h", h, vh_, (size_t) T * N, " T " + std::to_string(T));
                verify_q8("add_rms q8", h, T, " T " + std::to_string(T));
            }
            return;
        }
        k_add_rms<<<T, kThreads, 0, st>>>(x, y, L.post_norm, h, N, m->eps);
    }
    /// The shared expert's gate + up + SwiGLU + q8_1 image (into q8) for T <= 4 rows in one launch (q36k::k_sh_gu), else false.
    bool shared_gu(const Layer& L, int T) {
        const int FS = (int) m->g.n_ff_shexp;
        const Mat &a = L.sh_gate, &b = L.sh_up;
        if (!fast_k || T < 1 || T > 4 || a.type != 8 || b.type != 8 || a.n_in != b.n_in || a.n_out != FS || b.n_out != FS ||
            a.n_in % 32 != 0 || a.n_in / 32 > 64 || a.n_in / 32 <= 32 || FS % 32 != 0 ||
            (K::native_q8_0_packed_eligible(a.n_in, a.n_out) && K::native_q8_0_packed_enabled()))
            return false;
        const int nb = a.n_in / 32;
        const dim3 grid((unsigned) (FS / 32));
        const bool pkd = false;
        const uint8_t* wa = a.d;
        const uint8_t* wb = b.d;
        const half* da = nullptr;
        const half* db = nullptr;
#define Q36_SH(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (pkd)                                                                                             \
            q36k::k_sh_gu<NCV, 2, true><<<grid, 1024, 0, st>>>(wa, da, wb, db, (const q36k::QB*) xq,        \
                                                               (q36k::QB*) q8, nb, FS);                      \
        else                                                                                                 \
            q36k::k_sh_gu<NCV, 2, false><<<grid, 1024, 0, st>>>(wa, nullptr, wb, nullptr, (const q36k::QB*) xq, \
                                                                (q36k::QB*) q8, nb, FS);                     \
        break;
        switch (T) { Q36_SH(1) Q36_SH(2) Q36_SH(3) Q36_SH(4) }
#undef Q36_SH
        if (verify_k) {
            if (!vbuf) CK(cudaMalloc((void**) &vbuf, (size_t) 8 * 16384 * 4));
            if (!vq) CK(cudaMalloc(&vq, K::native_q8_1_bytes(m->g.n_embd, 1) * 8));
            float* vg = vbuf;
            float* vu = vbuf + (size_t) 8 * FS;
            mmq_ref(a, xq, vg, T);
            mmq_ref(b, xq, vu, T);
            K::native_swiglu_quantize_q8_1(vg, vu, vq, FS, T, st);
            verify_cmp("shared_gu q8", (const float*) q8, (const float*) vq, K::native_q8_1_bytes(FS, 1) * (size_t) T / 4,
                       " T " + std::to_string(T));
        }
        return true;
    }
    /// The old products (native MMVQ), for the checks.
    void mmq_ref(const Mat& w, const void* q, float* out, int T) {
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in, w.n_out, nc, st);
        }
    }
    void group_old(const Layer& L, const int32_t* slot_of, int T, unsigned long long* o_ptr, int32_t* o_start, int32_t* o_ng,
                   int32_t* o_dst, int32_t* o_tok) {
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (slot_of)
            k_group_hy<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, slot_of, L.blobs, L.stride, o_ptr, o_start,
                                               o_ng, o_dst, o_tok);
        else
            k_group<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, L.blobs, L.stride, o_ptr, o_start, o_ng, o_dst,
                                            o_tok);
    }
    /// VERIFY: the old grouping of the current `ids` into scratch, compared with grp_ptr / grp_start / n_groups / ent_dst / ent_tok.
    void group_verify(const Layer& L, const int32_t* slot_of, int T) {
        if (!vgrp) CK(cudaMalloc((void**) &vgrp, 2048));
        unsigned long long* o_ptr = (unsigned long long*) vgrp;
        int32_t* o_start = (int32_t*) (o_ptr + 65);
        int32_t* o_ng = o_start + 66;
        int32_t* o_dst = o_ng + 2;
        int32_t* o_tok = o_dst + 64;
        group_old(L, slot_of, T, o_ptr, o_start, o_ng, o_dst, o_tok);
        CK(cudaStreamSynchronize(st));
        int nn = 0, no = 0;
        CK(cudaMemcpy(&nn, n_groups, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&no, o_ng, 4, cudaMemcpyDeviceToHost));
        ++v_calls;
        bool bad = nn != no;
        if (!bad) {
            std::vector<unsigned long long> a((size_t) nn + 1), b((size_t) nn + 1);
            std::vector<int32_t> sa((size_t) nn + 1), sb((size_t) nn + 1), da(64), db(64), ta(64), tb(64);
            CK(cudaMemcpy(a.data(), grp_ptr, (size_t) nn * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), o_ptr, (size_t) nn * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sa.data(), grp_start, ((size_t) nn + 1) * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sb.data(), o_start, ((size_t) nn + 1) * 4, cudaMemcpyDeviceToHost));
            const int tot = sa[(size_t) nn];
            CK(cudaMemcpy(da.data(), ent_dst, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(db.data(), o_dst, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ta.data(), ent_tok, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(tb.data(), o_tok, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            bad = std::memcmp(a.data(), b.data(), (size_t) nn * 8) != 0 ||
                  std::memcmp(sa.data(), sb.data(), ((size_t) nn + 1) * 4) != 0 ||
                  std::memcmp(da.data(), db.data(), (size_t) tot * 4) != 0 ||
                  std::memcmp(ta.data(), tb.data(), (size_t) tot * 4) != 0;
        }
        if (bad) {
            ++v_bad;
            if (v_bad <= 20) std::fprintf(stderr, "strata-q36 VERIFY MISMATCH group (T %d, groups %d vs %d)\n", T, nn, no);
        }
    }
    bool grouped_by_route = false;   // route_group already filled the group arrays of this layer
    /// The router's top-k and the grouping of the picks in one launch (T <= 8, 256 experts, 8 picks); the logits are in logits_r.
    /// `slot_of`: the hybrid mode's slot table (null: every expert resident).  Returns false when the old kernels must run.
    bool route_group(const Layer& L, const int32_t* slot_of, int T, int ld) {
        const int Kp = (int) m->top_k;
        grouped_by_route = false;
        if (!fast_k || T > 8 || m->g.n_expert != 256 || Kp != 8) return false;
        if (slot_of)
            q36k::k_route_group<true><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, slot_of, L.blobs, L.stride, grp_ptr,
                                                                 grp_start, n_groups, ent_dst, ent_tok);
        else
            q36k::k_route_group<false><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, nullptr, L.blobs, L.stride, grp_ptr,
                                                                  grp_start, n_groups, ent_dst, ent_tok);
        grouped_by_route = true;
        if (verify_k) {
            // the old top-k into scratch, the ids / weights compared, then the old grouping of the new ids
            if (!vbuf) CK(cudaMalloc((void**) &vbuf, (size_t) 8 * 16384 * 4));
            int32_t* vid = (int32_t*) (vbuf + 8 * 8192);
            float* vw = vbuf + 8 * 8192 + 64;
            K::native_router_topk(logits_r, vid, vw, 256, Kp, T, st);
            verify_cmp("route ids", (const float*) ids, (const float*) vid, (size_t) T * Kp, " T " + std::to_string(T));
            verify_cmp("route weights", ewts, vw, (size_t) T * Kp, " T " + std::to_string(T));
            group_verify(L, slot_of, T);
        }
        return true;
    }
    /// The routed entries of T rows grouped per expert (k_group; with slot_of the hybrid mode's k_group_hy).
    void group_entries(const Layer& L, const int32_t* slot_of, int T) {
        if (grouped_by_route) {   // route_group did it
            grouped_by_route = false;
            return;
        }
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (fast_k && n_ent <= 64) {
            if (slot_of)
                q36k::k_group_small<true><<<1, 64, 0, st>>>(ids, n_ent, Kp, slot_of, L.blobs, L.stride, grp_ptr, grp_start,
                                                            n_groups, ent_dst, ent_tok);
            else
                q36k::k_group_small<false><<<1, 64, 0, st>>>(ids, n_ent, Kp, nullptr, L.blobs, L.stride, grp_ptr, grp_start,
                                                             n_groups, ent_dst, ent_tok);
            if (verify_k) group_verify(L, slot_of, T);
            return;
        }
        group_old(L, slot_of, T, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
    }
    void* vgrp = nullptr;
    float *vx_ = nullptr, *vh_ = nullptr;
    /// The MoE block's end (k_moe_out) for T rows.
    void moe_end(int T, const int32_t* prow, const float* shared, const float* sgate, const float* wn, float* hout, int sgs = 1) {
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        const bool small = fast_k && T <= 8 && N <= 2048 && N % 256 == 0 && Kp <= 15;
        if (small && verify_k) {
            if (!vx_) {
                CK(cudaMalloc((void**) &vx_, (size_t) 8 * 2048 * 4));
                CK(cudaMalloc((void**) &vh_, (size_t) 8 * 2048 * 4));
            }
            CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
            k_moe_out<<<T, kThreads, 0, st>>>(vx_, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, vh_, N, m->eps);
        }
        if (small) {
            const bool mkq = wn != nullptr && hout == h;   // the next layer's input, quantized here
            q36k::k_moe_out_small<<<T, 1024, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps,
                                                      mkq ? (q36k::QB*) xq : nullptr);
            xq_ready = mkq;
            if (verify_k) {
                if (mkq) verify_q8("moe_out q8", hout, T, " T " + std::to_string(T));
                verify_cmp("moe_out x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                if (wn) verify_cmp("moe_out h", hout, vh_, (size_t) T * N, " T " + std::to_string(T));
            }
            return;
        }
        xq_ready = false;
        k_moe_out<<<T, kThreads, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps);
    }
    /// y [T][n_out] = W x from x's q8_1 image, 8 columns per launch.
    void mmq(const Mat& w, const void* q, float* out, int T) {
        if (q8_small(w, q, out, T)) return;
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, nc, st);
        }
    }

    /// One layer for T rows: x (the residual) += mixer(h); then MoE; x += moe; `hout` = rms(x) * wn (the next norm).
    /// `li` indexes the layer's state; positions *dpos .. + T - 1 (device), p0 the same on the host (eager grids);
    /// `verify`: a verify window - the GDN state is read, not written (commit() writes it for the kept rows).
    void layer(const Layer& L, size_t li, int T, int64_t p0, const int32_t* dpos, bool graph_mode, bool verify,
               const float* wn, float* hout) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, D = (int) g.ssm_state_size;
        const int C = (int) g.ssm_conv_channels;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim;
        if (!L.attn) {
            float* qkv_b = verify ? v_qkv[li] : qkv;
            float* hc_b = verify ? v_hc[li] : hc;
            float* al_b = verify ? v_alpha[li] : alpha;
            float* be_b = verify ? v_beta[li] : beta;
            prep_in(T);
            apply(L.qkv, xq, qkv_b, T);
            apply(L.zg, xq, z, T);
            if (T <= kGemmT) {
                k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(L.alpha, L.beta, h, L.dt, L.ssm_a, al_b, be_b, N,
                                                                      HV);
            } else {
                f32mm(L.alpha, h, al_b, N, HV, T);
                f32mm(L.beta, h, be_b, N, HV, T);
                k_gdn_gb<<<grid1((int64_t) HV * T), kThreads, 0, st>>>(al_b, L.dt, L.ssm_a, be_b, HV, T);
            }
            if (rows_mode) {   // batch rows: a slot's rows (one, or a verify window's two) in sequence on its own state
                k_conv_seq<<<dim3(C / 128, rows_G), 128, 0, st>>>(nullptr, qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, 1,
                                                                  !verify, d_conv_tab + li * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, rows_G), dim3(128, 4), 0, st>>>(nullptr, hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps,
                                                                     gdn_y, HK, HV, 1, !verify, d_S_tab + li * n_sess,
                                                                     d_gslot, d_grow0, d_gcnt);
            } else {
                // one token, a window or a chunk: the same kernels (every token in one launch each), so a decode step and a
                // verify window compute bitwise the same rows and the MTP's greedy text is the plain greedy text.  (Strata's
                // fused single-token kernels are not used here: fused_gdn_conv_l2's 4-tap sum contracts into FMAs
                // differently from k_conv_seq's, a 1-ulp difference in h that the Q8_1 activation rounding can turn into a
                // flipped near-tie later; with them the KL is the same, 1.05x / 0.81x the floor, but MTP would not be exact.)
                // a prompt chunk: tiles of kConvTile tokens side by side (tile mode); else one block per head walks the tokens
                const int tiles = (fast_k && T > kGemmT && !verify) ? (T + kConvTile - 1) / kConvTile : 1;
                k_conv_seq<<<dim3(C / 128, tiles), 128, 0, st>>>(conv[li], qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, T, !verify);
                if (T > kGemmT && !verify) {   // a prompt chunk: the column-split delta rule, then the norm
                    k_gdn_seq_cols<<<dim3(HV, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(S[li], hc_b, C, al_b,
                                                                                                      be_b, gdn_y, HK, HV, T);
                    k_gdn_norm<<<dim3(HV, T), 128, 0, st>>>(gdn_y, z, L.ssm_norm, m->eps, gdn_y, HV);
                } else {   // decode, verify windows and commits: one kernel, so a window's rows are a decode step's rows
                    k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps, gdn_y, HK, HV,
                                                           T, !verify);
                }
            }
            mm(L.ssm_out, gdn_y, y, T);
        } else {
            prep_in(T);
            apply(L.wq, xq, qg, T);
            apply(L.wk, xq, kx, T);
            apply(L.wv, xq, vx, T);
            RowSel rs;
            if (rows_mode) {
                rs.cell = d_rcell;
                rs.slot = d_rslot;
                rs.delta = d_rdelta;
                rs.kc = d_kc_tab + li * n_sess;
                rs.vc = d_vc_tab + li * n_sess;
            }
            kv4([&](auto k8c, auto v8c) {
                constexpr bool K8 = decltype(k8c)::value, V8 = decltype(v8c)::value;
                k_attn_prep<K8, V8><<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, L.q_norm, L.k_norm, q, gate, rows_mode ? nullptr : kc[li],
                                                                          rows_mode ? nullptr : vc[li], H, HKV, HD, m->n_rot, dpos, ctx,
                                                                          m->rope_base, m->eps, d_delta, mtab_now, rs);
            });
            if (T > kGemmT) {
                // a prompt chunk: scores = Q K^T and out = P V as FP16 GEMMs (FP32 accumulation, FP32 scores) per KV
                // head (its query heads one strided batch over the shared K / V), the causal softmax in between writing
                // FP16 probabilities; the queries in slices whose scores fit
                const int P = (int) p0 + T;
                const int ldp = (P + 7) / 8 * 8;
                const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
                const int G = H / HKV;
                const float one = 1.0f, zero = 0.0f;
                if (kv16_blk > 0 && P > kv16_blk) {
                    // more cells than the FP16 K / V hold (kv16_blk): block by block, each block's K / V converted once;
                    // per block and query slice its scores, the online softmax (k_attn_softmax16_blk: FP16 exp relative to
                    // the row's running max, the output so far rescaled) and out += P V; the sum divides out with the gate
                    const int ldb = (kv16_blk + 7) / 8 * 8;
                    const int tqb = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldb))));
                    k_to_f16<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(q, q16, (int64_t) T * H * HD);
                    for (int c0 = 0; c0 < P; c0 += kv16_blk) {
                        const int nb = std::min(kv16_blk, P - c0);
                        kv4([&](auto k8c, auto v8c) {
                            constexpr bool K8 = decltype(k8c)::value, V8 = decltype(v8c)::value;
                            k_kv_f16_blk<K8, V8><<<grid1((int64_t) HKV * nb * HD), kThreads, 0, st>>>(kc[li], vc[li], k16, v16,
                                                                                                    ctx, c0, nb, HD, HKV);
                        });
                        for (int t0 = 0; t0 < T; t0 += tqb) {
                            const int nt = std::min(tqb, T - t0);
                            for (int kvh = 0; kvh < HKV; ++kvh) {
                                const __half* Kh = k16 + (size_t) kvh * nb * HD;
                                const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * HD;
                                float* Sh = att_s + (size_t) kvh * G * nt * ldb;
                                if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, nb, nt, HD, &one, Kh, CUDA_R_16F,
                                                               HD, 0, Qh, CUDA_R_16F, H * HD, HD, &zero, Sh, CUDA_R_32F, ldb,
                                                               (long long) nt * ldb, G, CUBLAS_COMPUTE_32F,
                                                               CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                                    std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                                    std::exit(1);
                                }
                            }
                            k_attn_softmax16_blk<<<dim3(nt, H), kThreads, 0, st>>>(
                                att_s, att_p16, ldb, nt, nb, c0, (int) p0 + t0, 1.0f / sqrtf((float) HD), att_m + (size_t) t0 * H,
                                att_l + (size_t) t0 * H, att + (size_t) t0 * H * HD, H, HD, c0 == 0);
                            for (int kvh = 0; kvh < HKV; ++kvh) {
                                const __half* Vh = v16 + (size_t) kvh * nb * HD;
                                const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldb;
                                float* Oh = att + ((size_t) t0 * H + kvh * G) * HD;
                                if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, nb, &one, Vh, CUDA_R_16F,
                                                               HD, 0, Sh, CUDA_R_16F, ldb, (long long) nt * ldb,
                                                               c0 == 0 ? &zero : &one, Oh, CUDA_R_32F, H * HD, HD, G,
                                                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                                    std::fprintf(stderr, "strata-q36: attention values GEMM failed\n");
                                    std::exit(1);
                                }
                            }
                        }
                    }
                    k_gate_mul_l<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(att, gate, att_l, HD, (int64_t) T * H * HD);
                } else {
                kv4([&](auto k8c, auto v8c) {
                    constexpr bool K8 = decltype(k8c)::value, V8 = decltype(v8c)::value;
                    k_kv_f16<K8, V8><<<grid1((int64_t) HKV * P * HD), kThreads, 0, st>>>(kc[li], vc[li], k16, v16, ctx, P, HD, HKV);
                });
                k_to_f16<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(q, q16, (int64_t) T * H * HD);
                for (int t0 = 0; t0 < T; t0 += tq) {
                    const int nt = std::min(tq, T - t0);
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Kh = k16 + (size_t) kvh * P * HD;
                        const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * HD;
                        float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, HD, &one, Kh, CUDA_R_16F, HD,
                                                       0, Qh, CUDA_R_16F, H * HD, HD, &zero, Sh, CUDA_R_32F, ldp,
                                                       (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                            std::exit(1);
                        }
                    }
                    k_attn_softmax16<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, P, (int) p0 + t0,
                                                                       1.0f / sqrtf((float) HD));
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Vh = v16 + (size_t) kvh * P * HD;
                        const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                        float* Oh = att + ((size_t) t0 * H + kvh * G) * HD;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, P, &one, Vh, CUDA_R_16F, HD,
                                                       0, Sh, CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F,
                                                       H * HD, HD, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-q36: attention values GEMM failed\n");
                            std::exit(1);
                        }
                    }
                }
                k_gate_mul<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(att, gate, (int64_t) T * H * HD);
                }
            } else {
                // split-K over 256-cell chunks; the chunk grid covers every cell a row can see (graph mode: the whole
                // context, past-the-end chunks return at once)
                const int rows_per = att_rows / H;
                // STRATA_Q36_ATTN_GQA=1: k_attn_gqa (one block per token, KV head and chunk of gqa_cc cells) and its merge
                const bool gqa = gqa_cc > 0 && H == kGqaG * HKV && HD == 256;
                for (int t0 = 0; gqa && t0 < T; t0 += rows_per) {
                    const int nt = std::min(rows_per, T - t0);
                    const int ncg = (ctx + gqa_cc - 1) / gqa_cc;   // <= n_chunk: the partials keep their row stride
                    const int nc = graph_mode ? ncg : (int) std::min<int64_t>(ncg, (p0 + t0 + nt + gqa_cc - 1) / gqa_cc);
                    kv4([&](auto k8c, auto v8c) {
                        constexpr bool K8 = decltype(k8c)::value, V8 = decltype(v8c)::value;
                        k_attn_gqa<K8, V8><<<dim3(nt, HKV, nc), kThreads, 0, st>>>(q, rows_mode ? nullptr : kc[li],
                                                                                  rows_mode ? nullptr : vc[li], dpos, t0, H, HKV, ctx,
                                                                                  1.0f / sqrtf((float) HD), n_chunk, gqa_cc, pacc, pm,
                                                                                  pl, rs);
                    });
                    k_attn_merge_cc<<<nt * H, HD, 0, st>>>(pacc, pm, pl, gate, dpos, t0, H, n_chunk, gqa_cc, att, rs);
                }
                for (int t0 = 0; !gqa && t0 < T; t0 += rows_per) {
                    const int nt = std::min(rows_per, T - t0);
                    const int nc = graph_mode ? n_chunk
                                              : (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                    kv4([&](auto k8c, auto v8c) {
                        constexpr bool K8 = decltype(k8c)::value, V8 = decltype(v8c)::value;
                        k_attn_part<K8, V8><<<dim3(nc, nt * H), kChunkCells, 0, st>>>(q, rows_mode ? nullptr : kc[li],
                                                                                     rows_mode ? nullptr : vc[li], dpos, t0, H, HKV, ctx,
                                                                                     1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    });
                    k_attn_merge<<<nt * H, HD, 0, st>>>(pacc, pm, pl, gate, dpos, t0, H, n_chunk, att, rs);
                }
            }
            mm(L.wo, att, y, T);
        }
        add_rms_post(L, T);
        // routed experts (one token: the router and the shared expert's gate in one product, the gate its last row)
        const bool one = T == 1;
        // a decode batch of up to 8 rows takes the router and the shared expert's gate in one product too (rows of 257) when the fused
        // router follows; the prompt chunks and the checks keep the separate products
        const bool ext = one || (fast_k && std::getenv("STRATA_Q36_NO_EXT") == nullptr && !verify_k && T <= kGemmT && g.n_expert == 256 && Kp == 8);
        if (ext) f32mm(L.router_ext, h, logits_r, N, (int) g.n_expert + 1, T);
        else f32mm(L.router, h, logits_r, N, (int) g.n_expert, T);
        // the hybrid rows path: a batch of up to hybrid_t rows; when only the MTP block is hybrid, its every batch
        // (its prompt pass too: no staging buffers then)
        const bool hy_rows = lhy(li) && (T <= hybrid_t || !main_hybrid);
        const bool mmq_rows = !hy_rows && T > kGemmT && use_mmq;
        if (!(!mmq_rows && route_group(L, hy_rows ? hy[li].slot_of_d : nullptr, T, ext ? (int) g.n_expert + 1 : (int) g.n_expert)))
            K::native_router_topk(logits_r, ids, ewts, (int) g.n_expert, Kp, T, st);
        const int64_t n_ent = (int64_t) T * Kp;
        const int32_t* prow = nullptr;
        if (hy_rows) {
            moe_hybrid_gpu(L, li, T);
        } else if (T > kGemmT && use_mmq) {
            // a prompt chunk: the experts on llama.cpp's MMQ (int8, the weights read once per expert): the entries
            // sorted by expert on the device, gate+up as one [2 FF] product per expert, SwiGLU, down into rows the
            // combine reads through entry -> sorted row
            const int ne = (int) g.n_expert;
            CK(cudaMemsetAsync(rcnt, 0, (size_t) ne * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, ne, rbounds, rcur, rmax);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
            CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
            if (lhy(li)) CK(cudaMemcpyAsync(h_cnt + li * g.n_expert, rcnt, (size_t) ne * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (lhy(li))
                for (int x = 0; x < ne; ++x) hy[li].score[(size_t) x] += h_cnt[li * g.n_expert + x];
            const int64_t maxr = hpin[10];
            const uint8_t* wbase = lhy(li) ? stream_take((int) li) : L.blobs;
            MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
            MMQ::Product gu;
            gu.w = wbase; gu.type = L.L.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = L.stride;
            gu.n = ne; gu.xq = mq_x; gu.bounds = rbounds; gu.ids = ident; gu.total_rows = n_ent; gu.max_rows = maxr;
            gu.dst = mq_gu; gu.ld_dst = 2 * FF;
            mctx->run(gu, st);
            MMQ::swiglu(mq_gu, mq_h, n_ent, FF, false, st);
            MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_ent, st);
            MMQ::Product dn;
            dn.w = wbase + L.L.down_off; dn.type = L.L.d_type; dn.w_rows = N; dn.w_cols = FF;
            dn.expert_bytes = L.stride; dn.n = ne; dn.xq = mq_hq; dn.bounds = rbounds; dn.ids = ident;
            dn.total_rows = n_ent; dn.max_rows = maxr; dn.dst = eparts; dn.ld_dst = N;
            mctx->run(dn, st);
            if (lhy(li)) stream_done();
            prow = rinv;
        } else {
            group_entries(L, nullptr, T);
            if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                     std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
        }
        // the shared expert, scaled by sigmoid(w . h)
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) N * T), kThreads, 0, st>>>(h, x16, (int64_t) N * T);
        if (T <= kGemmT) {
            if (!shared_gu(L, T)) {
                apply(L.sh_gate, xq, sg, T);
                apply(L.sh_up, xq, su, T);
                K::native_swiglu_quantize_q8_1(sg, su, q8, FS, T, st);   // SwiGLU and its q8_1 image in one launch
            }
            mmq(L.sh_down, q8, sy, T);
        } else {
            apply(L.sh_gate, xq, sg, T);
            apply(L.sh_up, xq, su, T);
            k_swiglu<<<grid1((int64_t) T * FS), kThreads, 0, st>>>(sg, su, shh, (int64_t) T * FS);
            mm(L.sh_down, shh, sy, T);
        }
        if (!ext) f32mm(L.sh_gate_inp, h, sgate, N, 1, T);
        if (hy_rows) moe_hybrid_cpu(li, T);   // the CPU pool's experts (while the GPU runs the rest), merged
        // the combine, the shared expert's gate, the residual add and the next norm in one launch
        moe_end(T, prow, sy, ext ? logits_r + g.n_expert : sgate, wn, hout, ext ? (int) g.n_expert + 1 : 1);
    }

    /// The main model over T rows (device `tok`, positions p0 .. p0 + T - 1, also at *pos0): hmain gets every row's
    /// final normed hidden; `heads` rows from the end get logits (rows [T - heads, T): logits row r - (T - heads)) and
    /// their greedy picks (`pick`).  `verify`: a verify window (the GDN state left for commit()).
    void forward(int T, int64_t p0, int heads, bool graph_mode, bool verify) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (main_hybrid && T > hybrid_t && use_mmq) {
            std::vector<int> seq;
            for (size_t l = 0; l < m->layers.size(); ++l)
                if (lhy(l)) seq.push_back((int) l);
            stream_begin(seq);
        }
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(x, d_img, d_map, N);   // a picture's cells take its rows
        k_rms<<<T, kThreads, 0, st>>>(x, m->layers[0].attn_norm, h, N, m->eps);   // later layers: from k_moe_out
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const bool last = l + 1 == m->layers.size();
            layer(m->layers[l], l, T, p0, pos0, graph_mode, verify, last ? m->out_norm : m->layers[l + 1].attn_norm,
                  last ? hmain : h);
        }
        if (heads > 0) head_rows(hmain + (size_t) (T - heads) * N, heads, logits, pick);
    }
    /// logits [n][V] = W_out hn rows, and each row's greedy pick.
    void head_rows(const float* hn, int n, float* lg, int32_t* pk) {
        const auto& g = m->g;
        mm(m->head, hn, lg, n);
        k_argmax<<<dim3(kArgBlocks, n), kThreads, 0, st>>>(lg, (int) g.n_vocab, pk, amv, ami, false);
        k_argmax<<<dim3(1, n), kThreads, 0, st>>>(lg, kArgBlocks, pk, amv, ami, true);
    }

    /// The draft head: the MTP block's pick over the first `draft_n` rows of the output matrix only (the lowest token ids are
    /// the most frequent ones).  A draft only proposes: the verify window decides, so a token outside the subset is a
    /// rejected draft and nothing else - the answer is the plain greedy answer whatever draft_n is.  0 = the whole vocabulary.
    int draft_n = 0;
    void head_draft(const float* hn) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        const Mat& hd = m->head;
        if (!fast_k || draft_n <= 0 || draft_n >= g.n_vocab || draft_n % 8 != 0 || hd.type != 8 || hd.n_in != N || N / 32 != 64) {
            head_rows(hn, 1, logits_m, pick_m);
            return;
        }
        quant(hn, N, 1, q8);
        const uint8_t* pk = nullptr;
        for (const auto& p : m->packed)
            if (std::get<0>(p) == hd.d) pk = std::get<1>(p);
        if (pk) {
            const half* dpl = reinterpret_cast<const half*>(pk + (size_t) hd.n_in * hd.n_out);
            q36k::launch_q8_rows<1, 2, true>(pk, dpl, q8, logits_m, 64, draft_n, st);
        } else {
            q36k::launch_q8_rows<1, 2, false>(hd.d, nullptr, q8, logits_m, 64, draft_n, st);
        }
        k_argmax<<<dim3(kArgBlocks, 1), kThreads, 0, st>>>(logits_m, draft_n, pick_m, amv, ami, false);
        k_argmax<<<dim3(1, 1), kThreads, 0, st>>>(logits_m, kArgBlocks, pick_m, amv, ami, true);
    }

    /// The MTP block over T rows: row t reads token tok_m[t] at position p0 + t (also at *pos0_m) and the seed hidden
    /// seeds[t] (the main model's normed hidden of the position before, or the block's own for a chained draft);
    /// hmtp gets each row's normed output.  `draft`: logits of the last row and its pick (`pick_m`).
    void forward_mtp(int T, int64_t p0, const float* seeds, bool draft, bool graph_mode) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (main_hybrid && lhy(m->layers.size()) && T > hybrid_t && use_mmq) stream_begin({(int) m->layers.size()});
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(y, d_img, d_map, N);
        k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);              // [e | ...]
        k_rms_ld<<<T, kThreads, 0, st>>>(seeds, N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);      // [... | h]
        mm(m->eh_proj, mtp_cat, x, T);
        k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
        layer(m->mtp, m->layers.size(), T, p0, pos0_m, graph_mode, false, m->mtp_shnorm, hmtp);
        if (draft) head_draft(hmtp + (size_t) (T - 1) * N);
    }

    /// After a verify window of T rows: the GDN state takes the first n_keep rows (conv history and recurrence run
    /// again over the rows' kept inputs).
    void commit(int n_keep) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& L = m->layers[l];
            if (L.attn) continue;
            k_conv_seq<<<C / 128, 128, 0, st>>>(conv[l], v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, n_keep, true);
            k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[l], v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm, m->eps,
                                                   nullptr, HK, HV, n_keep, true);
        }
    }

    // ---- the hybrid mode
    const Layer& layer_at(int li) const { return li < (int) m->layers.size() ? m->layers[(size_t) li] : m->mtp; }

    /// A streamed forward over the state indices `seq`: the first n_stage layers' experts start copying now.
    void stream_begin(const std::vector<int>& seq) {
        stream_seq = seq;
        stream_pos = 0;
        for (int i = 0; i < n_stage && i < (int) seq.size(); ++i) stream_kick(i);
    }
    void stream_kick(int i) {   // the i-th layer of the sequence into staging buffer i % n_stage, after its last reader
        const int b = i % n_stage;
        const Layer& L = layer_at(stream_seq[(size_t) i]);
        CK(cudaStreamWaitEvent(cst, ev_used[b], 0));
        // the experts with a VRAM slot come from it (device to device, at VRAM speed), only the others over PCIe; runs
        // of consecutive experts from the same source in one copy each.  STRATA_Q36_STREAM_ALL=1: the whole layer from
        // host memory (the first version: PCIe-bound)
        static const bool all = std::getenv("STRATA_Q36_STREAM_ALL") != nullptr;
        const int ne = (int) m->g.n_expert;
        const Hy& H = hy[(size_t) stream_seq[(size_t) i]];
        // fewer than half the experts resident (a 12 GB card): the one whole-layer copy is faster than many small ones
        // (RTX 3060, UD-Q4_K_XL at 37 of 256: pp2048 1970 whole vs 1540 in runs)
        if (all || 2 * L.n_slots < ne) {
            CK(cudaMemcpyAsync(stage[b], L.hblobs, L.stride * (size_t) ne, cudaMemcpyHostToDevice, cst));
        } else {
            for (int x = 0; x < ne;) {
                const bool dev = H.slot_of[(size_t) x] >= 0;
                int y = x + 1;
                if (dev)   // consecutive slots holding consecutive experts
                    while (y < ne && H.slot_of[(size_t) y] == H.slot_of[(size_t) y - 1] + 1) ++y;
                else
                    while (y < ne && H.slot_of[(size_t) y] < 0) ++y;
                const size_t n = (size_t) (y - x) * L.stride;
                if (dev)
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.blobs + (size_t) H.slot_of[(size_t) x] * L.stride,
                                       n, cudaMemcpyDeviceToDevice, cst));
                else
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.hblobs + (size_t) x * L.stride, n,
                                       cudaMemcpyHostToDevice, cst));
                x = y;
            }
        }
        CK(cudaEventRecord(ev_copy[b], cst));
    }
    const uint8_t* stream_take(int li) {
        if (stream_pos >= (int) stream_seq.size() || stream_seq[(size_t) stream_pos] != li) {
            std::fprintf(stderr, "strata-q36: the expert stream is out of step (layer %d)\n", li);
            std::exit(1);
        }
        const int b = stream_pos % n_stage;
        CK(cudaStreamWaitEvent(st, ev_copy[b], 0));
        return stage[b];
    }
    void stream_done() {
        const int b = stream_pos % n_stage;
        CK(cudaEventRecord(ev_used[b], st));
        if (stream_pos + n_stage < (int) stream_seq.size()) stream_kick(stream_pos + n_stage);
        ++stream_pos;
    }

    /// The GPU half of a hybrid MoE: the routing to the host, the resident experts' entries grouped and run.
    void moe_hybrid_gpu(const Layer& L, size_t li, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaMemcpyAsync(h_h, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaEventRecord(ev_route, st));
        group_entries(L, hy[li].slot_of_d, T);
        if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
    }
    /// The CPU half: the entries without a slot on the pool (ggml-cpu's arithmetic for the layer's formats), their
    /// rows uploaded and merged into the entries' rows.
    void moe_hybrid_cpu(size_t li, int T) {
#if defined(STRATA_Q36_CPU_POOL)
        namespace C = strata::kernels::cpu;
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        Hy& H = hy[li];
        const Layer& L = layer_at((int) li);
        while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
        }
        const auto t0 = Clock::now();
        // the missed entries per expert (up to MAXT tokens per job)
        jobs.clear();
        std::vector<char> quantized((size_t) T, 0);
        int n_miss = 0;
        std::map<int, size_t> open;   // expert -> its job with room
        const bool defer = rows_deferred && li < m->layers.size();
        if (defer) {
            if (row_ids.size() < m->layers.size()) row_ids.resize(m->layers.size());
            row_ids[li].assign(h_ids, h_ids + n_ent);
        }
        for (int e = 0; e < n_ent; ++e) {
            const int x = h_ids[e];
            if (!defer) H.score[(size_t) x] += 1.0;
            h_miss[e] = H.slot_of[(size_t) x] < 0;
            if (!h_miss[e]) continue;
            ++n_miss;
            const int t = e / Kp;
            if (!quantized[(size_t) t]) {
                C::native_quant_act(H.fmt, h_h + (size_t) t * N, acts + (size_t) t * C::kNativeActBytes);
                quantized[(size_t) t] = 1;
            }
            auto it = open.find(x);
            if (it == open.end() || jobs[it->second].nt == C::MAXT) {
                C::ExpertJobMulti j;
                j.blob = L.hblobs + (size_t) x * L.stride;
                jobs.push_back(j);
                it = open.insert_or_assign(x, jobs.size() - 1).first;
            }
            C::ExpertJobMulti& j = jobs[it->second];
            j.nact[j.nt] = acts + (size_t) t * C::kNativeActBytes;
            j.out[j.nt] = cpu_out + (size_t) e * N;
            ++j.nt;
        }
        hy_cpu += n_miss;
        hy_gpu += n_ent - n_miss;
        if (n_miss == 0) return;
        pool->run_split_multi_native(H.fmt, jobs.data(), (int) jobs.size());
        hy_cpu_ms += ms_since(t0);
        CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
#else
        (void) li;
        (void) T;
#endif
    }

    /// The slots to the experts the routing used most (the counts halved each time): at most max_swaps replacements
    /// over all layers, between forwards (the copies are on the engine's stream).
    void rebalance(int max_swaps) {
        static const bool off = std::getenv("STRATA_Q36_FIXED_SLOTS") != nullptr;   // the slots as placed (tests)
        if (!m->hybrid || off || warming) return;
        const int64_t ne = m->g.n_expert;
        int left = max_swaps;
        for (size_t li = 0; li < hy.size() && left > 0; ++li) {
            Hy& H = hy[li];
            const Layer& L = layer_at((int) li);
            const int S = L.n_slots;
            if (S <= 0 || S >= ne) continue;
            std::vector<int> order((size_t) ne);
            for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return H.score[(size_t) a] > H.score[(size_t) b]; });
            // residents outside the top S, lowest score first: the slots to give away
            std::vector<int> victims;
            for (int s = 0; s < S; ++s) victims.push_back(s);
            std::sort(victims.begin(), victims.end(), [&](int a, int b) {
                return H.score[(size_t) H.exp_in_slot[(size_t) a]] < H.score[(size_t) H.exp_in_slot[(size_t) b]];
            });
            size_t vi = 0;
            bool changed = false;
            for (int i = 0; i < S && left > 0; ++i) {
                const int x = order[(size_t) i];
                if (H.slot_of[(size_t) x] >= 0) continue;
                while (vi < victims.size()) {
                    const int s = victims[vi];
                    const int y = H.exp_in_slot[(size_t) s];
                    if (H.score[(size_t) y] >= H.score[(size_t) x]) { vi = victims.size(); break; }
                    // y must not be in the top S itself
                    ++vi;
                    H.slot_of[(size_t) y] = -1;
                    H.slot_of[(size_t) x] = s;
                    H.exp_in_slot[(size_t) s] = x;
                    CK(cudaMemcpyAsync(L.blobs + (size_t) s * L.stride, L.hblobs + (size_t) x * L.stride, L.L.bytes,
                                       cudaMemcpyHostToDevice, st));
                    --left;
                    ++hy_swaps;
                    changed = true;
                    break;
                }
                if (vi >= victims.size()) break;
            }
            if (changed)
                CK(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
            for (double& v : H.score) v *= 0.5;
        }
        CK(cudaStreamSynchronize(st));   // slot_of's host copy is pageable: the upload is done before it changes
    }

    void shrink_att(size_t floats) {
        CK(cudaDeviceSynchronize());
        CK(cudaFree(att_s));
        CK(cudaFree(att_p16));
        kAttScoreFloats = floats;
        att_s = dalloc(kAttScoreFloats);
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        CK(cudaDeviceSynchronize());
    }
    void reset_graphs() {
        for (GSet& gset : gsets) {
            for (cudaGraphExec_t* g : {&gset.g1, &gset.gm1})
                if (*g) { CK(cudaGraphExecDestroy(*g)); *g = nullptr; }
            for (auto& g : gset.gv)
                if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
        }
        for (auto& kv : row_graphs)
            if (kv.second) CK(cudaGraphExecDestroy(kv.second));
        row_graphs.clear();
    }

    template <class F> void graph_run(cudaGraphExec_t& gx, F body) {
        if (!use_graph) { body(); return; }
        if (!gx) {
            cudaGraph_t graph = nullptr;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            body();
            CK(cudaStreamEndCapture(st, &graph));
            CK(cudaGraphInstantiate(&gx, graph, 0));
            CK(cudaGraphDestroy(graph));
        }
        CK(cudaGraphLaunch(gx, st));
    }

    // ---- pictures (--vision).  A request with pictures hands the engine its rows (the encoder's, in prompt order), the
    // cell -> row map of its <|image_pad|> cells and the M-RoPE table of its prompt cells; the cells below the prompt
    // length take their rotary positions from the table, the text after it continues at cell + delta (what the pictures
    // did not use of their cells: llama.cpp's mtmd advances the text position by max(nx, ny) per picture).
    bool img_active = false;
    void set_images(const std::vector<float>& rows, const std::vector<int32_t>& cell_row, const std::vector<int32_t>& mrope,
                    int64_t n_prompt, int32_t delta) {
        const size_t need = rows.size();
        if (need > img_cap) {
            CK(cudaDeviceSynchronize());
            if (d_img) CK(cudaFree(d_img));
            CK(cudaMalloc((void**) &d_img, need * 4));
            img_cap = need;
        }
        CK(cudaMemcpy(d_img, rows.data(), need * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_mtab, mrope.data(), mrope.size() * 4, cudaMemcpyHostToDevice));
        img_map = cell_row;
        img_prompt_cells = n_prompt;
        img_active = true;
        set_delta(delta);
    }
    void clear_images() {
        img_map.clear();
        img_active = false;
        img_prompt_cells = 0;
        mtab_now = nullptr;
        map_on = false;
        set_delta(0);
    }
    /// A forward over the cells [c0, c0 + T): the picture rows of its cells, and the table when it is a prompt's.
    void chunk_pictures(int64_t c0, int T) {
        mtab_now = nullptr;
        map_on = false;
        if (!img_active) return;
        if (c0 < img_prompt_cells) mtab_now = d_mtab;
        bool any = false;
        for (int j = 0; j < T; ++j) {
            const int64_t c = c0 + j;
            const int32_t v = c < (int64_t) img_map.size() ? img_map[(size_t) c] : -1;
            h_map[j] = v;
            any = any || v >= 0;
        }
        if (any) {
            CK(cudaMemcpyAsync(d_map, h_map, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            map_on = true;
        }
    }

    /// The tokens at the current position: a prompt in chunks (eager), one token from the decode graph.  The last
    /// token's logits stay in `logits` (its greedy pick in `pick`), hlast its normed hidden.  With the MTP block the
    /// prompt also runs through it (its K / V for every position: row q reads h_{q-1} and token q).  on_chunk(done) runs
    /// after every chunk; false ends the read there (a cancelled request): the tokens processed are returned.
    template <class F> int64_t process(const int32_t* toks, int64_t n, F on_chunk_, bool rebal = true) {
        const int N = (int) m->g.n_embd;
        auto on_chunk = [&](int64_t done) -> bool {   // a callback returning void never cancels
            if constexpr (std::is_void_v<decltype(on_chunk_(done))>) {
                on_chunk_(done);
                return true;
            } else {
                return on_chunk_(done);
            }
        };
        if (n == 1) {
            CK(cudaStreamSynchronize(st));   // the pinned staging is free again
            hpin[0] = toks[0];
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            chunk_pictures(pos, 1);
            rows_deferred = true;
            // a prompt cell of a picture request (its rotary position is in the table, its row a picture's) runs eagerly
            if (mtab_now || map_on) forward(1, pos, 1, false, false);
            else graph_run(gs().g1, [&] { forward(1, pos, 1, use_graph, false); });
            mtab_now = nullptr;
            map_on = false;
            rows_deferred = false;
            CK(cudaMemcpyAsync(hlast, hmain, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += 1;
            after_rows(1);
            on_chunk(1);
            return 1;
        }
        for (int64_t i = 0; i < n; i += tmax) {
            const int T = (int) std::min<int64_t>(tmax, n - i);
            CK(cudaStreamSynchronize(st));
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            chunk_pictures(pos, T);
            forward(T, pos, i + T == n ? 1 : 0, false, false);
            if (m->has_mtp && mtp_d > 0) {
                // seeds: h_{q-1} for each row q: hlast, then this chunk's rows 0 .. T - 2 (in `y`, free here)
                CK(cudaMemcpyAsync(mtp_seed(), hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                if (T > 1)
                    CK(cudaMemcpyAsync(mtp_seed() + N, hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(tok_m, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
                forward_mtp(T, pos, mtp_seed(), false, false);
            }
            CK(cudaMemcpyAsync(hlast, hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += T;
            CK(cudaStreamSynchronize(st));
            mtab_now = nullptr;
            map_on = false;
            if (!on_chunk(i + T)) {
                if (m->hybrid) rebalance(hybrid_swaps_after_prompt);
                return i + T;
            }
        }
        if (m->hybrid && rebal) rebalance(hybrid_swaps_after_prompt);
        return n;
    }

    // ---- batch rows: one decode window for the next token of several sessions
    struct Row { int slot; int32_t tok; };
    long long win_count = 0;
    std::map<int, cudaGraphExec_t> row_graphs;   // by (rows, groups, kind): graphs of the batch windows
    /// A slot's rows of one window: `n` consecutive cells from the session's position (1: the next token; 2: also an MTP
    /// proposal the window verifies).
    struct Group { int slot; int n; int32_t tok[2]; };
    int32_t rows_pick[kVerify] = {};
    int rows_G_last = 0;   // groups of the window run last (commit_rows takes their keep counts)

    void run_graph_key(int key, const std::function<void()>& body) {
        if (!use_graph) { body(); return; }
        graph_run(row_graphs[key], body);
    }

    /// One window over the groups' rows.  `verify`: a verify window - the GDN state is read, not written (commit_rows writes
    /// it for the kept rows) and the sessions' positions stay.  The greedy picks end up in rows_pick (one per row), every
    /// row's logits in `logits`.  Without verify the sessions' positions advance by their rows.
    void run_rows(const Group* gr, int G, bool verify) {
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));   // the pinned staging is free again
        int64_t pmax = 0;
        int T = 0;
        for (int g = 0; g < G; ++g) {
            const Sess& ss = sess[(size_t) gr[g].slot];
            hrow[40 + g] = gr[g].slot;
            hrow[48 + g] = T;
            hrow[56 + g] = gr[g].n;
            for (int j = 0; j < gr[g].n; ++j) {
                hrow[T] = gr[g].tok[j];
                hrow[8 + T] = (int32_t) (ss.pos + j);
                hrow[16 + T] = gr[g].slot;
                hrow[24 + T] = ss.rdelta;
                ++T;
            }
            pmax = std::max<int64_t>(pmax, ss.pos + gr[g].n - 1);
        }
        CK(cudaMemcpyAsync(tok, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gslot, hrow + 40, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_grow0, hrow + 48, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = G;
        rows_deferred = true;
        run_graph_key((verify ? 1000 : 0) + G * 10 + (T - G), [&] { forward(T, pmax, T, use_graph, verify); });
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) rows_pick[r] = hrow[32 + r];
        if (!verify)
            for (int g = 0; g < G; ++g) sess[(size_t) gr[g].slot].pos += gr[g].n;
        pos = sess[(size_t) cur].pos;
        rows_G_last = G;
        if (m->hybrid) {
            count_rows(T);
            if (++win_count % kRebalanceEvery == 0) rebalance(8);
        }
    }
    /// One window for the next token of each of the rows' sessions (no proposals): the greedy picks in rows_pick.
    void step_rows(const Row* rows, int nb) {
        Group gr[kVerify];
        for (int r = 0; r < nb; ++r) gr[r] = {rows[r].slot, 1, {rows[r].tok, 0}};
        run_rows(gr, nb, false);
    }
    /// After run_rows(.., verify = true): the GDN state of group g takes its first keep[g] rows (the conv history and the
    /// recurrence run again over the rows' kept inputs, as commit() does) and the session advances by keep[g].
    void commit_rows(const Group* gr, const int* keep, int G) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        sess[(size_t) cur].pos = pos;
        for (int i = 0; i < G; ++i) hrow[56 + i] = keep[i];   // (hrow's slot / first-row words are the window's, unchanged)
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        rows_mode = true;
        rows_G = G;
        run_graph_key(2000 + G, [&] {
            for (size_t l = 0; l < m->layers.size(); ++l) {
                const Layer& L = m->layers[l];
                if (L.attn) continue;
                k_conv_seq<<<dim3(C / 128, G), 128, 0, st>>>(nullptr, v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, 1, true,
                                                             d_conv_tab + l * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, G), dim3(128, 4), 0, st>>>(nullptr, v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm,
                                                                m->eps, nullptr, HK, HV, 1, true, d_S_tab + l * n_sess,
                                                                d_gslot, d_grow0, d_gcnt);
            }
        });
        rows_mode = false;
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < G; ++i) sess[(size_t) gr[i].slot].pos += keep[i];
        pos = sess[(size_t) cur].pos;
    }

    /// The MTP block over rows of several sessions: row r reads token tok at cell `cell` of session `slot` and the seed
    /// hidden at `seed` (device, n_embd floats); its draft - the block's greedy pick after that row - ends up in
    /// mtp_pick[r], and the block's K / V cells are written.
    struct MtpRow { int slot; int32_t tok; int64_t cell; const float* seed; };
    int32_t mtp_pick[kVerify] = {};
    float* logits_mr = nullptr;
    int32_t* pick_mr = nullptr;
    void mtp_rows(const MtpRow* rows, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (!logits_mr) {
            logits_mr = dalloc((size_t) kVerify * g.n_vocab);
            pick_mr = dalloc_t<int32_t>(kVerify);
        }
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));
        int64_t pmax = 0;
        for (int r = 0; r < T; ++r) {
            hrow[r] = rows[r].tok;
            hrow[8 + r] = (int32_t) rows[r].cell;
            hrow[16 + r] = rows[r].slot;
            hrow[24 + r] = sess[(size_t) rows[r].slot].rdelta;
            pmax = std::max<int64_t>(pmax, rows[r].cell);
            CK(cudaMemcpyAsync(mtp_seed() + (size_t) r * N, rows[r].seed, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        }
        CK(cudaMemcpyAsync(tok_m, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = T;
        rows_deferred = true;
        run_graph_key(3000 + T, [&] {
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
            k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);                 // [e | ...]
            k_rms_ld<<<T, kThreads, 0, st>>>(mtp_seed(), N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);   // [... | h]
            mm(m->eh_proj, mtp_cat, x, T);
            k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
            layer(m->mtp, m->layers.size(), T, pmax, pos0_m, use_graph, false, m->mtp_shnorm, hmtp);
            mm(m->head, hmtp, logits_mr, T);
            k_argmax<<<dim3(kArgBlocks, T), kThreads, 0, st>>>(logits_mr, (int) g.n_vocab, pick_mr, amv, ami, false);
            k_argmax<<<dim3(1, T), kThreads, 0, st>>>(logits_mr, kArgBlocks, pick_mr, amv, ami, true);
        });
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick_mr, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) mtp_pick[r] = hrow[32 + r];
    }
    /// Row r's logits of the last step_rows, on the host.
    void host_logits_row(int r, std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits + (size_t) r * m->g.n_vocab, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }

    /// Session dst takes session src's state: the recurrent state, the cache's first src.pos cells, the position, the
    /// pictures' offset (for a conversation read in one session that continues in another).
    void copy_session(int src, int dst) {
        if (src == dst) return;
        sess[(size_t) cur].pos = pos;
        const auto& g = m->g;
        const Sess& a = sess[(size_t) src];
        Sess& b = sess[(size_t) dst];
        const size_t C = (size_t) g.ssm_conv_channels;
        const size_t row = (size_t) ctx * g.head_dim * 4;
        for (size_t l = 0; l < a.conv.size(); ++l) {
            if (a.conv[l]) {
                CK(cudaMemcpyAsync(b.conv[l], a.conv[l], C * (g.ssm_d_conv - 1) * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(b.S[l], a.S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4,
                                   cudaMemcpyDeviceToDevice, st));
            }
            if (a.kc[l] && a.pos > 0 && !kv_k8 && !kv_v8) {
                CK(cudaMemcpy2DAsync(b.kc[l], row, a.kc[l], row, (size_t) a.pos * g.head_dim * 4, (size_t) g.n_head_kv,
                                     cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpy2DAsync(b.vc[l], row, a.vc[l], row, (size_t) a.pos * g.head_dim * 4, (size_t) g.n_head_kv,
                                     cudaMemcpyDeviceToDevice, st));
            } else if (a.kc[l] && a.pos > 0) {   // --kv: an int8 store's first pos cells are its codes' and its scales'
                const int HKV_ = (int) g.n_head_kv, HD_ = (int) g.head_dim;
                auto cp = [&](float* d, const float* s_, bool i8) {
                    if (!i8) {
                        CK(cudaMemcpy2DAsync(d, row, s_, row, (size_t) a.pos * g.head_dim * 4, (size_t) g.n_head_kv,
                                             cudaMemcpyDeviceToDevice, st));
                        return;
                    }
                    const size_t ng = (size_t) HD_ / kKVG;
                    CK(cudaMemcpy2DAsync(d, (size_t) ctx * HD_, s_, (size_t) ctx * HD_, (size_t) a.pos * HD_, (size_t) HKV_,
                                         cudaMemcpyDeviceToDevice, st));
                    CK(cudaMemcpy2DAsync(kv8_scales(d, HKV_, ctx, HD_), (size_t) ctx * ng * 2, kv8_scales(s_, HKV_, ctx, HD_),
                                         (size_t) ctx * ng * 2, (size_t) a.pos * ng * 2, (size_t) HKV_,
                                         cudaMemcpyDeviceToDevice, st));
                };
                cp(b.kc[l], a.kc[l], kv_k8);
                cp(b.vc[l], a.vc[l], kv_v8);
            }
        }
        CK(cudaMemcpyAsync(b.hlast, a.hlast, (size_t) g.n_embd * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaStreamSynchronize(st));
        b.pos = a.pos;
        b.rdelta = a.rdelta;
        b.live = a.live;
        b.imgs = a.imgs;
        b.ck_n = 0;
        if (dst == cur) load_cur();
    }
    int hybrid_swaps_after_prompt = 2048;
    // STRATA_Q36_TURN_CKPT=1 (opt-in): a prompt is read to its last <|im_start|>, the recurrent state there (each GDN
    // layer's conv and S, hlast: ~63 MiB for Qwen3.6-35B-A3B) is copied to pinned host memory, then the new turn's header
    // is read.  The next turn of the chat re-renders that history the same way but not the header and the answer (the
    // template drops the empty <think></think> of earlier answers), so it resumes there instead of reading from token 0.
    bool turn_ckpt = false;
    /// The bytes of the current session's recurrent state: each GDN layer's conv and S, hlast.
    size_t rec_bytes() const {
        const auto& g = m->g;
        const size_t cb = (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4;
        const size_t sb = (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4;
        size_t bytes = (size_t) g.n_embd * 4;
        for (float* c : conv) bytes += c ? cb + sb : 0;
        return bytes;
    }
    /// The current session's recurrent state to (save) or from host memory `h` (rec_bytes() bytes).
    void rec_copy(void* host, bool save) {
        const auto& g = m->g;
        const size_t cb = (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4;
        const size_t sb = (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4;
        char* h = (char*) host;
        auto cp = [&](void* d, size_t n) {
            CK(cudaMemcpyAsync(save ? (void*) h : d, save ? d : (void*) h, n,
                               save ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, st));
            h += n;
        };
        for (size_t l = 0; l < conv.size(); ++l)
            if (conv[l]) { cp(conv[l], cb); cp(S[l], sb); }
        cp(hlast, (size_t) g.n_embd * 4);
        CK(cudaStreamSynchronize(st));
    }
    /// The current session's checkpoint buffer (pinned, rec_bytes()), allocated on first use.
    void* ck_buf() {
        Sess& ss = sess[(size_t) cur];
        if (!ss.ck_host) CK(cudaHostAlloc(&ss.ck_host, rec_bytes(), cudaHostAllocDefault));
        return ss.ck_host;
    }
    void ckpt_copy(bool save) { rec_copy(ck_buf(), save); }

    // STRATA_Q36_PARK_MIB=N (opt-in): conversations parked in host RAM (see Serve::park).  A parked image holds the
    // first n cells of every K / V store (layer by layer: K, then V; an int8 store's codes, then its scales), packed
    // with a host row pitch of `cap` cells, so a restore can bring back any n <= cap of them.
    /// Host bytes of `cells` cells of all the K / V stores.
    size_t kv_cells_bytes(int64_t cells) const {
        const size_t HKV = (size_t) m->g.n_head_kv, HD = (size_t) m->g.head_dim, ng = HD / kKVG;
        auto one = [&](bool i8) { return i8 ? HKV * (HD + ng * 2) : HKV * HD * 4; };
        size_t per = 0;
        for (float* k : kc) per += k ? one(kv_k8) + one(kv_v8) : 0;
        return per * (size_t) cells;
    }
    /// The current session's cells [0, n) to (save) or from a parked image of `cap` cells at `host`.
    void kv_copy(char* host, int64_t cap, int64_t n, bool save) {
        if (n <= 0) return;
        const int HKV = (int) m->g.n_head_kv, HD = (int) m->g.head_dim;
        const size_t ng = (size_t) HD / kKVG;
        char* h = host;
        auto cp2 = [&](void* d, size_t dpitch, size_t hpitch, size_t width) {
            if (save) CK(cudaMemcpy2DAsync(h, hpitch, d, dpitch, width, (size_t) HKV, cudaMemcpyDeviceToHost, st));
            else CK(cudaMemcpy2DAsync(d, dpitch, h, hpitch, width, (size_t) HKV, cudaMemcpyHostToDevice, st));
            h += hpitch * (size_t) HKV;
        };
        auto store = [&](float* d, bool i8) {
            if (!i8) {
                cp2(d, (size_t) ctx * HD * 4, (size_t) cap * HD * 4, (size_t) n * HD * 4);
                return;
            }
            cp2(d, (size_t) ctx * HD, (size_t) cap * HD, (size_t) n * HD);
            cp2(kv8_scales(d, HKV, ctx, HD), (size_t) ctx * ng * 2, (size_t) cap * ng * 2, (size_t) n * ng * 2);
        };
        for (size_t l = 0; l < kc.size(); ++l)
            if (kc[l]) { store(kc[l], kv_k8); store(vc[l], kv_v8); }
        CK(cudaStreamSynchronize(st));
    }
    /// The current session's state after its first n cells becomes its checkpoint.
    void ckpt_save(int64_t n) {
        ckpt_copy(true);
        sess[(size_t) cur].ck_n = n;
        sess[(size_t) cur].ck_rdelta = sess[(size_t) cur].rdelta;
    }
    /// The current session goes back to its checkpoint (the caller cuts `live` to ck_n).
    void ckpt_load() {
        Sess& ss = sess[(size_t) cur];
        ckpt_copy(false);
        pos = ss.ck_n;
        ss.pos = pos;
        set_delta(ss.ck_rdelta);
    }
    /// the MTP seed rows of a prompt chunk: a buffer the block's own forward does not use (`moe_out`, [tmax][N])
    float* mtp_seed() { return moe_out; }

    /// One speculative step with D MTP drafts (greedy): `x` is the next token (decided, not processed); `fix` the
    /// tokens of the rows the block must redo first (the drafts the last window accepted, seeded by hmain rows
    /// 0 .. fix.size() - 1 of that window).  Drafts d1 .. dD, then the window [x, d1 .. dD] through the main model;
    /// returns the accepted count a and writes the picks of rows 0 .. a (row a's pick is the token after them).
    /// `cancel`: a stop asked for before the state moves keeps only x (the caller then emits no draft), so that a request
    /// stopped mid-answer leaves the state at the tokens it has shown: a conversation that continues from them (the server's
    /// hand-over of a stopped request to a batch slot) reuses it instead of reading them again.
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks,
                  const std::atomic<bool>* cancel = nullptr, const std::function<int32_t(int)>* repick = nullptr) {
        const int N = (int) m->g.n_embd;
        if (main_hybrid) D = std::min<int>(D, (int) (kRebalanceEvery - pos % kRebalanceEvery) - 1);
        const int T0 = (int) fix.size() + 1;   // the redo rows and the row of x
        CK(cudaStreamSynchronize(st));
        // rows of the first MTP call: (hmain row i, fix[i]) at positions pos - fix.size() + i, then (hlast, x) at pos
        for (int i = 0; i < (int) fix.size(); ++i) hpin[16 + i] = fix[(size_t) i];
        hpin[16 + fix.size()] = x_tok;
        hpin[1] = (int32_t) (pos - (int64_t) fix.size());
        CK(cudaMemcpyAsync(tok_m, hpin + 16, (size_t) T0 * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        // seeds: hmain rows 0 .. fix - 1 are the h of the positions before the fixed rows; hlast is that before x
        float* seeds = mtp_seed();
        if (!fix.empty())
            CK(cudaMemcpyAsync(seeds, hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0, false);
        drafts.assign((size_t) D, 0);
        if (D > 0) {
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[0] = hpin[2];
        }
        int last_row = T0 - 1;   // the row of the previous MTP call whose output seeds the next draft
        for (int d = 1; d < D; ++d) {   // chained: (the block's own output, the last draft) at the next position
            hpin[0] = drafts[(size_t) d - 1];
            hpin[1] = (int32_t) (pos + d);
            CK(cudaMemcpyAsync(tok_m, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(seeds, hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            last_row = 0;
            if (mtp_graph) graph_run(gs().gm1, [&] { forward_mtp(1, pos + d, seeds, true, use_graph); });
            else forward_mtp(1, pos + d, seeds, true, false);
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[(size_t) d] = hpin[2];
        }
        // the verify window
        const int T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        hpin[1] = (int32_t) pos;
        CK(cudaMemcpyAsync(tok, hpin + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        rows_deferred = true;
        graph_run(gs().gv[T], [&] { forward(T, pos, T, use_graph, true); });
        rows_deferred = false;
        CK(cudaMemcpyAsync(hpin + 2, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        int a = 0;
        if (repick != nullptr) {   // greedy with penalties: each row's pick from its penalized logits (row r: after x, d1 .. dr)
            for (;;) {
                hpin[2 + a] = (*repick)(a);
                if (a == D || hpin[2 + a] != drafts[(size_t) a]) break;
                ++a;
            }
        } else {
            while (a < D && hpin[2 + a] == drafts[(size_t) a]) ++a;
        }
        if (cancel != nullptr && cancel->load()) a = 0;
        for (int i = 0; i <= a; ++i) picks[i] = hpin[2 + i];
        commit(a + 1);
        CK(cudaMemcpyAsync(hlast, hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        pos += a + 1;
        after_rows(a + 1);
        return a;
    }

    /// The first prompt after load ran at about half speed (pp4096 rep 0: ~1770 tok/s against ~3290): the GPU's code
    /// objects, the BLAS handles' kernels and the decode graph are made on first use.  A throw-away prompt of one
    /// chunk, a decode step and (with drafts) a speculative step at load do that before the first request; then the
    /// state, the cache position and the hybrid counters are as after load.
    bool warming = false;
    void warmup() {
        const auto t0 = Clock::now();
        warming = true;
        const int32_t filler = 198;   // a newline: any token works
        std::vector<int32_t> toks((size_t) tmax, filler);
        process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
        if (tmax > kGemmT + 1) process(toks.data(), kGemmT + 1, [](int64_t) {});   // the smallest GEMM chunk too
        process(&filler, 1, [](int64_t) {});
        if (mtp_d > 0) {
            std::vector<int32_t> fix, drafts;
            int32_t picks[16];
            spec_step(filler, fix, mtp_d, drafts, picks);
        }
        CK(cudaStreamSynchronize(st));
        warming = false;
        for (Hy& H : hy) std::fill(H.score.begin(), H.score.end(), 0.0);
        hy_cpu = hy_gpu = hy_swaps = 0;
        hy_cpu_ms = 0;
        reset();
        std::fprintf(stderr, "strata-q36: warmed up in %.1f s (one prompt chunk, a decode step%s)\n",
                     ms_since(t0) / 1000.0, mtp_d > 0 ? ", a speculative step" : "");
    }

    int greedy() {
        CK(cudaMemcpyAsync(hpin + 2, pick, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
/// The routed experts after the engine's buffers: all in VRAM when they fit in `vram_gib` (< 0: the free VRAM less a
/// reserve), else the hybrid mode - every expert in host memory, the same number of VRAM slots per layer, a decode
/// batch's other experts on the CPU pool and a prompt chunk's streamed through staging buffers (double-buffered, or
/// one with STRATA_Q36_SINGLE_STAGE=1 or an int8 K / V store short of VRAM).
bool place_experts(Model& m, Engine& E, double vram_gib, int threads, int hybrid_t, int slots, std::string& err) {
    const auto t0 = Clock::now();
    const auto& g = m.g;
    const int64_t ne = g.n_expert;
    std::vector<Layer*> all;
    for (Layer& L : m.layers) all.push_back(&L);
    if (m.has_mtp) all.push_back(&m.mtp);
    size_t total = 0, per_slot = 0, max_layer = 0;
    for (Layer* L : all) {
        total += L->stride * (size_t) ne + kMmqTail;
        per_slot += L->stride;
        max_layer = std::max(max_layer, L->stride * (size_t) ne);
    }
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    // STRATA_Q36_RESERVE_MIB=N: the VRAM kept free beside the slots (default 768) - what cuBLAS, MMQ and the first prompt
    // allocate later; a long-context 8 GB card trades it for expert slots (a decode token reads more experts from VRAM)
    const size_t reserve = (std::getenv("STRATA_Q36_RESERVE_MIB") ? (size_t) std::atoll(std::getenv("STRATA_Q36_RESERVE_MIB")) : (size_t) 768) << 20;
    const size_t budget = vram_gib >= 0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                        : (fr > reserve ? fr - reserve : 0);
    // every expert fits once the packed Q8_0 decode copies are given back (a file a few GiB larger than the card's
    // room, UD-Q6_K_XL on 32 GB): drop them - the decode reads the GGUF layout then, bitwise the same outputs, a little
    // slower - rather than run the hybrid mode with its CPU share and streamed prompt layers.  STRATA_Q36_KEEP_PACKED=1
    // keeps them (and the hybrid mode).
    size_t packed_bytes = 0;
    for (const auto& pk : m.packed) packed_bytes += std::get<2>(pk);
    size_t main_need = 0;   // the main layers' experts (the MTP block's can go to the CPU, see below)
    for (Layer& L : m.layers) main_need += L.stride * (size_t) ne + kMmqTail;
    const size_t slack = ((size_t) 512 << 20) + (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // reserve + scores
    const bool mtp_cpu_ok = m.has_mtp && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr;
    // On small cards the optional decode copies can leave too little room for the hybrid prompt's staging pair
    // and CPU-output scratch, even with no expert slots. Keep the same reserve as the placement budget below.
    const size_t hybrid_scratch = (size_t) E.tmax * (size_t) m.top_k * ((size_t) g.n_embd + 1) * 4;
    const bool staging_short = E.use_mmq && total > budget &&
                              (size_t) E.n_stage * (max_layer + kMmqTail) + hybrid_scratch + reserve > fr;
    if (slots <= 0 && vram_gib < 0 && total > budget && packed_bytes > 0 &&
        (total <= budget + packed_bytes || (mtp_cpu_ok && main_need <= budget + packed_bytes + slack) || staging_short) &&
        std::getenv("STRATA_Q36_KEEP_PACKED") == nullptr) {
        CK(cudaDeviceSynchronize());
        for (const auto& pk : m.packed) {
            K::native_q8_0_packed_unregister(std::get<0>(pk));
            CK(cudaFree(std::get<1>(pk)));
        }
        m.packed.clear();
        E.reset_graphs();
        CK(cudaMemGetInfo(&fr, &tot));
        if (staging_short)
            std::fprintf(stderr, "strata-q36: the packed Q8_0 decode copies (%.2f GiB) given back for the hybrid "
                         "prompt's staging buffers (STRATA_Q36_KEEP_PACKED=1 keeps them)\n",
                         (double) packed_bytes / (1 << 30));
        else std::fprintf(stderr, "strata-q36: the packed Q8_0 decode copies (%.2f GiB) given back so that every routed "
                     "expert fits in VRAM (STRATA_Q36_KEEP_PACKED=1 keeps them and runs the hybrid mode)\n",
                     (double) packed_bytes / (1 << 30));
    }
    size_t budget2 = vram_gib >= 0 ? budget : (fr > reserve ? fr - reserve : 0);
    // the MAIN layers' experts fit, the MTP block's not all (UD-Q6_K_XL on 32 GB): the main layers all in VRAM - the
    // text is the all-in-VRAM text, the decode graph stays - and the MTP block's experts on slots + the CPU pool (its
    // drafts only).  When that needs it: the attention scores buffer at 64 MB instead of 256 MB (more, smaller
    // slices in a long prompt) and the reserve at 256 MiB instead of 768 (said in the log; --expert-vram sets the
    // budget instead).  STRATA_Q36_NO_MTP_CPU=1: the plain hybrid mode for every layer.
    size_t total_main = 0;
    for (Layer& L : m.layers) total_main += L.stride * (size_t) ne + kMmqTail;
    bool mtp_only = false;
    if (slots <= 0 && vram_gib < 0 && m.has_mtp && total > budget2 && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr) {
        const size_t att_save = (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // att_s (4 B) + att_p16 (2 B)
        const size_t small_reserve = (size_t) 256 << 20;
        const size_t room = fr + att_save > small_reserve ? fr + att_save - small_reserve : 0;
        if (total_main + ((size_t) 64 << 20) <= budget2) {
            mtp_only = true;
        } else if (total_main + ((size_t) 64 << 20) <= room) {
            E.shrink_att((size_t) 16 << 20);
            CK(cudaMemGetInfo(&fr, &tot));
            budget2 = fr > small_reserve ? fr - small_reserve : 0;
            mtp_only = total_main + ((size_t) 64 << 20) <= budget2;
            std::fprintf(stderr, "strata-q36: the experts nearly fit: the attention scores buffer at 64 MB and the VRAM "
                         "reserve at 256 MiB (instead of 768) so that every main layer's experts fit; a driver or "
                         "another program needing more VRAM later can make it fail - --expert-vram GiB sets the experts' "
                         "budget instead\n");
        }
    }
    if (slots <= 0 && total <= budget2 && total + ((size_t) 256 << 20) <= fr) {
        std::vector<uint8_t> blob;
        for (Layer* L : all) {
            CK(cudaMalloc(&L->blobs, L->stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L->blobs, 0, L->stride * (size_t) ne + kMmqTail));
            blob.assign(L->L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(*L, e, blob.data());
                CK(cudaMemcpy(L->blobs + (size_t) e * L->stride, blob.data(), L->L.bytes, cudaMemcpyHostToDevice));
            }
            L->n_slots = (int) ne;
        }
        std::fprintf(stderr, "strata-q36: every routed expert in VRAM (%.2f GiB) in %.1f s\n", (double) total / (1 << 30),
                     ms_since(t0) / 1000.0);
        return true;
    }
#if !defined(STRATA_Q36_CPU_POOL)
    (void) threads;
    (void) hybrid_t;
    err = "the routed experts (" + std::to_string(total >> 20) + " MiB) do not fit in VRAM, and this build has no CPU "
          "expert pool (STRATA_NATIVE_EXPERTS=OFF)";
    return false;
#else
    namespace C = strata::kernels::cpu;
    m.hybrid = true;
    E.main_hybrid = !mtp_only;
    if (E.main_hybrid) E.use_graph = false;   // the host takes part in every layer
    E.mtp_graph = false;                      // the MTP block's layer is hybrid either way
    E.hybrid_t = std::max(1, std::min(hybrid_t, E.use_mmq ? E.tmax : E.tmax));
    if (!E.use_mmq) E.hybrid_t = E.tmax;   // no streamed prompt path: every batch on slots + the pool
    const bool kv_i8 = E.kv_k8 || E.kv_v8;   // an int8 K / V store (--kv): the long-context trades below
    // the rows the CPU pool's outputs (ecpu) and the native grouped experts' scratch (escr) serve: a hybrid batch of up to
    // hybrid_t rows, or a decode / verify batch of kGemmT; with the main layers hybrid and MMQ every bigger batch is a
    // streamed prompt chunk, which uses neither (the MTP-only placement runs every MTP batch on the pool: all rows).
    // --kv int8 sizes both by that instead of the chunk (2048 chunk, 128 rows: 120 MiB and 98 MiB less)
    const size_t hy_rows_max = kv_i8 && E.main_hybrid && E.use_mmq
                                   ? (size_t) std::min(E.tmax, std::max(E.hybrid_t, Engine::kGemmT)) : (size_t) E.tmax;
    if (hy_rows_max < (size_t) E.tmax) {
        CK(cudaDeviceSynchronize());
        CK(cudaFree(E.escr));
        CK(cudaMalloc(&E.escr, K::native_expert_scratch_bytes((int64_t) (hy_rows_max * m.top_k), (int64_t) g.n_ff)));
        CK(cudaMemGetInfo(&fr, &tot));
        budget2 = vram_gib >= 0 ? budget : (fr > reserve ? fr - reserve : 0);
    }
    const size_t ecpu_bytes = hy_rows_max * (size_t) m.top_k * (size_t) g.n_embd * 4;
    // --kv int8: a staging pair that leaves no room beside it for the hybrid buffers and the 512 MiB the slots keep free,
    // even with the scores buffer at 64 MB (a long context on a small card: 204800 cells on 8 GB): one staging buffer,
    // as STRATA_Q36_SINGLE_STAGE=1 (a layer's copy then waits for the one before to finish computing)
    if (kv_i8 && E.n_stage == 2 && E.use_mmq && E.main_hybrid) {
        const size_t att_save = E.kAttScoreFloats > ((size_t) 16 << 20) ? (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6 : 0;
        const size_t pair = 2 * (max_layer + kMmqTail) + ecpu_bytes + hy_rows_max * (size_t) m.top_k * 4 + ((size_t) 512 << 20);
        if (pair > fr + att_save) {
            E.n_stage = 1;
            std::fprintf(stderr, "strata-q36: the prompt's staging pair (%.2f GiB) does not fit in the %.2f GiB free: one "
                         "staging buffer (%.2f GiB), as STRATA_Q36_SINGLE_STAGE=1\n",
                         (double) (2 * (max_layer + kMmqTail)) / (1 << 30), (double) fr / (1 << 30),
                         (double) (max_layer + kMmqTail) / (1 << 30));
        }
    }
    const size_t staging = E.use_mmq && E.main_hybrid ? (size_t) E.n_stage * (max_layer + kMmqTail) : 0;
    // the streamed prompt path needs the staging buffer(s) plus E.ecpu (allocated below) to fit; where they only just
    // do not, shrink the attention scores buffer (256 -> 64 MB, the prompt attention then runs in more, smaller
    // slices) so the staging fits - the same trade the MTP-only path above makes.  (Opt-in only - --kv int8 or
    // STRATA_Q36_SINGLE_STAGE=1 - so the default path places exactly as before.  --kv int8 asks for 256 MiB beside them
    // for what the first prompt allocates on the way - cuBLAS's own workspace, MMQ's pool, local memory -; and a
    // blocked context where 64 MB is still short takes 32 MB, 32+ query rows per slice of a block.)
    if (staging > 0 && E.kAttScoreFloats > ((size_t) 16 << 20) && (kv_i8 || E.n_stage == 1)) {
        const size_t need = staging + ecpu_bytes + (kv_i8 ? ((size_t) 256 << 20) : ((size_t) 64 << 20));
        if (need > fr) {
            E.shrink_att((size_t) 16 << 20);
            CK(cudaMemGetInfo(&fr, &tot));
            budget2 = vram_gib >= 0 ? budget : (fr > reserve ? fr - reserve : 0);
            std::fprintf(stderr, "strata-q36: the prompt staging (%.2f GiB) did not fit: the attention scores buffer at "
                         "64 MB instead of 256 (more, smaller slices in a long prompt), %.2f GiB free now\n",
                         (double) staging / (1 << 30), (double) fr / (1 << 30));
        }
    }
    {
        const size_t att_cells = (size_t) ((E.kv16_blk > 0 ? E.kv16_blk : E.ctx) + 7) / 8 * 8;
        const size_t small = (size_t) 8 << 20;   // 32 MB
        if (staging > 0 && kv_i8 && E.kAttScoreFloats > small && (size_t) g.n_head * att_cells * 32 <= small &&
            staging + ecpu_bytes + ((size_t) 256 << 20) > fr) {
            E.shrink_att(small);
            CK(cudaMemGetInfo(&fr, &tot));
            budget2 = vram_gib >= 0 ? budget : (fr > reserve ? fr - reserve : 0);
            std::fprintf(stderr, "strata-q36: the prompt staging (%.2f GiB) still short: the attention scores buffer at 32 MB "
                         "(%d query rows per slice of a %zu-cell block), %.2f GiB free now\n", (double) staging / (1 << 30),
                         (int) std::min<size_t>((size_t) E.tmax, small / ((size_t) g.n_head * att_cells)), att_cells,
                         (double) fr / (1 << 30));
        }
    }
    size_t slot_bytes = budget2 > staging ? budget2 - staging : 0;
    if (slot_bytes + staging + ((size_t) 256 << 20) > fr) slot_bytes = fr > staging + ((size_t) 512 << 20) ? fr - staging - ((size_t) 512 << 20) : 0;
    // --expert-slots: that many per layer (tests: the same split with and without the MTP block)
    int S = slots > 0 ? (int) std::min<int64_t>(ne, slots) : (int) std::min<size_t>((size_t) ne, slot_bytes / per_slot);
    // --kv int8 at a context above the FP16 K / V block (E.kv16_blk): the scores buffer at 64 MB when that holds more
    // expert slots (the blocked prompt attention then runs 4x as many, 4x smaller query slices per block)
    if (slots <= 0 && vram_gib < 0 && !mtp_only && kv_i8 && E.kv16_blk > 0 && E.kAttScoreFloats > ((size_t) 16 << 20) &&
        S < (int) ne) {
        auto slots_at = [&](size_t f) {
            const size_t b2 = f > reserve ? f - reserve : 0;
            size_t sb = b2 > staging ? b2 - staging : 0;
            if (sb + staging + ((size_t) 256 << 20) > f) sb = f > staging + ((size_t) 512 << 20) ? f - staging - ((size_t) 512 << 20) : 0;
            return (int) std::min<size_t>((size_t) ne, sb / per_slot);
        };
        const size_t att_save = (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;
        if (slots_at(fr + att_save) > S) {
            E.shrink_att((size_t) 16 << 20);
            CK(cudaMemGetInfo(&fr, &tot));
            budget2 = fr > reserve ? fr - reserve : 0;
            const int was = S;
            S = slots_at(fr);
            std::fprintf(stderr, "strata-q36: the attention scores buffer at 64 MB instead of 256 for more expert slots (%d "
                         "per layer instead of %d), %.2f GiB free now\n", S, was, (double) fr / (1 << 30));
        }
    }
    if (mtp_only) {   // the main layers all, the MTP block what is left
        const size_t left = budget2 > total_main ? budget2 - total_main : 0;
        S = (int) std::min<size_t>((size_t) ne, left / m.mtp.stride);
    }
    {   // what the hybrid mode allocates next (the staging pair, the CPU part's outputs, the slots) against the VRAM the
        // sessions and the context left: said with the numbers instead of a cudaMalloc failure (8 slots on a 12 GB card)
        const size_t Tn = (size_t) E.tmax, Nn = (size_t) g.n_embd, Kn = (size_t) m.top_k;
        size_t need = staging + ecpu_bytes + Tn * Kn * 4;
        (void) Nn;
        for (Layer* L : all) {
            if (mtp_only && L != &m.mtp) need += L->stride * (size_t) ne + kMmqTail;
            else need += (S > 0 ? (size_t) S * L->stride + kMmqTail : 0);
        }
        if (need > fr) {
            err = "not enough VRAM left for the hybrid experts' buffers and slots (" + std::to_string((need >> 20)) +
                  " MiB needed, " + std::to_string(fr >> 20) + " MiB free after the weights, the context and the sessions): use "
                  "fewer slots (\"parallel\" / --batch), a shorter --max-context" +
                  (slots > 0 ? ", or fewer --expert-slots" : "");
            return false;
        }
    }
    // host memory: every expert, pinned where the driver allows it (the prompt's stream and the slot copies)
    size_t host = 0;
    for (Layer* L : all)
        if (!mtp_only || L == &m.mtp) host += L->stride * (size_t) ne + kMmqTail;
    uint8_t* arena = nullptr;
    bool pinned = cudaHostAlloc((void**) &arena, host, cudaHostAllocDefault) == cudaSuccess;
    if (!pinned) {
        (void) cudaGetLastError();
        arena = (uint8_t*) std::malloc(host);
        if (!arena) {
            err = "no host memory for the experts (" + std::to_string(host >> 20) + " MiB)";
            return false;
        }
    }
    size_t off = 0;
    E.hy.resize(all.size());
    for (size_t li = 0; li < all.size(); ++li) {
        Layer& L = *all[li];
        if (mtp_only && &L != &m.mtp) {   // a main layer: every expert in VRAM, as in the all-in-VRAM placement
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) ne + kMmqTail));
            std::vector<uint8_t> blob(L.L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(L, e, blob.data());
                CK(cudaMemcpy(L.blobs + (size_t) e * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice));
            }
            L.n_slots = (int) ne;
            continue;
        }
        L.hblobs = arena + off;
        off += L.stride * (size_t) ne + kMmqTail;
        std::memset(L.hblobs, 0, L.stride * (size_t) ne + kMmqTail);
        for (int64_t e = 0; e < ne; ++e) expert_blob(L, e, L.hblobs + (size_t) e * L.stride);
        L.n_slots = S;
        if (S > 0) {
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemcpy(L.blobs, L.hblobs, L.stride * (size_t) S, cudaMemcpyHostToDevice));
        }
        Engine::Hy& H = E.hy[li];
        H.slot_of.assign((size_t) ne, -1);
        H.exp_in_slot.assign((size_t) S, 0);
        for (int s = 0; s < S; ++s) {
            H.slot_of[(size_t) s] = s;
            H.exp_in_slot[(size_t) s] = s;
        }
        H.score.assign((size_t) ne, 0.0);
        CK(cudaMalloc(&H.slot_of_d, (size_t) ne * 4));
        CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
        if (!C::native_fmt(L.L.gu_type, L.L.d_type, g.n_embd, g.n_ff, H.fmt, err)) return false;
        if (H.fmt.up_off != L.L.up_off || H.fmt.down_off != L.L.down_off || H.fmt.bytes != L.L.bytes) {
            err = "the CPU and GPU expert layouts differ";
            return false;
        }
    }
    const size_t T = (size_t) E.tmax, N = (size_t) g.n_embd, Kp = (size_t) m.top_k;
    CK(cudaHostAlloc((void**) &E.h_h, T * N * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.cpu_out, T * Kp * N * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_ids, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_miss, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_cnt, all.size() * (size_t) ne * 4, cudaHostAllocDefault));
    CK(cudaMalloc(&E.ecpu, ecpu_bytes));
    CK(cudaMalloc(&E.d_miss, T * Kp * 4));
    E.acts = (uint8_t*) std::malloc(T * C::kNativeActBytes + 64);
    CK(cudaEventCreateWithFlags(&E.ev_route, cudaEventDisableTiming));
    if (std::getenv("STRATA_Q36_PLACE_DEBUG")) {
        size_t f2 = 0, t2 = 0;
        CK(cudaMemGetInfo(&f2, &t2));
        std::fprintf(stderr, "strata-q36: staging alloc: n_stage=%d, max_layer=%.2f GiB (+%zu B), want=%.2f GiB, free=%.2f GiB\n",
                     E.n_stage, (double) max_layer / (1 << 30), kMmqTail, (double) staging / (1 << 30),
                     (double) f2 / (1 << 30));
    }
    if (staging) {
        CK(cudaStreamCreateWithFlags(&E.cst, cudaStreamNonBlocking));
        for (int b = 0; b < E.n_stage; ++b) {
            CK(cudaMalloc(&E.stage[b], max_layer + kMmqTail));
            CK(cudaMemset(E.stage[b], 0, max_layer + kMmqTail));
            CK(cudaEventCreateWithFlags(&E.ev_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&E.ev_used[b], cudaEventDisableTiming));
            CK(cudaEventRecord(E.ev_used[b], E.st));
        }
    }
    E.pool = std::make_unique<C::ExpertPool>(threads);
    CK(cudaDeviceSynchronize());
    if (mtp_only)
        std::fprintf(stderr, "strata-q36: every main layer's routed experts in VRAM (%.2f GiB); the MTP block's: %d of "
                     "%lld in VRAM, the rest on the CPU pool (its drafts only)\n", (double) total_main / (1 << 30), S,
                     (long long) ne);
    else
    std::fprintf(stderr, "strata-q36: hybrid experts: %d of %lld per layer in VRAM (%.2f GiB), all %.2f GiB in %s host "
                 "memory, the rest of a batch of <= %d rows on %d CPU threads, bigger prompt chunks streamed (%.2f GiB "
                 "staging); %.1f s\n", S, (long long) ne, (double) S * per_slot / (1 << 30), (double) host / (1 << 30),
                 pinned ? "pinned" : "pageable", E.hybrid_t, E.pool->workers() + (E.pool->host_works() ? 1 : 0),
                 (double) staging / (1 << 30), ms_since(t0) / 1000.0);
    return true;
#endif
}

// ------------------------------------------------------------------------------------------------ sampling
using strata::q36_sampling::Dist;
using strata::q36_sampling::Sampling;
using strata::q36_sampling::sample;
using strata::q36_sampling::spec_row;

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

bool is_eos(int t) { return t == 248044 || t == 248046; }
constexpr int32_t kImStart = 248045;   // <|im_start|>: a turn boundary (STRATA_Q36_TURN_CKPT)

// ------------------------------------------------------------------------------------------------ pictures
constexpr int32_t kImagePad = 248056;   // <|image_pad|>

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

/// A request's pictures: the encoder's rows in prompt order, each picture's cell run, and the positions.
struct ImgReq {
    std::vector<float> rows;
    std::vector<ImgKey> imgs;
    std::vector<int32_t> cell_row;   // per prompt cell: its row in `rows`, or -1 (text)
    std::vector<int32_t> mrope;      // [cell][3] t, h, w of every prompt cell (llama.cpp's mtmd positions)
    int32_t delta = 0;               // the text after the prompt: rotary position = cell + delta
};

/// Reads a strata-vision file (one or more records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, in prompt
/// order) and lays the pictures over the prompt's <|image_pad|> runs.  Positions as mtmd's: text counts up by one; a
/// picture's cells take (t, h, w) = (p, p + row, p + column) and the text after it continues at p + max(nx, ny).  "" or
/// the error.  `keys`: the server's key for each picture (img_keys=), else the embeddings' hash.
std::string build_images(const std::string& path, const std::vector<int32_t>& ids, int64_t n_embd,
                         const std::vector<uint64_t>& keys, ImgReq& r) {
    struct Img { int64_t n, nx, ny; size_t off; };
    std::vector<Img> imgs;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string ve;
    while (ve.empty()) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 || (int64_t) hdr[2] * hdr[3] != hdr[1] ||
            hdr[4] != (int32_t) n_embd) {
            ve = "bad embeddings file (expected strata-vision records of width " + std::to_string((long long) n_embd) + ")";
            break;
        }
        const size_t off = r.rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
        r.rows.resize(off + cnt);
        if (std::fread(r.rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
    }
    std::fclose(f);
    if (!ve.empty()) return ve;
    const int64_t n = (int64_t) ids.size();
    r.cell_row.assign((size_t) n, -1);
    r.mrope.assign((size_t) n * 3, 0);
    auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        r.mrope[(size_t) c * 3] = (int32_t) t;
        r.mrope[(size_t) c * 3 + 1] = (int32_t) h;
        r.mrope[(size_t) c * 3 + 2] = (int32_t) w;
    };
    int64_t p = 0, i = 0;
    size_t k = 0;
    while (i < n) {
        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const Img& im = imgs[k++];
        uint64_t h;
        {   // what the conversation cache compares: a picture is its grid and its key (or its embeddings)
            const int64_t grid[3] = {im.n, im.nx, im.ny};
            h = fnv1a(grid, sizeof grid);
            if (keys.size() == imgs.size()) h = fnv1a(&keys[k - 1], sizeof(uint64_t), h);
            else h = fnv1a(r.rows.data() + im.off, (size_t) im.n * (size_t) n_embd * sizeof(float), h);
        }
        r.imgs.push_back({i, im.n, h});
        for (int64_t j = 0; j < im.n; ++j)
            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                       " rows but fewer <|image_pad|> tokens";
        for (int64_t j = 0; j < im.n; ++j) {
            const int64_t y = j / im.nx, x = j % im.nx;
            put(i + j, p, p + y, p + x);
            r.cell_row[(size_t) (i + j)] = (int32_t) ((im.off / (size_t) n_embd) + (size_t) j);
        }
        i += im.n;
        p += std::max(im.nx, im.ny);
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && ids[(size_t) (n - 1)] == kImagePad) return "the prompt cannot end in an image";
    r.delta = (int32_t) (p - n);
    return "";
}

// ------------------------------------------------------------------------------------------------ serve
/// A batch slot: a session decoding (its next token is `x`, not yet in the state).
struct Slot {
    bool active = false, stop = false;
    int32_t x = 0;
    int64_t produced = 0, max_new = 0;
    Sampling smp;
    std::mt19937_64 rng;
    std::vector<int32_t> hist;
    Clock::time_point t0;
    int32_t draft = 0;            // --batch-mtp: the MTP block's proposal for the token after `x`
    bool draft_ready = false;
    long long offered = 0, accepted = 0;
};

/// The engine protocol (serve/server.py).  One request at a time on the control lines: GEN / GENI (alone: speculative
/// steps with the MTP block), or with --batch N BGEN / BGENI <slot> (the request's prompt is read into that slot's
/// session - the decode windows of the other slots go on between its chunks - then its tokens come as BT lines, one
/// window for every slot decoding, until BDONE).  BSTOP <slot> ends a slot at its next window; STOP the request on the
/// control lines.
struct Serve {
    Engine& E;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::vector<Slot> slots;
    int n_active = 0;
    std::vector<float> lg;
    // decode windows of the other slots between two chunks of a prompt read: STRATA_Q36_READ_WINDOWS (default 1; a chunk
    // takes ~0.15 s at 512 tokens, a window ~0.012 s: more windows give the decoding slots more of the time a read takes)
    int read_windows = 1;

    // STRATA_Q36_PARK_MIB=N (opt-in, default 0 = off): the engine holds one conversation, so a client that puts short
    // side requests (a title, a topic check) between the turns of an agent's chat had the chat read again from token 0
    // after each of them.  With a budget, the conversation a request is about to overwrite is parked in host RAM (its
    // K / V cells, its recurrent state at the end and at its turn checkpoint), and a later prompt that starts with a
    // parked conversation's tokens brings it back instead of reading them.  A prompt read from the start also parks the
    // state at the end of its first message (the system prompt and tools, when that is at least
    // STRATA_Q36_PARK_ROOT tokens, default 2048) as a shared root: a new chat of the same client borrows it.
    //   STRATA_Q36_PARK_SLOTS (default 4): at most that many parked entries (oldest use leaves first, roots last)
    //   STRATA_Q36_PARK_MIN (default 1024): conversations shorter than this are not parked
    //   STRATA_Q36_PARK_FREE_MIB (default 1024): RAM (MemAvailable) that must be left after a park
    struct Parked {
        std::vector<int32_t> live;            // its tokens; the K / V image holds cells [0, cap)
        int64_t cap = 0;
        std::vector<char> kv;
        struct State { int64_t n; std::vector<char> rec; };
        std::vector<State> states;            // the recurrent state after live[0, n), n <= cap
        bool root = false;                    // a shared prefix: a match borrows it (it stays parked)
        uint64_t used = 0;
        size_t bytes() const {
            size_t b = kv.size();
            for (const State& s : states) b += s.rec.size();
            return b;
        }
    };
    std::vector<Parked> parked;
    size_t park_budget = 0, park_free = (size_t) 1024 << 20;
    int park_slots = 4;
    int64_t park_min = 1024, park_root = 2048;
    uint64_t park_clock = 0;

    explicit Serve(Engine& e) : E(e), slots((size_t) e.n_sess) {
        if (const char* v = std::getenv("STRATA_Q36_READ_WINDOWS")) read_windows = std::max(0, std::min(64, std::atoi(v)));
        if (const char* v = std::getenv("STRATA_Q36_PARK_MIB")) park_budget = (size_t) std::max(0L, std::atol(v)) << 20;
        if (const char* v = std::getenv("STRATA_Q36_PARK_SLOTS")) park_slots = std::max(1, std::atoi(v));
        if (const char* v = std::getenv("STRATA_Q36_PARK_MIN")) park_min = std::max(1L, std::atol(v));
        if (const char* v = std::getenv("STRATA_Q36_PARK_ROOT")) park_root = std::max(0L, std::atol(v));
        if (const char* v = std::getenv("STRATA_Q36_PARK_FREE_MIB")) park_free = (size_t) std::max(0L, std::atol(v)) << 20;
        if (park_budget > 0)
            std::fprintf(stderr, "strata serve: STRATA_Q36_PARK_MIB=%zu: up to %d conversations parked in host RAM "
                         "(min %lld tokens, system-prompt root from %lld tokens, %zu MiB left free)\n", park_budget >> 20,
                         park_slots, (long long) park_min, (long long) park_root, park_free >> 20);
    }

    /// MemAvailable from /proc/meminfo (bytes; SIZE_MAX when it cannot be read: no check).
    static size_t mem_available() {
        FILE* f = std::fopen("/proc/meminfo", "r");
        if (!f) return SIZE_MAX;
        char line[256];
        size_t kb = SIZE_MAX;
        while (std::fgets(line, sizeof line, f))
            if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
        std::fclose(f);
        return kb == SIZE_MAX ? kb : kb << 10;
    }
    size_t parked_bytes() const {
        size_t b = 0;
        for (const Parked& p : parked) b += p.bytes();
        return b;
    }
    /// The longest state of a parked entry whose tokens start `ids` (a proper start): its index and length, else -1 / 0.
    std::pair<int, int64_t> park_match(const std::vector<int32_t>& ids) const {
        int best = -1;
        int64_t bn = 0;
        for (size_t i = 0; i < parked.size(); ++i) {
            const Parked& p = parked[i];
            for (const Parked::State& s : p.states) {
                if (s.n <= bn || s.n >= (int64_t) ids.size()) continue;
                if (std::equal(p.live.begin(), p.live.begin() + s.n, ids.begin())) { bn = s.n; best = (int) i; }
            }
        }
        return {best, bn};
    }
    /// Room for `need` more bytes and one more entry: the least recently used leave (conversations before roots).
    bool park_room(size_t need) {
        if (need > park_budget) return false;
        while (!parked.empty() && ((int) parked.size() >= park_slots || parked_bytes() + need > park_budget)) {
            size_t v = 0;
            for (size_t i = 1; i < parked.size(); ++i) {
                const Parked &a = parked[i], &b = parked[v];
                if (a.root != b.root ? !a.root : a.used < b.used) v = i;
            }
            std::fprintf(stderr, "strata serve: park: evicted a %s of %zu tokens (%.0f MiB)\n",
                         parked[v].root ? "root" : "conversation", parked[v].live.size(), parked[v].bytes() / 1048576.0);
            parked.erase(parked.begin() + (long) v);
        }
        const size_t avail = mem_available();
        return avail == SIZE_MAX || avail >= need + park_free;
    }
    /// Parks the current session's state after toks[0, n) (its K / V cells and the device's recurrent state), and the
    /// checkpoint (ck_n, ck_rec) when ck_n is in (0, n).  n_end 0: the device state is not at n (only the checkpoint).
    void park_now(const int32_t* toks, int64_t n, bool at_end, int64_t ck_n, const void* ck_rec, bool root) {
        if (n <= 0 || (!at_end && ck_n <= 0)) return;
        const int64_t cap = at_end ? n : ck_n;
        // already parked (the same tokens up to the same state) or superseded copies of this conversation
        for (size_t i = 0; i < parked.size();) {
            Parked& p = parked[i];
            const int64_t m = std::min<int64_t>((int64_t) p.live.size(), cap);
            const bool prefix = std::equal(p.live.begin(), p.live.begin() + m, toks);
            if (prefix && (int64_t) p.live.size() >= cap && (p.root == root || p.root)) {
                bool has = false;
                for (const Parked::State& s : p.states) has |= s.n == cap;
                if (has) { p.used = ++park_clock; return; }
            }
            if (prefix && !p.root && !root && (int64_t) p.live.size() <= cap) parked.erase(parked.begin() + (long) i);
            else ++i;
        }
        const size_t rb = E.rec_bytes();
        const bool with_ck = at_end && ck_n > 0 && ck_n < n && ck_rec;
        const size_t need = E.kv_cells_bytes(cap) + (at_end ? rb : 0) + (with_ck || !at_end ? rb : 0);
        if (!park_room(need)) {
            std::fprintf(stderr, "strata serve: park: skipped a %s of %lld tokens (%.0f MiB: over the budget or the RAM "
                         "floor)\n", root ? "root" : "conversation", (long long) cap, need / 1048576.0);
            return;
        }
        const auto t0 = Clock::now();
        Parked p;
        try {
            p.live.assign(toks, toks + cap);
            p.kv.resize(E.kv_cells_bytes(cap));
            if (at_end) p.states.push_back({n, std::vector<char>(rb)});
            if (with_ck || !at_end) {
                p.states.push_back({ck_n, std::vector<char>(rb)});
                std::memcpy(p.states.back().rec.data(), ck_rec, rb);
            }
        } catch (const std::bad_alloc&) {
            std::fprintf(stderr, "strata serve: park: no memory for %.0f MiB\n", need / 1048576.0);
            return;
        }
        p.cap = cap;
        p.root = root;
        p.used = ++park_clock;
        E.kv_copy(p.kv.data(), cap, cap, true);
        if (at_end) E.rec_copy(p.states.front().rec.data(), true);
        std::fprintf(stderr, "strata serve: park: parked a %s of %lld tokens (%.0f MiB in %.0f ms; %zu parked, %.0f MiB)\n",
                     root ? "root" : "conversation", (long long) cap, need / 1048576.0, ms_since(t0), parked.size() + 1,
                     (parked_bytes() + need) / 1048576.0);
        parked.push_back(std::move(p));
    }
    /// The current session (about to be overwritten) to the park, when it is worth a place.
    void park_session(Engine::Sess& ss) {
        if (park_budget == 0 || ss.active || !ss.imgs.empty() || ss.rdelta != 0) return;
        const int64_t L = (int64_t) ss.live.size();
        if (L < park_min) return;
        const bool at_end = E.pos == L;
        const bool ck_ok = E.turn_ckpt && ss.ck_n > 0 && ss.ck_n <= L && ss.ck_host;
        if (!at_end && !ck_ok) return;
        park_now(ss.live.data(), at_end ? L : ss.ck_n, at_end, ck_ok ? ss.ck_n : 0, ck_ok ? ss.ck_host : nullptr, false);
    }
    /// Parked entry i's state of length n into the current session (a conversation leaves the park; a root stays).
    void park_restore(int i, int64_t n) {
        const auto t0 = Clock::now();
        Parked& p = parked[(size_t) i];
        Engine::Sess& ss = E.sess[(size_t) E.cur];
        const Parked::State* st = nullptr;
        const Parked::State* ck = nullptr;
        for (const Parked::State& s : p.states) {
            if (s.n == n) st = &s;
            else if (s.n < n && (!ck || s.n > ck->n)) ck = &s;
        }
        E.kv_copy(p.kv.data(), p.cap, n, false);
        E.rec_copy(const_cast<char*>(st->rec.data()), false);
        E.pos = n;
        ss.pos = n;
        E.set_delta(0);
        ss.live.assign(p.live.begin(), p.live.begin() + n);
        ss.imgs.clear();
        ss.ck_n = 0;
        if (ck && E.turn_ckpt) {   // the conversation's turn checkpoint comes back with it
            std::memcpy(E.ck_buf(), ck->rec.data(), ck->rec.size());
            ss.ck_n = ck->n;
            ss.ck_rdelta = 0;
        }
        std::fprintf(stderr, "strata serve: park: %s a %s at %lld of its %zu tokens in %.0f ms\n",
                     p.root ? "borrowed" : "restored", p.root ? "root" : "conversation", (long long) n, p.live.size(),
                     ms_since(t0));
        if (p.root) p.used = ++park_clock;
        else parked.erase(parked.begin() + i);
    }

    /// BSTOP / BYIELD lines among the queued ones: applied now (a window is a moment away at most).
    void poll_control() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end();) {
            if (it->rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(it->c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
                it = lines.erase(it);
            } else if (it->rfind("BYIELD", 0) == 0) {   // (a read here does not give way: the server then goes on)
                it = lines.erase(it);
            } else {
                ++it;
            }
        }
    }

    /// One decode window with MTP proposals (--batch-mtp): every active slot's next token, and for up to kVerify - S of the
    /// slots (the greedy ones, in turn) also the proposal its MTP block made: the window verifies both at once.  A slot whose
    /// proposal the main model confirms moves two tokens in the window.  The proposals for the next window come right after
    /// it, from the hidden states the window just produced (the first one of a slot from the state its prompt left).
    int mtp_turn = 0;
    void window_mtp() {
        poll_control();
        const int N = (int) E.m->g.n_embd;
        std::vector<int> act;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) act.push_back(b);
        const int S = (int) act.size();
        if (S == 0) return;
        // who verifies a proposal this window: the greedy slots in turn, as many as fit beside one row per slot
        const int room = Engine::kVerify - S;
        std::vector<char> cand((size_t) S, 0);
        int nc = 0;
        for (int k = 0; k < S && nc < room; ++k) {
            const int i = (mtp_turn + k) % S;
            if (!slots[(size_t) act[(size_t) i]].smp.needs_logits()) { cand[(size_t) i] = 1; ++nc; }
        }
        ++mtp_turn;
        // a candidate without a proposal gets its first one from the state its last row left
        std::vector<Engine::MtpRow> first;
        std::vector<int> first_of;
        for (int i = 0; i < S; ++i) {
            Slot& sl = slots[(size_t) act[(size_t) i]];
            if (cand[(size_t) i] && !sl.draft_ready) {
                Engine::Sess& ss = E.sess[(size_t) act[(size_t) i]];
                first.push_back({act[(size_t) i], sl.x, ss.pos, ss.hlast});
                first_of.push_back(i);
            }
        }
        if (!first.empty()) {
            E.mtp_rows(first.data(), (int) first.size());
            for (size_t k = 0; k < first.size(); ++k) {
                Slot& sl = slots[(size_t) act[(size_t) first_of[k]]];
                sl.draft = E.mtp_pick[k];
                sl.draft_ready = true;
            }
        }
        std::vector<Engine::Group> gr((size_t) S);
        std::vector<int> row0((size_t) S);
        int T = 0;
        for (int i = 0; i < S; ++i) {
            const Slot& sl = slots[(size_t) act[(size_t) i]];
            gr[(size_t) i] = {act[(size_t) i], cand[(size_t) i] ? 2 : 1, {sl.x, sl.draft}};
            row0[(size_t) i] = T;
            T += gr[(size_t) i].n;
        }
        E.run_rows(gr.data(), S, true);
        // the picks: emit, decide how many rows each slot keeps
        std::vector<int> keep((size_t) S, 1);
        std::vector<int32_t> last((size_t) S, 0);
        std::vector<char> alive((size_t) S, 0);
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int n = gr[(size_t) i].n, r0 = row0[(size_t) i];
            int y0 = E.rows_pick[r0];
            if (sl.smp.needs_logits()) {
                E.host_logits_row(r0, lg);
                y0 = sample(lg, sl.smp, sl.hist, sl.rng);
            }
            bool two = n == 2 && y0 == sl.draft && !is_eos(y0) && !sl.stop && sl.produced + 2 <= sl.max_new &&
                       ss.pos + 3 <= E.ctx;   // (ss.pos: the position of the slot's next token: the window did not move it)
            if (n == 2) { ++sl.offered; if (two) ++sl.accepted; }
            const int rows_kept = two ? 2 : 1;
            keep[(size_t) i] = rows_kept;
            const int32_t fed[2] = {sl.x, sl.draft};
            bool done = false;
            int y = y0;
            for (int j = 0; j < rows_kept; ++j) {
                if (j == 1) y = E.rows_pick[r0 + 1];
                ss.live.push_back(fed[j]);
                std::printf("BT %d %d\n", b, y);
                ++sl.produced;
                sl.hist.push_back(y);
                const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                  : ss.pos + j + 2 > E.ctx ? "length" : nullptr;
                if (fin) {
                    std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                    if (sl.offered > 0)
                        std::fprintf(stderr, "strata batch: slot %d: %lld of %lld MTP proposals accepted\n", b, sl.accepted,
                                     sl.offered);
                    sl.active = false;
                    ss.active = false;
                    --n_active;
                    done = true;
                    break;
                }
            }
            last[(size_t) i] = y;
            alive[(size_t) i] = !done;
            if (!done) sl.x = y;
        }
        E.commit_rows(gr.data(), keep.data(), S);
        // the hidden state after each slot's last kept row (the seed of a later first proposal), and the next proposals
        std::vector<Engine::MtpRow> nx;
        std::vector<int> nx_of, nx_last;
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int r0 = row0[(size_t) i];
            CK(cudaMemcpyAsync(ss.hlast, E.hmain + (size_t) (r0 + keep[(size_t) i] - 1) * N, (size_t) N * 4,
                               cudaMemcpyDeviceToDevice, E.st));
            Slot& sl = slots[(size_t) b];
            sl.draft_ready = false;
            if (!alive[(size_t) i] || !cand[(size_t) i]) continue;
            const int64_t p_new = ss.pos;   // commit_rows moved it: the cell of the token the next window feeds
            if (keep[(size_t) i] == 2) {
                nx.push_back({b, sl.draft, p_new - 1, E.hmain + (size_t) r0 * N});
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) (r0 + 1) * N});
                nx_last.push_back((int) nx.size() - 1);
            } else {
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) r0 * N});
                nx_last.push_back((int) nx.size() - 1);
            }
            nx_of.push_back(i);
        }
        std::fflush(stdout);
        if (!nx.empty()) {
            // (more rows than the window holds: the proposals come in two calls)
            for (size_t at = 0, k = 0; at < nx.size();) {
                size_t take = std::min<size_t>(nx.size() - at, Engine::kVerify);
                while (take > 0 && at + take < nx.size() && nx[at + take].slot == nx[at + take - 1].slot) --take;   // a slot's rows together
                E.mtp_rows(nx.data() + at, (int) take);
                for (; k < nx_last.size() && (size_t) nx_last[k] < at + take; ++k) {
                    Slot& sl = slots[(size_t) act[(size_t) nx_of[k]]];
                    sl.draft = E.mtp_pick[(size_t) nx_last[k] - at];
                    sl.draft_ready = true;
                }
                at += take;
            }
        }
    }

    /// One decode window over every active slot: one token each.
    void window() {
        poll_control();
        if (E.bmtp && E.mtp_d > 0) {   // proposals need a row to spare and a slot that decodes greedily
            int act = 0, greedy = 0;
            for (const Slot& sl : slots)
                if (sl.active) { ++act; greedy += !sl.smp.needs_logits(); }
            if (greedy > 0 && Engine::kVerify - act > 0) { window_mtp(); return; }
        }
        window_plain();
    }

    /// One token of every active slot, no proposals.
    void window_plain() {
        std::vector<Engine::Row> rows;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) rows.push_back({b, slots[(size_t) b].x});
        if (rows.empty()) return;
        E.step_rows(rows.data(), (int) rows.size());
        const size_t Nb = (size_t) E.m->g.n_embd * 4;
        for (size_t r = 0; r < rows.size(); ++r) {
            const int b = rows[r].slot;
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            if (E.bmtp)   // (the hidden state a later first proposal starts from)
                CK(cudaMemcpyAsync(ss.hlast, E.hmain + r * (size_t) E.m->g.n_embd, Nb, cudaMemcpyDeviceToDevice, E.st));
            sl.draft_ready = false;
            ss.live.push_back(sl.x);
            int y;
            if (sl.smp.needs_logits()) {
                E.host_logits_row((int) r, lg);
                y = sample(lg, sl.smp, sl.hist, sl.rng);
            } else {
                y = E.rows_pick[r];
            }
            std::printf("BT %d %d\n", b, y);
            ++sl.produced;
            sl.hist.push_back(y);
            const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                              : ss.pos + 1 > E.ctx ? "length" : nullptr;
            if (fin) {
                std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                sl.active = false;
                ss.active = false;
                --n_active;
            } else {
                sl.x = y;
            }
        }
        std::fflush(stdout);
    }

    /// The length of session ss's held tokens when they are a proper start of `ids` with the same pictures, else 0.
    size_t match_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        const size_t L = ss.live.size();
        if (L == 0 || L >= ids.size() || !std::equal(ss.live.begin(), ss.live.end(), ids.begin())) return 0;
        std::vector<ImgKey> in_live;
        for (const ImgKey& k : imgs) {
            if (k.cell + k.n <= (int64_t) L) in_live.push_back(k);
            else if (k.cell < (int64_t) L) return 0;   // the held tokens end inside a picture
        }
        return in_live == ss.imgs ? L : 0;
    }

    /// STRATA_Q36_TURN_CKPT: the length of session ss's checkpoint when its tokens are a proper start of `ids` and no
    /// picture is involved, else 0.
    size_t ck_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        const size_t c = (size_t) ss.ck_n;
        if (!E.turn_ckpt || c == 0 || c >= ids.size() || ss.live.size() < c || !ss.imgs.empty()) return 0;
        for (const ImgKey& k : imgs)
            if (k.cell < (int64_t) c) return 0;
        return std::equal(ss.live.begin(), ss.live.begin() + (long) c, ids.begin()) ? c : 0;
    }

    /// The session a request alone takes: the one holding the longest start of its prompt, else an empty one, else the
    /// least recently used (never a decoding slot).
    int pick_session(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        int best = -1, free_empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < E.n_sess; ++i) {
            const Engine::Sess& ss = E.sess[(size_t) i];
            if (ss.active) continue;
            const size_t l = std::max(match_len(ss, ids, imgs), ck_len(ss, ids, imgs));
            if (l > bl) { bl = l; best = i; }
            if (ss.live.empty() && free_empty < 0) free_empty = i;
            if (lru < 0 || ss.used < E.sess[(size_t) lru].used) lru = i;
        }
        return best >= 0 ? best : free_empty >= 0 ? free_empty : lru;
    }

    void request(const std::string& line) {
        const bool adm = line[0] == 'B';
        const bool geni = line.rfind(adm ? "BGENI " : "GENI ", 0) == 0;
        if (!adm && !geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            return;
        }
        char* p = const_cast<char*>(line.c_str()) + (adm ? (geni ? 6 : 5) : (geni ? 5 : 4));
        int slot = -1;
        if (adm) {
            char* e1 = nullptr;
            const long b = std::strtol(p, &e1, 10);
            p = e1;
            slot = (int) b;
            if (e1 == line.c_str() || b < 0 || b >= (long) slots.size() || slots[(size_t) b].active || E.n_sess < 2) {
                std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", E.n_sess < 2 ? 0 : E.n_sess);
                std::fflush(stdout);
                return;
            }
        }
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        std::vector<uint64_t> img_keys;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "img_keys") {
                const char* q = v;
                img_keys.clear();
                while (*q) {
                    char* qe = nullptr;
                    const unsigned long long x = std::strtoull(q, &qe, 16);
                    if (qe == q) { img_keys.clear(); break; }
                    img_keys.push_back((uint64_t) x);
                    q = *qe == ',' ? qe + 1 : qe;
                    if (*qe != ',' && *qe != '\0') { img_keys.clear(); break; }
                }
            }
        }
        std::string emb;
        if (geni) {
            while (*p == ' ') ++p;
            const char* e = p;
            while (*e && *e != ' ') ++e;
            emb.assign(p, (size_t) (e - p));
            p = const_cast<char*>(e);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty() || (geni && emb.empty())) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            return;
        }
        if (geni && !E.vision) {
            std::printf("ERR this engine was started without --vision\n");
            std::fflush(stdout);
            return;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            return;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) E.m->g.n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); return; }
        ImgReq ir;
        if (geni) {
            const std::string ve = build_images(emb, ids, E.m->g.n_embd, img_keys, ir);
            if (!ve.empty()) { std::printf("ERR %s\n", ve.c_str()); std::fflush(stdout); return; }
        }
        stop.store(false);
        // ---- the session: the held state continues when the request extends it (the recurrent state cannot step back)
        const int s = adm ? slot : pick_session(ids, ir.imgs);
        Engine::Sess& ss = E.sess[(size_t) s];
        size_t reuse = match_len(ss, ids, ir.imgs);
        if (adm) {   // another session holding more of this prompt (the conversation was read alone): copy its state
            int src = -1;
            size_t best = reuse;
            for (int i = 0; i < E.n_sess; ++i) {
                if (i == s || E.sess[(size_t) i].active) continue;
                const size_t l = match_len(E.sess[(size_t) i], ids, ir.imgs);
                if (l > best) { best = l; src = i; }
            }
            if (src >= 0) {
                E.copy_session(src, s);
                reuse = best;
            }
        }
        E.select(s);
        if (!adm && park_budget > 0 && ir.imgs.empty()) {   // STRATA_Q36_PARK_MIB: a parked conversation that holds more
            const size_t have = std::max(reuse, ck_len(ss, ids, ir.imgs));
            auto [pi, pn] = park_match(ids);
            if (pi >= 0 && (size_t) pn > have) {
                parked[(size_t) pi].used = ++park_clock;   // (the park below leaves it be)
                park_session(ss);
                std::tie(pi, pn) = park_match(ids);
                if (pi >= 0 && (size_t) pn > have) {
                    park_restore(pi, pn);
                    reuse = match_len(ss, ids, ir.imgs);
                }
            } else if (have == 0) {
                park_session(ss);   // this prompt reads from the start over it
            }
        }
        if (!adm && reuse == 0) {   // STRATA_Q36_TURN_CKPT: the held answer is not this prompt's, its turn boundary is
            const size_t c = ck_len(ss, ids, ir.imgs);
            if (c > 0) {
                E.ckpt_load();
                ss.live.resize(c);
                reuse = c;
            }
        }
        if (reuse == 0) {
            E.reset();
            ss.live.clear();
            ss.imgs.clear();
            ss.ck_n = 0;
        }
        if (geni) E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) ids.size(), ir.delta);
        else E.clear_images();
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        int64_t split = 0;   // STRATA_Q36_TURN_CKPT: the fresh tokens before the prompt's last <|im_start|> (0: no split)
        if (E.turn_ckpt && !adm && ir.imgs.empty()) {
            for (int64_t i = fresh - 1; i > 0; --i)
                if (ids[reuse + (size_t) i] == kImStart) { split = i; break; }
            // both parts take the chunked read (a 1-token part would take the decode step: no MTP rows, no rebalance)
            if (split < 2 || fresh - split < 2) split = 0;
        }
        auto on_read = [&](int64_t done) -> bool {
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
            if (n_active > 0 && done < fresh) {
                poll_control();
                // the slots decoding go on between a read's chunks (the last chunk's
                for (int w = 0; w < read_windows && n_active > 0; ++w) window();   // logits must stay for the first token)
            } else {
                poll_control();
            }
            if (stop.load() && done < fresh) { cancelled = true; return false; }
            return true;
        };
        // STRATA_Q36_PARK_MIB: a prompt whose first message (the system prompt and tools) ends at or after
        // STRATA_Q36_PARK_ROOT tokens and is not parked yet is read in two parts there, and the state between them is
        // parked as a shared root
        int64_t root = 0;
        if (park_budget > 0 && park_root > 0 && !adm && ir.imgs.empty()) {
            int64_t r = 0;
            for (size_t i = 1; i < ids.size(); ++i)
                if (ids[i] == kImStart) { r = (int64_t) i; break; }
            root = r - (int64_t) reuse;
            if (r < park_root || root < 2 || fresh - root < 2 || (split > 0 && split - root < 2 && split != root))
                root = 0;
            if (root > 0 && split == root) root = 0;   // the turn checkpoint is the same point: no second part
            for (const Parked& p : parked) {
                if (root == 0) break;
                if (!p.root || (int64_t) p.live.size() != r) continue;
                if (std::equal(p.live.begin(), p.live.end(), ids.begin())) root = 0;
            }
        }
        int64_t got;
        if (split == 0 && root == 0) {
            got = E.process(ids.data() + reuse, fresh, on_read);
        } else if (split == 0 || root == 0 || root > split) {   // one cut (or the root after the turn split: only that)
            const int64_t cut = split > 0 ? split : root;
            got = E.process(ids.data() + reuse, cut, on_read, false);
            if (got == cut && !cancelled) {
                if (split > 0) E.ckpt_save((int64_t) reuse + split);
                else park_now(ids.data(), (int64_t) reuse + root, true, 0, nullptr, true);
                got += E.process(ids.data() + reuse + cut, fresh - cut,
                                 [&](int64_t done) -> bool { return on_read(cut + done); });
            }
        } else {   // the root, then the turn checkpoint, then the rest
            got = E.process(ids.data() + reuse, root, on_read, false);
            if (got == root && !cancelled) {
                park_now(ids.data(), (int64_t) reuse + root, true, 0, nullptr, true);
                got += E.process(ids.data() + reuse + root, split - root,
                                 [&](int64_t done) -> bool { return on_read(root + done); }, false);
                if (got == split && !cancelled) {
                    E.ckpt_save((int64_t) reuse + split);
                    got += E.process(ids.data() + reuse + split, fresh - split,
                                     [&](int64_t done) -> bool { return on_read(split + done); });
                }
            }
        }
        ss.live.insert(ss.live.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) got);
        ss.imgs.clear();
        bool cut = false;
        for (const ImgKey& k : ir.imgs) {
            if (k.cell + k.n <= (int64_t) ss.live.size()) ss.imgs.push_back(k);
            else if (k.cell < (int64_t) ss.live.size()) cut = true;
        }
        if (cancelled && cut) {   // stopped inside a picture: the state cannot be continued, the next read starts over
            ss.live.clear();
            ss.imgs.clear();
            ss.ck_n = 0;
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse, (long long) got);
            if (adm) std::printf("BADM %d 0\n", slot);
            std::fflush(stdout);
            return;
        }
        std::mt19937_64 rng(smp.seed);
        std::vector<int32_t> hist = ids;
        const auto d0 = Clock::now();
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        // greedy requests alone with the MTP block: speculative steps of mtp_d drafts (verified, so the text is the greedy
        // text); sampled ones, and requests beside others, one token at a time
        // opt-in (STRATA_Q36_PENALTY_SPEC=1): greedy requests WITH penalties (the model card's presence_penalty 1.5) also
        // take the speculative steps; every verify row is picked on the host from its penalized logits, as one token at a time
        static const bool pen_spec_env = std::getenv("STRATA_Q36_PENALTY_SPEC") != nullptr &&
                                         std::string(std::getenv("STRATA_Q36_PENALTY_SPEC")) != "0";
        const bool pen_spec = pen_spec_env && smp.needs_logits() && smp.temperature <= 0.f;
        // opt-in (STRATA_Q36_SPEC_SAMPLE=1): sampled requests (temperature > 0) take the speculative steps too, as exact
        // speculative sampling: draft i is accepted with the probability the sampler gives it at its row, a rejection draws
        // the replacement from that row's distribution without the draft (spec_row), so the text is distributed as one
        // token at a time (not the same tokens for a seed: the draws differ)
        static const bool spec_sample_env = std::getenv("STRATA_Q36_SPEC_SAMPLE") != nullptr &&
                                            std::string(std::getenv("STRATA_Q36_SPEC_SAMPLE")) != "0";
        const bool samp_spec = spec_sample_env && smp.temperature > 0.f;
        const bool spec = E.mtp_d > 0 && (!smp.needs_logits() || pen_spec || samp_spec) && !adm;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        bool decided = false;   // nx came from a speculative step's (penalized) row: not sampled again at the loop's top
        const std::function<int32_t(int)> repick = [&](int r) -> int32_t {
            E.host_logits_row(r, lg);
            std::vector<int32_t> h = hist;   // hist ends with x; row r follows drafts 0 .. r - 1 too
            for (int i = 0; i < r; ++i) h.push_back(drafts[(size_t) i]);
            return (int32_t) sample(lg, smp, h, rng);
        };
        Dist spec_dist;
        std::vector<int32_t> spec_h;
        // sampled: row r is (accepted draft r, or its replacement) for r < D, the bonus token for r == D; a row's token
        // equals its draft exactly when the draft is accepted, which is what spec_step's loop tests
        const std::function<int32_t(int)> repick_s = [&](int r) -> int32_t {
            if (r == 0) {   // hist ends with x; row r follows drafts 0 .. r - 1 too
                spec_h = hist;
                spec_h.insert(spec_h.end(), drafts.begin(), drafts.end());
            }
            E.host_logits_row(r, lg);
            return (int32_t) spec_row(lg, smp, spec_h.data(), hist.size() + (size_t) r,
                                      r < (int) drafts.size() ? drafts[(size_t) r] : -1, rng, spec_dist);
        };
        int nx = smp.needs_logits() ? -1 : E.greedy();
        if (adm) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            std::printf("T %d\n", nx);
            ++produced;
            hist.push_back(nx);
            const char* fin = is_eos(nx) ? "stop" : stop.load() ? "cancel" : max_new <= 1 ? "length"
                              : E.pos + 1 > E.ctx ? "length" : nullptr;
            const bool cont = fin == nullptr;
            std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                        ms_since(d0), cont ? "length" : fin, reuse, (long long) fresh);
            if (cont) {
                Slot& sl = slots[(size_t) slot];
                sl = Slot{};
                sl.active = true;
                sl.x = nx;
                sl.produced = 1;
                sl.max_new = max_new;
                sl.smp = smp;
                sl.rng = rng;
                sl.hist = hist;
                sl.t0 = Clock::now();
                ss.active = true;
                ++n_active;
            }
            std::printf("BADM %d %d\n", slot, cont ? 1 : 0);
            std::fflush(stdout);
            std::fprintf(stderr, "strata batch: slot %d takes a prompt of %zu tokens = %zu reused + %lld read in %.0f ms (%.1f "
                         "tok/s)%s\n", slot, ids.size(), reuse, (long long) fresh, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, cont ? "" : ", done at once");
            return;
        }
        for (;;) {
            if (smp.needs_logits() && !decided) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            decided = false;
            auto emit = [&](int t) {   // false: the answer ends here
                std::printf("T %d\n", t);
                std::fflush(stdout);
                ++produced;
                hist.push_back(t);
                if (is_eos(t)) { finish = "stop"; return false; }
                if (produced >= max_new) return false;
                if (stop.load()) { finish = "cancel"; return false; }
                return true;
            };
            if (!emit(nx)) break;
            if (n_active > 0) window();
            if (spec && n_active == 0 && E.pos + E.mtp_d + 1 <= E.ctx) {
                const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks, &stop,
                                          samp_spec ? &repick_s : pen_spec ? &repick : nullptr);
                offered += E.mtp_d;
                accepted += a;
                // the state now holds x and every accepted draft, whether or not the answer ends inside them (an end of
                // answer, max_new): the held tokens say so, so a later request that continues this conversation matches them
                // (a prompt that has the same tokens next reuses the state, any other reads from the start) and never reads
                // a token twice
                ss.live.push_back(nx);
                for (int i = 0; i < a; ++i) ss.live.push_back(drafts[(size_t) i]);
                fix.assign(drafts.begin(), drafts.begin() + a);
                bool ended = false;
                for (int i = 0; i < a && !ended; ++i)
                    if (!emit(drafts[(size_t) i])) ended = true;
                if (ended) break;
                nx = picks[a];
                decided = true;
                continue;
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            ss.live.push_back(nx);
            fix.clear();
            if (!smp.needs_logits()) nx = E.greedy();
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, accepted, offered, reuse, (long long) fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "strata serve: session %d: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), "
                     "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld (%s)\n", s, ids.size(), reuse,
                     (long long) fresh, prompt_ms, prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced,
                     decode_ms, decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, accepted, offered, finish);
    }

    int run() {
        if (E.n_sess > 1) std::printf("INFO batch_slots=%d slot_cache=1\n", E.n_sess);
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            bool have = false;
            {
                std::unique_lock<std::mutex> lk(mu);
                if (n_active == 0) cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (!lines.empty()) {
                    line = lines.front();
                    lines.pop_front();
                    have = true;
                } else if (n_active == 0) {
                    return 0;
                }
            }
            if (!have) { window(); continue; }
            if (line.empty()) continue;
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
            } else if (line.rfind("BYIELD", 0) == 0) {
                // (a read that has ended)
            } else if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
                request(line);
            } else {
                std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
                std::fflush(stdout);
            }
            if (n_active > 0) window();
        }
    }
};

int serve(Engine& E) {
    if (const char* v = std::getenv("STRATA_Q36_TURN_CKPT")) E.turn_ckpt = std::atoi(v) != 0;
    if (E.turn_ckpt)
        std::fprintf(stderr, "strata serve: STRATA_Q36_TURN_CKPT=1: a prompt's last turn boundary keeps the recurrent state "
                             "(pinned host memory), so the next turn of the chat resumes there\n");
    Serve sv(E);
    return sv.run();
}

}  // namespace

int strata::qwen35::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full, emb_file, kv_opt;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0, mtp_d = 0, cpu_threads = 0, hybrid_rows = 128;
    double expert_vram = -1.0;
    bool warm = true, kv_k8 = false, kv_v8 = false;
    int expert_slots = 0, batch = 0, draft_vocab = -1;
    bool serve_mode = false, graph = true, vision = false, batch_mtp = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-q36: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        // strata's names where they mean the same thing (the family switch passes strata's command line on)
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--no-token-graph") graph = false;
        else if (a == "--pool-workers") cpu_threads = std::atoi(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA/HIP_VISIBLE_DEVICES)
        else if (a == "--greedy") {}            // --tokens decodes greedily here anyway
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") tokens = next();
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--emb") emb_file = next();   // --tokens with pictures: a strata-vision file (the ids carry <|image_pad|> runs)
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--no-graph") graph = false;
        else if (a == "--no-warmup") warm = false;
        else if (a == "--mtp") {
            // a number of drafts; Flash-Next's "--mtp <draft layer dir>" means nothing for this family
            const std::string v = next();
            if (!v.empty() && std::isdigit((unsigned char) v[0])) mtp_d = std::atoi(v.c_str());
            else std::fprintf(stderr, "strata-q36: ignoring --mtp %s (this family's MTP is the GGUF's own block: --mtp N "
                                      "drafts)\n", v.c_str());
        }
        else if (a == "--draft-vocab") draft_vocab = std::atoi(next().c_str());   // the MTP drafts pick among the first N token ids
        else if (a == "--vision") vision = true;                                  // pictures (GENI): the encoder is strata-vision's
        else if (a == "--batch" || a == "--slots") batch = std::atoi(next().c_str());   // sessions decoding together
        else if (a == "--batch-mtp") batch_mtp = true;                                  // ... each verifying an MTP proposal
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());   // GiB for the routed experts
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
        else if (a == "--kv") kv_opt = next();   // the K / V stores: fp32 (default), int8 / k8v8, k8v32, k32v8
        else if (serve_mode && a.rfind("--", 0) == 0) {
            // serve passes the config's args plus a few of its own: one this engine does not have is said and skipped
            // (with its value), not fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-q36: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-q36 --model GGUF [--max-context 32768] [--chunk 512] [--kv fp32|int8|k8v32|"
                         "k32v8] [--no-graph]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N "
                         "--bench-tg N)\n");
            return 2;
        }
    }
    // --kv (strata's name for the K / V stores) beats STRATA_Q36_KV; the stores are independent: int8 / k8v8 quantize
    // both, k8v32 the K alone, k32v8 the V alone.  q4_0 / k8v4 (PR #21's Hadamard rotation) are not in this engine.
    {
        std::string kv = kv_opt;
        if (kv.empty()) { const char* e = std::getenv("STRATA_Q36_KV"); kv = e != nullptr ? e : "fp32"; }
        if (kv == "fp32" || kv == "f32") {}   // both stores FP32 (the default)
        else if (kv == "int8" || kv == "k8v8") kv_k8 = kv_v8 = true;
        else if (kv == "k8v32") kv_k8 = true;
        else if (kv == "k32v8") kv_v8 = true;
        else if (serve_mode) {
            // serve passes the config's args: a Flash-Next value (fp16, q4_0, k8v4) was skipped before --kv existed
            // here, and still is (both stores FP32)
            std::fprintf(stderr, "strata-q36: ignoring --kv %s (this engine's stores: fp32, int8, k8v8, k8v32, k32v8)\n",
                         kv.c_str());
        } else {
            std::fprintf(stderr, "strata-q36: --kv must be fp32, int8, k8v8, k8v32 or k32v8 (q4_0 and k8v4 are PR "
                                 "#21's rotation kernels: not in this engine)\n");
            return 2;
        }
        if (kv_k8 || kv_v8)
            std::fprintf(stderr, "strata-q36: KV cache: K %s, V %s\n", kv_k8 ? "int8" : "fp32", kv_v8 ? "int8" : "fp32");
    }
    if (model.empty()) { std::fprintf(stderr, "strata-q36: --model is required\n"); return 2; }
#if !defined(_WIN32)
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);   // every kernel loaded with the context, not in the first prompt
    setenv("STRATA_Q8_PACKED", "1", 0);   // the packed Q8_0 decode copy unless STRATA_Q8_PACKED is set otherwise
    setenv("STRATA_MMQ_KQ", "1", 0);      // the K-quant MMQ prompt kernels (every expert of this family is a K-quant)
#else
    if (std::getenv("STRATA_MMQ_KQ") == nullptr) _putenv_s("STRATA_MMQ_KQ", "1");
    if (std::getenv("STRATA_Q8_PACKED") == nullptr) _putenv_s("STRATA_Q8_PACKED", "1");   // as on Linux (it was missing here)
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#endif
    if (mtp_d < 0 || mtp_d > 7) { std::fprintf(stderr, "strata-q36: --mtp takes 0 .. 7 drafts\n"); return 2; }
    if (chunk < 0 || chunk > 4096 || ctx < 256) { std::fprintf(stderr, "strata-q36: bad --chunk / --max-context\n"); return 2; }
    Model m;
    std::string err;
    try {
        if (!load(model, m, err, mtp_d > 0)) { std::fprintf(stderr, "strata-q36: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-q36: %s\n", e.what());
        return 1;
    }
    Engine E;
    E.kv_k8 = kv_k8;
    E.kv_v8 = kv_v8;
    E.use_graph = graph;
    E.packed_droppable = expert_slots <= 0 && expert_vram < 0 && std::getenv("STRATA_Q36_KEEP_PACKED") == nullptr;
    if (chunk == 0) {
        // auto: the largest chunk (4096 .. 1024) that fits beside every expert in VRAM; 2048 when they will not all fit (a prompt chunk streams every layer's
        // experts over PCIe once, so a bigger chunk pays that once for more tokens)
        size_t fr = 0, tot = 0, experts = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        for (const Layer& L : m.layers) experts += L.stride * (size_t) m.g.n_expert;
        const size_t fr_raw = fr;   // before the packed Q8_0 copies count as given back
        // (the MTP block's experts can go to the CPU, the packed Q8_0 copies can be given back: see place_experts)
        size_t packed = 0;
        for (const auto& pk : m.packed) packed += std::get<2>(pk);
        fr += packed;
        // the engine's buffers at 512 (~1.1 GiB on gfx1201 at 8K, less the scores buffer's shrink) + the small reserve
        const double want = (double) experts + 1.1 * (1ull << 30);
        const bool fits = expert_slots <= 0 && (expert_vram < 0 ? want <= (double) fr
                                                               : (double) experts <= expert_vram * (double) (1ull << 30));
        chunk = fits ? 512 : 2048;
        const double GiB = (double) (1ull << 30);
        if (fits && expert_vram < 0) {
            // a bigger prompt chunk feeds the MMQ expert kernels more rows per expert (R9700, UD-Q4_K_XL, pp4096: 512 3390, 1024
            // 4495, 2048 5353, 4096 5750 tok/s): the largest that fits beside the experts.  Its buffers grow by about 0.5 MiB per
            // token (measured 0.47: 1.66 GiB more at 4096 than at 512), the sessions' K / V caches by 50 KB per cell and session.
            // With several slots a long prompt would hold the other slots' windows for a whole chunk: 1024 at most there.
            // (measured on a 32 GB R9700 at 8192 cells, 512 chunk: 1.77 GiB of buffers + K / V; the free room after init must still hold
            // every expert and the 0.77 GiB reserve, or place_experts gives the packed decode copies back - slower decode - or goes hybrid)
            const int ns = std::max(1, batch);
            double kv_one = 45056.0;
            double kv_shared = 5120.0;   // per cell: a session's K / V, and the fp16 copies the prompt attention keeps
            if (kv_k8 || kv_v8) {   // --kv: an int8 store takes 1 + 2 / kKVG bytes per value instead of 4
                kv_one *= ((kv_k8 ? 1.03125 / 4.0 : 1.0) + (kv_v8 ? 1.03125 / 4.0 : 1.0)) / 2.0;
                // the FP16 copies (none past a block: Engine::kv16_blk) and the split-K partials (kGemmT rows instead of 64)
                const auto& g = m.g;
                kv_shared = (Engine::kv16_block_cells(true, ctx) > 0 ? 0.0 : (double) (g.n_head_kv * g.head_dim * 2 * 2)) +
                            (double) (g.n_head * Engine::kGemmT) * (double) (g.head_dim + 2) * 4.0 / kChunkCells;
            }
            const double sess_extra = std::max(0.0, (double) ctx * (kv_one * ns + kv_shared) - 8192.0 * (kv_one + kv_shared));
            const int cap = ns > 1 ? 1024 : 4096;
            for (int c : {4096, 2048, 1024})
                if (c <= cap && (double) experts + 2.75 * GiB + (double) (c - 512) * 0.5 * 1048576.0 + sess_extra <= (double) fr_raw) {   // (no give-back of the packed decode copies for a bigger chunk)
                    chunk = c;
                    break;
                }
        } else if (!fits) {
            // the hybrid mode: 2048 (a chunk streams every layer's experts over PCIe once); a small card, where the chunk's buffers
            // would take the slots' room, falls back to less
            const double fr_gib = (double) fr / GiB;
            chunk = fr_gib >= 7.5 ? 2048 : fr_gib >= 5.5 ? 1024 : 512;
        }
        if (const char* cv = std::getenv("STRATA_Q36_CHUNK")) chunk = std::max(256, std::min(4096, std::atoi(cv)));   // force a chunk (512: the old default)
        if (std::getenv("STRATA_Q36_PLACE_DEBUG"))
            std::fprintf(stderr, "strata-q36: chunk rule: free %.2f GiB (raw %.2f, with %.2f packed), main experts %.2f GiB, chunk %d\n",
                         fr / 1073741824.0, fr_raw / 1073741824.0, packed / 1073741824.0, experts / 1073741824.0, chunk);
    }
    E.init(m, ctx, chunk, batch, vision);
    if (!place_experts(m, E, expert_vram, cpu_threads, hybrid_rows, expert_slots, err)) {
        std::fprintf(stderr, "strata-q36: %s\n", err.c_str());
        return 1;
    }
    struct Report {
        Engine& E;
        ~Report() {
            if (E.verify_k) std::fprintf(stderr, "strata-q36 VERIFY: %lld comparisons, %lld mismatches\n", E.v_calls, E.v_bad);
            if (E.verify_k)
                for (const auto& kv : E.v_by)
                    std::fprintf(stderr, "strata-q36 VERIFY   %s: %lld mismatches\n", kv.first.c_str(), kv.second);
            if (!E.m->hybrid) return;
            const long long n = E.hy_cpu + E.hy_gpu;
            std::fprintf(stderr, "strata-q36: hybrid experts: %lld of %lld batch entries on the GPU (%.1f%%), %lld on the "
                         "CPU pool in %.0f ms, %lld slot replacements\n", E.hy_gpu, n, n ? 100.0 * E.hy_gpu / n : 0.0,
                         E.hy_cpu, E.hy_cpu_ms, E.hy_swaps);
        }
    } report{E};
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
    if (draft_vocab < 0 && std::getenv("STRATA_Q36_DRAFT_VOCAB") != nullptr) draft_vocab = std::atoi(std::getenv("STRATA_Q36_DRAFT_VOCAB"));
    E.draft_n = std::max(0, draft_vocab);
    E.bmtp = (batch_mtp || (std::getenv("STRATA_BATCH_MTP") != nullptr && std::string(std::getenv("STRATA_BATCH_MTP")) != "0")) && E.n_sess > 1 && E.mtp_d > 0;
    if (batch_mtp && !E.bmtp) std::fprintf(stderr, "strata-q36: --batch-mtp needs --batch N (2 or more) and --mtp N: ignored\n");
    if (warm) E.warmup();
    if (serve_mode) return serve(E);
    if (bench_pp > 0 || bench_tg > 0) {
        // llama-bench's shapes: a prompt of bench_pp tokens from a fresh state, then bench_tg tokens one at a time
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (rng() % 200000u);
        for (int rep = 0; rep < 3; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            CK(cudaStreamSynchronize(E.st));
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                E.process(&one, 1, [](int64_t) {});
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.1f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg,
                        bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
            std::fflush(stdout);
        }
        return 0;
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-q36: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    if (!emb_file.empty()) {
        if (!vision) { std::fprintf(stderr, "strata-q36: --emb needs --vision\n"); return 2; }
        ImgReq ir;
        const std::string ve = build_images(emb_file, prompt, m.g.n_embd, {}, ir);
        if (!ve.empty()) { std::fprintf(stderr, "strata-q36: %s\n", ve.c_str()); return 2; }
        E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) prompt.size(), ir.delta);
        std::fprintf(stderr, "strata-q36: %zu picture(s), %zu rows, text after them at cell + %d\n", ir.imgs.size(),
                     ir.rows.size() / (size_t) m.g.n_embd, (int) ir.delta);
    }
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (E.mtp_d > 0 && forced.empty() && !ff) {
        long long offered = 0, accepted = 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = E.greedy();
        while ((int) gen.size() < n_gen) {
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks);
            offered += E.mtp_d;
            accepted += a;
            fix.assign(drafts.begin(), drafts.begin() + a);
            for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) gen.push_back(drafts[(size_t) i]);
            nx = picks[a];
        }
        gen.resize((size_t) n_gen);
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.1f tok/s), MTP %d drafts: "
                     "%lld of %lld accepted (%.0f%%)\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), E.mtp_d, accepted, offered,
                     offered ? 100.0 * accepted / offered : 0.0);
        return 0;
    }
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) {});
    }
    if (ff) std::fclose(ff);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.1f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}
