// tools/qwen35/kv_v4_ref.cpp - CPU reference of the --kv k8v4 V store (g++ -O2 -std=c++17 kv_v4_ref.cpp && ./a.out): mirrors k_attn_prep's V4 branch (rotation, Q4_0 group, pack), nib4_i8x4 (copied verbatim from q36_engine.cu: keep them equal), the dequant of k_kv_f16 / k_attn_gqa, and the FWHT butterfly of fwht256_block.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
typedef _Float16 half_t;
static inline uint32_t nib4_i8x4(uint32_t h) {   // verbatim from q36_engine.cu
    const uint32_t w = (h & 0x000Fu) | ((h & 0x00F0u) << 4) | ((h & 0x0F00u) << 8) | ((h & 0xF000u) << 12);
    return ((w | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
}
// fwht256_block as the kernel runs it, per-thread lockstep emulated: 5 shuffle stages then 3 shared-memory stages
static void fwht256(float* v) {
    float x[256]; std::memcpy(x, v, sizeof x);
    for (int h = 1; h < 32; h <<= 1) { float y[256]; for (int i = 0; i < 256; ++i) { float o = x[i ^ h]; y[i] = (i & h) ? o - x[i] : x[i] + o; } std::memcpy(x, y, sizeof x); }
    for (int h = 32; h < 256; h <<= 1) { float y[256]; for (int i = 0; i < 256; ++i) { float o = x[i ^ h]; y[i] = (i & h) ? o - x[i] : x[i] + o; } std::memcpy(x, y, sizeof x); }
    for (int i = 0; i < 256; ++i) v[i] = x[i] * 0.0625f;
}
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while (0)

// the kernel's store of one cell: returns packed bytes [128] and scales [8] (as half)
static void store_cell(const float* vin, bool rot, uint8_t* bytes, half_t* sc) {
    float v[256]; std::memcpy(v, vin, sizeof v);
    if (rot) fwht256(v);
    int q[256];
    for (int g = 0; g < 8; ++g) {
        float am = 0, mv = 0; bool first = true;
        for (int l = 0; l < 32; ++l) { float x = v[g * 32 + l]; float a = std::fabs(x);
            if (first || a > am || (a == am && x > mv)) { am = a; mv = x; first = false; } }
        half_t dh = (half_t) (mv != 0.f ? mv / -8.0f : 1.0f);
        float d = (float) dh != 0.f ? (float) dh : 1.0f;
        for (int l = 0; l < 32; ++l) q[g * 32 + l] = (int) std::fmin(std::fmax(std::floor(v[g * 32 + l] / d + 8.5f), 0.0f), 15.0f);
        sc[g] = (half_t) d;
    }
    for (int i = 0; i < 256; i += 2) bytes[i >> 1] = (uint8_t) (q[i] | (q[i + 1] << 4));
}
// decode paths: (a) k_kv_f16: nibble extraction; (b) gqa MV: 2 bytes -> nib4_i8x4 -> signed bytes
static void load_cell_a(const uint8_t* bytes, const half_t* sc, float* out) {
    for (int d = 0; d < 256; ++d) { int nib = (bytes[d / 2] >> ((d & 1) * 4)) & 15; out[d] = (float) (half_t) ((float) sc[d / 32] * (float) (nib - 8)); }
}
static void load_cell_b(const uint8_t* bytes, const half_t* sc, float* out) {
    for (int g = 0; g < 8; ++g) for (int gg = 0; gg < 8; ++gg) {   // lane g's 4 values 32 w + 4 g ..
        unsigned short us; std::memcpy(&us, bytes + 16 * g + 2 * gg, 2);
        uint32_t x = nib4_i8x4(us);
        for (int k = 0; k < 4; ++k) out[32 * g + 4 * gg + k] = (float) (half_t) ((float) sc[g] * (float) (int8_t) ((x >> (8 * k)) & 255));
    }
}
int main() {
    // 1. nib4_i8x4 over all 65536 inputs
    for (uint32_t h = 0; h < 65536; ++h) {
        uint32_t x = nib4_i8x4(h);
        for (int k = 0; k < 4; ++k) CHECK((int8_t) ((x >> (8 * k)) & 255) == (int) ((h >> (4 * k)) & 15) - 8, "nib4 h=%u k=%d", h, k);
    }
    // 2. FWHT: orthonormal, self-inverse, equals the Sylvester matrix
    std::mt19937 rng(7); std::normal_distribution<float> N(0, 1);
    float v[256], w[256]; for (auto& x : v) x = N(rng);
    std::memcpy(w, v, sizeof v); fwht256(w);
    for (int i = 0; i < 256; ++i) { double s = 0; for (int j = 0; j < 256; ++j) s += ((__builtin_popcount(i & j) & 1) ? -1.0 : 1.0) * v[j]; s /= 16.0; CHECK(std::fabs(s - w[i]) < 1e-4, "fwht[%d] %g vs %g", i, s, w[i]); }
    fwht256(w); for (int i = 0; i < 256; ++i) CHECK(std::fabs(w[i] - v[i]) < 1e-5, "fwht inverse %d", i);
    // 3. store -> load: both decode paths agree bit for bit; the stored max element is exact; error stats
    double se = 0, sn = 0; int ncell = 20000;
    for (int rot = 0; rot < 2; ++rot) {
        se = sn = 0;
        for (int c = 0; c < ncell; ++c) {
            float cell[256]; for (int i = 0; i < 256; ++i) cell[i] = N(rng) * (i % 61 == 0 ? 8.f : 1.f);   // a few outlier channels
            if (c % 7 == 0) std::fill(cell, cell + 256, 0.f);   // all-zero cells
            uint8_t b[128]; half_t sc[8]; store_cell(cell, rot, b, sc);
            float a[256], bb[256]; load_cell_a(b, sc, a); load_cell_b(b, sc, bb);
            CHECK(std::memcmp(a, bb, sizeof a) == 0, "decode paths differ, cell %d", c);
            float r[256]; std::memcpy(r, rot ? a : a, sizeof r);
            if (rot) fwht256(r);   // the merge's rotation back is on the output, equivalent by linearity
            for (int i = 0; i < 256; ++i) { se += (r[i] - cell[i]) * (r[i] - cell[i]); sn += cell[i] * cell[i]; }
        }
        std::printf("rot=%d: relative RMS error of stored V (outlier-channel test data): %.4f\n", rot, std::sqrt(se / sn));
    }
    // 4. the max element of each group is stored exactly (to the FP16 rounding of d)
    { float cell[256]; for (auto& x : cell) x = N(rng); uint8_t b[128]; half_t sc[8]; store_cell(cell, false, b, sc); float a[256]; load_cell_a(b, sc, a);
      for (int g = 0; g < 8; ++g) { int im = 0; for (int l = 0; l < 32; ++l) if (std::fabs(cell[g * 32 + l]) > std::fabs(cell[g * 32 + im])) im = l;
        CHECK(std::fabs(a[g * 32 + im] - cell[g * 32 + im]) <= 8e-4f * std::fabs(cell[g * 32 + im]), "group max %d: %g vs %g", g, a[g * 32 + im], cell[g * 32 + im]); } }
    std::printf(fails ? "FAILED (%d)\n" : "all checks passed\n", fails);
    return fails != 0;
}
