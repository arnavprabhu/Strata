// The Qwen3.6 engine's host sampler (q36_engine.cu includes this; tools/q36_spec_sample_test.cpp tests it without a GPU).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <random>
#include <vector>

namespace strata::q36_sampling {

struct Sampling {
    float temperature = 0.f, top_p = 1.f, min_p = 0.f, repeat = 1.f, freq = 0.f, present = 0.f;
    int top_k = 20, last_n = 0;
    unsigned long long seed = 0;
    bool needs_logits() const { return temperature > 0.f || repeat != 1.f || freq != 0.f || present != 0.f; }
};

/// The repetition / frequency / presence penalties over the last `last_n` of the `n` tokens in `h`, applied to `lg`.
inline void apply_penalties(std::vector<float>& lg, const Sampling& s, const int32_t* h, size_t n) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= n ? 0 : n - (size_t) s.last_n;
        for (size_t i = from; i < n; ++i) ++cnt[h[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
}

/// The token distribution of a sampled request (temperature > 0) after the penalties: the `kk` highest logits (idx, by
/// descending logit), their weights exp((l - l_max) / T) in p, the first `keep` of them surviving min_p and top_p, and
/// tot = the sum of those `keep` weights.  The probability of token idx[i] is p[i] / tot for i < keep, 0 for any other.
/// fast: pick the kk with one pass and a heap (the order of equal logits is by token id) instead of a partial sort of
/// all ids; the same set of kept tokens and weights, but for the order among equal logits.
struct Dist {
    std::vector<int> idx;
    std::vector<double> p;
    size_t keep = 0;
    double tot = 0;
};

inline void top_ids(const std::vector<float>& lg, int kk, bool fast, std::vector<int>& idx) {
    if (!fast) {
        idx.resize(lg.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        idx.resize((size_t) kk);
        return;
    }
    idx.resize((size_t) kk);
    for (int i = 0; i < kk; ++i) idx[(size_t) i] = i;
    const auto above = [&](int a, int b) { return lg[(size_t) a] > lg[(size_t) b]; };
    std::make_heap(idx.begin(), idx.end(), above);   // the lowest kept logit on top
    float thr = lg[(size_t) idx[0]];
    for (size_t i = (size_t) kk; i < lg.size(); ++i) {
        if (!(lg[i] > thr)) continue;
        std::pop_heap(idx.begin(), idx.end(), above);
        idx.back() = (int) i;
        std::push_heap(idx.begin(), idx.end(), above);
        thr = lg[(size_t) idx[0]];
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        const float x = lg[(size_t) a], y = lg[(size_t) b];
        return x > y || (x == y && a < b);
    });
}

inline void filter_dist(const std::vector<float>& lg, const Sampling& s, bool fast, Dist& d) {
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    top_ids(lg, kk, fast, d.idx);
    std::vector<double>& p = d.p;
    p.resize(d.idx.size());
    const double mx = lg[(size_t) d.idx[0]];
    double sum = 0;
    for (size_t i = 0; i < d.idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) d.idx[i]] - mx) / s.temperature));
    size_t keep = d.idx.size();
    double cum = 0;
    for (size_t i = 0; i < d.idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    d.keep = std::max<size_t>(1, keep);
    d.tot = 0;
    for (size_t i = 0; i < d.keep; ++i) d.tot += p[i];
}

inline int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    apply_penalties(lg, s, hist.data(), hist.size());
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    Dist d;
    filter_dist(lg, s, false, d);
    std::uniform_real_distribution<double> u(0.0, d.tot);
    double r = u(rng);
    for (size_t i = 0; i < d.keep; ++i) {
        if ((r -= d.p[i]) <= 0) return d.idx[i];
    }
    return d.idx[d.keep - 1];
}

/// One row of a speculative step (STRATA_Q36_SPEC_SAMPLE), temperature > 0.  The draft is the point mass at `draft`
/// (the MTP head's argmax), so the exact rule is: accept it with probability p(draft), where p is the row's distribution
/// that sample() would draw from (same penalties over h[0 .. n), same top_k / min_p / top_p); on a rejection draw the
/// replacement from p with the draft removed (p renormalized over the rest), which is never the draft.  With draft < 0
/// (the row after the last draft) it is a plain draw from p.  Together: every returned token, accepted or replaced, is
/// distributed as p, and a returned token equals `draft` exactly when the draft is accepted.
inline int spec_row(std::vector<float>& lg, const Sampling& s, const int32_t* h, size_t n, int draft,
                    std::mt19937_64& rng, Dist& d) {
    apply_penalties(lg, s, h, n);
    filter_dist(lg, s, true, d);
    double pd = 0;   // the draft's weight in p
    for (size_t i = 0; i < d.keep && draft >= 0; ++i)
        if (d.idx[i] == draft) { pd = d.p[i]; break; }
    if (draft >= 0) {
        std::uniform_real_distribution<double> acc(0.0, 1.0);
        if (acc(rng) * d.tot < pd) return draft;
    }
    const double rest = d.tot - pd;
    if (draft >= 0 && !(rest > d.tot * 1e-12)) return draft;   // nothing else has weight (a rounding edge): keep the draft
    std::uniform_real_distribution<double> u(0.0, rest);
    double r = u(rng);
    int last = -1;
    for (size_t i = 0; i < d.keep; ++i) {
        if (d.idx[i] == draft) continue;
        last = d.idx[i];
        if ((r -= d.p[i]) <= 0) return last;
    }
    return last;
}

}  // namespace strata::q36_sampling
