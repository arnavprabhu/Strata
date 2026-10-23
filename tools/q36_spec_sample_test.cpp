// CPU-only test of src/qwen35/q36_sampling.h (no GPU, no model):
//   1. sample() still draws what the pre-refactor code drew (same rng stream, same token), also with penalties;
//   2. the fast top-k selection keeps the same tokens and weights as the partial sort;
//   3. spec_row(): over many draws with random logits and random drafts, the token it returns is distributed as the
//      direct sampler's distribution, for several temperature / top_k / top_p / min_p / penalty settings, and it equals
//      the draft exactly when the draft is accepted (the acceptance rate is p(draft));
//   4. a whole speculative loop (D drafts per step, the accept chain of q36_engine.cu's spec_step) writes sequences
//      distributed as one token at a time, on a Markov "model" whose row logits depend on the previous token.
// Build and run:  c++ -O2 -std=c++17 -I src/qwen35 tools/q36_spec_sample_test.cpp -o /tmp/q36_spec_sample_test && /tmp/q36_spec_sample_test
#include "q36_sampling.h"

#include <cstdio>
#include <cstdlib>
#include <map>

using namespace strata::q36_sampling;

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// the code of q36_engine.cu's sample() before it moved into q36_sampling.h, verbatim
int sample_ref(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    if (s.last_n != 0 && (s.repeat != 1.f || s.freq != 0.f || s.present != 0.f)) {
        std::map<int, int> cnt;
        const size_t from = s.last_n < 0 || (size_t) s.last_n >= hist.size() ? 0 : hist.size() - (size_t) s.last_n;
        for (size_t i = from; i < hist.size(); ++i) ++cnt[hist[i]];
        for (const auto& [id, c] : cnt) {
            float& v = lg[(size_t) id];
            if (s.repeat != 1.f) v = v > 0 ? v / s.repeat : v * s.repeat;
            v -= (float) c * s.freq + s.present;
        }
    }
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return idx[keep - 1];
}

std::vector<float> random_logits(std::mt19937_64& g, int V, float spread) {
    std::normal_distribution<float> n(0.f, spread);
    std::vector<float> v((size_t) V);
    for (float& x : v) x = n(g);
    return v;
}

// the exact distribution sample() draws from: the kept tokens' weights over tot, others 0
std::vector<double> exact_p(std::vector<float> lg, const Sampling& s, const std::vector<int32_t>& h) {
    apply_penalties(lg, s, h.data(), h.size());
    Dist d;
    filter_dist(lg, s, false, d);
    std::vector<double> p(lg.size(), 0.0);
    for (size_t i = 0; i < d.keep; ++i) p[(size_t) d.idx[i]] = d.p[i] / d.tot;
    return p;
}

struct Cfg { float T, top_p, min_p; int top_k; float rep, freq, pres; int last_n; };

}  // namespace

int main() {
    std::mt19937_64 g(12345);
    const Cfg cfgs[] = {
        {0.7f, 1.0f, 0.0f, 20, 1.f, 0.f, 0.f, 0},          // temperature and top_k only
        {0.6f, 0.95f, 0.0f, 20, 1.f, 0.f, 0.f, 0},         // + top_p
        {1.0f, 0.8f, 0.05f, 40, 1.f, 0.f, 0.f, 0},         // + min_p
        {0.7f, 0.95f, 0.0f, 0, 1.1f, 0.2f, 1.5f, 64},      // top_k 0 (the sampler's 64) and every penalty
        {1.3f, 1.0f, 0.0f, 5, 1.f, 0.f, 1.5f, -1},         // presence penalty over the whole history, small top_k
        {0.3f, 0.5f, 0.1f, 8, 1.f, 0.f, 0.f, 0},          // a sharp one: draft often outside the kept set
    };
    const int V = 40;

    // 1 + 2: the refactored sample() against the old code; fast against slow selection
    {
        bool same = true, same_set = true;
        for (int it = 0; it < 3000; ++it) {
            const Cfg& c = cfgs[it % 6];
            Sampling s;
            s.temperature = c.T; s.top_p = c.top_p; s.min_p = c.min_p; s.top_k = c.top_k;
            s.repeat = c.rep; s.freq = c.freq; s.present = c.pres; s.last_n = c.last_n;
            std::vector<float> lg = random_logits(g, V, 2.f), lg2 = lg, lg3 = lg;
            std::vector<int32_t> h;
            for (int i = 0; i < 100; ++i) h.push_back((int32_t) (g() % V));
            std::mt19937_64 r1(it), r2(it);
            same &= sample(lg, s, h, r1) == sample_ref(lg2, s, h, r2) && r1() == r2();
            Dist a, b;
            apply_penalties(lg3, s, h.data(), h.size());
            filter_dist(lg3, s, false, a);
            filter_dist(lg3, s, true, b);
            same_set &= a.keep == b.keep && a.tot == b.tot && a.idx == b.idx && a.p == b.p;
        }
        check(same, "sample() == the pre-refactor code (token and rng state), 3000 random cases");
        check(same_set, "fast top-k selection == partial sort (ids, weights, keep, tot), 3000 random cases");
    }

    // 3: one row: spec_row's token is distributed as the sampler's, whatever the draft
    for (size_t ci = 0; ci < 6; ++ci) {
        const Cfg& c = cfgs[ci];
        Sampling s;
        s.temperature = c.T; s.top_p = c.top_p; s.min_p = c.min_p; s.top_k = c.top_k;
        s.repeat = c.rep; s.freq = c.freq; s.present = c.pres; s.last_n = c.last_n;
        double worst = 0, worst_acc = 0;
        bool iff = true;
        for (int trial = 0; trial < 6; ++trial) {
            const std::vector<float> lg = random_logits(g, V, 2.f);
            std::vector<int32_t> h;
            for (int i = 0; i < 30; ++i) h.push_back((int32_t) (g() % V));
            const std::vector<double> p = exact_p(lg, s, h);
            int draft;   // trial 0: the argmax of p; 1: a token outside the kept set if there is one; else random
            if (trial == 0) draft = (int) (std::max_element(p.begin(), p.end()) - p.begin());
            else if (trial == 1) { draft = 0; for (int i = 0; i < V; ++i) if (p[(size_t) i] == 0.0) { draft = i; break; } }
            else draft = (int) (g() % V);
            const int N = 400000;
            std::vector<double> cnt((size_t) V, 0.0);
            double acc = 0;
            Dist d;
            std::mt19937_64 rng(1000 + trial);
            for (int i = 0; i < N; ++i) {
                std::vector<float> l = lg;
                const int t = spec_row(l, s, h.data(), h.size(), draft, rng, d);
                cnt[(size_t) t] += 1;
                acc += t == draft;
                if (t != draft && p[(size_t) t] == 0.0) iff = false;   // a token outside the kept set
            }
            for (int t = 0; t < V; ++t) {
                const double sd = std::sqrt(std::max(p[(size_t) t], 1e-9) * (1 - p[(size_t) t]) / N);
                worst = std::max(worst, std::fabs(cnt[(size_t) t] / N - p[(size_t) t]) / sd);
                if (p[(size_t) t] == 0.0 && cnt[(size_t) t] > 0) worst = 1e9;
            }
            worst_acc = std::max(worst_acc, std::fabs(acc / N - p[(size_t) draft]) /
                                                 std::sqrt(std::max(p[(size_t) draft] * (1 - p[(size_t) draft]), 1e-9) / N));
        }
        char buf[200];
        std::snprintf(buf, sizeof buf, "config %zu: spec_row token ~ p (worst deviation %.1f sigma), acceptance = p(draft) "
                      "(%.1f sigma), no token outside the kept set", ci, worst, worst_acc);
        check(worst < 5.0 && worst_acc < 5.0 && iff, buf);
    }
    // no draft (the bonus row) is a plain draw
    {
        Sampling s; s.temperature = 0.7f; s.top_p = 0.95f; s.top_k = 20;
        const std::vector<float> lg = random_logits(g, V, 2.f);
        const std::vector<int32_t> h;
        const std::vector<double> p = exact_p(lg, s, h);
        std::vector<double> cnt((size_t) V, 0.0);
        std::mt19937_64 rng(7);
        Dist d;
        const int N = 400000;
        for (int i = 0; i < N; ++i) { std::vector<float> l = lg; cnt[(size_t) spec_row(l, s, h.data(), 0, -1, rng, d)] += 1; }
        double worst = 0;
        for (int t = 0; t < V; ++t)
            worst = std::max(worst, std::fabs(cnt[(size_t) t] / N - p[(size_t) t]) / std::sqrt(std::max(p[(size_t) t], 1e-9) * (1 - p[(size_t) t]) / N));
        char buf[120];
        std::snprintf(buf, sizeof buf, "bonus row (no draft) ~ p (worst deviation %.1f sigma)", worst);
        check(worst < 5.0, buf);
    }

    // 4: a whole speculative loop against one token at a time, on a small Markov "model" (V = 5, row logits = L[last token]);
    // the draft of a step is G[last token] (the row's argmax, but a random guess for the last token); sequences of 4
    {
        const int Vc = 5, LEN = 4;
        std::vector<std::vector<float>> L((size_t) Vc);
        for (auto& r : L) r = random_logits(g, Vc, 2.2f);
        std::vector<int> G((size_t) Vc);
        for (int t = 0; t < Vc; ++t) {
            const int am = (int) (std::max_element(L[(size_t) t].begin(), L[(size_t) t].end()) - L[(size_t) t].begin());
            G[(size_t) t] = t != Vc - 1 ? am : (int) (g() % Vc);
        }
        Sampling s; s.temperature = 0.8f; s.top_k = 4; s.top_p = 0.97f;
        auto row = [&](int last) { return L[(size_t) last]; };
        const int start = 0;
        auto code = [&](const std::vector<int>& q) { int c = 0; for (int x : q) c = c * Vc + x; return c; };
        int ncode = 1;
        for (int i = 0; i < LEN; ++i) ncode *= Vc;
        // exact: product of the one-token-at-a-time probabilities
        std::vector<double> pe((size_t) ncode, 0.0);
        for (int c = 0; c < ncode; ++c) {
            std::vector<int> q((size_t) LEN);
            int cc = c;
            for (int i = LEN - 1; i >= 0; --i) { q[(size_t) i] = cc % Vc; cc /= Vc; }
            double pr = 1;
            int last = start;
            for (int i = 0; i < LEN; ++i) { pr *= exact_p(row(last), s, {})[(size_t) q[(size_t) i]]; last = q[(size_t) i]; }
            pe[(size_t) c] = pr;
        }
        const int N = 1000000;
        for (int D = 1; D <= 3; ++D) {
            std::vector<double> cnt((size_t) ncode, 0.0);
            std::mt19937_64 rng(99 + D);
            Dist d;
            double accepted = 0, offered = 0;
            for (int it = 0; it < N; ++it) {
                std::vector<int> out;
                int last = start;
                while ((int) out.size() < LEN) {   // one step: drafts from the draft model, rows from the target model
                    std::vector<int> dr;
                    int prev = last;
                    for (int j = 0; j < D; ++j) { dr.push_back(G[(size_t) prev]); prev = dr.back(); }
                    int a = 0, tok;
                    for (;;) {   // spec_step's loop: row a follows x and drafts 0 .. a - 1
                        std::vector<float> l = row(a == 0 ? last : dr[(size_t) a - 1]);
                        tok = spec_row(l, s, nullptr, 0, a < D ? dr[(size_t) a] : -1, rng, d);
                        if (a == D || tok != dr[(size_t) a]) break;
                        ++a;
                    }
                    accepted += a; offered += D;
                    for (int j = 0; j < a; ++j) out.push_back(dr[(size_t) j]);
                    out.push_back(tok);
                    last = tok;
                }
                out.resize((size_t) LEN);
                cnt[(size_t) code(out)] += 1;
            }
            double worst = 0;
            for (int c = 0; c < ncode; ++c) {
                const double sd = std::sqrt(std::max(pe[(size_t) c], 1e-9) * (1 - pe[(size_t) c]) / N);
                worst = std::max(worst, std::fabs(cnt[(size_t) c] / N - pe[(size_t) c]) / sd);
                if (pe[(size_t) c] == 0.0 && cnt[(size_t) c] > 0) worst = 1e9;
            }
            char buf[160];
            std::snprintf(buf, sizeof buf, "speculative loop, D = %d: 4-token sequences ~ one token at a time (%d outcomes, worst "
                          "%.1f sigma; drafts accepted %.0f%%)", D, ncode, worst, 100.0 * accepted / offered);
            check(worst < 5.0, buf);
        }
    }
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
