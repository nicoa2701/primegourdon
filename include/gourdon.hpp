// gourdon.hpp — public entry points for Gourdon's prime-counting algorithm.
//
// π(x) = Φ0(x) + Σ(x) + AC(x) + B(x) + D(x)
//
// Each term is exposed individually so it can be timed and validated on its
// own (CLI flags --Phi0, --Sigma, --AC, --B, --D), and pi_gourdon assembles
// them into the full count.
#ifndef PRIMECOUNT_GOURDON_HPP
#define PRIMECOUNT_GOURDON_HPP

#include "int_types.hpp"

namespace primecount {

// Tiny-x fast path: below this limit pi_gourdon returns an exact segmented sieve
// (count_primes) instead of running Gourdon. count_primes(x) is sub-millisecond
// here (~µs up to 1e5, ~1 ms at 1e6) — faster than Gourdon's irreducible ~2 ms
// setup floor (build_ctx + B presieve) even after the small-x smoothing (#5/#6),
// and exact, so it sidesteps the degenerate-parameter regime entirely. Above 1e6
// count_primes' linear cost overtakes Gourdon, so the cutoff stays at 1e6.
constexpr int64_t kSieveCutoff = 1000000; // 1e6

// Full prime count π(x) via Gourdon's algorithm.
maxint_t pi_gourdon(maxint_t x, int threads);

// Per-term breakdown with HONEST timings: the shared context (build_ctx) is built
// ONCE (as in pi_gourdon), then each term timed against it. The public per-term
// wrappers each rebuild build_ctx (AC twice) — 6 build_ctx vs 1 — so a naive
// --verbose loop over them overcounts the total by +40-50% and buries the cheap
// terms (Phi0/Sigma) under build_ctx. Values are identical to the per-term
// wrappers; only timings are corrected. `b` is the SIGNED contribution (=-B_impl,
// negative), so pi == phi0 + sigma + ac + b + d.
struct PiBreakdown {
  maxint_t phi0 = 0, sigma = 0, ac = 0, b = 0, d = 0, pi = 0;
  double build_us = 0, phi0_us = 0, sigma_us = 0, ac_us = 0, b_us = 0, d_us = 0;
};
PiBreakdown pi_gourdon_verbose(maxint_t x, int threads);

// Individual terms of the decomposition.
maxint_t Phi0(maxint_t x, int threads);
maxint_t Sigma(maxint_t x, int threads);
maxint_t AC(maxint_t x, int threads);
maxint_t B(maxint_t x, int threads);
maxint_t D(maxint_t x, int threads);

} // namespace primecount

#endif // PRIMECOUNT_GOURDON_HPP
