// gterms.hpp — the true Gourdon terms (gourdon.md). Implemented and validated
// in isolation before the public flags / pi_gourdon are switched over to them.
//
//   pi(x) = A - B + C + D + Phi0 + Sigma
#ifndef PRIMECOUNT_GTERMS_HPP
#define PRIMECOUNT_GTERMS_HPP

#include "int_types.hpp"

namespace primecount {

// threads <= 0 means "use all available cores".
maxint_t g_Phi0(int128_t x, int threads = 0);  // ordinary leaves, gourdon.md §3
maxint_t g_Sigma(int128_t x, int threads = 0); // seven trivial terms, §4
maxint_t g_A(int128_t x, int threads = 0);     // easy two-prime leaves, §5
maxint_t g_B(int128_t x, int threads = 0);     // non-trivial part of P2, §8
maxint_t g_C(int128_t x, int threads = 0);     // easy special leaves C1+C2, §6
maxint_t g_D(int128_t x, int threads = 0);     // hard special leaves D1+D2, §7

// pi(x) = A - B + C + D + Phi0 + Sigma (gourdon.md §9).
maxint_t g_pi(int128_t x, int threads = 0);

} // namespace primecount

#endif // PRIMECOUNT_GTERMS_HPP
