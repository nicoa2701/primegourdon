#include "gourdon.hpp"
#include "gterms.hpp"

namespace primecount {

// Full prime count via Gourdon's algorithm (gourdon.md §9):
//   pi(x) = A - B + C + D + Phi0 + Sigma
maxint_t pi_gourdon(maxint_t x, int threads) { return g_pi(x, threads); }

} // namespace primecount
