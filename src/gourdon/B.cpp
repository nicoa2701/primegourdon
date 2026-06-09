#include "gourdon.hpp"
#include "gterms.hpp"

namespace primecount {

// B — non-trivial part of P2 (gourdon.md §8). In Gourdon's assembly B is
// subtracted (pi = A - B + C + D + Phi0 + Sigma); we expose the SIGNED
// contribution -g_B so that Phi0 + Sigma + AC + B + D == pi(x) directly.
maxint_t B(maxint_t x, int threads) { return -g_B(x, threads); }

} // namespace primecount
