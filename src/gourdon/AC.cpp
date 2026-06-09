#include "gourdon.hpp"
#include "gterms.hpp"

namespace primecount {

// AC — the easy special leaves: Gourdon's A (§5) plus C = C1+C2 (§6).
maxint_t AC(maxint_t x, int threads) { return g_A(x, threads) + g_C(x, threads); }

} // namespace primecount
