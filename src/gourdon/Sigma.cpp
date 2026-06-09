#include "gourdon.hpp"
#include "gterms.hpp"

namespace primecount {

// Sigma — the seven trivial terms Sigma_0..Sigma_6 (gourdon.md §4).
maxint_t Sigma(maxint_t x, int threads) { return g_Sigma(x, threads); }

} // namespace primecount
