#include "gourdon.hpp"
#include "gterms.hpp"

namespace primecount {

// D — the hard special leaves D1+D2 (gourdon.md §7).
maxint_t D(maxint_t x, int threads) { return g_D(x, threads); }

} // namespace primecount
