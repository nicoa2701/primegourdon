#include "gourdon.hpp"
#include "gterms.hpp"

namespace primecount {

// Phi0 — ordinary leaves (gourdon.md §3). See g_Phi0 in gourdon_terms.cpp.
maxint_t Phi0(maxint_t x, int threads) { return g_Phi0(x, threads); }

} // namespace primecount
