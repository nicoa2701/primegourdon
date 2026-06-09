// phi_pi.hpp — partial sieve phi(x, a) accelerated with a pi-table cutoff.
//
// Adds the Legendre identity phi(t, a) = pi(t) - a + 1 (valid once p_{a+1} > √t)
// to the recurrence, using an existing PiTable to answer it in O(1). This prunes
// the recursion enough that building the per-segment phi(x, i) start counts for
// the parallel hard-leaves sieve (D) stops being a bottleneck. Distinct from the
// plain phi() in phi.hpp, which has no pi oracle.
#ifndef PRIMECOUNT_PHI_PI_HPP
#define PRIMECOUNT_PHI_PI_HPP

#include "int_types.hpp"

#include <cstdint>
#include <vector>

namespace primecount {

class PiTable;

// phi(x, a) = #{ n in [1, x] : n coprime to the first a primes }. `primes` is
// 0-indexed (primes[0] = 2); `pi` must cover at least sqrt of the largest x
// queried for the cutoff to fire. Result fits int64 for the ranges used in D.
int64_t phi_pi(int128_t x, int a, const std::vector<int64_t>& primes,
               const PiTable& pi);

} // namespace primecount

#endif // PRIMECOUNT_PHI_PI_HPP
