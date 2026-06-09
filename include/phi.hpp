// phi.hpp — the partial sieve (Legendre) function φ(x, a).
//
// φ(x, a) = #{ n in [1, x] : n is not divisible by any of the first a primes }.
// Small a is answered in O(1) by a precomputed wheel (PhiTiny); larger a uses
// the recurrence φ(x, a) = φ(x, a-1) − φ(x/p_a, a-1).
#ifndef PRIMECOUNT_PHI_HPP
#define PRIMECOUNT_PHI_HPP

#include "int_types.hpp"

namespace primecount {

// Exact φ(x, a) for x >= 0 and a >= 0. Result can reach x (when a == 0), hence
// the 128-bit return type.
int128_t phi(int128_t x, int a);

} // namespace primecount

#endif // PRIMECOUNT_PHI_HPP
