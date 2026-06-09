#include "phi_pi.hpp"
#include "phi.hpp"
#include "pitable.hpp"
#include "util.hpp"

namespace primecount {

// primes is 0-indexed: primes[i] = p_{i+1}. So p_a = primes[a-1] and the next
// prime p_{a+1} = primes[a].
// All-int64 recursion, used once x <= INT64_MAX (always true in the hard-leaf
// callers when x itself fits int64). The quotient x/primes[a-1] only shrinks, so
// the subtree stays int64 — no 128-bit division (__divti3) per recursion level,
// which perf annotate flagged as the top hard-leaf cost via the phi_start
// recompute. phi(x) now has its own int64 fast path, so the base is int64 too.
static int64_t phi_pi64(int64_t x, int a, const std::vector<int64_t>& primes,
                        const PiTable& pi) {
  if (x <= 0)
    return 0;
  if (a <= 7)
    return static_cast<int64_t>(phi(x, a)); // PhiTiny base, O(1)
  const int64_t p_next = primes[a];
  if (x < p_next)
    return 1;
  if (x <= pi.limit() &&
      static_cast<int128_t>(p_next) * p_next > x)
    return pi(x) - a + 1;
  return phi_pi64(x, a - 1, primes, pi) -
         phi_pi64(x / primes[a - 1], a - 1, primes, pi);
}

int64_t phi_pi(int128_t x, int a, const std::vector<int64_t>& primes,
               const PiTable& pi) {
  if (x <= 0)
    return 0;
  if (x <= INT64_MAX) // common case: int64 recursion, no 128-bit division
    return phi_pi64(static_cast<int64_t>(x), a, primes, pi);
  if (a <= 7)
    return static_cast<int64_t>(phi(x, a)); // PhiTiny base, O(1)

  const int64_t p_next = primes[a]; // p_{a+1}
  if (x < p_next)
    return 1; // no prime in (p_a, x] survives besides the integer 1
  // Legendre cutoff: once p_{a+1} > √x, no composite <= x is coprime to the
  // first a primes, so phi(x,a) = pi(x) - a + 1. Needs pi(x), i.e. x <= limit.
  if (x <= pi.limit() &&
      static_cast<int128_t>(p_next) * p_next > x)
    return pi(static_cast<int64_t>(x)) - a + 1;

  return phi_pi(x, a - 1, primes, pi) -
         phi_pi(x / primes[a - 1], a - 1, primes, pi);
}

} // namespace primecount
