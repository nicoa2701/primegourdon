// util.hpp — exact integer roots and logarithms.
//
// All functions return mathematically exact results (no floating-point error
// in the final value): a fast floating-point seed is corrected by integer
// comparisons so the answer is guaranteed correct for the whole input range.
#ifndef PRIMECOUNT_UTIL_HPP
#define PRIMECOUNT_UTIL_HPP

#include "int_types.hpp"

#include <cmath>
#include <cstdint>

namespace primecount {

// Largest integer s with s*s <= x.
inline int64_t isqrt(int128_t x) {
  if (x <= 0)
    return 0;

  int128_t s = static_cast<int128_t>(std::sqrt(static_cast<double>(x)));
  // Correct a possibly off-by-a-little floating-point seed.
  while (s > 0 && s * s > x)
    --s;
  while ((s + 1) * (s + 1) <= x)
    ++s;
  return static_cast<int64_t>(s);
}

// Largest integer r with r^N <= x.
template <int N>
inline int64_t iroot(int128_t x) {
  static_assert(N >= 1, "iroot requires N >= 1");
  if (x <= 0)
    return 0;
  if (N == 1)
    return static_cast<int64_t>(x);

  auto pow_n = [](int128_t base) {
    int128_t p = 1;
    for (int i = 0; i < N; ++i)
      p *= base;
    return p;
  };

  int128_t r = static_cast<int128_t>(std::pow(static_cast<double>(x), 1.0 / N));
  if (r < 1)
    r = 1;
  while (r > 1 && pow_n(r) > x)
    --r;
  while (pow_n(r + 1) <= x)
    ++r;
  return static_cast<int64_t>(r);
}

// Floor of the natural logarithm's integer companion: largest e with 2^e <= x.
inline int ilog2(int128_t x) {
  int e = 0;
  while (x > 1) {
    x >>= 1;
    ++e;
  }
  return e;
}

} // namespace primecount

#endif // PRIMECOUNT_UTIL_HPP
