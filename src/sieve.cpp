#include "sieve.hpp"
#include "util.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace primecount {

namespace {

// Odd primes p with 3 <= p <= limit, used as sieving primes for segments.
std::vector<int64_t> simple_odd_primes(int64_t limit) {
  std::vector<int64_t> primes;
  if (limit < 3)
    return primes;

  // is_composite[i] represents the odd number 2*i + 3.
  int64_t n = (limit - 3) / 2 + 1;
  std::vector<bool> is_composite(n, false);
  for (int64_t i = 0; i < n; ++i) {
    if (is_composite[i])
      continue;
    int64_t p = 2 * i + 3;
    primes.push_back(p);
    // Mark multiples p*p, p*p+2p, ... (only odd multiples matter).
    for (int64_t j = (p * p - 3) / 2; j < n; j += p)
      is_composite[j] = true;
  }
  return primes;
}

} // namespace

std::vector<int64_t> generate_primes(int64_t limit) {
  std::vector<int64_t> primes;
  if (limit < 2)
    return primes;
  // Reserve ~pi(limit) up front so the push_back loop never reallocates. Without
  // this the vector grows by doubling: its final CAPACITY can reach ~2x the prime
  // count and the last realloc transiently holds old+new buffers — a large waste
  // at high x (the list is the O(sqrt x) RAM peak: ~407 MB at 1e18, ~3.6 GB at
  // 1e20). Dusart 2010: pi(n) <= n/(ln n - 1.1) for n >= 60184 (a tight upper
  // bound, ~0.3% over the true count) — so the reserve is near-exact, never short.
  {
    const double n = static_cast<double>(limit);
    const size_t est = limit < 60184
                           ? 6200 // pi(60184)=6097; small fixed reserve below that
                           : static_cast<size_t>(n / (std::log(n) - 1.1)) + 16;
    primes.reserve(est);
  }
  primes.push_back(2);

  int64_t sqrt_limit = isqrt(limit);
  std::vector<int64_t> base = simple_odd_primes(sqrt_limit);

  // Segmented odd sieve over (sqrt_limit, limit], plus the small odd primes
  // already found that are <= limit.
  for (int64_t p : base)
    if (p <= limit)
      primes.push_back(p);

  const int64_t SEG = 1 << 18; // odd numbers per segment
  int64_t low = sqrt_limit + 1;
  if (low % 2 == 0)
    ++low; // start on an odd number

  std::vector<bool> seg(SEG);
  for (; low <= limit; low += 2 * SEG) {
    std::fill(seg.begin(), seg.end(), false);
    int64_t high = std::min(low + 2 * SEG - 2, limit); // last odd in segment

    for (int64_t p : base) {
      if (p * p > high)
        break;
      // First odd multiple of p that is >= low.
      int64_t start = (low + p - 1) / p * p;
      if (start < p * p)
        start = p * p;
      if (start % 2 == 0)
        start += p; // keep it odd
      for (int64_t m = start; m <= high; m += 2 * p)
        seg[(m - low) / 2] = true;
    }

    for (int64_t i = 0; i < SEG; ++i) {
      int64_t value = low + 2 * i;
      if (value > high)
        break;
      if (!seg[i])
        primes.push_back(value);
    }
  }

  return primes;
}

int64_t count_primes(int64_t x) {
  if (x < 2)
    return 0;

  int64_t count = 1; // the prime 2
  int64_t sqrt_x = isqrt(x);
  std::vector<int64_t> base = simple_odd_primes(sqrt_x);

  // Count small odd primes <= x that are also part of the base set.
  for (int64_t p : base)
    if (p <= x)
      ++count;

  const int64_t SEG = 1 << 18;
  int64_t low = sqrt_x + 1;
  if (low % 2 == 0)
    ++low;

  std::vector<bool> seg(SEG);
  for (; low <= x; low += 2 * SEG) {
    std::fill(seg.begin(), seg.end(), false);
    int64_t high = std::min(low + 2 * SEG - 2, x);

    for (int64_t p : base) {
      if (p * p > high)
        break;
      int64_t start = (low + p - 1) / p * p;
      if (start < p * p)
        start = p * p;
      if (start % 2 == 0)
        start += p;
      for (int64_t m = start; m <= high; m += 2 * p)
        seg[(m - low) / 2] = true;
    }

    for (int64_t i = 0; i < SEG; ++i) {
      int64_t value = low + 2 * i;
      if (value > high)
        break;
      if (!seg[i])
        ++count;
    }
  }

  return count;
}

} // namespace primecount
