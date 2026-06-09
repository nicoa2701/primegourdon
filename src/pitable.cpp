#include "pitable.hpp"
#include "sieve.hpp"
#include "util.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <vector>

#include <omp.h>

namespace primecount {

// Self-sieving build: mark the odd-prime bits by a SEGMENTED ODD SIEVE using base
// sieving primes <= sqrt(limit) (only ~x^(1/4) of them) — instead of marking from
// a materialised O(sqrt x) prime LIST. The PiTable (sqrt(x)/16 bytes, bit-packed)
// then encodes all primes <= sqrt x in ~6.5x less memory than the int64 list, and
// can itself serve prime enumeration (walk the set bits) — removing the list as
// the O(sqrt x) RAM peak.
PiTable::PiTable(int64_t limit, int nt) : limit_(limit < 0 ? 0 : limit) {
  const int64_t words = (limit_ / 2) / 64 + 1;
  odd_bits_.assign(words, 0);

  if (limit_ >= 3) {
    const int64_t sqrtlim = isqrt(limit_);
    const std::vector<int64_t> base = generate_primes(sqrtlim); // <= x^(1/4), small
    // PARALLEL segmented odd sieve. Segments partition the bit array on WHOLE-WORD
    // boundaries (SEGW words each => 64*SEGW consecutive odd indices): segment s
    // owns odd_bits_ words [s*SEGW, (s+1)*SEGW), DISJOINT across s, so the survivor
    // writes (|=) race-free with no atomics. (Partitioning by value, as the old
    // serial loop did, made adjacent segments share the boundary word — unsafe in
    // parallel.) Per-thread `seg` byte buffer; `base` is read-only and shared.
    const int64_t SEGW = 1 << 12;            // words per segment (64*SEGW odds)
    const int64_t SEG = 64 * SEGW;           // odd numbers per segment
    const int64_t nseg = (words + SEGW - 1) / SEGW;
    const int threads = (nt > 0) ? nt : omp_get_max_threads();
#pragma omp parallel num_threads(threads)
    {
      std::vector<uint8_t> seg(SEG); // byte-per-odd: faster marking than vector<bool>
#pragma omp for schedule(dynamic)
      for (int64_t s = 0; s < nseg; ++s) {
        const int64_t wlo = s * SEGW;
        const int64_t whi = std::min<int64_t>(wlo + SEGW, words);
        const int64_t ilo = wlo * 64;          // first odd index owned by this segment
        const int64_t ihi = whi * 64;          // one past the last
        const int64_t low = 2 * ilo + 1;       // first odd VALUE
        int64_t high = 2 * (ihi - 1) + 1;      // last odd value in the word range
        if (high > limit_)
          high = limit_;
        const int64_t n = (high - low) / 2 + 1; // odds in [low, high]
        std::fill(seg.begin(), seg.begin() + n, uint8_t(0));
        for (int64_t p : base) {
          if (p == 2)
            continue;
          if (p * p > high)
            break;
          int64_t start = (low + p - 1) / p * p; // first multiple of p >= low
          if (start < p * p)
            start = p * p;
          if ((start & 1) == 0)
            start += p; // keep odd
          for (int64_t m = start; m <= high; m += 2 * p)
            seg[(m - low) / 2] = 1;
        }
        for (int64_t k = 0; k < n; ++k) {
          const int64_t value = low + 2 * k;
          if (value < 3)
            continue; // 1 is not prime (only reachable in segment 0)
          if (!seg[k]) { // survivor => odd prime
            const int64_t i = (value - 1) / 2;
            odd_bits_[i >> 6] |= (uint64_t(1) << (i & 63));
          }
        }
      }
    }
  }

  prefix_.assign(words, 0);
  int64_t count = 0;
  for (int64_t w = 0; w < words; ++w) {
    prefix_[w] = static_cast<uint32_t>(count);
    count += std::popcount(odd_bits_[w]);
  }
}

PiTable::PiTable(int64_t limit, const std::vector<int64_t>& primes)
    : limit_(limit < 0 ? 0 : limit) {
  // Odd index i represents the odd number 2i+1, for 0 <= i <= limit_/2.
  int64_t words = (limit_ / 2) / 64 + 1;
  odd_bits_.assign(words, 0);

  // Mark each odd prime (the prime 2 is handled at query time).
  for (int64_t p : primes) {
    if (p > limit_)
      break;
    if (p == 2)
      continue;
    int64_t i = (p - 1) / 2; // p == 2i+1
    odd_bits_[i >> 6] |= (uint64_t(1) << (i & 63));
  }

  // prefix_[w] = number of odd primes among 2i+1 with i < 64*w.
  prefix_.assign(words, 0);
  int64_t count = 0;
  for (int64_t w = 0; w < words; ++w) {
    prefix_[w] = static_cast<uint32_t>(count);
    count += std::popcount(odd_bits_[w]);
  }
}

int64_t PiTable::prime_le(int64_t n) const {
  if (n > limit_)
    n = limit_;
  if (n < 2)
    return 0;
  if (n >= 3) {
    // largest odd index i with 2i+1 <= n
    int64_t i = (n % 2 == 0) ? (n - 2) / 2 : (n - 1) / 2;
    int64_t w = i >> 6;
    const unsigned bit = static_cast<unsigned>(i & 63);
    const uint64_t mask =
        (bit == 63) ? ~uint64_t(0) : ((uint64_t(1) << (bit + 1)) - 1);
    uint64_t word = odd_bits_[w] & mask;
    while (true) {
      if (word) {
        const int hb = 63 - std::countl_zero(word); // highest set bit in word
        return 2 * (w * 64 + hb) + 1;
      }
      if (w == 0)
        break;
      word = odd_bits_[--w];
    }
  }
  return 2; // only the prime 2 is <= n
}

int64_t PiTable::prime_gt(int64_t n) const {
  if (n < 2)
    return 2;
  const int64_t maxi = limit_ / 2;
  int64_t i = (n + 1) / 2; // smallest odd index i with 2i+1 > n
  if (i > maxi)
    return 0;
  int64_t w = i >> 6;
  const unsigned bit = static_cast<unsigned>(i & 63);
  uint64_t word = odd_bits_[w] & (~uint64_t(0) << bit);
  const int64_t words = static_cast<int64_t>(odd_bits_.size());
  while (true) {
    if (word) {
      const int lb = std::countr_zero(word); // lowest set bit
      const int64_t j = w * 64 + lb;
      return (j > maxi) ? 0 : 2 * j + 1;
    }
    if (++w >= words)
      return 0;
    word = odd_bits_[w];
  }
}

int64_t PiTable::operator()(int64_t n) const {
  if (n < 2)
    return 0;
  if (n > limit_)
    n = limit_; // defensive; callers should respect limit()

  int64_t i = (n - 1) / 2; // largest odd index with 2i+1 <= n
  int64_t w = i >> 6;
  unsigned bit = unsigned(i & 63);
  uint64_t mask = (bit == 63) ? ~uint64_t(0) : ((uint64_t(1) << (bit + 1)) - 1);
  // 1 for the prime 2, plus the odd primes <= n.
  return 1 + static_cast<int64_t>(prefix_[w]) +
         std::popcount(odd_bits_[w] & mask);
}

} // namespace primecount
