// pitable.hpp — O(1) prime-counting lookup π(n) for n up to a fixed limit.
//
// Several Gourdon terms need many small-π evaluations. PiTable precomputes the
// prime bitmap once and answers each π(n) query in constant time using a
// per-word prefix count plus a popcount of the partial word.
#ifndef PRIMECOUNT_PITABLE_HPP
#define PRIMECOUNT_PITABLE_HPP

#include <bit>
#include <cstdint>
#include <vector>

namespace primecount {

// Precomputes π(n) for all 0 <= n <= limit and answers each query in O(1).
// ODD-ONLY: only odd integers get a bit (the prime 2 is added at query time),
// so memory is ≈ limit/8 bytes — half the naive table, which fits the last-level
// cache better and cuts the random-lookup misses in the easy-leaf terms.
class PiTable {
public:
  // Self-sieving build. nt = thread count for the parallel segmented sieve
  // (0 => OpenMP default / all cores). The segments partition the bit array on
  // WHOLE-WORD boundaries, so the parallel writes to odd_bits_ never share a word.
  explicit PiTable(int64_t limit, int nt = 0);

  // Build from an already-generated, ascending list of primes (those <= limit
  // are used). Avoids re-sieving when the caller already holds the prime list.
  PiTable(int64_t limit, const std::vector<int64_t>& primes);

  // Number of primes p with p <= n, for 0 <= n <= limit() (clamped otherwise).
  int64_t operator()(int64_t n) const;

  // Unchecked variant for hot callers that GUARANTEE 2 <= n <= limit_ (the
  // easy-leaf terms A/C, whose pi arguments are bounded by construction). Drops
  // the n<2 / n>limit guards (a branch + cmov) that the profile showed in the
  // per-leaf lookup. Out-of-range n is undefined.
  int64_t at(int64_t n) const {
    const int64_t i = (n - 1) >> 1; // largest odd index with 2i+1 <= n
    const int64_t w = i >> 6;
    const unsigned bit = static_cast<unsigned>(i & 63);
    const uint64_t mask =
        (bit == 63) ? ~uint64_t(0) : ((uint64_t(1) << (bit + 1)) - 1);
    return 1 + static_cast<int64_t>(prefix_[w]) +
           std::popcount(odd_bits_[w] & mask);
  }

  int64_t limit() const { return limit_; }

  // Prime enumeration straight from the bitmap, so callers that walk the primes
  // <= limit (e.g. the B term over (y, sqrt x]) need not keep a separate O(sqrt x)
  // int64 prime LIST — the bit-packed PiTable (~ limit/16 bytes) serves both pi()
  // and iteration. prime_le: largest prime <= n (0 if none). prime_gt: smallest
  // prime > n (0 if none <= limit). Both clamp n to limit_.
  int64_t prime_le(int64_t n) const;
  int64_t prime_gt(int64_t n) const;

private:
  int64_t limit_;
  std::vector<uint64_t> odd_bits_; // bit i set => the odd number 2i+1 is prime
  // prefix_[w] = #odd primes 2i+1 with i < 64*w. uint32 (not int64): the count is
  // pi(sqrt x) < 2^31 up to x ~ 1e21 — halves this (the larger non-bitmap) half of
  // the PiTable. (Bump to uint64 above ~1e21 if ever needed.)
  std::vector<uint32_t> prefix_;
};

} // namespace primecount

#endif // PRIMECOUNT_PITABLE_HPP
