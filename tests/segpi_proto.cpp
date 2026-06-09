// segpi_proto.cpp — ISOLATED prototype/benchmark for the SegmentedPiTable
// Standalone experiment. NOT wired into the library.
//
// Question it answers: is sweeping a sliding pi-window over [3, sqrt x] as cheap
// as ONE full sieve (so fill cost ≈ 1 sieve, not N), and how small is the RAM
// (window vs full)? Also checks segPi[v] == full PiTable pi(v).
//
// Build: g++ -O3 -std=gnu++20 -mpopcnt -I include tests/segpi_proto.cpp \
//        src/pitable.cpp src/sieve.cpp src/util.cpp -o /tmp/segpi_proto
#include "pitable.hpp"
#include "sieve.hpp"
#include "util.hpp"

#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace primecount;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Sliding odd-only pi window. base primes (<= sqrt(limit)) computed once; each
// init(low) fills [low, low+window) and tracks pi(low-1) across sequential calls.
class SegmentedPiTable {
public:
  SegmentedPiTable(int64_t limit, int64_t window)
      : limit_(limit), window_(window) {
    base_ = generate_primes(isqrt(limit_)); // <= x^(1/4), small, ONCE
    bits_.assign(window_ / 2 / 64 + 2, 0);
    prefix_.assign(window_ / 2 / 64 + 2, 0);
  }

  // Fill window [low, high). Must be called with strictly increasing, window-
  // aligned `low` starting at the first call; pi_base_ accumulates pi(low-1).
  void init(int64_t low, int64_t pi_base) {
    low_ = low;
    high_ = std::min(low + window_, limit_ + 1);
    pi_base_ = pi_base;
    const int64_t nbits = (high_ - low_ + 1) / 2 + 1;
    const int64_t words = nbits / 64 + 1;
    std::fill(bits_.begin(), bits_.begin() + words + 1, 0);
    // mark odd primes in [low_, high_) : bit i <=> odd number low_ + 2i is prime
    // (we align low_ odd). cross off composites with base odd primes.
    std::vector<uint8_t> seg((high_ - low_) / 2 + 2, 0);
    for (int64_t p : base_) {
      if (p == 2) continue;
      if (p * p >= high_) break;
      int64_t start = (low_ + p - 1) / p * p;
      if (start < p * p) start = p * p;
      if ((start & 1) == 0) start += p; // odd multiple
      for (int64_t m = start; m < high_; m += 2 * p)
        seg[(m - low_) / 2] = 1;
    }
    int64_t cnt = 0;
    for (int64_t k = 0; (low_ + 2 * k) < high_; ++k) {
      prefix_[k >> 6] = (k & 63) == 0 ? cnt : prefix_[k >> 6];
      if (!seg[k]) { bits_[k >> 6] |= (uint64_t(1) << (k & 63)); cnt++; }
    }
    // recompute prefix per word (clean)
    cnt = 0;
    for (int64_t w = 0; w < words + 1; ++w) {
      prefix_[w] = cnt;
      cnt += std::popcount(bits_[w]);
    }
    window_primes_ = cnt;
  }
  int64_t window_primes() const { return window_primes_; }
  int64_t low() const { return low_; }
  int64_t high() const { return high_; }

  // pi(v) for low_ <= v < high_. low_ is odd; bit k <=> odd number low_+2k.
  int64_t operator[](int64_t v) const {
    if (v < 2) return 0;
    int64_t k = (v - low_) / 2;          // largest bit index with low_+2k <= v
    int64_t w = k >> 6;
    unsigned bit = (unsigned)(k & 63);
    uint64_t mask = (bit == 63) ? ~uint64_t(0) : ((uint64_t(1) << (bit + 1)) - 1);
    return 1 /*prime 2*/ + pi_base_ + prefix_[w] +
           std::popcount(bits_[w] & mask);
  }

private:
  int64_t limit_, window_, low_ = 1, high_ = 1, pi_base_ = 0, window_primes_ = 0;
  std::vector<int64_t> base_;
  std::vector<uint64_t> bits_;
  std::vector<int64_t> prefix_;
};

int main(int argc, char** argv) {
  // sqrt(x) target; default ~ sqrt(1e18)=1e9
  int64_t sqrtx = (argc > 1) ? (int64_t)atof(argv[1]) : 1000000000LL;
  int64_t window = (argc > 2) ? (int64_t)atof(argv[2]) : (1 << 21); // ints/window

  std::printf("sqrtx=%lld  window=%lld ints (%lld KB odd-bitmap)\n",
              (long long)sqrtx, (long long)window, (long long)(window / 2 / 8 / 1024));

  // --- baseline: ONE full PiTable build (the current approach) ---
  auto t0 = clk::now();
  PiTable full(sqrtx);
  auto t1 = clk::now();
  std::printf("full PiTable(sqrtx) build: %.1f ms  (RAM ~%lld MB)\n",
              ms(t0, t1), (long long)(sqrtx / 2 / 8 / (1 << 20)));

  // --- sweep the sliding window over [3, sqrtx] ---
  auto t2 = clk::now();
  SegmentedPiTable seg(sqrtx, window);
  int64_t pi_base = 0;          // pi(low-1) accumulator across windows
  int64_t checks = 0, fails = 0;
  // align first low to odd
  for (int64_t low = 3; low <= sqrtx; low += window) {
    if ((low & 1) == 0) low++; // keep odd (window even so stays odd)
    seg.init(low, pi_base);
    // validate a few queries against the full table
    for (int64_t v = low; v < seg.high() && v <= sqrtx; v += (window / 7 + 1)) {
      checks++;
      if (seg[v] != full(v)) {
        if (fails < 5)
          std::printf("  MISMATCH v=%lld seg=%lld full=%lld\n",
                      (long long)v, (long long)seg[v], (long long)full(v));
        fails++;
      }
    }
    pi_base += seg.window_primes();
  }
  auto t3 = clk::now();
  std::printf("SWEEP fill+query [3,sqrtx] in windows: %.1f ms  (RAM ~%lld KB window)\n",
              ms(t2, t3), (long long)(window / 2 / 8 / 1024));
  std::printf("checks=%lld  fails=%lld  => %s\n", (long long)checks,
              (long long)fails, fails == 0 ? "CORRECT" : "WRONG");
  std::printf("ratio sweep/full = %.2fx\n", ms(t2, t3) / ms(t0, t1));
  return fails == 0 ? 0 : 1;
}
