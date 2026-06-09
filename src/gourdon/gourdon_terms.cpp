#include "gterms.hpp"
#include "gourdon.hpp"
#include "gparams.hpp"
#include "phi.hpp"
#include "phi_pi.hpp"
#include "pitable.hpp"
#include "sieve.hpp"
#include "util.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include <immintrin.h>
#include <omp.h>
#include <unistd.h>

namespace primecount {

namespace {

// Reduction over the 128-bit accumulator (built-in reductions don't cover it).
#pragma omp declare reduction(i128add:maxint_t : omp_out += omp_in)            \
    initializer(omp_priv = 0)

int resolve_threads(int threads) {
  return threads > 0 ? threads : omp_get_max_threads();
}

// First index i with primes[i] > v (primes ascending).
size_t upper_index(const std::vector<int64_t>& primes, int64_t v) {
  return static_cast<size_t>(
      std::upper_bound(primes.begin(), primes.end(), v) - primes.begin());
}

// Fast non-negative 64/64 division emitting `divl` (64-bit / 32-bit -> 32-bit)
// when BOTH the divisor and the quotient fit in 32 bits (the runtime test
// high32(x) < d guarantees no #DE overflow). On pre-2020 x86 with a slow,
// non-pipelined 64-bit `divq`, that division dominates the easy-leaf terms A/C
// (IPC ~0.78); `divl` is much faster there: measured A/C -14.5%, full pi(x)
// -21.8% at 1e16 on such a host.
// The A/C leaves have quotient ~sqrt(x)
// (< 2^32 up to ~1e19) and a prime/product divisor, so divl covers most of them.
//
// GATED behind ENABLE_DIV32 (CMake -DWITH_DIV32=ON), OFF by default. Unlike the
// reference -- whose template selects divl at COMPILE time from a 32-bit divisor
// TYPE, paying nothing extra -- our divisor is a 64-bit runtime value, so the
// fast path costs a `d <= 2^32 && high < d` branch per call. On a fast divider
// (divq IPC ~2.3) divl buys nothing and that branch is pure overhead: measured
// A/C +2.7% (1e16) / +3.2% (1e17) REGRESSION on a recent big-core x86. So
// slow-divider hosts build with -DWITH_DIV32=ON, fast-divider hosts keep the
// default (plain x/d, byte-identical to pristine) -- the configure-time probe
// picks the right one automatically.

#if defined(ENABLE_DIV32)
#pragma message("A/C division backend: 32-bit divl (ENABLE_DIV32 / WITH_DIV32=ON)")
#else
#pragma message("A/C division backend: plain 64-bit divq (WITH_DIV32=OFF)")
#endif
inline int64_t fast_div_u(uint64_t x, uint64_t d) {
#if defined(ENABLE_DIV32) && defined(__x86_64__) &&                            \
    (defined(__GNUC__) || defined(__clang__))
  uint32_t high = static_cast<uint32_t>(x >> 32);
  if (d <= 0xffffffffu && high < static_cast<uint32_t>(d)) {
    uint32_t low = static_cast<uint32_t>(x);
    uint32_t dd = static_cast<uint32_t>(d);
    __asm__("divl %[dd]" : "+a"(low), "+d"(high) : [dd] "r"(dd));
    return static_cast<int64_t>(low);
  }
#endif
  return static_cast<int64_t>(x / d);
}

// Fast x/d for the hot leaf/bound computations. When x <= INT64_MAX (every x
// below ~9.2e18) and the divisor fits in 64 bits, fast_div_u replaces the
// ~35-cycle libgcc __divti3 (128-bit) with a 64-bit division, itself narrowed to
// `divl` when possible (see above). The quotient x/d <= x always fits in int64;
// for x > 2^63 it falls back to the 128-bit division so 1e19 stays correct.
inline int64_t divx(int128_t x, int128_t d) {
  if (x <= INT64_MAX && d <= INT64_MAX)
    return fast_div_u(static_cast<uint64_t>(static_cast<int64_t>(x)),
                      static_cast<uint64_t>(static_cast<int64_t>(d)));
  return static_cast<int64_t>(x / d);
}

// One pre-sieve group: a period of its "coprime to this group's primes" bit
// pattern (period bits + >= kPrePad words of wrap padding so SIMD blocks can read
// past the period; the padding is a true continuation since the pattern is
// periodic). phase0 = the segment's start phase low%period is set per segment.
constexpr int64_t kPrePad = 16; // padding words (covers the AVX/AVX-512 overshoot)
struct PreTable {
  std::vector<uint64_t> bits;
  int64_t period;
};

// FUSED pre-sieve fill: out[w] = AND over all groups of their pattern at integer
// low+64w, in ONE pass over out[] (each word written once). Per word, phase
// advances 64 bits so each group's off = phase&63 is CONSTANT until that group's
// period wraps; we cut the segment into runs bounded by the nearest wrap, and
// inside a run every group is a fixed funnel-shift — AND them all in registers
// and store the accumulator once. (srl/sll by an xmm count give 0 at count 64,
// so off==0 needs no special case.) The run/wrap control + scalar tail are shared
// by the AVX2 and AVX-512 bodies; the SIMD inner block differs only in width.
#define PRESIEVE_FILL_BODY(SIMD_BLOCK)                                          \
  int64_t phase[32];                                                           \
  for (int g = 0; g < ng; ++g)                                                  \
    phase[g] = low % pre[g].period;                                            \
  int64_t w = 0;                                                               \
  while (w < nw) {                                                             \
    int64_t runw = nw - w;                                                     \
    for (int g = 0; g < ng; ++g) {                                            \
      const int64_t until = (pre[g].period - phase[g] + 63) / 64;             \
      if (until < runw)                                                        \
        runw = until;                                                          \
    }                                                                          \
    if (runw < 1)                                                              \
      runw = 1;                                                                \
    const uint64_t* base[32];                                                  \
    __m128i sr[32], sl[32];                                                    \
    int off[32];                                                               \
    for (int g = 0; g < ng; ++g) {                                            \
      off[g] = static_cast<int>(phase[g] & 63);                               \
      base[g] = pre[g].bits.data() + (phase[g] >> 6);                          \
      sr[g] = _mm_cvtsi64_si128(off[g]);                                       \
      sl[g] = _mm_cvtsi64_si128(64 - off[g]);                                  \
    }                                                                          \
    int64_t j = 0;                                                            \
    SIMD_BLOCK                                                                 \
    for (; j < runw; ++j) {                                                    \
      uint64_t acc = ~uint64_t(0);                                            \
      for (int g = 0; g < ng; ++g)                                            \
        acc &= off[g]                                                          \
                   ? ((base[g][j] >> off[g]) | (base[g][j + 1] << (64 - off[g])))\
                   : base[g][j];                                              \
      out[w + j] = acc;                                                       \
    }                                                                          \
    w += runw;                                                                \
    for (int g = 0; g < ng; ++g) {                                            \
      phase[g] += 64 * runw;                                                   \
      while (phase[g] >= pre[g].period)                                        \
        phase[g] -= pre[g].period;                                            \
    }                                                                          \
  }

void presieve_fill_avx2(uint64_t* out, int64_t nw, const PreTable* pre, int ng,
                        int64_t low) {
  PRESIEVE_FILL_BODY(for (; j + 4 <= runw; j += 4) {
    __m256i acc = _mm256_set1_epi64x(-1);
    for (int g = 0; g < ng; ++g) {
      const __m256i lo =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base[g] + j));
      const __m256i hi =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(base[g] + j + 1));
      acc = _mm256_and_si256(
          acc, _mm256_or_si256(_mm256_srl_epi64(lo, sr[g]),
                               _mm256_sll_epi64(hi, sl[g])));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + w + j), acc);
  })
}

// 8-wide variant for AVX-512F (CPUs that support it); dispatched at runtime.
__attribute__((target("avx512f"))) void
presieve_fill_avx512(uint64_t* out, int64_t nw, const PreTable* pre, int ng,
                     int64_t low) {
  PRESIEVE_FILL_BODY(for (; j + 8 <= runw; j += 8) {
    __m512i acc = _mm512_set1_epi64(-1);
    for (int g = 0; g < ng; ++g) {
      const __m512i lo =
          _mm512_loadu_si512(reinterpret_cast<const void*>(base[g] + j));
      const __m512i hi =
          _mm512_loadu_si512(reinterpret_cast<const void*>(base[g] + j + 1));
      acc = _mm512_and_si512(
          acc, _mm512_or_si512(_mm512_srl_epi64(lo, sr[g]),
                               _mm512_sll_epi64(hi, sl[g])));
    }
    _mm512_storeu_si512(reinterpret_cast<void*>(out + w + j), acc);
  })
}
#undef PRESIEVE_FILL_BODY

inline void presieve_fill(uint64_t* out, int64_t nw, const PreTable* pre, int ng,
                          int64_t low) {
  static const bool has512 = __builtin_cpu_supports("avx512f");
  if (has512)
    presieve_fill_avx512(out, nw, pre, ng, low);
  else
    presieve_fill_avx2(out, nw, pre, ng, low);
}

// Resources shared by every Gourdon term, built once per pi(x) computation:
//   pi       — PiTable up to x/y (largest pi() argument, from B; also covers
//              sqrt x for A/Sigma and x*^2 for the easy leaves).
//   primes   — all primes <= sqrt x.
//   mp/pmax  — mu(m)*pmin(m) and largest prime factor of m, for m <= z.
//              int32 (values are <= z < 2^31 for x <= ~1e19) halves their RAM.
// Packed 24-bit signed store for mu(m)*pmin(m): 3 bytes/entry (vs int32's 4 = -25%
// RAM). Safe because pmin is only ever compared to p_b <= x*, so it is capped at
// x*+1 at build time (|value| < 2^23 for x < ~5e27). Used by the low-mem context;
// the default keeps the plain int32 `mp` (no decode cost on the hot D path).
inline void mp3_set(std::vector<uint8_t>& d, int64_t m, int32_t v) {
  const uint32_t u = static_cast<uint32_t>(v) & 0xFFFFFFu;
  d[3 * m] = static_cast<uint8_t>(u);
  d[3 * m + 1] = static_cast<uint8_t>(u >> 8);
  d[3 * m + 2] = static_cast<uint8_t>(u >> 16);
}
inline int32_t mp3_get(const std::vector<uint8_t>& d, int64_t m) {
  const uint32_t u = static_cast<uint32_t>(d[3 * m]) |
                     (static_cast<uint32_t>(d[3 * m + 1]) << 8) |
                     (static_cast<uint32_t>(d[3 * m + 2]) << 16);
  return (u & 0x800000u) ? static_cast<int32_t>(u | 0xFF000000u)
                         : static_cast<int32_t>(u);
}

// --- Wheel-2310 coprime indexing for the D factor table (PC_DFACTOR) ---
// Christian Bau's compressed-factor-table idea (as in primecount's FactorTableD):
// store one entry per integer coprime to 2,3,5,7,11. to_index/to_number map between
// a number and its dense coprime index (480 coprimes per 2310-period). Lets D's leaf
// scan walk CONTIGUOUS indices (no wheel-stepping arithmetic) and read a contiguous
// uint16 array, instead of wheel-stepping over mp3 with a 3-byte decode.
struct Wheel2310 {
  uint16_t coprime[480];   // coprime[i] = i-th integer coprime to 2310 in [1,2310]
  int16_t cindex[2310];    // cindex[r] = index of the largest coprime <= r (or -1)
  Wheel2310() {
    int i = -1;
    for (int r = 0; r < 2310; ++r) {
      if (r % 2 && r % 3 && r % 5 && r % 7 && r % 11)
        coprime[++i] = static_cast<uint16_t>(r);
      cindex[r] = static_cast<int16_t>(i);
    }
  }
};
inline const Wheel2310& wheel2310() {
  static const Wheel2310 t;
  return t;
}
inline int64_t dfac_to_index(int64_t n) {
  const Wheel2310& t = wheel2310();
  return 480 * (n / 2310) + t.cindex[n % 2310];
}
inline int64_t dfac_to_number(int64_t idx) {
  const Wheel2310& t = wheel2310();
  return 2310 * (idx / 480) + t.coprime[idx % 480];
}

// Sum of set bits over n bytes at p, uint64-chunked (8 bytes per popcnt). D's phi
// counter (count_le / prefix_bytes) popcounts ~24 bytes per survivor, so this is ~8x
// fewer popcnt ops + 8x fewer loads than byte-by-byte. Inlined + portable (POPCNT64,
// so the AVX2-only i5 benefits too). AVX512 vpopcntdq is NOT used here: the per-call
// run (~24 B) is below its 64-B granularity and a target-attr function would not
// inline (a call per survivor would cost more than it saves).
inline int64_t popcount_bytes(const uint8_t* p, int64_t n) {
  int64_t c = 0, i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w;
    std::memcpy(&w, p + i, 8);
    c += std::popcount(w);
  }
  for (; i < n; ++i)
    c += std::popcount(p[i]);
  return c;
}

struct GCtx {
  GParams p;
  PiTable pi;
  std::vector<int64_t> primes;
  std::vector<int32_t> mp;   // mu(m)*pmin(m), int32 (default path)
  std::vector<int32_t> pmax; // largest prime factor (only when z != y)
  std::vector<uint8_t> mp3;  // packed 24-bit variant of mp (low-mem path; mp empty)

  // mu(m)*min(pmin(m), x*+1), 0 if m not squarefree. Reads whichever store is
  // populated; the branch is loop-invariant so it predicts away on the hot path.
  int32_t mpv(int64_t m) const {
    return mp.empty() ? mp3_get(mp3, m) : mp[m];
  }
};

GCtx build_ctx(int128_t x, int nt = 0) {
  const bool tb = std::getenv("PC_TIMEBUILD") != nullptr;
  auto tnow = [] { return std::chrono::steady_clock::now(); };
  auto tms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto t0 = tnow();

  GParams p = make_gparams(x);
  const int64_t z = p.z;

  std::vector<int32_t> mp(z + 1, 1);
  for (int64_t j = 2; j <= z; ++j)
    if (mp[j] == 1)
      for (int64_t i = j; i <= z; i += j)
        mp[i] = (mp[i] == 1) ? static_cast<int32_t>(-j) : -mp[i];
  for (int64_t j = 2; j * j <= z; ++j)
    if (mp[j] == -j)
      for (int64_t i = j * j; i <= z; i += j * j)
        mp[i] = 0;
  auto t_mp = tnow();

  // pmax[m] (largest prime factor of m) is consulted by D1 ONLY to test
  // P^+(m) <= y, and that test is skipped entirely when z == y (checkpmax), the
  // default tuning (alpha_z = 1): for m <= z = y every prime factor is <= m <= y
  // automatically. So when z == y, don't build pmax at all — it is never read.
  // This drops ~one third of peak RSS at 1e19 (266 MB) plus its O(z log log z)
  // sieve pass. When z != y (PC_AZ override), build it as before.
  std::vector<int32_t> pmax;
  if (z != p.y) {
    pmax.assign(z + 1, 1);
    for (int64_t pp = 2; pp <= z; ++pp)
      if (mp[pp] == -pp)
        for (int64_t i = pp; i <= z; i += pp)
          pmax[i] = static_cast<int32_t>(pp);
  }

  // The shared PiTable only needs to reach the largest pi() argument among A,
  // C and Sigma — that is max(sqrt x, x*^2). B no longer uses it (it sieves),
  // which keeps this table near sqrt x instead of x/y.
  const int64_t piLimit =
      std::max(p.sqrtx, static_cast<int64_t>(static_cast<int128_t>(p.xstar) * p.xstar));
  // Generate the primes <= piLimit ONCE (piLimit >= sqrt x covers A/B), and
  // build the PiTable from that list instead of re-sieving inside its ctor.
  // The PiTable still spans piLimit (~sqrt x) for the pi() queries (A/C/B) and now
  // SELF-SIEVES, so it no longer needs the prime list to build. The prime LIST is
  // shrunk to only the small-prime consumers: A's 2nd prime q <= isqrt(x/xstar),
  // C/D/Phi0 primes <= y, B's crossing primes <= x^(1/3) — all << sqrt x. B walks
  // its (y, sqrt x] primes from the PiTable bitmap (prime_le) instead. This drops
  // the O(sqrt x) int64 list (407 MB @1e18 / 3.6 GB @1e20) to O(x^0.4) (~3 MB @1e18).
  const int64_t piSmall = std::max<int64_t>(
      {p.x13, p.y, p.z,
       static_cast<int64_t>(isqrt(static_cast<int128_t>(x) / p.xstar)) + 2});
  auto t_pmax = tnow();
  std::vector<int64_t> primes = generate_primes(piSmall);
  auto t_primes = tnow();
  PiTable pi(piLimit, nt); // self-sieving, parallel segmented (nt=0 => all cores)
  auto t_pi = tnow();
  if (tb)
    std::fprintf(stderr,
                 "[build] z=%lld piLimit=%lld | mp=%.0fms pmax=%.0fms "
                 "primes=%.0fms PiTable=%.0fms | total=%.0fms\n",
                 (long long)z, (long long)piLimit, tms(t0, t_mp),
                 tms(t_mp, t_pmax), tms(t_pmax, t_primes), tms(t_primes, t_pi),
                 tms(t0, t_pi));
  return GCtx{p, std::move(pi), std::move(primes), std::move(mp), std::move(pmax)};
}

// piSmall: the largest prime LIST argument any term needs (<< sqrt x). Shared by
// the full build_ctx and the phased partial builds below.
inline int64_t ctx_pi_small(const GParams& p, int128_t x) {
  return std::max<int64_t>(
      {p.x13, p.y, p.z,
       static_cast<int64_t>(isqrt(static_cast<int128_t>(x) / p.xstar)) + 2});
}

// Phase-1 context for the PHASED pi (opt-in PC_PHASE): the full PiTable + prime
// list, but NO mp[]/pmax[]. Among the six terms, mp[] is consulted ONLY by D, so
// A, C, Sigma, B and Phi0 all run here against the table alone. Freeing this ctx
// before phase 2 means PiTable (~0.094*sqrt x) and mp[] (4z) never coexist.
GCtx build_ctx_pi_only(int128_t x, int nt) {
  GParams p = make_gparams(x);
  const int64_t piLimit = std::max(
      p.sqrtx, static_cast<int64_t>(static_cast<int128_t>(p.xstar) * p.xstar));
  std::vector<int64_t> primes = generate_primes(ctx_pi_small(p, x));
  PiTable pi(piLimit, nt);
  return GCtx{p, std::move(pi), std::move(primes), {}, {}};
}

// Phase-2 context for the PHASED pi: mp[] (+pmax) + prime list + a SMALL PiTable
// of limit d_pi_limit. D is the only phase-2 term; it needs mp[] for its squarefree
// leaves and a PiTable only to ACCELERATE phi_pi's Legendre cutoff (phi_pi stays
// correct with any limit — it just recurses when the argument exceeds it, see
// phi_pi.cpp). A smaller table keeps phase-2 peak at ~4z instead of 4z + the full
// PiTable, at the cost of more phi_pi recursion (measured via PC_DPILIM).
GCtx build_ctx_d_only(int128_t x, int nt, int64_t d_pi_limit) {
  GParams p = make_gparams(x);
  const int64_t z = p.z;
  std::vector<int32_t> mp(z + 1, 1);
  for (int64_t j = 2; j <= z; ++j)
    if (mp[j] == 1)
      for (int64_t i = j; i <= z; i += j)
        mp[i] = (mp[i] == 1) ? static_cast<int32_t>(-j) : -mp[i];
  for (int64_t j = 2; j * j <= z; ++j)
    if (mp[j] == -j)
      for (int64_t i = j * j; i <= z; i += j * j)
        mp[i] = 0;
  std::vector<int32_t> pmax;
  if (z != p.y) {
    pmax.assign(z + 1, 1);
    for (int64_t pp = 2; pp <= z; ++pp)
      if (mp[pp] == -pp)
        for (int64_t i = pp; i <= z; i += pp)
          pmax[i] = static_cast<int32_t>(pp);
  }
  std::vector<int64_t> primes = generate_primes(ctx_pi_small(p, x));
  const int64_t piLimit = std::max(
      p.sqrtx, static_cast<int64_t>(static_cast<int128_t>(p.xstar) * p.xstar));
  PiTable pi(std::min<int64_t>(std::max<int64_t>(d_pi_limit, 2), piLimit), nt);
  return GCtx{p, std::move(pi), std::move(primes), std::move(mp), std::move(pmax)};
}

// Largest pi() argument Sigma needs OTHER than pi(sqrt x) (gourdon.md §4). Every
// swept arg (x/(q*y), x/q^2, isqrt(x/q) for xstar < q <= x^(1/3)) and singleton
// (pi(y), pi(x13), pi(isqrt(x/y)), pi(x*)) is <= this bound, which is O(x^(3/8)) —
// far below sqrt x. pi(sqrt x) itself is supplied as a scalar via pi_count_seg.
inline int64_t sigma_pi_limit(const GParams& p, int128_t x) {
  const int64_t sxy = isqrt(x / p.y);
  return std::max<int64_t>(
      {p.x13, p.y, sxy, p.xstar,
       divx(x, static_cast<int128_t>(p.xstar) * p.y), // max x/(q*y), q > x*
       isqrt(x / p.xstar)});                          // max isqrt(x/q), q > x*
}

// LOW-MEMORY unified context (no O(sqrt x) PiTable): mp[] (O(x^(1/3)), needed by C
// and D) + small prime list + a small PiTable covering Sigma's args and D's phi_pi
// acceleration (limit = max(sigma_pi_limit, d_pi_limit), all O(x^(3/8))). A, B, C
// and Sigma never touch a sqrt-x structure here — they sieve windows on demand. So
// peak RSS ~ mp[] (O(x^(1/3))) instead of O(sqrt x). Single context, no phasing.
// C-sparse split threshold T: leaves with v <= T use the O(1) PiTable (bulk),
// v > T use the value-segmented tail. T defaults to z; PC_CSPLIT=mult raises it to
// mult*x13 (capped at x*^2, beyond which no C leaf exists). The tail dense-scan
// cost is ~ (x/T)*Σ(1/pb), so larger T shrinks the tail ~1/T, at the price of a
// PiTable up to T (~T/10 bytes). Keep T = O(x^(1/3)) to preserve the x^(1/3) RAM
// slope (T grows as x^(1/3) via x13). Shared by build_ctx_lowmem (table size) and
// c_impl_sparse (split point) so they always agree.
inline int64_t c_split_T(const GParams& p) {
  int64_t mult = 64; // default: T = 64*x13 (still O(x^(1/3))), tail ~4x smaller
  if (const char* e = std::getenv("PC_CSPLIT"))
    mult = std::atoll(e); // 0 disables the raise (T = z)
  int64_t T = p.z;
  if (mult > 0)
    T = std::max<int64_t>(T, mult * p.x13);
  const int64_t vmax =
      static_cast<int64_t>(static_cast<int128_t>(p.xstar) * p.xstar);
  return std::min<int64_t>(T, vmax);
}

GCtx build_ctx_lowmem(int128_t x, int nt, int64_t d_pi_limit) {
  GParams p = make_gparams(x);
  const int64_t z = p.z;
  std::vector<int64_t> primes = generate_primes(ctx_pi_small(p, x));
  const int64_t lim = std::max<int64_t>(
      {sigma_pi_limit(p, x), d_pi_limit, c_split_T(p), 2});
  PiTable pi(lim, nt);

  if (z != p.y) {
    // Non-default tuning (alpha_z != 1): pmax[] is needed and requires primes up to
    // z (> x*+1 cap), which the capped store cannot identify. Fall back to int32 mp.
    std::vector<int32_t> mp(z + 1, 1);
    for (int64_t j = 2; j <= z; ++j)
      if (mp[j] == 1)
        for (int64_t i = j; i <= z; i += j)
          mp[i] = (mp[i] == 1) ? static_cast<int32_t>(-j) : -mp[i];
    for (int64_t j = 2; j * j <= z; ++j)
      if (mp[j] == -j)
        for (int64_t i = j * j; i <= z; i += j * j)
          mp[i] = 0;
    std::vector<int32_t> pmax(z + 1, 1);
    for (int64_t pp = 2; pp <= z; ++pp)
      if (mp[pp] == -pp)
        for (int64_t i = pp; i <= z; i += pp)
          pmax[i] = static_cast<int32_t>(pp);
    return GCtx{p, std::move(pi), std::move(primes), std::move(mp),
                std::move(pmax), {}};
  }

  // Default tuning (z == y): build the PACKED 24-bit mp directly (no transient
  // int32 array), pmin capped at x*+1 (safe: every reader compares pmin to p_b <=
  // x*; the squarefree pass uses only primes <= sqrt z < x*, unaffected by the cap).
  const int64_t cap = static_cast<int64_t>(p.xstar) + 1;
  std::vector<uint8_t> mp3(3 * (z + 1));
  for (int64_t i = 0; i <= z; ++i)
    mp3_set(mp3, i, 1);
  for (int64_t j = 2; j <= z; ++j)
    if (mp3_get(mp3, j) == 1)
      for (int64_t i = j; i <= z; i += j) {
        const int32_t cur = mp3_get(mp3, i);
        mp3_set(mp3, i,
                (cur == 1) ? -static_cast<int32_t>(std::min<int64_t>(j, cap))
                           : -cur);
      }
  for (int64_t j = 2; j * j <= z; ++j)
    if (mp3_get(mp3, j) == -j)
      for (int64_t i = j * j; i <= z; i += j * j)
        mp3_set(mp3, i, 0);
  return GCtx{p, std::move(pi), std::move(primes), {}, {}, std::move(mp3)};
}

// --- Phi0 (gourdon.md §3) ---

void phi0_rec(int128_t x, const std::vector<int64_t>& primes, int k, int64_t y,
              int64_t z, size_t start, int64_t n, int sign, maxint_t& acc) {
  for (size_t i = start; i < primes.size(); ++i) {
    const int64_t p = primes[i];
    if (p > y)
      break;
    if (static_cast<int128_t>(n) * p > z)
      break;
    const int64_t nn = n * p;
    const int s = -sign;
    acc += static_cast<maxint_t>(s * static_cast<int128_t>(phi(x / nn, k)));
    phi0_rec(x, primes, k, y, z, i + 1, nn, s, acc);
  }
}

maxint_t phi0_impl(int128_t x, const GCtx& c) {
  maxint_t acc = static_cast<maxint_t>(phi(x, c.p.k)); // n = 1
  phi0_rec(x, c.primes, c.p.k, c.p.y, c.p.z, static_cast<size_t>(c.p.k), 1, +1,
           acc);
  return acc;
}

// Count-only segmented sieve: returns pi(n) WITHOUT materializing an O(n) table.
// `base` must contain every prime <= isqrt(n) (a small list, <= x^(1/4) here).
// O(n) time (parallel over segments), O(window) memory. Used to get pi(sqrt x)
// in the low-memory path where no full sqrt-x PiTable exists (gourdon.md §4: the
// single pi(sqrt x) value Sigma0 needs).
int64_t pi_count_seg(int64_t n, const std::vector<int64_t>& base, int nt) {
  if (n < 2)
    return 0;
  const size_t nsieve = upper_index(base, isqrt(n));
  int64_t Slog = 20;
  if (const char* e = std::getenv("PC_PICOUNTLOG"))
    Slog = std::atoi(e);
  const int64_t S = int64_t(1) << Slog;
  const int64_t V0 = 2;
  const int64_t nseg = (n - V0) / S + 1;
  int64_t total = 0;
#pragma omp parallel num_threads(nt) reduction(+ : total)
  {
    const int64_t NW = (S + 63) / 64;
    std::vector<uint64_t> bits(NW + 1);
#pragma omp for schedule(dynamic)
    for (int64_t seg = 0; seg < nseg; ++seg) {
      const int64_t low = V0 + seg * S;
      const int64_t high = std::min<int64_t>(low + S, n + 1);
      const int64_t len = high - low;
      const int64_t nw = (len + 63) / 64;
      std::fill(bits.begin(), bits.begin() + nw, ~uint64_t(0));
      if (len & 63)
        bits[nw - 1] &= (uint64_t(1) << (len & 63)) - 1;
      for (size_t si = 0; si < nsieve; ++si) {
        const int64_t pp = base[si];
        if (pp * pp >= high)
          break;
        int64_t k = (low + pp - 1) / pp;
        if (k < 2)
          k = 2;
        for (int64_t m = pp * k; m < high; m += pp) {
          const int64_t pos = m - low;
          bits[pos >> 6] &= ~(uint64_t(1) << (pos & 63));
        }
      }
      int64_t run = 0;
      for (int64_t w = 0; w < nw; ++w)
        run += std::popcount(bits[w]);
      total += run;
    }
  }
  return total;
}

// --- Sigma (gourdon.md §4) ---

// pisqrtx: pi(sqrt x). Pass >= 0 to supply it as a scalar (low-mem path, where
// c.pi only reaches ~O(x^(1/3)) and cannot answer pi(sqrt x)); pass -1 to read it
// from the full c.pi (legacy path). Every OTHER pi() arg here is <= sigmaLimit
// (gourdon.md §4), so the small c.pi covers them.
maxint_t sigma_impl(int128_t x, const GCtx& c, int128_t pisqrtx = -1) {
  const GParams& p = c.p;
  const PiTable& pi = c.pi;

  const int128_t a = pi(p.y);
  const int128_t b = pi(p.x13);
  const int64_t sxy = isqrt(x / p.y);
  const int128_t cc = pi(sxy);
  const int128_t dd = pi(p.xstar);
  const int128_t pisx = (pisqrtx >= 0) ? pisqrtx : pi(p.sqrtx);

  int128_t S0 = a - 1 + pisx * (pisx - 1) / 2 - a * (a - 1) / 2;
  int128_t S1 = (a - b) * (a - b - 1) / 2;
  int128_t S2 = a * (b - cc - cc * (cc - 3) / 2 + dd * (dd - 3) / 2);
  int128_t S3 = (b * (b - 1) * (2 * b - 1) / 6 - b) -
                (dd * (dd - 1) * (2 * dd - 1) / 6 - dd);

  int128_t S4 = 0, S5 = 0, S6 = 0;
  for (int64_t q : c.primes) {
    if (q > p.x13)
      break;
    if (q <= p.xstar)
      continue;
    if (q <= sxy)
      S4 += pi(static_cast<int64_t>(x / (static_cast<int128_t>(q) * p.y)));
    else
      S5 += pi(static_cast<int64_t>(x / (static_cast<int128_t>(q) * q)));
    int128_t pr = pi(isqrt(x / q));
    S6 += pr * pr;
  }
  return static_cast<maxint_t>(S0 + S1 + S2 + S3 + a * S4 + S5 - S6);
}

// --- A (gourdon.md §5): easy two-prime leaves, weighted pi sums ---

// Segmented-PiTable variant of A (opt-in: env PC_ASEG). Computes the SAME A as
// a_impl but WITHOUT consulting the shared full PiTable: it sweeps a segmented
// prime sieve over the leaf values v in [2, Vmax] (Vmax = max leaf = x/pb_min^2),
// keeping a running prime count so pi(v) = running + #primes in [low, v]. Each
// leaf (p_b, q) is processed in the one segment its value v = x/(p_b q) falls in;
// the q-range landing in [low, high) is q in (x/(p_b high), x/(p_b low)] intersected
// with A's own (p_b, isqrt(x/p_b)] range, and the candidate p_b are bounded to
// [~x/high^2, isqrt(x/low)] so we don't rescan every p_b per segment. This is the
// proof-of-correctness step for dropping the O(sqrt x) PiTable (gourdon.md §5);
// once C/B/Sigma are segmented too, build_ctx need not build the full table.
maxint_t a_impl_seg(int128_t x, const GCtx& c, int nt) {
  const GParams& p = c.p;
  const size_t lo = upper_index(c.primes, p.xstar); // p_b > x*
  const size_t hi = upper_index(c.primes, p.x13);   // p_b <= x^(1/3)
  if (lo >= hi)
    return 0;
  const int64_t pb_min = c.primes[lo];
  const int64_t Vmax = divx(x, static_cast<int128_t>(pb_min) * pb_min);
  if (Vmax < 2)
    return 0;

  int64_t Slog = 20; // segment size 2^Slog (bitmap S/8 + word-prefix S/8)
  if (const char* e = std::getenv("PC_ASEGLOG"))
    Slog = std::atoi(e);
  const int64_t S = int64_t(1) << Slog;
  const int64_t V0 = 2;
  const int64_t nseg = (Vmax - V0) / S + 1;
  const size_t nsieve = upper_index(c.primes, isqrt(Vmax)); // sieving primes <= sqrt(Vmax)

  std::vector<int64_t> seg_partial(nseg, 0), seg_wsum(nseg, 0), seg_pcount(nseg, 0);

#pragma omp parallel num_threads(nt)
  {
    const int64_t NW = (S + 63) / 64;
    std::vector<uint64_t> bits(NW + 1);
    std::vector<int64_t> wpref(NW + 1);
#pragma omp for schedule(dynamic)
    for (int64_t seg = 0; seg < nseg; ++seg) {
      const int64_t low = V0 + seg * S;
      const int64_t high = std::min<int64_t>(low + S, Vmax + 1);
      const int64_t len = high - low;
      const int64_t nw = (len + 63) / 64;

      // Sieve [low, high): bit set => integer is prime. Tail bits (>= len) cleared.
      std::fill(bits.begin(), bits.begin() + nw, ~uint64_t(0));
      if (len & 63)
        bits[nw - 1] &= (uint64_t(1) << (len & 63)) - 1;
      for (size_t si = 0; si < nsieve; ++si) {
        const int64_t pp = c.primes[si];
        if (pp * pp >= high)
          break;
        int64_t k = (low + pp - 1) / pp; // ceil(low/pp)
        if (k < 2)
          k = 2; // never clear the prime pp itself (its smallest multiple is 2*pp)
        for (int64_t m = pp * k; m < high; m += pp) {
          const int64_t pos = m - low;
          bits[pos >> 6] &= ~(uint64_t(1) << (pos & 63));
        }
      }

      // Per-word prefix prime counts: wpref[w] = #primes in words [0, w).
      int64_t run = 0;
      for (int64_t w = 0; w < nw; ++w) {
        wpref[w] = run;
        run += std::popcount(bits[w]);
      }
      seg_pcount[seg] = run;

      // p_b whose leaves can land in [low, high): x/high^2 < p_b <= isqrt(x/low).
      size_t ib_end = std::min(hi, upper_index(c.primes, isqrt(x / low)));
      const int64_t pblo = divx(x, static_cast<int128_t>(high) * high);
      size_t ib_start = std::max(lo, upper_index(c.primes, pblo));
      if (ib_start > lo)
        --ib_start; // one-index safety margin against floor/isqrt slack

      int64_t part = 0, wsum = 0;
      for (size_t ib = ib_start; ib < ib_end; ++ib) {
        const int64_t pb = c.primes[ib];
        const int128_t xp = x / pb;
        const int64_t s = isqrt(xp);                  // q <= s (A's range)
        const int64_t thresh = divx(xp, p.y);         // weight 1 if q <= thresh else 2
        int64_t q_hi = divx(x, static_cast<int128_t>(pb) * low);      // v >= low
        int64_t q_lo = divx(x, static_cast<int128_t>(pb) * high) + 1; // v < high
        if (q_lo <= pb)
          q_lo = pb + 1; // q > p_b
        if (q_hi > s)
          q_hi = s;
        if (q_hi < q_lo)
          continue;
        for (size_t qi = upper_index(c.primes, q_lo - 1);
             qi < c.primes.size(); ++qi) {
          const int64_t q = c.primes[qi];
          if (q > q_hi)
            break;
          const int64_t v = divx(xp, q); // = floor(x/(pb q)), in [low, high)
          const int64_t pos = v - low;
          const unsigned bit = static_cast<unsigned>(pos & 63);
          const uint64_t mask =
              (bit == 63) ? ~uint64_t(0) : ((uint64_t(1) << (bit + 1)) - 1);
          const int64_t lc =
              wpref[pos >> 6] + std::popcount(bits[pos >> 6] & mask);
          const int64_t w = (q <= thresh) ? 1 : 2;
          part += w * lc;
          wsum += w;
        }
      }
      seg_partial[seg] = part;
      seg_wsum[seg] = wsum;
    }
  }

  // Sequential fold of the running prime count across segments:
  //   pi(v) = running + local_count(v), so sum_leaves w*pi(v) over a segment
  //         = seg_partial + running * seg_wsum.
  maxint_t A = 0;
  int64_t running = 0;
  for (int64_t seg = 0; seg < nseg; ++seg) {
    A += static_cast<maxint_t>(seg_partial[seg]) +
         static_cast<int128_t>(running) * seg_wsum[seg];
    running += seg_pcount[seg];
  }
  return A;
}

maxint_t a_impl(int128_t x, const GCtx& c, int nt) {
  if (std::getenv("PC_ASEG")) // opt-in: segmented A (no full PiTable). Validate vs default.
    return a_impl_seg(x, c, nt);
  const GParams& p = c.p;
  const size_t lo = upper_index(c.primes, p.xstar); // p_b > x*
  const size_t hi = upper_index(c.primes, p.x13);   // p_b <= x^(1/3)
  // NB: libdivide (branchfree reciprocal per prime q) was tried here to replace the
  // per-leaf divq — it REGRESSED on a fast-divider CPU (AC 2.64->2.95s @1e18). A
  // modern hardware divq (~14 cyc, pipelined) is fast enough that AC is NOT
  // division-bound there (the ~3% `div` samples are throughput, not a stall);
  // libdivide just adds a recip load + multiply chain. The reference's ~2.5x AC lead
  // is AVX512 + a segmented (cache-local) PiTable, not libdivide here. Reverted.

  maxint_t sum = 0;
#pragma omp parallel for schedule(dynamic, 16) reduction(i128add : sum)        \
    num_threads(nt)
  for (size_t ib = lo; ib < hi; ++ib) {
    const int64_t pb = c.primes[ib];
    const int128_t xp = x / pb;
    const int64_t s = isqrt(xp);
    const int64_t thresh = divx(xp, p.y);
    // 64-bit per-p_b accumulator (one p_b's leaves x max pi(sqrt x) << 2^63);
    // promoted to the int128 term once per p_b, so the hot loop stays int64
    // (the profile showed int128 shld/add:adc per leaf dominating a_impl).
    int64_t local = 0;
    for (size_t i = ib + 1; i < c.primes.size(); ++i) {
      const int64_t q = c.primes[i];
      if (q > s)
        break;
      const int64_t val = c.pi.at(divx(xp, q)); // sqrt(xp) <= u < sqrt(x): valid
      local += (q <= thresh ? 1 : 2) * val;
    }
    sum += static_cast<maxint_t>(local);
  }
  return sum;
}

// --- Local prime helpers for the low-memory B (no sqrt-x PiTable bitmap) ---
//
// B iterates the primes p in (y, sqrt x] and needs pi(sqrt x). The legacy path
// reads them from the shared O(sqrt x) PiTable bitmap; the low-mem path derives
// them on the fly. `base` must hold every prime <= isqrt(n) (n <= sqrt x here, so
// isqrt(n) <= x^(1/4) <= base max — base = c.primes covers it).

// Largest prime <= n (0 if none). Trial division; n is small (<= sqrt x).
inline int64_t prime_le_local(int64_t n, const std::vector<int64_t>& base) {
  for (int64_t m = n; m >= 2; --m) {
    bool prime = true;
    for (int64_t pp : base) {
      if (pp * pp > m)
        break;
      if (m % pp == 0) { prime = false; break; }
    }
    if (prime)
      return m;
  }
  return 0;
}

// Smallest prime > n, searching up to cap (0 if none in (n, cap]).
inline int64_t prime_gt_local(int64_t n, int64_t cap,
                              const std::vector<int64_t>& base) {
  for (int64_t m = n + 1; m <= cap; ++m) {
    bool prime = true;
    for (int64_t pp : base) {
      if (pp * pp > m)
        break;
      if (m % pp == 0) { prime = false; break; }
    }
    if (prime)
      return m;
  }
  return 0;
}

// All primes in [lo, hi] (ascending) into `out`, via a segmented bit-sieve crossing
// the base primes <= isqrt(hi). `bits` is a reusable scratch buffer. The leaf p-range
// of one B segment has width <= S, so this is O(S) per segment — same order as B's
// own sieve, and O(window) memory (no sqrt-x structure).
inline void primes_in_range(int64_t lo, int64_t hi,
                            const std::vector<int64_t>& base,
                            std::vector<uint64_t>& bits,
                            std::vector<int64_t>& out) {
  out.clear();
  if (hi < 2 || hi < lo)
    return;
  if (lo < 2)
    lo = 2;
  const int64_t len = hi - lo + 1;
  const int64_t nw = (len + 63) / 64;
  if (static_cast<int64_t>(bits.size()) < nw)
    bits.resize(nw);
  std::fill(bits.begin(), bits.begin() + nw, ~uint64_t(0));
  if (len & 63)
    bits[nw - 1] &= (uint64_t(1) << (len & 63)) - 1;
  for (int64_t pp : base) {
    if (pp * pp > hi)
      break;
    int64_t k = (lo + pp - 1) / pp;
    if (k < 2)
      k = 2;
    for (int64_t m = pp * k; m <= hi; m += pp) {
      const int64_t pos = m - lo;
      bits[pos >> 6] &= ~(uint64_t(1) << (pos & 63));
    }
  }
  for (int64_t w = 0; w < nw; ++w) {
    uint64_t b = bits[w];
    while (b) {
      const int bit = std::countr_zero(b);
      out.push_back(lo + (w << 6) + bit);
      b &= b - 1;
    }
  }
}

// --- B, wheel-30 sieve variant (gourdon.md §8) ---
//
// Same algorithm as b_impl but the segment sieve uses a WHEEL-30 layout: only the
// 8 residues coprime to 30 in each 30-integer window are stored (1 byte/window,
// bit k = residue {1,7,11,13,17,19,23,29}[k]). This stores 8/30 of the integers,
// so crossing a prime q clears 3.75x fewer bits (only its coprime-to-30 multiples)
// and counting scans 3.75x fewer bytes. Crossing steps through the 8-state wheel
// with per-prime precomputed (byte-delta, clear-mask) tables, so the inner loop
// has NO division or %30 (the bit-per-integer b_impl crossing — the bit-clears —
// was ~37% of B). Selected with PC_BWHEEL.
// pisqrtx: pi(sqrt x). Pass >= 0 for the LOW-MEM path (no sqrt-x PiTable): primes
// in (y, sqrt x] and the base pi(sqrt x) are then derived by local sieving instead
// of reading c.pi. Pass -1 for the legacy path (reads the shared bitmap).
maxint_t b_impl_wheel(int128_t x, const GCtx& c, int nt, int128_t pisqrtx = -1) {
  const GParams& p = c.p;
  const int64_t sqrtx = p.sqrtx;
  const int64_t y = p.y;
  const bool lowmem = pisqrtx >= 0;
  // B walks the primes p in (y, sqrt x]. Legacy: via the PiTable bitmap (prime_le /
  // pi()). Low-mem: via local sieves (prime_le_local / primes_in_range) so no
  // O(sqrt x) structure is needed. Small crossing primes (<= x^(1/3)) come from
  // c.primes in both cases.
  const int64_t p_top =
      lowmem ? prime_le_local(sqrtx, c.primes) : c.pi.prime_le(sqrtx);
  if (p_top <= y)
    return 0;
  const int64_t base = lowmem ? static_cast<int64_t>(pisqrtx) : c.pi(sqrtx);

  // Wheel-30 cannot represent the primes 2, 3 and 5. They land inside the B
  // sieve range (sqrt x, Wmax] only when sqrt x < 5 (i.e. x < 25), where the
  // wheel would silently miss them — e.g. B(15) needs pi(5) but the sieve never
  // counts the 5, undercounting by 1 (this made pi(15) come out as 7, not 6).
  // For such tiny x the sum is just a handful of leaves, so evaluate it directly
  // with a trivial prime count instead of the wheel sieve. (Threshold 7 rather
  // than 5 keeps the guard obviously safe; the path is unmeasurably cheap here.)
  if (sqrtx < 7) {
    auto pi_tiny = [](int64_t w) {
      int64_t cnt = 0;
      for (int64_t n = 2; n <= w; ++n) {
        bool prime = true;
        for (int64_t d = 2; d * d <= n; ++d)
          if (n % d == 0) { prime = false; break; }
        if (prime) ++cnt;
      }
      return cnt;
    };
    maxint_t Bt = 0;
    for (int64_t pp = p_top; pp > y;
         pp = lowmem ? prime_le_local(pp - 1, c.primes) : c.pi.prime_le(pp - 1))
      Bt += pi_tiny(divx(x, pp));
    return Bt;
  }

  maxint_t B = 0;
  int64_t cur_p = p_top; // primes p with w=x/p <= sqrt x are evaluated directly
  while (cur_p > y) {
    int64_t w = divx(x, cur_p);
    if (w > sqrtx)
      break;
    // w = floor(x/cur_p) >= sqrt x (cur_p <= sqrt x) and <= sqrt x here, so w ==
    // sqrt x exactly and pi(w) = pi(sqrt x). Low-mem uses the scalar directly.
    B += lowmem ? static_cast<maxint_t>(pisqrtx) : static_cast<maxint_t>(c.pi(w));
    cur_p = lowmem ? prime_le_local(cur_p - 1, c.primes) : c.pi.prime_le(cur_p - 1);
  }
  if (cur_p <= y)
    return B;

  const int64_t Wmax = divx(
      x, lowmem ? prime_gt_local(y, 2 * y + 100, c.primes) : c.pi.prime_gt(y));
  const int64_t W0 = sqrtx + 1;

  // Wheel-30 residue tables (built once per call; 30 entries).
  static constexpr int W8[8] = {1, 7, 11, 13, 17, 19, 23, 29};
  static constexpr int GAP[8] = {6, 4, 2, 4, 2, 4, 6, 2};
  int8_t slotT[30];
  uint8_t bmT[30], startT[30], endT[30];
  for (int r = 0; r < 30; ++r) {
    slotT[r] = -1;
    bmT[r] = 0;
  }
  for (int k = 0; k < 8; ++k) {
    slotT[W8[k]] = static_cast<int8_t>(k);
    bmT[W8[k]] = static_cast<uint8_t>(1u << k);
  }
  for (int r = 0; r < 30; ++r) { // startT[r]: residues >= r; endT[r]: residues <= r
    uint8_t s = 0, e = 0;
    for (int j = 0; j < 30; ++j)
      if (bmT[j]) {
        if (j >= r)
          s |= bmT[j];
        if (j <= r)
          e |= bmT[j];
      }
    startT[r] = s;
    endT[r] = e;
  }

  // Segment size: the wheel sieve's working set is bytes = S/30, so size S so
  // that fits ~L1d (B has no per-segment phi_start, so don't grow it with x).
  const long l1d_b = sysconf(_SC_LEVEL1_DCACHE_SIZE);
  int64_t Bslog = ilog2(60 * ((l1d_b > 0) ? l1d_b : 48 * 1024)); // S/30 ~= 2*L1d
  while (((Wmax - W0) >> Bslog) > (int64_t(1) << 18) && Bslog < 24)
    ++Bslog;
  if (const char* e = std::getenv("PC_BSEGLOG"))
    Bslog = std::atoi(e);
  const int64_t S = int64_t(1) << Bslog;
  const int64_t nseg = (Wmax - W0) / S + 1;
  std::vector<int64_t> seg_count(nseg, 0), seg_partial(nseg, 0),
      seg_leaves(nseg, 0);

  // Wheel-30 pre-sieve pattern: one period of "coprime to 30 AND to the primes
  // 7..kBWheelPre", in the byte layout. Copied into each segment (instead of
  // fill(0xFF) + crossing those small, dense primes every segment — they have the
  // most multiples). Period = product of the pre-sieved primes (>5); 7..19 ->
  // 323323 bytes (~316 KB, L2-resident). Crossing then starts above kBWheelPre.
  // The pre-sieve pattern size adapts to x. For kBWheelPre=19 the pattern spans
  // 7*11*13*17*19 = 323323 bytes and building it costs ~30*patP (~10M) byte-ops
  // REGARDLESS of x. Below KSieveCutOff the B sieve range is far smaller than
  // that pattern, so the fixed build dwarfs the actual sieving — this was the
  // bulk of the small-x fixed cost (B(1e8) = 5ms with pre=19). A smaller pre=13
  // pattern (7*11*13 = 1001 bytes, L1-resident) collapses it to 69us, and in fact
  // wins on B everywhere up to ~3e17. Only for very large x (>= KSieveCutOff)
  // does the denser pre=19 pattern amortize its build and win (~8% on B at 1e18).
  // Crossover measured between 3e17 (pre=13 faster) and 1e18 (pre=19 faster).
  static const int128_t KSieveCutOff =
      static_cast<int128_t>(500000000000000000LL); // 5e17
  int64_t kBWheelPre = (x < KSieveCutOff) ? 13 : 19;
  if (const char* e = std::getenv("PC_BWHEELPRE"))
    kBWheelPre = std::atoi(e);
  int64_t patP = 1;
  for (size_t i = upper_index(c.primes, 5);
       i < c.primes.size() && c.primes[i] <= kBWheelPre; ++i)
    patP *= c.primes[i];
  std::vector<uint8_t> pat(patP, uint8_t(0xFF));
  for (size_t i = upper_index(c.primes, 5);
       i < c.primes.size() && c.primes[i] <= kBWheelPre; ++i) {
    const int64_t q = c.primes[i];
    for (int64_t n = q; n < 30 * patP; n += q) {
      const uint8_t bm = bmT[n % 30];
      if (bm)
        pat[n / 30] &= static_cast<uint8_t>(~bm);
    }
  }

  // Per-prime wheel tables for the crossed primes q in (kBWheelPre, sqrt(Wmax)]
  // (smaller primes + 2,3,5 are handled by the pre-sieve pattern / wheel layout).
  // For each of the 8 wheel states k: dmask = the byte AND-mask clearing q's
  // multiple's slot, dbyte = the byte advance. Depend only on q — precompute once.
  const int64_t qmax = isqrt(Wmax);
  const size_t qfirst = upper_index(c.primes, kBWheelPre); // first prime > pre
  const size_t qend = upper_index(c.primes, qmax);
  const size_t nq = (qend > qfirst) ? qend - qfirst : 0;
  std::vector<std::array<int32_t, 8>> DBYTE(nq);
  std::vector<std::array<uint8_t, 8>> DMASK(nq);
  for (size_t i = 0; i < nq; ++i) {
    const int64_t q = c.primes[qfirst + i];
    const int qm = static_cast<int>(q % 30);
    for (int k = 0; k < 8; ++k) {
      const int rho = (qm * W8[k]) % 30; // residue of q*c when c == W8[k] (mod 30)
      DMASK[i][k] = static_cast<uint8_t>(~bmT[rho]);
      DBYTE[i][k] = static_cast<int32_t>((rho + GAP[k] * q) / 30);
    }
  }

#pragma omp parallel num_threads(nt)
  {
    std::vector<uint8_t> sb(S / 30 + 32);
    std::vector<uint64_t> pbits; // low-mem: scratch for primes_in_range
    std::vector<int64_t> pbuf;   // low-mem: primes p of this segment (ascending)
#pragma omp for schedule(dynamic)
    for (int64_t seg = 0; seg < nseg; ++seg) {
      const int64_t low = W0 + seg * S;
      const int64_t high = std::min<int64_t>(low + S, Wmax + 1);
      const int64_t wlo = low / 30;
      const int64_t whi = (high - 1) / 30;
      const int64_t nbytes = whi - wlo + 1;

      // Fill from the pre-sieve pattern (periodic, period patP bytes) at phase
      // wlo%patP — copies "coprime to 30 and to 7..kBWheelPre" in one or two
      // memcpys, replacing fill(0xFF) + crossing those small primes.
      int64_t phase = wlo % patP;
      int64_t done = std::min<int64_t>(nbytes, patP - phase);
      std::memcpy(sb.data(), pat.data() + phase, done);
      while (done < nbytes) {
        const int64_t chunk = std::min<int64_t>(nbytes - done, patP);
        std::memcpy(sb.data() + done, pat.data(), chunk);
        done += chunk;
      }
      sb[0] &= startT[low % 30];                // drop integers < low
      sb[nbytes - 1] &= endT[(high - 1) % 30];  // drop integers >= high

      for (size_t i = 0; i < nq; ++i) {
        const int64_t q = c.primes[qfirst + i];
        if (q * q >= high)
          break;
        int64_t c0 = (low + q - 1) / q; // ceil(low/q): smallest multiplier, q*c0>=low
        int r = static_cast<int>(c0 % 30);
        while (slotT[r] < 0) { // advance to the next coprime-to-30 multiplier
          ++c0;
          r = (r + 1 == 30) ? 0 : r + 1;
        }
        int64_t lb = (q * c0) / 30 - wlo;
        int k = slotT[r];
        const int32_t* db = DBYTE[i].data();
        const uint8_t* dm = DMASK[i].data();
        uint8_t* sbp = sb.data();
        // Step single states until k wraps to 0, then UNROLL the 8 wheel states
        // (deltas/masks in registers, no k&7, no indexed loads — the indexed-load
        // dependency chain was the crossing bottleneck). A full turn advances lb
        // by exactly q bytes.
        while (k != 0 && lb < nbytes) {
          sbp[lb] &= dm[k];
          lb += db[k];
          k = (k + 1) & 7;
        }
        const int32_t d0 = db[0], d1 = db[1], d2 = db[2], d3 = db[3], d4 = db[4],
                      d5 = db[5], d6 = db[6], d7 = db[7];
        const uint8_t m0 = dm[0], m1 = dm[1], m2 = dm[2], m3 = dm[3], m4 = dm[4],
                      m5 = dm[5], m6 = dm[6], m7 = dm[7];
        const int64_t lim = nbytes - q; // a full turn (advance q) stays in range
        while (lb < lim) {
          sbp[lb] &= m0; lb += d0;
          sbp[lb] &= m1; lb += d1;
          sbp[lb] &= m2; lb += d2;
          sbp[lb] &= m3; lb += d3;
          sbp[lb] &= m4; lb += d4;
          sbp[lb] &= m5; lb += d5;
          sbp[lb] &= m6; lb += d6;
          sbp[lb] &= m7; lb += d7;
        }
        for (k = 0; lb < nbytes;) { // tail (< one turn left)
          sbp[lb] &= dm[k];
          lb += db[k];
          k = (k + 1) & 7;
        }
      }

      // Primes p whose w = x/p lands in [low, high): p in (x/high, x/low]. Count
      // primes in [low,w] via the running byte cursor, popcounting 8 windows at a
      // time as a uint64 (the byte sieve packs one window per byte).
      auto pc8 = [&](int64_t off) {
        uint64_t word;
        std::memcpy(&word, &sb[off], sizeof(word));
        return std::popcount(word);
      };
      const int64_t p_low = std::max(y, divx(x, high));     // leaves: p > p_low
      const int64_t p_high = std::min(cur_p, divx(x, low));  //         p <= p_high
      int64_t acc = 0, cb = 0, part = 0, nl = 0;
      // One leaf p (w = x/p ascending => p descending), counting pi(w) via the
      // running byte cursor: pi(w) = base + (primes counted so far this segment).
      auto leaf = [&](int64_t pp) {
        const int64_t w = divx(x, pp);
        const int64_t wb = w / 30 - wlo;
        for (; cb + 8 <= wb; cb += 8)
          acc += pc8(cb);
        for (; cb < wb; ++cb)
          acc += std::popcount(static_cast<unsigned>(sb[cb]));
        part += base + acc +
                std::popcount(static_cast<unsigned>(sb[wb] & endT[w % 30]));
        ++nl;
      };
      if (lowmem) {
        // Local sieve of this segment's p-range (width <= S), then descend.
        primes_in_range(p_low + 1, p_high, c.primes, pbits, pbuf);
        for (size_t pj = pbuf.size(); pj-- > 0;)
          leaf(pbuf[pj]); // pbuf ascending -> iterate reverse for p descending
      } else {
        for (int64_t pp = c.pi.prime_le(p_high); pp > p_low;
             pp = c.pi.prime_le(pp - 1))
          leaf(pp);
      }
      int64_t sc = acc, b2 = cb;
      for (; b2 + 8 <= nbytes; b2 += 8)
        sc += pc8(b2);
      for (; b2 < nbytes; ++b2)
        sc += std::popcount(static_cast<unsigned>(sb[b2]));
      seg_count[seg] = sc;
      seg_partial[seg] = part;
      seg_leaves[seg] = nl;
    }
  }

  int64_t running = 0;
  for (int64_t seg = 0; seg < nseg; ++seg) {
    B += static_cast<maxint_t>(seg_partial[seg]) +
         static_cast<int128_t>(running) * seg_leaves[seg];
    running += seg_count[seg];
  }
  return B;
}

// --- B (gourdon.md §8): sum_{y < p <= sqrt x} pi(x/p) ---
//
// The arguments w = x/p lie in [sqrt x, x/y); rather than a PiTable up to x/y,
// sweep a segmented sieve over (sqrt x, x/y] keeping a running prime count, so
// pi(w) = pi(sqrt x) + #primes in (sqrt x, w]. Primes p are walked by DECREASING
// p (hence w ascending), in lockstep with the ascending sieve.
maxint_t b_impl(int128_t x, const GCtx& c, int nt) {
  if (!std::getenv("PC_NOBWHEEL")) // wheel-30 sieve is the default; opt out to compare
    return b_impl_wheel(x, c, nt);
  const GParams& p = c.p;
  const int64_t sqrtx = p.sqrtx;
  const size_t i_lo = upper_index(c.primes, p.y);   // first prime > y
  const size_t i_hi = upper_index(c.primes, sqrtx); // first prime > sqrt x
  if (i_lo >= i_hi)
    return 0;
  const int64_t base = c.pi(sqrtx); // pi(sqrt x)

  maxint_t B = 0;
  // Largest primes p may give w = x/p <= sqrt x; evaluate those directly.
  size_t cur = i_hi; // primes [i_lo, cur) are the "main" ones (w > sqrt x)
  while (cur > i_lo) {
    int64_t w = divx(x, c.primes[cur - 1]);
    if (w > sqrtx)
      break;
    B += c.pi(w);
    --cur;
  }

  if (cur > i_lo) {
    const int64_t Wmax = divx(x, c.primes[i_lo]); // largest w
    const int64_t W0 = sqrtx + 1;
    // Segment size — same cache-level-adaptive rule as d_impl (see its comment):
    // B's per-thread working set is the bit sieve (S/8 bytes). On big-L2 / high-
    // bandwidth cores an L1d-resident S wins (measured optimum 2^17-2^18), on
    // small-L2 cores the L2-resident S (2^20). Branch on L2 size; override
    // with PC_BSEGLOG.
    const long l1d_b = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    const long l2_b = sysconf(_SC_LEVEL2_CACHE_SIZE);
    int64_t Bslog, bcap;
    if (l2_b > 0 && l2_b <= 384 * 1024) {
      Bslog = 20; // L2-resident (small-L2 tuning)
      bcap = 1024;
    } else {
      // Unlike D, B has no per-segment phi_start recompute (just two binary
      // searches + the cheap pre-sieve fill), so it tolerates many small
      // segments: keep an L1d-resident S and DON'T grow it with x (measured flat
      // 2^17..2^21 at 1e17, optimum 2^18 at 1e16). bcap is a loose backstop only.
      const long l1 = (l1d_b > 0) ? l1d_b : 48 * 1024;
      Bslog = ilog2(8 * l1); // S/8 ~= L1d (48 KB -> 2^18)
      bcap = int64_t(1) << 18;
    }
    while (((Wmax - W0) >> Bslog) > bcap && Bslog < 22)
      ++Bslog;
    if (const char* e = std::getenv("PC_BSEGLOG"))
      Bslog = std::atoi(e);
    const int64_t S = int64_t(1) << Bslog;
    const int64_t nseg = (Wmax - W0) / S + 1;
    // Per-segment results so segments are independent (parallel): the prime
    // count, the local sum of (base + #primes in [low, w_p]) over this
    // segment's leaves, and the leaf count. The cross-segment running prime
    // count is folded in afterwards by a cheap sequential scan.
    std::vector<int64_t> seg_count(nseg, 0), seg_partial(nseg, 0),
        seg_leaves(nseg, 0);

    // Multi-table pre-sieve up to L = 163: "coprime to every prime <= L" is the
    // AND of several wheels, one per group of primes (a single wheel for all of
    // them would have an astronomically large period). Per segment the base = AND
    // over the tables (fused, see presieve_fill); then we cross only primes > L.
    // This removes the per-segment crossing of the dense primes 2..163 (the small
    // ones have the most multiples); B's range is > sqrt x so none lies in it.
    int64_t kPreLimit = 163;
    if (const char* e = std::getenv("PC_PRELIMIT"))
      kPreLimit = std::atoi(e);
    constexpr int64_t kPreCap = int64_t(1) << 22; // per-table period cap (bits)
    std::vector<PreTable> pre;
    for (size_t qi = 0; qi < c.primes.size() && c.primes[qi] <= kPreLimit;) {
      int64_t period = 1;
      std::vector<int64_t> grp;
      while (qi < c.primes.size() && c.primes[qi] <= kPreLimit &&
             (grp.empty() || period * c.primes[qi] <= kPreCap)) {
        period *= c.primes[qi];
        grp.push_back(c.primes[qi++]);
      }
      PreTable t;
      t.period = period;
      t.bits.assign(period / 64 + kPrePad + 8, ~uint64_t(0));
      // cross into the padding too: multiples are periodic, so the pad words are a
      // true continuation of the pattern (lets SIMD blocks read past the wrap).
      for (int64_t q : grp)
        for (int64_t j = 0; j < period + kPrePad * 64; j += q)
          t.bits[j >> 6] &= ~(uint64_t(1) << (j & 63));
      pre.push_back(std::move(t));
    }
    const size_t firstCross = upper_index(c.primes, kPreLimit);
    const int ng = static_cast<int>(pre.size());

#pragma omp parallel num_threads(nt)
    {
      std::vector<uint64_t> bits((S + 63) / 64);
#pragma omp for schedule(dynamic)
      for (int64_t seg = 0; seg < nseg; ++seg) {
        const int64_t low = W0 + seg * S;
        const int64_t high = std::min<int64_t>(low + S, Wmax + 1);
        const int64_t len = high - low;
        const int64_t nw = (len + 63) / 64;

        // base = AND over all pre-sieve tables, fused into one pass over bits[].
        presieve_fill(bits.data(), nw, pre.data(), ng, low);
        if (len & 63)
          bits[nw - 1] &= (uint64_t(1) << (len & 63)) - 1;
        for (size_t qi = firstCross; qi < c.primes.size(); ++qi) {
          const int64_t q = c.primes[qi];
          if (q * q >= high)
            break;
          for (int64_t j = ((low + q - 1) / q) * q; j < high; j += q) {
            const int64_t pos = j - low;
            bits[pos >> 6] &= ~(uint64_t(1) << (pos & 63));
          }
        }

        // Primes p whose w = x/p lands in [low, high): p in (x/high, x/low].
        const size_t lo_idx =
            std::max(i_lo, upper_index(c.primes, divx(x, high)));
        const size_t hi_idx =
            std::min(cur, upper_index(c.primes, divx(x, low)));
        int64_t acc = 0, cw = 0, part = 0, nl = 0;
        for (size_t ip = hi_idx; ip-- > lo_idx;) { // w ascending (p descending)
          const int64_t w = divx(x, c.primes[ip]);
          const int64_t t = w - low, tw = t >> 6;
          while (cw < tw)
            acc += std::popcount(bits[cw++]);
          const unsigned bit = static_cast<unsigned>(t & 63);
          const uint64_t mask =
              bit == 63 ? ~uint64_t(0) : ((uint64_t(1) << (bit + 1)) - 1);
          part += base + acc + std::popcount(bits[tw] & mask);
          ++nl;
        }
        int64_t sc = acc;
        for (int64_t w2 = cw; w2 < nw; ++w2)
          sc += std::popcount(bits[w2]);
        seg_count[seg] = sc;
        seg_partial[seg] = part;
        seg_leaves[seg] = nl;
      }
    }

    // Sequential scan: add the running prime count from earlier segments.
    int64_t running = 0;
    for (int64_t seg = 0; seg < nseg; ++seg) {
      B += static_cast<maxint_t>(seg_partial[seg]) +
           static_cast<int128_t>(running) * seg_leaves[seg];
      running += seg_count[seg];
    }
  }
  return B;
}

// --- C: easy special leaves p_k < p_b <= x* with v < p_b^2 (gourdon.md §6) ---
// leaf value via the identity phi(v, b-1) = pi(v) - b + 2; parallel over p_b.

// Per-p_b recursion state. The invariants (x, primes, pi, b, p_b, the m-bounds)
// are FIXED for one p_b, so they live here instead of being passed — and
// stack-spilled — on every recursive call; only (start, m, sign) vary. This cut
// c_rec's per-call overhead (the profile showed ~14 args spilled to the stack
// each call). The leaf sum is a 64-bit local (one p_b's leaves: count x
// max|pi(v)| << 2^63) promoted to the int128 term once per p_b, so the hot path
// is int64 not int128 (no shld/imulq:sbb sequences per leaf).
struct CRec {
  int128_t x;
  bool x64;
  int64_t x64v;
  const std::vector<int64_t>& primes;
  const PiTable& pi;
  int b;
  int64_t pb;
  int64_t mlo;
  int64_t mhi;
  int64_t y;
  int64_t sum = 0;

  // Recursively enumerate ONLY the valid leaf multipliers m: squarefree products
  // of distinct primes in (p_b, y], with m <= mhi (pruned), emitting those with
  // m > mlo. One node per valid leaf instead of scanning (and rejecting ~95% of)
  // every integer in (mlo, mhi]. mu = (-1)^#factors; leaf adds -mu*(pi(v)-b+2).
  // MFITS=true: m*q is known to fit int64 (mhi*y < 2^63, i.e. x <= ~1e20 with
  // z=y), so the product/compare stay int64 — the int128 mul+cmp was ~26% of
  // c_impl on a fast-divider host. MFITS=false keeps the int128 path (high-x / huge z), so
  // there is no overflow even if the x>1e20 guardrail is bypassed.
  template <bool MFITS>
  void rec(size_t start, int64_t m, int sign) {
    for (size_t i = start; i < primes.size(); ++i) {
      const int64_t q = primes[i];
      if (q > y)
        break;
      int64_t nm;
      if constexpr (MFITS) {
        nm = m * q;
        if (nm > mhi)
          break; // primes ascending: m*q only grows
      } else {
        const int128_t nm128 = static_cast<int128_t>(m) * q;
        if (nm128 > mhi)
          break;
        nm = static_cast<int64_t>(nm128);
      }
      const int s = -sign;
      if (nm > mlo) {
        const int64_t pbm = pb * nm;
        const int64_t v =
            x64 ? fast_div_u(static_cast<uint64_t>(x64v),
                             static_cast<uint64_t>(pbm))
                : static_cast<int64_t>(x / static_cast<int128_t>(pbm));
        const int64_t leafval = pi.at(v) - b + 2; // p_b <= v < p_b^2 <= sqrt x
        sum += -static_cast<int64_t>(s) * leafval;
      }
      rec<MFITS>(i + 1, nm, s);
    }
  }
};

// Segmented-PiTable variant of C (opt-in: env PC_CSEG). Same value as c_impl, but
// WITHOUT the shared full PiTable: it sweeps a segmented prime sieve over the leaf
// values v in [2, Vmax] (Vmax = x*^2, since every C leaf has v < p_b^2 <= x*^2),
// keeping a running prime count so pi(v) = running + #primes in [low, v].
//
// The CRec squarefree RECURSION is replaced by a LINEAR scan of m over the per-
// (segment, p_b) range, using the precomputed mp[m] = mu(m)*pmin(m) (0 if m is not
// squarefree) to filter: squarefree (mp!=0), P^-(m) > p_b (|mp[m]| > p_b), and
// P^+(m) <= y (auto when z==y; else via pmax[m]). This is the value-ordered traversal
// segmentation forces (gourdon.md §6/§7.3); it revisits dense integers instead of the
// recursion's sparse leaves, the documented C trade-off.
//
// Fold subtlety: the leaf is coeff*(pi(v) - b + 2) with coeff = -mu(m) = (mp[m]<0?+1:-1)
// and b = ib+1 depends on p_b (NOT the segment). So pi(v)=running+lc(v) splits the
// per-segment fold into THREE accumulators: sum coeff*lc(v) (the local-count part),
// sum coeff (multiplies running), and sum coeff*(2-b) (the per-p_b constant part).
// Value-segmented C (gourdon.md §6) with no full PiTable. Optional [v_from, .)
// restriction: when v_from > 2 only leaves with v >= v_from are emitted (the
// value sweep starts at v_from), and running_init must be pi(v_from - 1) so the
// running prime count is correct for the first segment. Defaults reproduce the
// full C over [2, x*^2] (the oracle for PC_CSEG and c_impl_sparse's tail).
maxint_t c_impl_seg(int128_t x, const GCtx& c, int nt, int64_t v_from = 2,
                    int64_t running_init = 0) {
  const GParams& p = c.p;
  const size_t lo = static_cast<size_t>(p.k);     // p_b > p_k
  const size_t hi = upper_index(c.primes, p.xstar); // p_b <= x*
  if (lo >= hi)
    return 0;
  const int64_t y = p.y;
  const bool checkpmax = (p.z != p.y); // P^+(m) <= y automatic when z == y
  const int64_t Vmax =
      static_cast<int64_t>(static_cast<int128_t>(p.xstar) * p.xstar); // v < x*^2
  if (Vmax < 2)
    return 0;

  int64_t Slog = 20;
  if (const char* e = std::getenv("PC_CSEGLOG"))
    Slog = std::atoi(e);
  const int64_t S = int64_t(1) << Slog;
  const int64_t V0 = std::max<int64_t>(2, v_from);
  if (V0 > Vmax)
    return 0;
  const int64_t nseg = (Vmax - V0) / S + 1;
  const size_t nsieve = upper_index(c.primes, isqrt(Vmax));

  std::vector<maxint_t> seg_partial(nseg, 0), seg_csum(nseg, 0), seg_cbsum(nseg, 0);
  std::vector<int64_t> seg_pcount(nseg, 0);

#pragma omp parallel num_threads(nt)
  {
    const int64_t NW = (S + 63) / 64;
    std::vector<uint64_t> bits(NW + 1);
    std::vector<int64_t> wpref(NW + 1);
#pragma omp for schedule(dynamic)
    for (int64_t seg = 0; seg < nseg; ++seg) {
      const int64_t low = V0 + seg * S;
      const int64_t high = std::min<int64_t>(low + S, Vmax + 1);
      const int64_t len = high - low;
      const int64_t nw = (len + 63) / 64;

      std::fill(bits.begin(), bits.begin() + nw, ~uint64_t(0));
      if (len & 63)
        bits[nw - 1] &= (uint64_t(1) << (len & 63)) - 1;
      for (size_t si = 0; si < nsieve; ++si) {
        const int64_t pp = c.primes[si];
        if (pp * pp >= high)
          break;
        int64_t k = (low + pp - 1) / pp;
        if (k < 2)
          k = 2;
        for (int64_t m = pp * k; m < high; m += pp) {
          const int64_t pos = m - low;
          bits[pos >> 6] &= ~(uint64_t(1) << (pos & 63));
        }
      }
      int64_t run = 0;
      for (int64_t w = 0; w < nw; ++w) {
        wpref[w] = run;
        run += std::popcount(bits[w]);
      }
      seg_pcount[seg] = run;

      // p_b with a leaf in [low, high): v >= p_b and v < p_b^2 give
      // isqrt(low) < p_b < high.
      size_t ib_end = std::min(hi, upper_index(c.primes, high - 1));
      size_t ib_start = std::max(lo, upper_index(c.primes, isqrt(low)));
      if (ib_start > lo)
        --ib_start; // safety margin against floor/isqrt slack

      maxint_t part = 0, csum = 0, cbsum = 0;
      for (size_t ib = ib_start; ib < ib_end; ++ib) {
        const int64_t pb = c.primes[ib];
        const int b = static_cast<int>(ib) + 1;
        const int128_t pb2 = static_cast<int128_t>(pb) * pb;
        const int128_t pb3 = pb2 * pb;
        const int64_t mlo = std::max(static_cast<int64_t>(p.z / pb), divx(x, pb3));
        const int64_t mhi = std::min<int64_t>(p.z, divx(x, pb2));
        if (mhi <= mlo)
          continue;
        // m-range whose leaf v = x/(pb m) lands in [low, high), intersected with C's.
        int64_t m_hi = std::min(mhi, divx(x, static_cast<int128_t>(pb) * low));
        int64_t m_lo =
            std::max(mlo + 1, divx(x, static_cast<int128_t>(pb) * high) + 1);
        if (m_hi < m_lo)
          continue;
        const int64_t twob = 2 - b;
        int64_t part_pb = 0, csum_pb = 0, cbsum_pb = 0;
        for (int64_t m = m_lo; m <= m_hi; ++m) {
          const int32_t mv = c.mpv(m);
          if (mv == 0)
            continue; // not squarefree
          const int64_t pmin = mv < 0 ? -mv : mv;
          if (pmin <= pb)
            continue; // P^-(m) <= p_b
          if (checkpmax && c.pmax[m] > y)
            continue; // P^+(m) > y
          const int64_t v = divx(x, static_cast<int128_t>(pb) * m); // in [low, high)
          const int64_t pos = v - low;
          const unsigned bit = static_cast<unsigned>(pos & 63);
          const uint64_t mask =
              (bit == 63) ? ~uint64_t(0) : ((uint64_t(1) << (bit + 1)) - 1);
          const int64_t lc =
              wpref[pos >> 6] + std::popcount(bits[pos >> 6] & mask);
          const int64_t coeff = (mv < 0) ? 1 : -1; // -mu(m)
          part_pb += coeff * lc;
          csum_pb += coeff;
          cbsum_pb += coeff * twob;
        }
        part += static_cast<maxint_t>(part_pb);
        csum += static_cast<maxint_t>(csum_pb);
        cbsum += static_cast<maxint_t>(cbsum_pb);
      }
      seg_partial[seg] = part;
      seg_csum[seg] = csum;
      seg_cbsum[seg] = cbsum;
    }
  }

  // Fold: C = sum_seg [ seg_partial + running*seg_csum + seg_cbsum ];
  // pi(v)=running+lc, so sum coeff*(pi(v)-b+2) = sum coeff*lc + running*sum coeff
  //                                              + sum coeff*(2-b).
  maxint_t C = 0;
  int64_t running = running_init;
  for (int64_t seg = 0; seg < nseg; ++seg) {
    C += seg_partial[seg] + static_cast<int128_t>(running) * seg_csum[seg] +
         seg_cbsum[seg];
    running += seg_pcount[seg];
  }
  return C;
}

// SPARSE C (low-mem): inverts the dense per-(p_b, m) scan to m-outer, removing
// both the ~2720x re-scan of each m and the ~94% squarefree/lpf rejection. Split
// by leaf value v = x/(p_b*m):
//   * BULK (v <= z, ~90% of leaves, growing with x): enumerate squarefree m once
//     (m = 2..z, skip mpv==0); for each, the valid p_b form a contiguous prime
//     range; emit coeff*(pi(v) - b + 2) with pi(v) = c.pi.at(v) (O(1), the small
//     PiTable up to z that build_ctx_lowmem already holds; no running-count needed
//     since pi(v) is absolute). Membership decided by the ORIGINAL integer
//     predicates (mlo<m<=mhi, pmin>pb) so it is bit-exact with the dense scan.
//   * TAIL (v > z, ~10%): the value-segmented dense scan over (z, x*^2], reusing
//     c_impl_seg with running_init = pi(z). Its value-segmentation visits only the
//     small-m (v>z) sub-ranges, ~15% of the original m-scan.
// C = C_bulk + C_tail (partition by v; every valid leaf counted exactly once).
maxint_t c_impl_sparse(int128_t x, const GCtx& c, int nt) {
  const GParams& p = c.p;
  const int64_t z = p.z;
  const int64_t y = p.y;
  const int64_t xstar = p.xstar;
  const size_t lo = static_cast<size_t>(p.k);       // p_b > p_k  (index >= lo)
  const size_t hi = upper_index(c.primes, xstar);   // p_b <= x*
  if (lo >= hi || z < 2)
    return 0;
  const int64_t T = c_split_T(p); // bulk serves v <= T, tail serves v > T
  // Bulk needs pi(v) for v <= T. build_ctx_lowmem sizes pi.limit() >= T; fall back
  // to the dense oracle if some tuning left it short (rare).
  if (c.pi.limit() < T)
    return c_impl_seg(x, c, nt);
  const bool checkpmax = (p.z != p.y);
  const bool prof = std::getenv("PC_LMPROF") != nullptr;
  auto pclk = [] { return std::chrono::steady_clock::now(); };
  auto pms = [](std::chrono::steady_clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  auto tb0 = pclk();

  // ---- C_bulk: v <= T, inverted m-outer ----
  maxint_t C = 0;
#pragma omp parallel for schedule(dynamic, 1 << 12) reduction(i128add : C)        \
    num_threads(nt)
  for (int64_t m = 2; m <= z; ++m) {
    const int32_t mv = c.mpv(m);
    if (mv == 0)
      continue; // not squarefree
    if (checkpmax && c.pmax[m] > y)
      continue; // P^+(m) > y
    const int64_t pmin = mv < 0 ? -mv : mv; // lpf(m)
    const int64_t coeff = (mv < 0) ? 1 : -1; // -mu(m)
    const int128_t xm = x / m;               // floor(x/m)
    // The valid p_b for this m form a CONTIGUOUS prime range; every bound below is
    // the exact integer equivalent of the dense scan's per-p_b predicate (so this
    // is bit-exact, no per-p_b re-test):
    //   m <= floor(x/pb^2)   <=> pb <= isqrt(floor(x/m))
    //   m >  floor(x/pb^3)   <=> pb >  iroot3(floor(x/m))
    //   m >  floor(z/pb)     <=> pb >  floor(z/m)
    //   v=floor(x/(pb m))<=T <=> pb >  floor(x/((T+1) m))   (bulk = v<=T half)
    //   P^-(m) > pb          <=> pb <  lpf(m);   plus p_k < pb <= x*.
    const int64_t pbHi = std::min<int64_t>({isqrt(xm), pmin - 1, xstar});
    const int64_t pbLo = std::max<int64_t>(
        {static_cast<int64_t>(z / m), iroot<3>(xm),
         divx(x, static_cast<int128_t>(T + 1) * m)});
    if (pbHi <= pbLo)
      continue;
    const size_t ib_start = std::max(lo, upper_index(c.primes, pbLo));
    const size_t ib_end = std::min(hi, upper_index(c.primes, pbHi));
    int64_t part = 0;
    for (size_t ib = ib_start; ib < ib_end; ++ib) {
      const int64_t pb = c.primes[ib];
      const int64_t v = divx(x, static_cast<int128_t>(pb) * m);
      const int b = static_cast<int>(ib) + 1;
      part += coeff * (c.pi.at(v) - b + 2);
    }
    C += static_cast<maxint_t>(part);
  }
  auto tb1 = pclk();

  // ---- C_tail: v > T, value-segmented over (T, x*^2], running starts at pi(T) ----
  const maxint_t Ctail = c_impl_seg(x, c, nt, T + 1, c.pi(T));
  C += Ctail;
  if (prof)
    std::fprintf(stderr,
        "[LMPROF c_sparse] T=%lld (z=%lld) bulk-wall=%.0fms tail-wall=%.0fms\n",
        (long long)T, (long long)z, pms(tb1 - tb0), pms(pclk() - tb1));
  return C;
}

maxint_t c_impl(int128_t x, const GCtx& c, int nt) {
  if (std::getenv("PC_CSEG")) // opt-in: segmented C (no full PiTable). Validate vs default.
    return c_impl_seg(x, c, nt);
  const GParams& p = c.p;
  const size_t lo = static_cast<size_t>(p.k);
  const size_t hi = upper_index(c.primes, p.xstar);
  const bool x64 = x <= INT64_MAX; // fast 64-bit division path
  const int64_t x64v = x64 ? static_cast<int64_t>(x) : 0;

  maxint_t C = 0;
#pragma omp parallel for schedule(dynamic, 8) reduction(i128add : C)           \
    num_threads(nt)
  for (size_t ib = lo; ib < hi; ++ib) {
    const int64_t pb = c.primes[ib];
    const int b = static_cast<int>(ib) + 1;
    const int128_t pb2 = static_cast<int128_t>(pb) * pb;
    const int128_t pb3 = pb2 * pb;
    // easy leaves: m > x/p_b^3 (v < p_b^2), m <= x/p_b^2 (v >= p_b), m > z/p_b
    // (n > z), m <= z.
    const int64_t mlo = std::max(static_cast<int64_t>(p.z / pb), divx(x, pb3));
    const int64_t mhi = std::min<int64_t>(p.z, divx(x, pb2));
    if (mhi > mlo) {
      CRec cr{x, x64, x64v, c.primes, c.pi, b, pb, mlo, mhi, p.y};
      // int64 vs int128 m*q is MACHINE-DIVERGENT (auto-gated on the divider probe,
      // INVERSE of the divl lever): a SLOW divider (ENABLE_DIV32 on) masks the
      // int128 mul+compare so it is ~free and int64 templating is a net loss
      // (measured +2%) -> always int128. A FAST divider (ENABLE_DIV32 off) exposes
      // the int128 cost -> int64 where it can't overflow (x<=~1e20), int128 above
      // (-0.6%). int128 (rec<false>) is always correct.
#if defined(ENABLE_DIV32)
      cr.rec<false>(ib + 1, 1, +1); // slow divider: int128 m*q is ~free
#else
      if (p.y == 0 || mhi <= INT64_MAX / p.y)
        cr.rec<true>(ib + 1, 1, +1); // fast divider: int64 where no overflow
      else
        cr.rec<false>(ib + 1, 1, +1);
#endif
      C += static_cast<maxint_t>(cr.sum);
    }
  }
  return C;
}

// --- D: hard special leaves p_k < p_b <= x* with v >= p_b^2 (gourdon.md §7) ---
// Sweep a sieve of [1, x/z]: process p_b in increasing order while the sieve
// holds phi(., b-1) (integers coprime to the first b-1 primes), so each leaf
// value phi(v, b-1) is a prefix count; after each p_b cross off its multiples
// to reach phi(., b).
//
// The sieve is BIT-PACKED (one bit per integer) with a Fenwick tree at WORD
// granularity holding each 64-bit word's popcount. This is the hot path
// (perf: ~half of the whole run): crossing a number off touches one word plus
// an O(log) Fenwick path 64x shorter/denser than a per-integer tree, cutting
// the cache misses that dominated the previous per-position Fenwick.

// Dense (1 bit/integer) sieve + linear block counter. Correct for any k; used
// as the fallback for tiny x (k < 3) where the wheel-30 layout (which excludes
// 2,3,5) cannot represent "coprime to the first k primes". d_impl dispatches.
maxint_t d_impl_dense(int128_t x, const GCtx& c, int nt) {
  const GParams& p = c.p;
  const int64_t N = static_cast<int64_t>(x / p.z); // sieve domain [1, N]
  if (N < 1)
    return 0;
  const size_t bend = upper_index(c.primes, p.xstar); // pi(x*)
  if (static_cast<size_t>(p.k) >= bend)
    return 0;

  // PARALLEL over segments: each is independent because its start counts
  // phi(low-1, b-1) are recomputed directly via the recurrence
  //   phi(L, i) = phi(L, i-1) - phi(L/p_i, i-1)
  // using phi_pi() (pi-table Legendre cutoff) so that recompute is cheap. The
  // Segment size — keep each thread's working set (bits S/8 + word-Fenwick S/16
  // ~= 3S/16 bytes) cache-resident so the dominant cost (the Fenwick's
  // erase/count memory traffic) stays in cache; each segment also recomputes
  // phi_start (O(bend)), so too small an S means too many segments. The optimum
  // cache LEVEL is microarchitecture-dependent, not just a function of size:
  //   * Small-L2, few-core, bandwidth-starved (e.g. ~256 KB L2, 4 cores): an
  //     L2-resident S (2^20, ~192 KB) wins — fewer, larger segments minimise the
  //     phi_start recompute that few cores can't hide. Measured D -42% at 1e15
  //     vs 2^22; optimum 2^20 at 1e15, 2^21 at 1e16.
  //   * Big-L2 / high-bandwidth, many-core (e.g. ~1 MB L2, 8 cores, AVX-512):
  //     an L1d-resident S (~3S/16 = L1d/2, i.e. 2^17) wins — abundant bandwidth
  //     and cores make the extra phi_start cheap, and the tighter working set
  //     cuts the dominant Fenwick traffic. Measured D -24% at 1e15, -22% at 1e16
  //     (optima 2^17 small x .. 2^21 at 1e17, reached via the nseg cap below).
  // We pick the regime by L2 size — the one signal that separates the two
  // measured classes: <=384 KB L2 keeps the L2-resident rule; otherwise the
  // L1d-resident rule (a sane default for modern big-core CPUs — re-measure &
  // pin PC_DSEGLOG on a new machine if it differs). PC_DSEGLOG overrides everything.
  const long l1d = sysconf(_SC_LEVEL1_DCACHE_SIZE);
  const long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
  int64_t Slog, capnseg;
  if (l2 > 0 && l2 <= 384 * 1024) {
    Slog = 20;       // L2-resident (small-L2 tuning)
    capnseg = 1024;
  } else {
    const long l1 = (l1d > 0) ? l1d : 48 * 1024; // fallback 48 KB
    Slog = ilog2(8 * l1 / 3);                     // 3S/16 ~= L1d/2 (48 KB -> 2^17)
    capnseg = 2750;
  }
  while ((N >> Slog) > capnseg && Slog < 22)
    ++Slog;
  if (const char* e = std::getenv("PC_DSEGLOG"))
    Slog = std::atoi(e);
  const int64_t S = int64_t(1) << Slog;
  const int64_t NW = (S + 63) / 64;
  // Counter-block geometry (replaces the per-word Fenwick): ~sqrt(NW) words per
  // block, so absorbing whole blocks and popcounting the sub-block remainder are
  // balanced. Power-of-2 block size -> block index = wordIndex >> LOG_WPB.
  const int64_t LOG_WPB = std::max<int64_t>(1, ilog2(NW) / 2);
  const int64_t WPB = int64_t(1) << LOG_WPB; // words per counter block
  const int64_t BLKBITS = WPB << 6;          // bits per counter block
  const int64_t nseg = (N + S - 1) / S;
  const bool x64 = x <= INT64_MAX;
  const int64_t x64v = x64 ? static_cast<int64_t>(x) : 0;
  // pmax[m] <= y is automatic when z == y (m <= z = y => all factors <= y), the
  // default (alpha_z = 1) — skip the pmax load+test per m in that common case.
  const bool checkpmax = (p.z != p.y);
  (void)phi(2, 8); // warm up phi()'s function-local statics before threading

  // Precompute one period of the first-k-primes wheel as a bit pattern. phi(.,k)
  // is periodic mod W = p_1*...*p_k, so each segment's coprime-to-first-k base
  // becomes a shift-and-copy of this pattern instead of re-crossing every small
  // prime's multiples per segment. wheel bit i (0 <= i <= W+63) = 1 iff (i mod W)
  // is coprime to the first k primes; clearing multiples over [0, W+63] is
  // already periodic because each of the first k primes divides W.
  int64_t W = 1;
  for (int i = 0; i < p.k; ++i)
    W *= c.primes[i];
  std::vector<uint64_t> wheel((W + 128) / 64 + 2, ~uint64_t(0));
  for (int i = 0; i < p.k; ++i) {
    const int64_t pp = c.primes[i];
    for (int64_t j = 0; j <= W + 63; j += pp)
      wheel[j >> 6] &= ~(uint64_t(1) << (j & 63));
  }

  maxint_t D = 0;
#pragma omp parallel num_threads(nt) reduction(i128add : D)
  {
    std::vector<uint64_t> bits(NW);
    std::vector<int32_t> cnt(((NW + WPB - 1) >> LOG_WPB) + 1); // block popcounts
    std::vector<int64_t> phi_start(bend + 1); // phi_start[b] = phi(low-1, b-1)

#pragma omp for schedule(dynamic)
    for (int64_t seg = 0; seg < nseg; ++seg) {
      const int64_t low = 1 + seg * S;
      const int64_t high = std::min<int64_t>(low + S, N + 1);
      const int64_t len = high - low;
      const int64_t nw = (len + 63) / 64;
      const int64_t L = low - 1;

      // phi_start[b] = phi(L, b-1) for b in [k+1, bend].
      int64_t prev = static_cast<int64_t>(phi(L, p.k)); // phi(L, k)
      for (size_t b = static_cast<size_t>(p.k) + 1; b <= bend; ++b) {
        phi_start[b] = prev;
        prev -= phi_pi(L / c.primes[b - 1], static_cast<int>(b) - 1, c.primes, c.pi);
      }

      // bits = positions [0,len) coprime to the first k primes, filled by
      // shift-copying the precomputed wheel pattern. The integer at position 0
      // is `low`, so the pattern phase is (low mod W); each 64-bit word reads 64
      // consecutive pattern bits and the phase advances by 64 (mod W) per word.
      int64_t ph = low % W;
      for (int64_t w = 0; w < nw; ++w) {
        const int64_t wd = ph >> 6;
        const int off = ph & 63;
        uint64_t val = wheel[wd] >> off;
        if (off)
          val |= wheel[wd + 1] << (64 - off);
        bits[w] = val;
        ph += 64;
        if (ph >= W)
          ph -= W;
      }
      if (len & 63)
        bits[nw - 1] &= (uint64_t(1) << (len & 63)) - 1;
      // cnt[j] = #set bits in word-block j (WPB words). Replaces the per-word
      // Fenwick: O(1) erase (a single decrement) and an O(1)-amortised monotone
      // count, dropping the Fenwick's O(log) scattered-write erase that the
      // profile showed dominating D (cache misses + branch mispredicts).
      const int64_t nblk = (nw + WPB - 1) >> LOG_WPB;
      std::fill(cnt.begin(), cnt.begin() + nblk, 0);
      for (int64_t w = 0; w < nw; ++w)
        cnt[w >> LOG_WPB] += std::popcount(bits[w]);

      auto erase = [&](int64_t pos) {
        const int64_t W = pos >> 6;
        const uint64_t m = uint64_t(1) << (pos & 63);
        if (bits[W] & m) {
          bits[W] &= ~m;
          --cnt[W >> LOG_WPB];
        }
      };

      // #set bits in the inclusive bit range [lo, hi].
      auto popcnt_range = [&](int64_t lo, int64_t hi) -> int64_t {
        if (lo > hi)
          return 0;
        const int64_t wl = lo >> 6, wh = hi >> 6;
        const uint64_t lmask = ~uint64_t(0) << (lo & 63);
        const unsigned hb = static_cast<unsigned>(hi & 63);
        const uint64_t hmask =
            (hb == 63) ? ~uint64_t(0) : ((uint64_t(1) << (hb + 1)) - 1);
        if (wl == wh)
          return std::popcount(bits[wl] & lmask & hmask);
        int64_t s = std::popcount(bits[wl] & lmask);
        for (int64_t w = wl + 1; w < wh; ++w)
          s += std::popcount(bits[w]);
        return s + std::popcount(bits[wh] & hmask);
      };

      // Monotone forward counter : count_mono(pos) = #set bits
      // in [0, pos] for pos ASCENDING within a fixed b. Whole blocks are absorbed
      // from cnt[] (O(1) each), the sub-block remainder counted by a short
      // popcount of the gap since the previous query; count_ persists across
      // calls. Reset per b (the previous b's erase mutated bits[]). The first
      // leaf of each b is near `low`, so the reset re-walk is cheap.
      int64_t c_i = 0, c_sum = 0, c_stop = 0, count_ = 0, prev_stop = -1;
      auto count_mono = [&](int64_t pos) -> int64_t {
        int64_t start = prev_stop + 1;
        prev_stop = pos;
        while (c_stop <= pos) {
          start = c_stop;
          c_stop += BLKBITS;
          c_sum += cnt[c_i++];
          count_ = c_sum;
        }
        count_ += popcnt_range(start, pos);
        return count_;
      };

      for (size_t b = static_cast<size_t>(p.k) + 1; b <= bend; ++b) {
        // reset the monotone counter for this b (its erase below mutates bits[])
        c_i = 0, c_sum = 0, c_stop = BLKBITS, count_ = 0, prev_stop = -1;
        const int64_t pb = c.primes[b - 1];
        // floor(x/(pb*k)) = floor(floor(x/pb)/k) for positive integers: compute
        // xpb = floor(x/pb) ONCE per b, then EVERY per-segment bound — and the
        // per-leaf v = x/(pb*m) = xpb/m — is a plain int64 division of a value
        // that already fits int64. The old code divided x by the int128 products
        // pb^3 / pb*high / pb*low, keeping those quotients (and emit's per-leaf
        // x/pbm when x>2^63) on the 128-bit __divti3 path. xpb <= x/p_{k+1} <=
        // x/19 fits int64 for every supported x, so the chained divides below are
        // exact and 128-bit-free. Part of a ~2% full-pi win (with the phi/phi_pi
        // int64 fast paths); it grows with x as nseg*bend setup work does.
        const int64_t xpb = x64 ? x64v / pb : static_cast<int64_t>(x / pb);
        const int64_t mlo = static_cast<int64_t>(p.z / pb);
        const int64_t mhi = std::min<int64_t>(p.z, (xpb / pb) / pb);
        const int64_t ma = std::max<int64_t>(mlo + 1, xpb / high + 1);
        const int64_t mb = std::min<int64_t>(mhi, xpb / low);
        // mp[m] = mu(m)*pmin(m), and 0 for non-squarefree. pmin = |v0| is 0
        // there, so a single `pmin <= pb` test rejects both non-squarefree m AND
        // m with a too-small factor (one fewer branch per m). The pmax<=y test is
        // hoisted OUT of the per-m loop: only z != y needs it, so split into two
        // loops (the common z==y path carries no per-m pmax check at all).
        const int64_t ps = phi_start[b];
        // Accumulate this b's leaves in an int64 local (mu=+/-1 so the leaf is
        // +/-phival), promoted to the int128 D once per b — keeps the hot loop
        // int64 (no per-leaf int128 mul/add). Same trick as a_impl/c_impl.
        int64_t dloc = 0;
        auto emit = [&](int64_t m, int64_t v0) {
          const int mu = v0 < 0 ? -1 : 1;
          // = floor(x/(pb*m)), int64. Routed through fast_div_u so the divl
          // lever (WITH_DIV32) also narrows this D hot-leaf division, like A/C.
          const int64_t v = fast_div_u(static_cast<uint64_t>(xpb),
                                       static_cast<uint64_t>(m));
          const int64_t phival = ps + count_mono(v - low);
          dloc -= static_cast<int64_t>(mu) * phival;
        };
        if (checkpmax) {
          for (int64_t m = mb; m >= ma; --m) {
            const int64_t v0 = c.mpv(m);
            const int64_t pmin = v0 < 0 ? -v0 : v0;
            if (pmin <= pb || c.pmax[m] > p.y)
              continue;
            emit(m, v0);
          }
        } else {
          for (int64_t m = mb; m >= ma; --m) {
            const int64_t v0 = c.mpv(m);
            const int64_t pmin = v0 < 0 ? -v0 : v0;
            if (pmin <= pb)
              continue;
            emit(m, v0);
          }
        }
        D += static_cast<maxint_t>(dloc); // promote once per b
        for (int64_t j = ((low + pb - 1) / pb) * pb; j < high; j += pb)
          erase(j - low);
      }
    }
  }
  return D;
}

// D, WHEEL-30 variant (gourdon.md §7): same algorithm as d_impl_dense, but the
// segment sieve uses the WHEEL-30 byte layout (8 residues coprime to 30 per
// 30-int byte) — it stores only 8/30 of the integers, so crossing p_b touches
// only its coprime-to-30 multiples (~3.75x fewer than the dense bit sieve) and
// counting popcounts ~3.75x fewer bytes. The linear block counter is kept, now
// over bytes. Requires k >= 3 (the first k primes must include 2,3,5, which the
// layout excludes); d_impl dispatches to d_impl_dense otherwise.
// WIDE=false: xpb = x/pb fits int64 (x <= ~1.6e20) — the fast canonical path
// (fast_div_u/divl leaf). WIDE=true: xpb kept int128 (x/pb exceeds INT64_MAX for
// the smallest pb at x>~1.6e20), leaf division int128 — corrects pi(x) at 1e21+
// at ~+2% on D, run ONLY when needed. `if constexpr` => WIDE=false codegen is
// identical to before (perf-neutral on the whole validated <=1e20 regime).
template <bool WIDE>
maxint_t d_impl_wheel(int128_t x, const GCtx& c, int nt) {
  const GParams& p = c.p;
  const int64_t N = static_cast<int64_t>(x / p.z);
  if (N < 1)
    return 0;
  const size_t bend = upper_index(c.primes, p.xstar);
  if (static_cast<size_t>(p.k) >= bend)
    return 0;

  // Segment sizing: same cache rule as d_impl_dense (the wheel working set is
  // now ~S/30 bytes + the byte-block counter, far smaller, so this is safe).
  const long l1d = sysconf(_SC_LEVEL1_DCACHE_SIZE);
  const long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
  int64_t Slog, capnseg;
  if (l2 > 0 && l2 <= 384 * 1024) {
    Slog = 20;
    capnseg = 1024;
  } else {
    const long l1 = (l1d > 0) ? l1d : 48 * 1024;
    Slog = ilog2(8 * l1 / 3);
    capnseg = 2750;
  }
  while ((N >> Slog) > capnseg && Slog < 22)
    ++Slog;
  if (const char* e = std::getenv("PC_DSEGLOG"))
    Slog = std::atoi(e);
  const int64_t S = int64_t(1) << Slog;
  const int64_t NB = S / 30 + 2;             // bytes per segment
  const int64_t LOG_BPB = std::max<int64_t>(1, ilog2(NB) / 2);
  const int64_t BPB = int64_t(1) << LOG_BPB; // bytes per counter block
  const int64_t nseg = (N + S - 1) / S;
  const bool x64 = x <= INT64_MAX;
  const int64_t x64v = x64 ? static_cast<int64_t>(x) : 0;
  const bool checkpmax = (p.z != p.y);
  (void)phi(2, 8); // warm up phi()'s function-local statics before threading

  // Wheel-30 residue tables (8 residues coprime to 30 per 30-int byte).
  static constexpr int W8[8] = {1, 7, 11, 13, 17, 19, 23, 29};
  static constexpr int GAP[8] = {6, 4, 2, 4, 2, 4, 6, 2};
  // GAPDN[s] = step DOWN from residue W8[s] to the previous coprime-to-30
  // residue (W8[s] - W8[(s-1)&7] mod 30). Used by the m-loop to visit only m
  // coprime to 30: every m divisible by 2,3,5 has lpf<=5<7<=pb so it is always
  // rejected (pmin<=pb) — stepping over them is exact and ~3.75x fewer iters.
  static constexpr int GAPDN[8] = {2, 6, 4, 2, 4, 2, 4, 6};
  int8_t slotT[30];
  uint8_t bmT[30], startT[30], endT[30];
  for (int r = 0; r < 30; ++r) {
    slotT[r] = -1;
    bmT[r] = 0;
  }
  for (int s = 0; s < 8; ++s) {
    slotT[W8[s]] = static_cast<int8_t>(s);
    bmT[W8[s]] = static_cast<uint8_t>(1u << s);
  }
  for (int r = 0; r < 30; ++r) { // startT[r]: residues >= r; endT[r]: <= r
    uint8_t sge = 0, sle = 0;
    for (int j = 0; j < 30; ++j)
      if (bmT[j]) {
        if (j >= r)
          sge |= bmT[j];
        if (j <= r)
          sle |= bmT[j];
      }
    startT[r] = sge;
    endT[r] = sle;
  }

  // m-loop wheel (SEPARATE from the mod-30 value-sieve wheel above): the leaf
  // m-scan visits only m coprime to the first min(k,5) primes. Every such skipped
  // m has lpf(m) <= 11 < p_{k+1} <= pb, so it is ALWAYS rejected (pmin <= pb) —
  // skipping it is exact. Capped at primes <= 11 (Wm <= 2310) to keep the tables
  // tiny and x-independent (the full k-wheel modulus explodes with k). vs the
  // mod-30 scan this drops ~22% of the m-steps at 1e18 (k>=5). PC_DMWHEEL=0 forces
  // the old mod-30 scan (oracle).
  int nwp = std::min<int>(p.k, 5); // wheel primes = first nwp of {2,3,5,7,11}
  if (const char* e = std::getenv("PC_DMWHEEL"))
    nwp = std::min<int>(std::max<int>(3, std::atoi(e)), 5); // 3 => mod-30 oracle
  static constexpr int kWP[5] = {2, 3, 5, 7, 11};
  int Wm = 1;
  for (int i = 0; i < nwp; ++i)
    Wm *= kWP[i];
  // FIXED stack arrays (Wm<=2310, Nm=phi(2310)=480): no heap/pointer indirection
  // in the hot step (a std::vector here regressed the step ~3%). Sized to the cap.
  int16_t slotTm[2310];          // residue -> coprime slot, or -1
  int32_t RmTbl[480];            // coprime residues ascending
  int Nm = 0;
  for (int r = 0; r < Wm; ++r) {
    bool cop = true;
    for (int i = 0; i < nwp; ++i)
      if (r % kWP[i] == 0) { cop = false; break; }
    slotTm[r] = cop ? static_cast<int16_t>(Nm) : int16_t(-1);
    if (cop) RmTbl[Nm++] = r;
  }
  int32_t GAPDNm[480]; // step DOWN from slot s to the previous coprime residue
  for (int s = 0; s < Nm; ++s)
    GAPDNm[s] = (s == 0) ? (RmTbl[0] + Wm - RmTbl[Nm - 1])
                         : (RmTbl[s] - RmTbl[s - 1]);

  // Pre-sieve pattern: one period of "coprime to the first k primes" in the
  // wheel-30 byte layout. 2,3,5 are excluded by the layout, so only the primes
  // p_6..p_k = primes[3..k-1] are crossed; period = (p_1..p_k)/30 bytes.
  int64_t patBytes = 1;
  for (int i = 3; i < p.k; ++i)
    patBytes *= c.primes[i];
  std::vector<uint8_t> patk(patBytes + 8, uint8_t(0xFF));
  for (int i = 3; i < p.k; ++i) {
    const int64_t q = c.primes[i];
    for (int64_t n = q; n < 30 * patBytes; n += q) {
      const uint8_t bm = bmT[n % 30];
      if (bm)
        patk[n / 30] &= static_cast<uint8_t>(~bm);
    }
  }

  // Per-prime crossing tables for p_b, b in [k+1, bend] (primes[k..bend-1]): for
  // each wheel state s, DMASK clears p_b's multiple's slot, DBYTE is the byte
  // advance for one wheel step (no %30 / indexed load in the hot crossing).
  const size_t qfirst = static_cast<size_t>(p.k);
  const size_t nq = bend - qfirst;
  std::vector<std::array<int32_t, 8>> DBYTE(nq);
  std::vector<std::array<uint8_t, 8>> DMASK(nq);
  for (size_t i = 0; i < nq; ++i) {
    const int64_t q = c.primes[qfirst + i];
    const int qm = static_cast<int>(q % 30);
    for (int s = 0; s < 8; ++s) {
      const int rho = (qm * W8[s]) % 30;
      DMASK[i][s] = static_cast<uint8_t>(~bmT[rho]);
      DBYTE[i][s] = static_cast<int32_t>((rho + GAP[s] * q) / 30);
    }
  }

  // Adaptive load balancer: threads grab CONTIGUOUS segment ranges from an
  // atomic cursor, so phi_start can be recomputed once per range and CARRIED
  // FORWARD across the range's segments (instead of recomputed every segment,
  // ~9% of D). D's per-segment cost falls steeply with the segment index, so a
  // fixed chunk size piles the heavy early segments onto one thread (measured
  // 2-3x slower). Instead the grab size GROWS with the start index s0:
  // chunk = clamp(s0/LBK + 1, 1, LBMAX) — tiny chunks over the heavy head (fine
  // balance), large chunks over the light tail (recompute amortised). If
  // cost(seg) ~ 1/seg then cost-per-chunk ~ 1/LBK ~ constant.
  // LBK=16 (chunk = s0/16 + 1) measured optimal on a modern 8-core part across
  // 1e15-1e18 (D -6% to -18% vs per-segment); robust at this constant. Re-tune
  // via PC_DLBK on a different core count if needed.
  int64_t LBK = 16;
  if (const char* e = std::getenv("PC_DLBK"))
    LBK = std::max<int64_t>(1, std::atoll(e));
  int64_t LBMAX = 4096;
  if (const char* e = std::getenv("PC_DLBMAX"))
    LBMAX = std::max<int64_t>(1, std::atoll(e));
  std::atomic<int64_t> nextSeg{0};

  // PC_DFACTOR: wheel-2310 compressed factor table for D's leaf scan. The scan then
  // walks CONTIGUOUS coprime indices reading a uint16 array with a single-comparison
  // filter (factor[idx] > pb), instead of wheel-stepping mp3 with a 3-byte decode.
  // Derived from c.mpv so it reproduces the exact (pmin>pb) filter (cap included):
  //   enc = 0                       if not squarefree
  //       = dcap (odd, > x*)        if mu=-1 and lpf > x*  (always a prime here)
  //       = lpf  (odd)              if mu=-1 and lpf <= x*
  //       = lpf-1 (even)            if mu=+1 (then lpf <= sqrt z < x*)
  // so factor[idx] > pb  <=>  lpf > pb (consecutive primes differ by >=2, pb>=19),
  // and mu = (factor[idx] & 1) ? -1 : +1. Only for z==y (!checkpmax, mpf<=y is then
  // automatic) and while dcap fits uint16 (x* < ~65534, x < ~1.8e19); else the mp3
  // wheel scan below is used (still correct, just not accelerated).
  const int64_t dcap = (p.xstar + 2) | 1; // odd, > x*+1; encodes capped primes (mu=-1)
  // DEFAULT ON: the factor-table scan is bit-exact with and ~4x faster than the mp3
  // wheel scan. Auto-falls back to the wheel path when z!=y (PC_AZ) or x*+1 exceeds
  // uint16 (x >~ 1.8e19). PC_NODFACTOR forces the wheel path (oracle).
  const bool usefac =
      std::getenv("PC_NODFACTOR") == nullptr && !checkpmax && dcap < 65536;
  std::vector<uint16_t> dfac;
  if (usefac) {
    const int64_t nidx = dfac_to_index(p.z);
    dfac.assign(static_cast<size_t>(nidx) + 1, 0);
    for (int64_t idx = 0; idx <= nidx; ++idx) {
      const int64_t m = dfac_to_number(idx);
      if (m < 2 || m > p.z)
        continue; // idx 0 => m=1 (never a leaf); guard the tail past z
      const int32_t v = c.mpv(m);
      if (v == 0)
        continue; // not squarefree -> 0 (already)
      const int64_t mag = v < 0 ? -v : v; // |mpv| = min(lpf, x*+1)
      uint16_t enc;
      if (v < 0)                          // mu = -1
        enc = (mag > p.xstar) ? static_cast<uint16_t>(dcap)  // capped prime (lpf>x*)
                              : static_cast<uint16_t>(mag);  // lpf (odd) <= x*
      else                                // mu = +1 (lpf <= sqrt z < x*, never capped)
        enc = static_cast<uint16_t>(mag - 1);
      dfac[idx] = enc;
    }
  }

  maxint_t D = 0;
  int64_t dlocmax = 0; // max |per-b int64 accumulator|, to confirm no overflow
  // Opt-in profiling (PC_DPROF): per-b time split fill / leaf-loop+flush / wheel
  // crossing, to locate D's hot phase. Per-b clock reads, negligible overhead.
  const bool dprof = std::getenv("PC_DPROF") != nullptr;
  double pf_d_fill = 0, pf_d_leaf = 0, pf_d_cross = 0, pf_d_flush = 0;
  double pf_d_div = 0, pf_d_cnt = 0;
  int64_t pf_d_nleaf = 0, pf_d_nsurv = 0, pf_d_cntb = 0;
#pragma omp parallel num_threads(nt) reduction(i128add : D) reduction(max : dlocmax) \
    reduction(+ : pf_d_fill, pf_d_leaf, pf_d_cross, pf_d_flush, pf_d_div, pf_d_cnt, pf_d_nleaf, pf_d_nsurv, pf_d_cntb)
  {
    auto dclk = [] { return std::chrono::steady_clock::now(); };
    auto dms = [](std::chrono::steady_clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };
    std::vector<uint8_t> sb(NB + 32);
    std::vector<int32_t> cnt((NB >> LOG_BPB) + 2); // byte-block popcounts
    std::vector<int64_t> phi_start(bend + 1);
    // Branchless survivor buffer for the m-loop : the hot
    // scan over [ma,mb] APPENDS surviving m branchlessly, then processes them in
    // batches — all divisions first (independent => pipelined), then the counts.
    // This kills the per-leaf reject branch (~19% of D, line "if pmin<=pb") and
    // exposes ILP across the leaf divisions. MBUF chunk fits L1.
    constexpr int64_t MBUF = 256;
    std::vector<int64_t> mbuf(MBUF);  // surviving m, sign carries mu
    std::vector<int64_t> vbuf(MBUF);  // batched xpb/m

    for (;;) {
      // Grab the next contiguous range [segLo, segHi) of segments.
      int64_t segLo = nextSeg.load(std::memory_order_relaxed);
      int64_t segHi;
      bool got = false;
      while (segLo < nseg) {
        const int64_t chunk =
            std::min<int64_t>(LBMAX, std::max<int64_t>(1, segLo / LBK + 1));
        segHi = std::min<int64_t>(segLo + chunk, nseg);
        if (nextSeg.compare_exchange_weak(segLo, segHi,
                                          std::memory_order_relaxed)) {
          got = true;
          break;
        }
      }
      if (!got)
        break;

      // phi_start[b] = phi(low0-1, b-1), recomputed ONCE at the range start;
      // carried forward across the range's segments below.
      {
        const int64_t L0 = segLo * S; // (1 + segLo*S) - 1
        int64_t prev = static_cast<int64_t>(phi(L0, p.k));
        for (size_t b = static_cast<size_t>(p.k) + 1; b <= bend; ++b) {
          phi_start[b] = prev;
          prev -= phi_pi(L0 / c.primes[b - 1], static_cast<int>(b) - 1, c.primes,
                         c.pi);
        }
      }

      for (int64_t seg = segLo; seg < segHi; ++seg) {
      const int64_t low = 1 + seg * S;
      const int64_t high = std::min<int64_t>(low + S, N + 1);
      std::chrono::steady_clock::time_point tfill0;
      if (dprof) tfill0 = dclk();

      // Fill the byte sieve from the periodic pre-sieve pattern (coprime to the
      // first k primes), then mask integers < low and >= high. Byte bi holds the
      // 8 coprime-to-30 residues of integers around 30*(WLO+bi).
      const int64_t WLO = low / 30;
      const int64_t whi = (high - 1) / 30;
      const int64_t nbytes = whi - WLO + 1;
      int64_t phase = WLO % patBytes;
      int64_t done = std::min<int64_t>(nbytes, patBytes - phase);
      std::memcpy(sb.data(), patk.data() + phase, done);
      while (done < nbytes) {
        const int64_t chunk = std::min<int64_t>(nbytes - done, patBytes);
        std::memcpy(sb.data() + done, patk.data(), chunk);
        done += chunk;
      }
      sb[0] &= startT[low % 30];
      sb[nbytes - 1] &= endT[(high - 1) % 30];

      // cnt[j] = #set bits in byte-block j (BPB bytes).
      const int64_t nblk = (nbytes + BPB - 1) >> LOG_BPB;
      std::fill(cnt.begin(), cnt.begin() + nblk, 0);
      int64_t total = 0; // #set bits in the segment at the current b-1 state
      for (int64_t bi = 0; bi < nbytes; ++bi) {
        const int pc = std::popcount(sb[bi]);
        cnt[bi >> LOG_BPB] += pc;
        total += pc;
      }

      // Monotone byte-prefix: prefix_bytes(wb) = #set bits in bytes [0, wb), wb
      // ASCENDING within a b. Whole blocks absorbed from cnt[] (O(1)); the
      // remainder popcounted byte-by-byte since the last query. Reset per b.
      int64_t cb_i = 0, cb_sum = 0, cb_stop = 0, cb_count = 0, cb_prev = 0;
      auto prefix_bytes = [&](int64_t wb) -> int64_t {
        while (cb_stop <= wb) {
          cb_sum += cnt[cb_i++];
          cb_count = cb_sum;
          cb_prev = cb_stop;
          cb_stop += BPB;
        }
        if (dprof) pf_d_cntb += (wb - cb_prev);
        cb_count += popcount_bytes(sb.data() + cb_prev, wb - cb_prev);
        cb_prev = wb;
        return cb_count;
      };
      // count_le(v) = #set bits among integers in [low, v] = phi(v,b-1) - ps.
      auto count_le = [&](int64_t v) -> int64_t {
        const int64_t wb = v / 30 - WLO;
        return prefix_bytes(wb) +
               std::popcount(static_cast<uint8_t>(sb[wb] & endT[v % 30]));
      };

      if (dprof) pf_d_fill += dms(dclk() - tfill0);
      for (size_t b = static_cast<size_t>(p.k) + 1; b <= bend; ++b) {
        std::chrono::steady_clock::time_point tleaf0;
        if (dprof) tleaf0 = dclk();
        cb_i = 0, cb_sum = 0, cb_stop = BPB, cb_count = 0, cb_prev = 0; // reset
        const int64_t pb = c.primes[b - 1];
        // xpb = floor(x/pb). int128 only when WIDE (x>~1.6e20); else int64 (fast).
        using XPBT = std::conditional_t<WIDE, int128_t, int64_t>;
        XPBT xpb;
        if constexpr (WIDE)
          xpb = x / pb;
        else
          xpb = x64 ? x64v / pb : static_cast<int64_t>(x / pb);
        const int64_t mlo = static_cast<int64_t>(p.z / pb);
        // mhi/ma/mb fit int64 (<= z); cast is a no-op when xpb is already int64.
        const int64_t mhi =
            std::min<int64_t>(p.z, static_cast<int64_t>((xpb / pb) / pb));
        const int64_t ma =
            std::max<int64_t>(mlo + 1, static_cast<int64_t>(xpb / high) + 1);
        const int64_t mb = std::min<int64_t>(mhi, static_cast<int64_t>(xpb / low));
        const int64_t ps = phi_start[b];
        // int64 local accumulator (mu=+/-1 => leaf is +/-phival), promoted to the
        // int128 D once per b — keeps the hot loop int64 (no per-leaf int128
        // mul/add). Same trick as a_impl/c_impl. phival = phi(v,b-1) can be much
        // larger than A's pi(sqrt x), so dloc's magnitude is tracked (dlocmax) to
        // confirm it stays well under 2^63 (see PC_DMAX print + overflow guard).
        int64_t dloc = 0;
        int64_t bn = 0; // buffered survivor count
        // Process a full chunk of survivors: divisions first (pipelined), then
        // counts (count_le's monotone cursor advances since v0..v(bn-1) ascend:
        // survivors are appended in m-DESCENDING order => xpb/m ASCENDING).
        // Survivors are stored as signed m: sign bit carries mu (m>0 always), so
        // the hot scan does a SINGLE store per iteration (no separate mu buffer).
        // mbuf holds either signed m (wheel path: sign=mu) or a coprime INDEX
        // (factor path: m=to_number(idx), mu from dfac[idx]&1). Decode per path;
        // `usefac` is loop-invariant so the branch predicts away.
        auto flush = [&]() {
          std::chrono::steady_clock::time_point tfl0, tfl1;
          if (dprof) { tfl0 = dclk(); pf_d_nsurv += bn; }
          for (int64_t i = 0; i < bn; ++i) {
            const int64_t sm = mbuf[i];
            const int64_t mm = usefac ? dfac_to_number(sm) : (sm < 0 ? -sm : sm);
            if constexpr (WIDE)
              vbuf[i] = static_cast<int64_t>(xpb / mm); // int128 div (rare)
            else
              vbuf[i] = fast_div_u(static_cast<uint64_t>(xpb),
                                   static_cast<uint64_t>(mm));
          }
          if (dprof) { tfl1 = dclk(); pf_d_div += dms(tfl1 - tfl0); }
          for (int64_t i = 0; i < bn; ++i) {
            const int mu = usefac ? ((dfac[mbuf[i]] & 1) ? -1 : 1)
                                  : (mbuf[i] < 0 ? -1 : 1);
            const int64_t phival = ps + count_le(vbuf[i]);
            dloc -= static_cast<int64_t>(mu) * phival;
          }
          bn = 0;
          if (dprof) { pf_d_cnt += dms(dclk() - tfl1); pf_d_flush += dms(dclk() - tfl0); }
        };
        if (usefac) {
          // FACTOR path: walk CONTIGUOUS coprime indices [lo_idx, hi_idx] (= the
          // coprime m in [ma, mb]) reading dfac, single-comparison branchless append.
          const int64_t hi_idx = dfac_to_index(mb);
          const int64_t lo_idx = dfac_to_index(ma - 1) + 1; // first coprime >= ma
          for (int64_t idx = hi_idx; idx >= lo_idx; --idx) {
            mbuf[bn] = idx;                  // store the index; decode in flush
            bn += (dfac[idx] > pb);          // factor[idx] > pb  <=>  lpf > pb
            if (bn >= MBUF - 1)
              flush();
          }
        } else {
          // WHEEL path (oracle): step the mod-Wm wheel DOWN over m in [ma, mb].
          int64_t m = mb;
          int sm = slotTm[static_cast<int>(m % Wm)];
          while (sm < 0 && m >= ma) { // align to a coprime-to-Wm residue
            --m;
            sm = slotTm[static_cast<int>(((m % Wm) + Wm) % Wm)];
          }
          if (checkpmax) {
            for (; m >= ma; m -= GAPDNm[sm], sm = (sm == 0 ? Nm - 1 : sm - 1)) {
              const int64_t v0 = c.mpv(m);
              const int64_t pmin = v0 < 0 ? -v0 : v0;
              mbuf[bn] = (v0 < 0) ? -m : m; // sign = mu, magnitude = m
              bn += (pmin > pb) & (c.pmax[m] <= p.y); // branchless append
              if (bn >= MBUF - 1)
                flush();
            }
          } else {
            for (; m >= ma; m -= GAPDNm[sm], sm = (sm == 0 ? Nm - 1 : sm - 1)) {
              const int64_t v0 = c.mpv(m);
              const int64_t pmin = v0 < 0 ? -v0 : v0;
              mbuf[bn] = (v0 < 0) ? -m : m; // sign = mu, magnitude = m
              bn += (pmin > pb); // branchless append
              if (bn >= MBUF - 1)
                flush();
            }
          }
        }
        flush();
        std::chrono::steady_clock::time_point tcross0;
        if (dprof) { pf_d_leaf += dms(dclk() - tleaf0); if (mb >= ma) pf_d_nleaf += (mb - ma + 1); tcross0 = dclk(); }
        D += static_cast<maxint_t>(dloc); // promote once per b
        { const int64_t aa = dloc < 0 ? -dloc : dloc; if (aa > dlocmax) dlocmax = aa; }
        // Carry phi_start forward: phi(high-1,b-1) = phi(low-1,b-1) + #set in the
        // segment at the (b-1) state (= total, before crossing p_b). After `ps`
        // is consumed, before the cross below.
        phi_start[b] = ps + total;
        // Cross p_b's coprime-to-30 multiples; branchless counter decrement
        // (popcount of before^after is 0 or 1, so no per-step branch).
        const std::array<int32_t, 8>& db = DBYTE[b - 1 - p.k];
        const std::array<uint8_t, 8>& dm = DMASK[b - 1 - p.k];
        int64_t c0 = (low + pb - 1) / pb; // ceil(low/pb)
        int rr = static_cast<int>(c0 % 30);
        while (slotT[rr] < 0) { // advance to a coprime-to-30 multiplier
          ++c0;
          rr = (rr + 1 == 30) ? 0 : rr + 1;
        }
        int64_t lb = (pb * c0) / 30 - WLO;
        int s = slotT[rr];
        // 8-way unrolled wheel crossing (ref D's structure): hoist the per-slot
        // advance/mask into locals with COMPILE-TIME indices so each step's
        // `lb += dbN` has no dependency on a running slot var (breaks the
        // s->db[s]->lb chain) and drops the `s=(s+1)&7` update + dynamic index.
        // switch(s) enters at the right slot; the for(;;) wraps slot 7->0.
        const int32_t db0 = db[0], db1 = db[1], db2 = db[2], db3 = db[3];
        const int32_t db4 = db[4], db5 = db[5], db6 = db[6], db7 = db[7];
        const uint8_t dm0 = dm[0], dm1 = dm[1], dm2 = dm[2], dm3 = dm[3];
        const uint8_t dm4 = dm[4], dm5 = dm[5], dm6 = dm[6], dm7 = dm[7];
#define DXOFF(J)                                                               \
  do {                                                                         \
    if (lb >= nbytes)                                                          \
      goto cross_done;                                                         \
    const uint8_t before = sb[lb];                                             \
    const uint8_t after = before & dm##J;                                      \
    sb[lb] = after;                                                            \
    const int32_t cleared = static_cast<int32_t>(before != after);            \
    cnt[lb >> LOG_BPB] -= cleared;                                             \
    total -= cleared;                                                          \
    lb += db##J;                                                               \
  } while (0)
        switch (s) {
          for (;;) {
          case 0: DXOFF(0); [[fallthrough]];
          case 1: DXOFF(1); [[fallthrough]];
          case 2: DXOFF(2); [[fallthrough]];
          case 3: DXOFF(3); [[fallthrough]];
          case 4: DXOFF(4); [[fallthrough]];
          case 5: DXOFF(5); [[fallthrough]];
          case 6: DXOFF(6); [[fallthrough]];
          case 7: DXOFF(7);
          }
        }
      cross_done:;
#undef DXOFF
        if (dprof) pf_d_cross += dms(dclk() - tcross0);
      } // b loop
      } // seg loop
    } // range loop (for(;;))
  }
  // Overflow guard diagnostic: max |per-b int64 accumulator| vs 2^63 (set PC_DMAX).
  // phi(v,b-1) >> pi(sqrt x), so we confirm headroom rather than assume it.
  if (std::getenv("PC_DMAX"))
    std::fprintf(stderr, "[D] max|dloc|=%lld  (2^63=%lld, ratio=%.2e)\n",
                 static_cast<long long>(dlocmax),
                 static_cast<long long>(INT64_MAX),
                 static_cast<double>(dlocmax) / 9.223372036854776e18);
  if (dprof) {
    const double tot = pf_d_fill + pf_d_leaf + pf_d_cross;
    std::fprintf(stderr,
        "[DPROF] thread-ms: fill=%.0f(%.1f%%) leaf+flush=%.0f(%.1f%%) "
        "cross=%.0f(%.1f%%)\n"
        "  within leaf+flush: scan=%.0f flush=%.0f (div=%.0f count_le=%.0f) | "
        "m-steps=%.3eG survivors=%.3eG (%.1f%% pass) cntbytes=%.3eG (%.1f B/surv)\n",
        pf_d_fill, tot ? 100 * pf_d_fill / tot : 0, pf_d_leaf,
        tot ? 100 * pf_d_leaf / tot : 0, pf_d_cross,
        tot ? 100 * pf_d_cross / tot : 0, pf_d_leaf - pf_d_flush, pf_d_flush,
        pf_d_div, pf_d_cnt,
        pf_d_nleaf / 1e9, pf_d_nsurv / 1e9,
        pf_d_nleaf ? 100.0 * pf_d_nsurv / pf_d_nleaf : 0,
        pf_d_cntb / 1e9, pf_d_nsurv ? (double)pf_d_cntb / pf_d_nsurv : 0);
  }
  return D;
}

// D dispatcher: wheel-30 needs the first k primes to include 2,3,5 (k >= 3);
// tiny x with k < 3 falls back to the dense bit sieve.
maxint_t d_impl(int128_t x, const GCtx& c, int nt) {
  if (c.p.k < 3)
    return d_impl_dense(x, c, nt);
  // The largest xpb is x/pb for the smallest pb = primes[k]. If it overflows
  // int64 (only for x > ~1.6e20) take the int128 (WIDE) path, else the fast one.
  const bool wide = (x / c.primes[c.p.k]) > static_cast<int128_t>(INT64_MAX);
  return wide ? d_impl_wheel<true>(x, c, nt) : d_impl_wheel<false>(x, c, nt);
}

} // namespace

maxint_t g_Phi0(int128_t x, int t) {
  return x < 1 ? 0 : phi0_impl(x, build_ctx(x, resolve_threads(t)));
}
maxint_t g_Sigma(int128_t x, int t) {
  return x < 2 ? 0 : sigma_impl(x, build_ctx(x, resolve_threads(t)));
}
maxint_t g_A(int128_t x, int t) {
  const int nt = resolve_threads(t);
  return x < 2 ? 0 : a_impl(x, build_ctx(x, nt), nt);
}
maxint_t g_B(int128_t x, int t) {
  const int nt = resolve_threads(t);
  return x < 4 ? 0 : b_impl(x, build_ctx(x, nt), nt);
}

maxint_t g_C(int128_t x, int t) {
  const int nt = resolve_threads(t);
  return x < 2 ? 0 : c_impl(x, build_ctx(x, nt), nt);
}

maxint_t g_D(int128_t x, int t) {
  const int nt = resolve_threads(t);
  return x < 2 ? 0 : d_impl(x, build_ctx(x, nt), nt);
}

// Full assembly — builds the shared context ONCE for all six terms.
// PHASED assembly (opt-in PC_PHASE): never holds the full PiTable and mp[] at the
// same time. Phase 1 builds the PiTable (no mp) and runs the five terms that need
// pi — A, C, Sigma, B, Phi0 — then frees it; phase 2 builds mp[] (+ a small pi for
// phi_pi) and runs D. Peak RSS = max(PiTable, mp) + transients instead of their sum
// (~0.30*sqrt x -> ~0.20*sqrt x), at the cost of D's phi_pi recursing more (PC_DPILIM
// tunes phase-2's pi limit). Result is identical to g_pi (same terms, same values).
maxint_t pi_gourdon_phased(int128_t x, int nt) {
  GParams p = make_gparams(x);
  // Phase 2's pi-table limit: small by default (phase-2 peak ~= mp[] alone). Larger
  // values speed up D's phi_pi but raise phase-2 peak toward the full table.
  int64_t d_pi_limit = p.x13;
  if (const char* e = std::getenv("PC_DPILIM"))
    d_pi_limit = std::atoll(e);

  maxint_t a, c_term, sigma, b, phi0;
  {
    GCtx c1 = build_ctx_pi_only(x, nt); // PiTable live, no mp[]
    a = a_impl(x, c1, nt);
    c_term = c_impl(x, c1, nt);
    sigma = sigma_impl(x, c1);
    b = b_impl(x, c1, nt);
    phi0 = phi0_impl(x, c1);
  } // c1 (PiTable) freed here
  maxint_t d;
  {
    GCtx c2 = build_ctx_d_only(x, nt, d_pi_limit); // mp[] live, small pi
    d = d_impl(x, c2, nt);
  } // c2 (mp[]) freed here
  return a - b + c_term + d + phi0 + sigma;
}

// LOW-MEMORY assembly: ONE unified context with NO O(sqrt x) PiTable. A and C use
// the segmented sweeps (a_impl_seg / c_impl_seg), B sieves its own primes, Sigma
// uses a small table + the pi(sqrt x) scalar. Peak RSS ~ mp[] = O(x^(1/3)), so it
// grows ~x^(1/3) (not ~sqrt x). Numerically identical to g_pi (same six terms).
// FUSED segmented sweep over the leaf-value domain [2, Vmax]: computes A
// (gourdon.md §5), C (§6) and pi(sqrt x) (Sigma0 / B base) in ONE pass, instead of
// re-sieving [2, sqrt x] three times (a_impl_seg + c_impl_seg + pi_count_seg). Each
// segment is sieved once; A's and C's leaves landing in it are folded with
// pi(v) = running + local_count(v), and the running prime count yields pi(sqrt x).
// Same values as a_impl_seg / c_impl_seg / pi_count_seg (their inner bodies, merged).
void ac_pi_sweep(int128_t x, const GCtx& c, int nt, maxint_t& A_out,
                 maxint_t& C_out, int128_t& pisqrtx_out) {
  const GParams& p = c.p;
  const int64_t y = p.y;
  const int64_t sqrtx = p.sqrtx;
  const size_t a_lo = upper_index(c.primes, p.xstar);    // A: pb > x*
  const size_t a_hi = upper_index(c.primes, p.x13);      //    pb <= x^(1/3)
  const bool haveA = a_lo < a_hi;
  const int64_t a_pbmin = haveA ? c.primes[a_lo] : 0;
  const int64_t Vmax_A =
      haveA ? divx(x, static_cast<int128_t>(a_pbmin) * a_pbmin) : 0;
  const size_t c_lo = static_cast<size_t>(p.k);          // C: pb > p_k
  const size_t c_hi = upper_index(c.primes, p.xstar);    //    pb <= x*
  const bool haveC = c_lo < c_hi;
  const int64_t Vmax_C =
      static_cast<int64_t>(static_cast<int128_t>(p.xstar) * p.xstar); // v < x*^2
  const bool checkpmax = (p.z != p.y);

  const int64_t Vmax = std::max<int64_t>({Vmax_A, haveC ? Vmax_C : 0, sqrtx});
  if (Vmax < 2) {
    A_out = 0;
    C_out = 0;
    pisqrtx_out = 0;
    return;
  }

  int64_t Slog = 20;
  if (const char* e = std::getenv("PC_ACSEGLOG"))
    Slog = std::atoi(e);
  const int64_t S = int64_t(1) << Slog;
  const int64_t V0 = 2;
  const int64_t nseg = (Vmax - V0) / S + 1;
  const size_t nsieve = upper_index(c.primes, isqrt(Vmax));

  std::vector<int64_t> seg_pcount(nseg, 0), seg_pi_lo(nseg, 0);
  std::vector<int64_t> segA_part(nseg, 0), segA_wsum(nseg, 0);
  std::vector<maxint_t> segC_part(nseg, 0), segC_csum(nseg, 0),
      segC_cbsum(nseg, 0);

  // Opt-in profiling (PC_LMPROF): fill-ratio of C's dense m-scan and a coarse
  // sieve/A/C time split. Bounds the theoretical speedup of a sparse C rewrite.
  const bool prof = std::getenv("PC_LMPROF") != nullptr;
  // PC_LMPROF_NOEMIT: skip the lc_at/divx emit work (value becomes WRONG) to
  // isolate the pure scan+decode+reject cost. Diff vs full run = emit cost.
  const bool noemit = std::getenv("PC_LMPROF_NOEMIT") != nullptr;
  // C is computed by the sparse c_impl_sparse by DEFAULT; the sweep then yields
  // A + pi(sqrt x) only (C_out = 0). PC_NOCSPARSE reverts to the dense fused C
  // (the bit-exact oracle).
  const bool skipC = std::getenv("PC_NOCSPARSE") == nullptr;
  int64_t pf_c_total = 0, pf_c_sf = 0, pf_c_emit = 0;
  int64_t pf_v_le = 0, pf_v_gt = 0, pf_v_gt_pbhi = 0;
  double pf_t_sieve = 0, pf_t_A = 0, pf_t_C = 0;
  const int64_t pf_z = p.z;
  const int64_t pf_pbmid = p.xstar / 2; // "large pb" = pb > x*/2

#pragma omp parallel num_threads(nt) reduction(+:pf_c_total,pf_c_sf,pf_c_emit,pf_v_le,pf_v_gt,pf_v_gt_pbhi,pf_t_sieve,pf_t_A,pf_t_C)
  {
    const int64_t NW = (S + 63) / 64;
    std::vector<uint64_t> bits(NW + 1);
    std::vector<int64_t> wpref(NW + 1);
    auto pclk = [] { return std::chrono::steady_clock::now(); };
    auto pms = [](std::chrono::steady_clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };
#pragma omp for schedule(dynamic)
    for (int64_t seg = 0; seg < nseg; ++seg) {
      const int64_t low = V0 + seg * S;
      const int64_t high = std::min<int64_t>(low + S, Vmax + 1);
      const int64_t len = high - low;
      const int64_t nw = (len + 63) / 64;
      auto ts0 = pclk();
      std::fill(bits.begin(), bits.begin() + nw, ~uint64_t(0));
      if (len & 63)
        bits[nw - 1] &= (uint64_t(1) << (len & 63)) - 1;
      for (size_t si = 0; si < nsieve; ++si) {
        const int64_t pp = c.primes[si];
        if (pp * pp >= high)
          break;
        int64_t k = (low + pp - 1) / pp;
        if (k < 2)
          k = 2;
        for (int64_t m = pp * k; m < high; m += pp) {
          const int64_t pos = m - low;
          bits[pos >> 6] &= ~(uint64_t(1) << (pos & 63));
        }
      }
      int64_t run = 0;
      for (int64_t w = 0; w < nw; ++w) {
        wpref[w] = run;
        run += std::popcount(bits[w]);
      }
      seg_pcount[seg] = run;
      if (prof) pf_t_sieve += pms(pclk() - ts0);

      auto lc_at = [&](int64_t v) -> int64_t {
        const int64_t pos = v - low;
        const unsigned bit = static_cast<unsigned>(pos & 63);
        const uint64_t mask =
            (bit == 63) ? ~uint64_t(0) : ((uint64_t(1) << (bit + 1)) - 1);
        return wpref[pos >> 6] + std::popcount(bits[pos >> 6] & mask);
      };

      // pi(sqrt x): primes in [low, min(high-1, sqrt x)] (at most one straddling seg).
      if (low <= sqrtx)
        seg_pi_lo[seg] = (high - 1 <= sqrtx) ? run : lc_at(sqrtx);

      // --- A emitter (gourdon.md §5) ---
      auto tA0 = pclk();
      if (haveA && low <= Vmax_A) {
        size_t ib_end = std::min(a_hi, upper_index(c.primes, isqrt(x / low)));
        const int64_t pblo = divx(x, static_cast<int128_t>(high) * high);
        size_t ib_start = std::max(a_lo, upper_index(c.primes, pblo));
        if (ib_start > a_lo)
          --ib_start;
        int64_t part = 0, wsum = 0;
        for (size_t ib = ib_start; ib < ib_end; ++ib) {
          const int64_t pb = c.primes[ib];
          const int128_t xp = x / pb;
          const int64_t s = isqrt(xp);
          const int64_t thresh = divx(xp, y);
          int64_t q_hi = divx(x, static_cast<int128_t>(pb) * low);
          int64_t q_lo = divx(x, static_cast<int128_t>(pb) * high) + 1;
          if (q_lo <= pb)
            q_lo = pb + 1;
          if (q_hi > s)
            q_hi = s;
          if (q_hi < q_lo)
            continue;
          for (size_t qi = upper_index(c.primes, q_lo - 1);
               qi < c.primes.size(); ++qi) {
            const int64_t q = c.primes[qi];
            if (q > q_hi)
              break;
            const int64_t lc = lc_at(divx(xp, q));
            const int64_t w = (q <= thresh) ? 1 : 2;
            part += w * lc;
            wsum += w;
          }
        }
        segA_part[seg] = part;
        segA_wsum[seg] = wsum;
      }
      if (prof) pf_t_A += pms(pclk() - tA0);

      // --- C emitter (gourdon.md §6) ---
      auto tC0 = pclk();
      if (haveC && !skipC && low <= Vmax_C) {
        size_t ib_end = std::min(c_hi, upper_index(c.primes, high - 1));
        size_t ib_start = std::max(c_lo, upper_index(c.primes, isqrt(low)));
        if (ib_start > c_lo)
          --ib_start;
        maxint_t part = 0, csum = 0, cbsum = 0;
        for (size_t ib = ib_start; ib < ib_end; ++ib) {
          const int64_t pb = c.primes[ib];
          const int b = static_cast<int>(ib) + 1;
          const int128_t pb2 = static_cast<int128_t>(pb) * pb;
          const int128_t pb3 = pb2 * pb;
          const int64_t mlo =
              std::max(static_cast<int64_t>(p.z / pb), divx(x, pb3));
          const int64_t mhi = std::min<int64_t>(p.z, divx(x, pb2));
          if (mhi <= mlo)
            continue;
          int64_t m_hi = std::min(mhi, divx(x, static_cast<int128_t>(pb) * low));
          int64_t m_lo =
              std::max(mlo + 1, divx(x, static_cast<int128_t>(pb) * high) + 1);
          if (m_hi < m_lo)
            continue;
          const int64_t twob = 2 - b;
          int64_t part_pb = 0, csum_pb = 0, cbsum_pb = 0;
          if (prof) pf_c_total += (m_hi - m_lo + 1);
          for (int64_t m = m_lo; m <= m_hi; ++m) {
            const int32_t mv = c.mpv(m);
            if (mv == 0)
              continue;
            if (prof) ++pf_c_sf;
            const int64_t pmin = mv < 0 ? -mv : mv;
            if (pmin <= pb)
              continue;
            if (checkpmax && c.pmax[m] > y)
              continue;
            if (prof) ++pf_c_emit;
            const int64_t vv = divx(x, static_cast<int128_t>(pb) * m);
            if (prof) {
              if (vv <= pf_z) ++pf_v_le;
              else { ++pf_v_gt; if (pb > pf_pbmid) ++pf_v_gt_pbhi; }
            }
            const int64_t lc = noemit ? 0 : lc_at(vv);
            const int64_t coeff = (mv < 0) ? 1 : -1; // -mu(m)
            part_pb += coeff * lc;
            csum_pb += coeff;
            cbsum_pb += coeff * twob;
          }
          part += static_cast<maxint_t>(part_pb);
          csum += static_cast<maxint_t>(csum_pb);
          cbsum += static_cast<maxint_t>(cbsum_pb);
        }
        segC_part[seg] = part;
        segC_csum[seg] = csum;
        segC_cbsum[seg] = cbsum;
      }
      if (prof) pf_t_C += pms(pclk() - tC0);
    }
  }

  if (prof) {
    const double tot = pf_t_sieve + pf_t_A + pf_t_C;
    std::fprintf(stderr,
        "[LMPROF ac_sweep] thread-ms: sieve=%.0f A=%.0f C=%.0f (sum=%.0f)\n"
        "  C dense-scan fill: m_total=%lld squarefree=%lld(%.1f%%) emit=%lld(%.2f%%)\n"
        "  C emit v-split: v<=z=%lld(%.1f%%) v>z=%lld(%.1f%%) [of v>z: pb>x*/2 = %lld(%.1f%%)]\n",
        pf_t_sieve, pf_t_A, pf_t_C, tot,
        (long long)pf_c_total, (long long)pf_c_sf,
        pf_c_total ? 100.0 * pf_c_sf / pf_c_total : 0.0,
        (long long)pf_c_emit,
        pf_c_total ? 100.0 * pf_c_emit / pf_c_total : 0.0,
        (long long)pf_v_le, pf_c_emit ? 100.0 * pf_v_le / pf_c_emit : 0.0,
        (long long)pf_v_gt, pf_c_emit ? 100.0 * pf_v_gt / pf_c_emit : 0.0,
        (long long)pf_v_gt_pbhi,
        pf_v_gt ? 100.0 * pf_v_gt_pbhi / pf_v_gt : 0.0);
  }

  // Sequential folds sharing one running prime count (pi(low-1)).
  maxint_t A = 0, C = 0;
  int64_t running = 0, pisx = 0;
  for (int64_t seg = 0; seg < nseg; ++seg) {
    A += static_cast<maxint_t>(segA_part[seg]) +
         static_cast<int128_t>(running) * segA_wsum[seg];
    C += segC_part[seg] + static_cast<int128_t>(running) * segC_csum[seg] +
         segC_cbsum[seg];
    pisx += seg_pi_lo[seg];
    running += seg_pcount[seg];
  }
  A_out = A;
  C_out = C;
  pisqrtx_out = pisx;
}

maxint_t pi_gourdon_lowmem(int128_t x, int nt) {
  GParams p = make_gparams(x);
  int64_t d_pi_limit = p.x13;
  if (const char* e = std::getenv("PC_DPILIM"))
    d_pi_limit = std::atoll(e);
  const bool prof = std::getenv("PC_LMPROF") != nullptr;
  using pclk = std::chrono::steady_clock;
  auto pms = [](pclk::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
  };
  if (prof)
    std::fprintf(stderr,
        "[LMPROF params] x13=%lld xstar=%lld z=%lld y=%lld sqrtx=%lld k=%lld\n",
        (long long)p.x13, (long long)p.xstar, (long long)p.z, (long long)p.y,
        (long long)p.sqrtx, (long long)p.k);
  auto t0 = pclk::now();
  GCtx c = build_ctx_lowmem(x, nt, d_pi_limit);
  auto t1 = pclk::now();
  maxint_t a, c_term;
  int128_t pisqrtx;
  ac_pi_sweep(x, c, nt, a, c_term, pisqrtx); // A (+C if PC_NOCSPARSE) + pi(sqrt x)
  if (std::getenv("PC_NOCSPARSE") == nullptr) // default: sweep gave C=0, do it sparsely
    c_term = c_impl_sparse(x, c, nt);
  auto t2 = pclk::now();
  const maxint_t sigma = sigma_impl(x, c, pisqrtx);
  auto t3 = pclk::now();
  const maxint_t b = b_impl_wheel(x, c, nt, pisqrtx);
  auto t4 = pclk::now();
  const maxint_t phi0 = phi0_impl(x, c);
  auto t5 = pclk::now();
  const maxint_t d = d_impl(x, c, nt);
  auto t6 = pclk::now();
  if (prof)
    std::fprintf(stderr,
        "[LMPROF lowmem] wall-ms: build=%.0f AC=%.0f sigma=%.0f B=%.0f "
        "phi0=%.0f D=%.0f | total=%.0f\n",
        pms(t1 - t0), pms(t2 - t1), pms(t3 - t2), pms(t4 - t3),
        pms(t5 - t4), pms(t6 - t5), pms(t6 - t0));
  return a - b + c_term + d + phi0 + sigma;
}

maxint_t g_pi(int128_t x, int t) {
  if (x < 2)
    return 0;
  // Tiny-x fast path (see kSieveCutoff in gourdon.hpp): an exact segmented sieve
  // beats Gourdon's ~2 ms setup floor below 1e6 and avoids the degenerate-params
  // regime. Placed in g_pi so every caller (CLI, tests, verbose) shares it.
  if (x < kSieveCutoff)
    return count_primes(static_cast<int64_t>(x));
  const int nt = resolve_threads(t);
  // LOW-MEM build (CLI --ram / --auto, or PC_LOWMEM): no O(sqrt x) PiTable, packed
  // 24-bit mp[]; peak RSS ~ O(x^(1/3)) (slope sqrt x -> x^(1/3)). With the SPARSE C
  // (default here; see c_impl_sparse) it costs only ~1.2x time vs the default path
  // at 1e18 (was ~6x with the dense m-scan). Opt-in so the default (perf) path stays
  // the fastest; prefer this once RAM is the binding constraint.
  if (std::getenv("PC_LOWMEM"))
    return pi_gourdon_lowmem(x, nt);
  // Phased build is the DEFAULT: the full PiTable and mp[] never coexist, cutting
  // peak RSS ~42% (1675->984 MiB @1e20) for ~0% time. PC_NOPHASE reverts to the
  // legacy single-context build (PiTable + mp[] live together) for comparison.
  if (!std::getenv("PC_NOPHASE"))
    return pi_gourdon_phased(x, nt);
  GCtx c = build_ctx(x, nt);
  return a_impl(x, c, nt) - b_impl(x, c, nt) + c_impl(x, c, nt) + d_impl(x, c, nt) +
         phi0_impl(x, c) + sigma_impl(x, c);
}

// Same as g_pi but builds the shared context ONCE and times each term against it
// (honest per-term breakdown for --verbose). pi == g_pi exactly.
PiBreakdown pi_gourdon_verbose(maxint_t x, int t) {
  using clk = std::chrono::steady_clock;
  auto us = [](clk::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
  };
  PiBreakdown r;
  if (x < 2)
    return r;
  const int nt = resolve_threads(t);
  auto t0 = clk::now();
  GCtx c = build_ctx(x, nt);
  auto t1 = clk::now();
  r.build_us = us(t1 - t0);
  // Time each term against the shared ctx. AC = A + C (one group, as displayed).
  auto sa = clk::now();
  const maxint_t A = a_impl(x, c, nt);
  const maxint_t C = c_impl(x, c, nt);
  auto ea = clk::now();
  r.ac = A + C;
  r.ac_us = us(ea - sa);
  auto sb = clk::now();
  const maxint_t Bimpl = b_impl(x, c, nt);
  auto eb = clk::now();
  r.b = -Bimpl; // signed contribution (pi subtracts b_impl)
  r.b_us = us(eb - sb);
  auto sd = clk::now();
  const maxint_t D = d_impl(x, c, nt);
  auto ed = clk::now();
  r.d = D;
  r.d_us = us(ed - sd);
  auto sp = clk::now();
  r.phi0 = phi0_impl(x, c);
  auto ep = clk::now();
  r.phi0_us = us(ep - sp);
  auto ss = clk::now();
  r.sigma = sigma_impl(x, c);
  auto es = clk::now();
  r.sigma_us = us(es - ss);
  r.pi = A - Bimpl + C + D + r.phi0 + r.sigma; // identical to g_pi
  return r;
}

} // namespace primecount
