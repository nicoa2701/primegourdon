#include "phi.hpp"
#include "sieve.hpp"

#include <vector>

namespace primecount {

namespace {

// First primes handled directly by the wheel. a = 7 -> primorial 510510, a
// ~2 MB int32 table: a good balance between table size and recursion pruning.
constexpr int kTinyMax = 7;

// First primes, 1-indexed: small_primes()[1] = 2, [2] = 3, [3] = 5, ...
const std::vector<int64_t>& small_primes() {
  static const std::vector<int64_t> p = [] {
    std::vector<int64_t> v = {0}; // index 0 unused so p_a == v[a]
    for (int64_t q : generate_primes(100000))
      v.push_back(q);
    return v;
  }();
  return p;
}

// PhiTiny wheel: for each a in [0, kTinyMax], the primorial period P_a, the
// count of integers coprime to the first a primes in one period (totient), and
// table[a][r] = #{ 1 <= i <= r : i coprime to the first a primes } for r < P_a.
struct PhiTiny {
  int64_t primorial[kTinyMax + 1];
  int64_t totient[kTinyMax + 1];
  std::vector<int32_t> table[kTinyMax + 1];

  PhiTiny() {
    const auto& pr = small_primes();
    for (int a = 0; a <= kTinyMax; ++a) {
      int64_t P = 1;
      for (int i = 1; i <= a; ++i)
        P *= pr[i];
      primorial[a] = P;

      std::vector<int32_t>& t = table[a];
      t.assign(P, 0); // t[0] = 0 (empty range)

      // Build table[a] by SIEVING the multiples of p_1..p_a (pure strided stores,
      // NO division) then a single prefix-count pass — instead of testing
      // coprime(r) with `a` modulos per r. The old test was ~3.5M hardware idiv
      // at a=7 (P=510510) with a RUNTIME divisor, which dominated the small-x
      // fixed cost on slow-divider CPUs (i5-9300HF: PhiTiny ctor ~13 ms). Same
      // table values, built division-free.
      std::vector<char> nc(P, 0); // nc[r] = 1 iff some p_i divides r
      for (int i = 1; i <= a; ++i) {
        const int64_t pp = pr[i];
        for (int64_t j = 0; j < P; j += pp)
          nc[j] = 1;
      }
      int32_t run = 0;
      for (int64_t r = 1; r < P; ++r) {
        run += !nc[r];
        t[r] = run; // #coprime in [1, r]
      }
      // One full period spans [1, P]; the integer P itself is coprime to the
      // first a primes only when a == 0 (for a >= 1, P is divisible by every p_i).
      totient[a] = run + (a == 0 ? 1 : 0);
    }
  }

  // φ(x, a) for a <= kTinyMax, x > 0, in O(1).
  int128_t phi(int128_t x, int a) const {
    int64_t P = primorial[a];
    int128_t full = x / P;
    int64_t rem = static_cast<int64_t>(x % P);
    return full * totient[a] + table[a][rem];
  }

  // Same, all int64 — for x <= INT64_MAX (the common case). Avoids the 128-bit
  // div+mod (__divti3/__divmodti4) the int128 form compiles to; the result
  // phi(x,a) <= x <= INT64_MAX so int64 cannot overflow.
  int64_t phi64(int64_t x, int a) const {
    int64_t P = primorial[a];
    return (x / P) * totient[a] + table[a][x % P];
  }
};

const PhiTiny& phi_tiny() {
  static const PhiTiny t;
  return t;
}

} // namespace

// All-int64 recursion, used once x <= INT64_MAX. The quotient x/pr[a] only
// shrinks, so the whole subtree stays int64 — no per-level "fits?" branch and
// no 128-bit division. phi(x,a) <= x <= INT64_MAX, so no int64 overflow.
static int64_t phi64(int64_t x, int a) {
  if (x <= 0)
    return 0;
  if (a <= kTinyMax)
    return phi_tiny().phi64(x, a);
  const auto& pr = small_primes();
  if (x <= pr[a])
    return 1;
  return phi64(x, a - 1) - phi64(x / pr[a], a - 1);
}

int128_t phi(int128_t x, int a) {
  if (x <= 0)
    return 0;
  if (x <= INT64_MAX) // common case: run the int64 recursion (no 128-bit div)
    return phi64(static_cast<int64_t>(x), a);
  if (a <= kTinyMax)
    return phi_tiny().phi(x, a);

  // When x <= p_a, every integer in [2, x] is divisible by some prime <= x,
  // all of which are among the first a; only the integer 1 survives.
  const auto& pr = small_primes();
  if (x <= pr[a])
    return 1;

  return phi(x, a - 1) - phi(x / pr[a], a - 1);
}

} // namespace primecount
