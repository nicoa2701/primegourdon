// selftest.cpp — fast correctness checks, validated at small limits against
// brute force / the exact sieve oracle. Build target `selftest`; run
// `./build/selftest`. Covers the shared core (PiTable, phi) and every Gourdon
// term (gourdon.md) in isolation, plus the full assembly pi(x).
#include "gourdon.hpp"
#include "int_types.hpp"
#include "phi.hpp"
#include "phi_pi.hpp"
#include "pitable.hpp"
#include "sieve.hpp"
#include "util.hpp"

#include "../src/gourdon/gparams.hpp"
#include "../src/gourdon/gterms.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace primecount;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++g_failures;
  }
}

// --- Shared core ---

int64_t phi_brute(int64_t x, int a, const std::vector<int64_t>& primes) {
  int64_t c = 0;
  for (int64_t i = 1; i <= x; ++i) {
    bool ok = true;
    for (int j = 0; j < a; ++j)
      if (i % primes[j] == 0) {
        ok = false;
        break;
      }
    if (ok)
      ++c;
  }
  return c;
}

void test_pitable() {
  PiTable pi(1000000);
  for (int64_t n = 0; n <= 3000; ++n)
    check(pi(n) == count_primes(n), "PiTable " + std::to_string(n));
  for (int64_t n : {int64_t(7919), int64_t(100000), int64_t(999983), int64_t(1000000)})
    check(pi(n) == count_primes(n), "PiTable " + std::to_string(n));
}

void test_phi() {
  std::vector<int64_t> primes = generate_primes(100);
  for (int a = 0; a <= 12; ++a)
    for (int64_t x : {int64_t(0), int64_t(1), int64_t(5), int64_t(30),
                      int64_t(1000), int64_t(5000), int64_t(12345)})
      check(static_cast<int64_t>(phi(x, a)) == phi_brute(x, a, primes),
            "phi(" + std::to_string(x) + "," + std::to_string(a) + ")");
}

// phi_pi (pi-table accelerated phi) must match the plain phi().
void test_phi_pi() {
  const int64_t L = 100000;
  std::vector<int64_t> primes = generate_primes(L);
  PiTable pi(L);
  for (int a = 0; a <= 40; ++a)
    for (int64_t xx : {int64_t(0), int64_t(1), int64_t(5), int64_t(100),
                       int64_t(1000), int64_t(50000), int64_t(99991), L})
      check(phi_pi(xx, a, primes, pi) == static_cast<int64_t>(phi(xx, a)),
            "phi_pi(" + std::to_string(xx) + "," + std::to_string(a) + ")");
}

// --- Gourdon terms (gourdon.md), each validated in isolation ---

// Phi0 (§3): direct enumeration of squarefree n <= z, P-(n) > p_k, P+(n) <= y.
int128_t g_phi0_brute(int128_t x) {
  GParams p = make_gparams(x);
  std::vector<int64_t> sp = generate_primes(64);
  int64_t pk = sp[p.k - 1];

  int128_t sum = phi(x, p.k);
  for (int64_t n = 2; n <= p.z; ++n) {
    int64_t m = n, pmin = 0, pmax = 0;
    int omega = 0;
    bool sf = true;
    for (int64_t d = 2; d * d <= m; ++d)
      if (m % d == 0) {
        if (pmin == 0)
          pmin = d;
        pmax = d;
        ++omega;
        m /= d;
        if (m % d == 0) {
          sf = false;
          break;
        }
      }
    if (!sf)
      continue;
    if (m > 1) {
      if (pmin == 0)
        pmin = m;
      pmax = m;
      ++omega;
    }
    if (pmin <= pk || pmax > p.y)
      continue;
    sum += ((omega & 1) ? -1 : 1) * phi(x / n, p.k);
  }
  return sum;
}

void test_g_phi0() {
  for (int64_t x : {int64_t(1000000), int64_t(10000000), int64_t(100000000),
                    int64_t(1000000000)})
    check(static_cast<int128_t>(g_Phi0(x)) == g_phi0_brute(x),
          "g_Phi0(" + std::to_string(x) + ")");
}

// Sigma (§4): same formulas, computed with the independent oracle.
int128_t g_sigma_brute(int128_t x) {
  GParams p = make_gparams(x);
  auto P = [](int64_t t) { return static_cast<int128_t>(count_primes(t)); };
  int128_t a = P(p.y), b = P(p.x13);
  int64_t sxy = isqrt(x / p.y);
  int128_t cc = P(sxy), dd = P(p.xstar), pisx = P(p.sqrtx);

  int128_t S0 = a - 1 + pisx * (pisx - 1) / 2 - a * (a - 1) / 2;
  int128_t S1 = (a - b) * (a - b - 1) / 2;
  int128_t S2 = a * (b - cc - cc * (cc - 3) / 2 + dd * (dd - 3) / 2);
  int128_t S3 = (b * (b - 1) * (2 * b - 1) / 6 - b) -
                (dd * (dd - 1) * (2 * dd - 1) / 6 - dd);
  int128_t S4 = 0, S5 = 0, S6 = 0;
  for (int64_t q : generate_primes(p.x13)) {
    if (q <= p.xstar)
      continue;
    if (q <= sxy)
      S4 += P(static_cast<int64_t>(x / (static_cast<int128_t>(q) * p.y)));
    else
      S5 += P(static_cast<int64_t>(x / (static_cast<int128_t>(q) * q)));
    int128_t pr = P(isqrt(x / q));
    S6 += pr * pr;
  }
  return S0 + S1 + S2 + S3 + a * S4 + S5 - S6;
}

void test_g_sigma() {
  for (int64_t x : {int64_t(1000000), int64_t(100000000), int64_t(1000000000)})
    check(static_cast<int128_t>(g_Sigma(x)) == g_sigma_brute(x),
          "g_Sigma(" + std::to_string(x) + ")");
}

// A (§5): same weighted two-prime formula via the oracle.
int128_t g_a_brute(int128_t x) {
  GParams p = make_gparams(x);
  std::vector<int64_t> P = generate_primes(p.sqrtx);
  int128_t sum = 0;
  for (size_t ib = 0; ib < P.size(); ++ib) {
    int64_t pb = P[ib];
    if (pb <= p.xstar)
      continue;
    if (pb > p.x13)
      break;
    int128_t xp = x / pb;
    int64_t s = isqrt(xp);
    for (size_t i = ib + 1; i < P.size(); ++i) {
      int64_t q = P[i];
      if (q > s)
        break;
      int128_t leaf = x / (static_cast<int128_t>(pb) * q);
      int w = (leaf >= p.y) ? 1 : 2;
      sum += static_cast<int128_t>(w) * count_primes(static_cast<int64_t>(leaf));
    }
  }
  return sum;
}

void test_g_A() {
  for (int64_t x : {int64_t(1000000), int64_t(100000000), int64_t(1000000000)})
    check(static_cast<int128_t>(g_A(x)) == g_a_brute(x),
          "g_A(" + std::to_string(x) + ")");
}

// B (§8): sum_{y<p<=sqrt x} pi(x/p) via the oracle.
int128_t g_b_brute(int128_t x) {
  GParams p = make_gparams(x);
  int128_t sum = 0;
  for (int64_t q : generate_primes(p.sqrtx))
    if (q > p.y)
      sum += count_primes(static_cast<int64_t>(x / q));
  return sum;
}

void test_g_B() {
  for (int64_t x : {int64_t(1000000), int64_t(10000000), int64_t(100000000),
                    int64_t(1000000000)})
    check(static_cast<int128_t>(g_B(x)) == g_b_brute(x),
          "g_B(" + std::to_string(x) + ")");
}

// The public flags expose exactly the Gourdon terms, and they sum to pi(x).
void test_global() {
  for (int64_t x : {int64_t(1000), int64_t(10000), int64_t(100000),
                    int64_t(1000000), int64_t(10000000), int64_t(100000000)}) {
    int128_t expect = count_primes(x);
    check(static_cast<int128_t>(pi_gourdon(x, 1)) == expect,
          "pi_gourdon(" + std::to_string(x) + ")");
    int128_t sum = static_cast<int128_t>(Phi0(x, 1)) + Sigma(x, 1) + AC(x, 1) +
                   B(x, 1) + D(x, 1);
    check(sum == expect, "Phi0+Sigma+AC+B+D at " + std::to_string(x));
  }
}

// Dense small-x sweep: pi_gourdon(x) must equal the exact sieve oracle for
// EVERY x in 1..5000. The 10^n-only tests above skip the tiny-x corner cases
// (empty Sigma sums, the wheel-30 B sieve range dipping down onto the primes
// 2/3/5) where off-by-one bugs hide — e.g. pi(15) once returned 7 instead of 6.
void test_small_dense() {
  for (int64_t x = 1; x <= 5000; ++x)
    check(static_cast<int128_t>(pi_gourdon(x, 1)) == count_primes(x),
          "pi_gourdon(" + std::to_string(x) + ") vs sieve");
}

// Known pi(10^n) values (no slow oracle needed at these sizes).
void test_table() {
  struct {
    int64_t x, pi;
  } T[] = {{1000000000LL, 50847534LL},
           {10000000000LL, 455052511LL},
           {100000000000LL, 4118054813LL}};
  for (auto& t : T)
    check(static_cast<int128_t>(pi_gourdon(t.x, 1)) == t.pi,
          "pi_gourdon(" + std::to_string(t.x) + ") vs table");
}

} // namespace

int main() {
  test_pitable();
  test_phi_pi();
  test_phi();
  test_g_phi0();
  test_g_sigma();
  test_g_A();
  test_g_B();
  test_small_dense();
  test_global();
  test_table();

  if (g_failures == 0) {
    std::cout << "All self-tests passed.\n";
    return 0;
  }
  std::cout << g_failures << " self-test(s) FAILED.\n";
  return 1;
}
