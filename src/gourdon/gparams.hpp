// gparams.hpp — Gourdon's parameters y, z, x*, k (gourdon.md §1).
//
// These differ from the interim Deleglise-Rivat params (params.hpp): here
// z = alpha_z * y ~ x^(1/3) (a SMALL bound), and x* is the pivot between the
// easy leaves (A / C2) and the hard leaves (D). We start in the degenerate
// regime alpha_y = alpha_z = 1 (y = z = x^(1/3)) recommended by gourdon.md §10
// for first validation; the real alpha heuristic is layered on during tuning.
#ifndef PRIMECOUNT_GPARAMS_HPP
#define PRIMECOUNT_GPARAMS_HPP

#include "int_types.hpp"
#include "sieve.hpp"
#include "util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace primecount {

struct GParams {
  int128_t x;
  int64_t x13;   // floor(x^(1/3))
  int64_t x14;   // floor(x^(1/4))
  int64_t sqrtx; // floor(x^(1/2))
  int64_t y;
  int64_t z;
  int64_t xstar; // x*
  int k;         // PhiTiny wheel size
};

// alpha_yz = alpha_y * alpha_z heuristic (gourdon.md §1.4). Any value >= 1 is
// correct; this one just aims for speed. Capped at x^(1/6) so y < sqrt(x).
inline double gourdon_alpha_yz(int128_t x) {
  double L = std::log(static_cast<double>(x));
  double a = (x <= 100000000000LL)
                 ? 0.078173 * L + 1.0
                 : 0.00526934 * L * L * L - 0.495545 * L * L + 16.5791 * L -
                       183.836;
  if (a < 1.0)
    a = 1.0;
  double cap = std::pow(static_cast<double>(x), 1.0 / 6.0);
  if (a > cap)
    a = cap;
  return a;
}

inline GParams make_gparams(int128_t x) {
  GParams p;
  p.x = x;
  p.x13 = iroot<3>(x);
  p.x14 = iroot<4>(x);
  p.sqrtx = isqrt(x);

  // y = alpha_y * x^(1/3), z = alpha_z * y (gourdon.md §1.1). z = y
  // (alpha_z = 1) is best here. alpha_y = 0.9x the generic cubic factor after
  // the D speedups (counter + wheel-30 + adaptive LB + divl): the smaller y
  // rebalances work back into D and shrinks A/C, with measured end-to-end wins
  // across the benchmark range. Sweep with PC_AY for a specific machine (any
  // value >= 1 is correct, only speed changes).
  double az = 1.0;
  double ay = 0.9 * gourdon_alpha_yz(x);

  // Env overrides for local tuning (PC_AY / PC_AZ); any value is correct, only
  // speed changes.
  if (const char* e = std::getenv("PC_AY"))
    ay = std::atof(e);
  if (const char* e = std::getenv("PC_AZ"))
    az = std::atof(e);

  int64_t y = static_cast<int64_t>(ay * static_cast<double>(p.x13));
  y = std::max(y, p.x13 + 1);
  y = std::min(y, p.sqrtx - 1);
  if (y < 2)
    y = 2;
  p.y = y;

  int64_t z = static_cast<int64_t>(az * static_cast<double>(y));
  z = std::max(z, y);
  z = std::min(z, p.sqrtx - 1);
  if (z < y)
    z = y;
  p.z = z;

  // x* = max(x^(1/4), ceil(x / y^2)), bounded by y and floor(sqrt(x/y)).
  int128_t y2 = static_cast<int128_t>(p.y) * p.y;
  int128_t xy2 = x / y2 + ((x % y2) ? 1 : 0); // ceil(x / y^2)
  int64_t xstar = std::max(p.x14, static_cast<int64_t>(xy2));
  xstar = std::min<int64_t>(xstar, p.y);
  int64_t sxy = isqrt(x / p.y); // floor(sqrt(x/y))
  xstar = std::min(xstar, sxy);
  if (xstar < 1)
    xstar = 1;
  p.xstar = xstar;

  // k = pi(x^(1/4)), capped to pi(min(x*, sqrt(x/y))) and to k_max = 7.
  const int kmax = 7;
  int64_t cap_arg = std::min(p.xstar, sxy);
  int k = static_cast<int>(count_primes(p.x14));
  int kcap = static_cast<int>(count_primes(cap_arg));
  k = std::min(k, kcap);
  k = std::min(k, kmax);
  if (k < 1)
    k = 1;
  p.k = k;
  return p;
}

} // namespace primecount

#endif // PRIMECOUNT_GPARAMS_HPP
