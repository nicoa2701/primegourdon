// Configure-time micro-benchmark: is the 32-bit `divl` path of fast_div_u()
// actually faster than the plain 64-bit `divq` on this build host? Decides the
// WITH_DIV32 default in CMakeLists.txt.
//
// It times the two REAL code paths head to head:
//   ON  = the guarded divl: `if (d<=2^32 && high32(x)<d) divl else x/d`
//   OFF = plain `x / d`
// over representative A/C-leaf operands (64-bit dividend, quotient ~sqrt < 2^32,
// prime-ish divisor < 2^32 so the guard passes and exercises divl). The ON path
// pays a per-call guard branch that the OFF path doesn't — exactly the trade-off
// that makes divl a win on some slow dividers and a small loss on fast ones.
//
// Exit 0  => divl meaningfully faster (>=2%): enable ENABLE_DIV32.
// Exit !0 => not worth it (or non-x86): leave it off (plain x/d).

#include <cstdint>
#include <cstdio>

#if !defined(__x86_64__) || !(defined(__GNUC__) || defined(__clang__))
int main() { return 1; } // no divl available on this target -> default OFF
#else

#include <chrono>

static inline int64_t fast_div_on(uint64_t x, uint64_t d) {
  uint32_t high = static_cast<uint32_t>(x >> 32);
  if (d <= 0xffffffffu && high < static_cast<uint32_t>(d)) {
    uint32_t low = static_cast<uint32_t>(x);
    uint32_t dd = static_cast<uint32_t>(d);
    __asm__("divl %[dd]" : "+a"(low), "+d"(high) : [dd] "r"(dd));
    return static_cast<int64_t>(low);
  }
  return static_cast<int64_t>(x / d);
}

static inline int64_t fast_div_off(uint64_t x, uint64_t d) {
  return static_cast<int64_t>(x / d);
}

int main() {
  constexpr int N = 2048;
  static uint64_t xs[N], ds[N];

  // Deterministic xorshift (no rand, reproducible). Build operands so that
  // d < 2^31 and the dividend x = q*d + r with q,r in [.,d) -> high32(x) < d
  // always holds (divl never faults), matching the covered A/C leaves.
  uint64_t s = 0x9e3779b97f4a7c15ull;
  auto nxt = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
  for (int i = 0; i < N; ++i) {
    uint32_t d = 1000003u + static_cast<uint32_t>(nxt() % 2000000000u); // < 2^31
    uint32_t q = 1u + static_cast<uint32_t>(nxt() % (d - 1));           // q < d
    uint64_t r = nxt() % d;                                             // r < d
    ds[i] = d;
    xs[i] = static_cast<uint64_t>(q) * d + r;
  }

  using clk = std::chrono::steady_clock;
  constexpr int REPS = 3000;
  volatile int64_t sink = 0;

  // warm up I-cache / branch predictor
  for (int i = 0; i < N; ++i) sink += fast_div_on(xs[i], ds[i]);

  double best_on = 1e30, best_off = 1e30;
  for (int t = 0; t < 5; ++t) {
    auto a = clk::now();
    int64_t acc = 0;
    for (int r = 0; r < REPS; ++r)
      for (int i = 0; i < N; ++i) acc += fast_div_on(xs[i], ds[i]);
    auto b = clk::now();
    sink += acc;
    double dt = std::chrono::duration<double>(b - a).count();
    if (dt < best_on) best_on = dt;
  }
  for (int t = 0; t < 5; ++t) {
    auto a = clk::now();
    int64_t acc = 0;
    for (int r = 0; r < REPS; ++r)
      for (int i = 0; i < N; ++i) acc += fast_div_off(xs[i], ds[i]);
    auto b = clk::now();
    sink += acc;
    double dt = std::chrono::duration<double>(b - a).count();
    if (dt < best_off) best_off = dt;
  }

  double ratio = best_on / best_off;
  std::fprintf(stderr, "divl=%.4fs divq=%.4fs ratio(on/off)=%.3f -> %s",
               best_on, best_off, ratio, ratio < 0.98 ? "ON" : "OFF");
  // Enable only if divl is at least 2% faster (margin avoids flapping when it's
  // a wash; on a fast divider the ON path is >= OFF, so this correctly stays off).
  return (best_on < best_off * 0.98) ? 0 : 1;
}
#endif
