// main.cpp — command-line front end for primecount (Gourdon's algorithm).
//
//   ./primecount X            print π(X)
//   ./primecount X --Phi0     print only the Φ0 term
//   ./primecount X --Sigma    print only the Σ term
//   ./primecount X --AC       print only the A+C term
//   ./primecount X --B        print only the B term
//   ./primecount X --D        print only the D term
//   ./primecount X -t N / --threads N   use N threads (default: all cores)
//
// X accepts plain integers ("123456") and scientific notation ("1e19",
// "1.5e18"); it is parsed exactly into a 128-bit integer.
#include "gourdon.hpp"
#include "int_types.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace {

using primecount::int128_t;
using primecount::maxint_t;

// Adaptive elapsed-time string: microseconds under 1 ms, milliseconds under
// 1 s, otherwise seconds with 3 decimals.
std::string fmt_time(double us) {
  char buf[32];
  if (us < 1000.0)
    std::snprintf(buf, sizeof(buf), "%lld µs",
                  static_cast<long long>(us + 0.5));
  else if (us < 1000000.0)
    std::snprintf(buf, sizeof(buf), "%lld ms",
                  static_cast<long long>(us / 1000.0 + 0.5));
  else
    std::snprintf(buf, sizeof(buf), "%.3f s", us / 1000000.0);
  return buf;
}

// Human-readable byte count in binary units (GiB/MiB/KiB), matching `free -h`,
// htop and the rest of the Linux toolchain so the numbers line up.
std::string fmt_bytes(double b) {
  char buf[32];
  const char* unit = "B";
  double v = b;
  if (v >= 1024.0 * 1024 * 1024) { v /= 1024.0 * 1024 * 1024; unit = "GiB"; }
  else if (v >= 1024.0 * 1024) { v /= 1024.0 * 1024; unit = "MiB"; }
  else if (v >= 1024.0) { v /= 1024.0; unit = "KiB"; }
  std::snprintf(buf, sizeof(buf), "%.2f %s", v, unit);
  return buf;
}

// Free RAM the OS can hand out without swapping, in bytes (Linux MemAvailable).
// Returns 0 if /proc/meminfo is absent/unreadable (non-Linux) — caller then
// skips the RAM guardrail rather than guessing.
uint64_t mem_available_bytes() {
  std::ifstream f("/proc/meminfo");
  std::string key;
  uint64_t kb = 0;
  while (f >> key >> kb) {
    if (key == "MemAvailable:")
      return kb * 1024ull;
    std::string rest; // skip the rest of the line (the "kB" unit suffix)
    std::getline(f, rest);
  }
  return 0;
}

// Peak-RSS model coefficient: peak_bytes ~= kPeakRssPerSqrtX * sqrt(x).
// Recalibrated for the PHASED build (now the default), where the full PiTable and
// mp[] never coexist — peak is dominated by the PiTable alone (~0.094*sqrt x) plus
// transients, NOT their sum. Measured phased peaks: coeff = peak/sqrt(x) =
// 0.124@1e18, 0.106@1e19, 0.103@1e20 (asymptotes to ~0.10 as fixed overhead fades).
// 0.14 is a deliberate conservative over-estimate (+13%@1e18 .. +36%@1e20) so the
// guardrail refuses up front rather than OOM-killing mid-run, while staying ~2.5x
// tighter than the old 0.30 (which modelled the non-phased PiTable+mp[] sum).
// (PC_NOPHASE reverts to the non-phased build, whose peak is ~0.30*sqrt x — there
// the guardrail under-estimates; --force overrides if ever needed.)
constexpr double kPeakRssPerSqrtX = 0.14;

double estimate_peak_rss(int128_t x) {
  if (x < 1)
    return 0.0;
  return kPeakRssPerSqrtX * std::sqrt(static_cast<double>(x));
}

// Low-RAM path (--ram / PC_LOWMEM) peak model. With the O(sqrt x) PiTable gone,
// peak RSS is dominated by the packed mp[] + small tables and tracks ~O(x^(1/3)).
// Calibrated conservatively from measured low-mem peaks (~109 MiB @1e18,
// ~270 MiB @1e19, ~700 MiB @1e20): coeff = peak / x^(1/3) ~ 114..150 bytes; 160
// over-estimates at every measured point so --auto/guardrail refuse up front
// rather than OOM-killing mid-run.
constexpr double kPeakRssPerCbrtX = 160.0;

double estimate_peak_rss_lowmem(int128_t x) {
  if (x < 1)
    return 0.0;
  return kPeakRssPerCbrtX * std::cbrt(static_cast<double>(x));
}

[[noreturn]] void usage(int code) {
  std::cout
      << "Usage: primecount X [option]\n"
         "  X                 integer or scientific notation (e.g. 1e19)\n"
         "  --Phi0            compute only the Phi0 term\n"
         "  --Sigma           compute only the Sigma term\n"
         "  --AC              compute only the A+C term\n"
         "  --B               compute only the B term\n"
         "  --D               compute only the D term\n"
         "  --perf            optimize for speed (default)\n"
         "  --ram             optimize for low peak RAM (~x^1/3) over speed\n"
         "  --auto            perf if its peak fits in free RAM, else ram\n"
         "  -v, --verbose     print every term with its own timing\n"
         "  -t, --threads N   number of threads (default: all cores)\n"
         "  --force           run even if the estimated peak RSS exceeds free RAM\n"
         "  -h, --help        show this help\n";
  std::exit(code);
}

// Parse a non-negative integer given as plain digits or scientific notation.
// Returns false on malformed input or a non-integer value.
bool parse_number(const std::string& s, int128_t& out) {
  if (s.empty())
    return false;

  std::string mantissa;     // all significant digits, decimal point removed
  int frac_digits = 0;      // number of digits after the decimal point
  int64_t exponent = 0;     // value following 'e'/'E'
  bool seen_point = false;
  size_t i = 0;

  for (; i < s.size(); ++i) {
    char c = s[i];
    if (c >= '0' && c <= '9') {
      mantissa.push_back(c);
      if (seen_point)
        ++frac_digits;
    } else if (c == '.') {
      if (seen_point)
        return false;
      seen_point = true;
    } else if (c == 'e' || c == 'E') {
      exponent = std::strtoll(s.c_str() + i + 1, nullptr, 10);
      break;
    } else {
      return false;
    }
  }

  if (mantissa.empty())
    return false;

  // Value = (mantissa as integer) * 10^(exponent - frac_digits).
  int64_t scale = exponent - frac_digits;
  if (scale < 0)
    return false; // would not be an integer

  int128_t value = 0;
  for (char c : mantissa)
    value = value * 10 + (c - '0');
  for (int64_t k = 0; k < scale; ++k)
    value *= 10;

  out = value;
  return true;
}

} // namespace

int main(int argc, char** argv) {
  // RAM/speed trade-off selector (default perf). --auto is resolved against free
  // RAM below, once x is parsed.
  enum class Mode { Perf, Ram, Auto };
  Mode mode = Mode::Perf;

  std::string x_arg;
  std::string term;
  bool verbose = false;
  bool force = false;
  bool threads_explicit = false;
  int threads = static_cast<int>(std::thread::hardware_concurrency());
  if (threads < 1)
    threads = 1;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(0);
    } else if (a == "-v" || a == "--verbose") {
      verbose = true;
    } else if (a == "--force") {
      force = true;
    } else if (a == "--perf") {
      mode = Mode::Perf;
    } else if (a == "--ram") {
      mode = Mode::Ram;
    } else if (a == "--auto") {
      mode = Mode::Auto;
    } else if (a == "--Phi0" || a == "--Sigma" || a == "--AC" || a == "--B" ||
               a == "--D") {
      if (!term.empty()) {
        std::cerr << "error: only one term flag may be given\n";
        return 1;
      }
      term = a.substr(2);
    } else if (a == "--threads" || a == "-t") {
      if (++i >= argc) {
        std::cerr << "error: --threads requires a value\n";
        return 1;
      }
      threads = std::atoi(argv[i]);
      if (threads < 1)
        threads = 1;
      threads_explicit = true;
    } else if (!a.empty() && a[0] == '-' && a != "-") {
      std::cerr << "error: unknown option '" << a << "'\n";
      return 1;
    } else if (x_arg.empty()) {
      x_arg = a;
    } else {
      std::cerr << "error: unexpected argument '" << a << "'\n";
      return 1;
    }
  }

  if (x_arg.empty())
    usage(1);

  int128_t x;
  if (!parse_number(x_arg, x)) {
    std::cerr << "error: invalid number '" << x_arg << "'\n";
    return 1;
  }

  // Machine-adaptive RAM guardrail. Peak RSS grows ~sqrt(x); with the phased build
  // (default) it is the PiTable alone, ~0.14*sqrt x conservatively (118 MB @1e18,
  // 984 MB @1e20, ~3 GB @1e21, ~10 GB @1e22). A fixed x cutoff is wrong per machine
  // — on a RAM-limited box a too-large x would OOM-kill mid-run.
  // So estimate the peak and compare against what the OS can actually give us;
  // refuse up front (overridable with --force) instead of dying after minutes.
  // Serial fast-path for small x (opt A — "smooth the small-prime fixed cost").
  // Below ~1e11 the actual Gourdon work is sub-millisecond, but entering the
  // first OpenMP parallel region spins up the thread pool (~15ms one-time) and
  // the parallel terms add scheduling overhead that dwarfs the work — measured
  // crossover where multi-thread finally wins is ~3e11 (e.g. 1e8: mono 6ms vs
  // multi 14ms; 1e7: 9ms vs 57ms). primecount sidesteps this by dispatching
  // small x to non-Gourdon methods (legendre/meissel) entirely; we keep the
  // single Gourdon path but drop to one thread. Skipped when the user pinned
  // -t explicitly (so mono/multi can still be benchmarked). Done before the
  // verbose block so the printed thread count reflects what actually runs.
  if (!threads_explicit && threads > 1 &&
      x <= static_cast<int128_t>(100000000000LL)) // 1e11
    threads = 1;

  const uint64_t avail = mem_available_bytes(); // 0 == unknown (non-Linux)

  // Resolve the RAM/speed mode. --auto runs the fast (perf) path when its estimated
  // peak fits in free RAM, otherwise falls back to the low-RAM path instead of
  // refusing. --ram forces low-RAM; --perf (default) forces the fast path. The
  // low-RAM path is the PC_LOWMEM build inside g_pi; the flag just opts in via the
  // env (PC_LOWMEM=1 still works directly). It governs the full pi(x) only — the
  // per-term flags (--AC/--B/--D/...) always use the default build.
  const double est_perf = estimate_peak_rss(x);
  const double est_ram = estimate_peak_rss_lowmem(x);
  // The low-RAM path is honored ONLY by the plain full pi(x) (g_pi -> PC_LOWMEM).
  // Per-term flags (--AC/--B/...) and the --verbose breakdown always build the full
  // O(sqrt x) PiTable, so they run at the PERF peak no matter the mode. The guardrail
  // must size against what ACTUALLY runs — else it under-estimates and lets an OOM
  // through (e.g. --ram --Phi0 at huge x estimated the low-mem peak, then the term
  // allocated the full PiTable and got OOM-killed).
  const bool lowmem_honored = term.empty() && !verbose;
  bool want_lowmem = (mode == Mode::Ram);
  if (mode == Mode::Auto)
    want_lowmem = (avail != 0 && est_perf > static_cast<double>(avail));
  const bool use_lowmem = want_lowmem && lowmem_honored;
  if (use_lowmem)
    setenv("PC_LOWMEM", "1", 1);
  const double est_peak = use_lowmem ? est_ram : est_perf;
  if (mode == Mode::Ram && !lowmem_honored)
    std::cerr << "note: --ram applies only to the full pi(x) run; this invocation "
                 "uses the default build (full PiTable).\n";

  if (verbose) {
    std::cout << "RAM       available " << fmt_bytes(static_cast<double>(avail))
              << ", estimated peak ~" << fmt_bytes(est_peak) << " (mode "
              << (use_lowmem ? "ram" : "perf")
              << (mode == Mode::Auto ? ", auto" : "") << ")";
    if (avail != 0) {
      // Invert the active peak model (perf: C*sqrt x; ram: C*cbrt x) -> largest x
      // that fits in free RAM under the selected mode.
      const double x_max =
          use_lowmem
              ? std::pow(static_cast<double>(avail) / kPeakRssPerCbrtX, 3.0)
              : std::pow(static_cast<double>(avail) / kPeakRssPerSqrtX, 2.0);
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.1e", x_max);
      std::cout << ", max x ~" << buf;
    }
    std::cout << "\n";

    // Active build switches on one line: DIV32 is compile-time (ENABLE_DIV32 /
    // -DWITH_DIV32); AVX512 is the runtime presieve dispatch (path is always
    // compiled in, used only if the CPU supports it); AVX2/BMI2/POPCNT are
    // baked into the release flags.
    std::cout << "build     DIV32 "
#if defined(ENABLE_DIV32)
                 "ON"
#else
                 "OFF"
#endif
              << ", AVX512 "
              << (__builtin_cpu_supports("avx512f") ? "ON" : "OFF")
              << ", AVX2/BMI2/POPCNT on, threads " << threads << "\n";
  }

  if (avail != 0 && est_peak > static_cast<double>(avail) && !force) {
    std::cerr << "error: estimated peak RSS ~" << fmt_bytes(est_peak)
              << " exceeds available RAM " << fmt_bytes(static_cast<double>(avail))
              << " — would swap-thrash or be OOM-killed. Pass --force to run "
                 "anyway.\n";
    return 1;
  }

  // Overflow note, independent of RAM: validated correct up to 1e21
  // (pi(1e21)==ref). Above 1e21 some int64 leaf accumulators are untested and
  // may eventually overflow (the D WIDE dispatch handles division, not these).
  if (x > static_cast<int128_t>(1000000000) * 1000000000000) // x > 1e21
    std::cerr << "warning: x > 1e21 — int64 leaf accumulators untested above "
                 "1e21 and may overflow.\n";

  auto now = [] { return std::chrono::steady_clock::now(); };
  auto micros = [](auto d) {
    return std::chrono::duration<double, std::micro>(d).count();
  };

  // Verbose full count: HONEST per-term timing. Builds the shared context ONCE
  // (like pi_gourdon) and times each term against it — the per-term wrappers each
  // rebuild build_ctx (AC twice), so a naive loop overcounts the total by +40-50%
  // and hides cheap terms behind build_ctx. A dedicated "build_ctx (shared once)"
  // line is shown; the term times + build sum to the real pi(x) wall time.
  if (verbose && term.empty()) {
    // Below the tiny-x sieve cutoff there are no Gourdon terms to break down —
    // the result comes straight from the exact segmented sieve (see g_pi).
    if (x < primecount::kSieveCutoff) {
      std::cout << "pi(x)  = " << primecount::to_string(primecount::pi_gourdon(x, threads))
                << "  (tiny-x exact sieve, no Gourdon breakdown)\n";
      return 0;
    }
    auto r = primecount::pi_gourdon_verbose(x, threads);
    std::printf("%-7s %s  [%s]\n", "build", "(shared ctx, built once)",
                fmt_time(r.build_us).c_str());
    std::printf("%-7s= %s  [%s]\n", "Phi0",
                primecount::to_string(r.phi0).c_str(), fmt_time(r.phi0_us).c_str());
    std::printf("%-7s= %s  [%s]\n", "Sigma",
                primecount::to_string(r.sigma).c_str(), fmt_time(r.sigma_us).c_str());
    std::printf("%-7s= %s  [%s]\n", "AC",
                primecount::to_string(r.ac).c_str(), fmt_time(r.ac_us).c_str());
    std::printf("%-7s= %s  [%s]\n", "B",
                primecount::to_string(r.b).c_str(), fmt_time(r.b_us).c_str());
    std::printf("%-7s= %s  [%s]\n", "D",
                primecount::to_string(r.d).c_str(), fmt_time(r.d_us).c_str());
    double terms = r.phi0_us + r.sigma_us + r.ac_us + r.b_us + r.d_us;
    std::printf("%-7s= %s  [%s, +%s build]\n", "pi(x)",
                primecount::to_string(r.pi).c_str(), fmt_time(terms).c_str(),
                fmt_time(r.build_us).c_str());
    return 0;
  }

  auto start = now();
  maxint_t result;
  if (term.empty())
    result = primecount::pi_gourdon(x, threads);
  else if (term == "Phi0")
    result = primecount::Phi0(x, threads);
  else if (term == "Sigma")
    result = primecount::Sigma(x, threads);
  else if (term == "AC")
    result = primecount::AC(x, threads);
  else if (term == "B")
    result = primecount::B(x, threads);
  else // "D"
    result = primecount::D(x, threads);
  double us = micros(now() - start);

  std::cout << result << "  [" << fmt_time(us) << "]\n";
  return 0;
}
