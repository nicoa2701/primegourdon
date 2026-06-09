# primecount — Gourdon's algorithm for π(x)

[![selftest](https://github.com/nicoa2701/primegourdon/actions/workflows/selftest.yml/badge.svg)](https://github.com/nicoa2701/primegourdon/actions/workflows/selftest.yml)

*[Version française : LISEZMOI.md](LISEZMOI.md)*

A from-scratch C++20 implementation of **Xavier Gourdon's algorithm** for the
prime-counting function π(x) — the number of primes ≤ x — computed analytically,
without enumerating the primes.

It rests on Gourdon's identity:

```
π(x) = A − B + C + D + Φ₀ + Σ
```

Each of the six terms is computed independently (and can be evaluated in
isolation from the command line), then combined. The full mathematical
specification — every formula, summation bound, sign and integer-arithmetic
pitfall — is in **[gourdon.md](gourdon.md)** (in French); the code is a faithful,
heavily optimized realization of that document.

## Highlights

- **Exact** integer arithmetic throughout; 128-bit dispatch above ~1.6·10²⁰ so
  results stay correct past 2⁶³.
- **Parallel** (OpenMP) with cache- and bandwidth-aware segment sizing.
- **Wheel-30 segmented sieves** for the two dominant terms (B and D), with
  branch-free, 8-way-unrolled crossing and SIMD presieve.
- **Runtime SIMD dispatch** (AVX2 / AVX-512) — one binary runs everywhere.
- **Configure-time micro-benchmark** picks the fastest 32-bit/64-bit division
  backend for the build host.
- **Phased memory build**: the π-table and the Möbius/least-prime-factor table
  never coexist, roughly halving peak RSS at no time cost.
- **Machine-adaptive RAM guardrail**: refuses up front rather than being
  OOM-killed mid-run.

## Build

Requires CMake ≥ 3.16, a C++20 compiler (GCC/Clang) and OpenMP.

```sh
./run.sh                 # configure + build (Release)
./run.sh --selftest      # build, then run the test suite
./run.sh 1e18 -v         # build, then run with a per-term breakdown
```

`run.sh` is robust to the tree being moved (it wipes a stale CMake cache and
reconfigures). The binary is placed at the project root as `./primecount`.

### Build options

The A/C easy-leaf code can use a guarded 32-bit division path (`divl`) on x86-64.
By default CMake runs a tiny configure-time benchmark and enables it only when it
is faster on the build host.

```sh
./run.sh -DWITH_DIV32=ON --clean   # force the divl backend
./run.sh -DWITH_DIV32=OFF --clean  # force plain 64-bit division
```

Configure options are cached by CMake; use `--clean` when changing them. For
local-only builds, you may also replace the release flags in `CMakeLists.txt`
with `-march=native` to let the compiler target your exact CPU.

## Usage

```
Usage: primecount X [option]
  X                 integer or scientific notation (e.g. 1e19)
  --Phi0            compute only the Phi0 term
  --Sigma           compute only the Sigma term
  --AC              compute only the A+C term
  --B               compute only the B term
  --D               compute only the D term
  --perf            optimize for speed (default)
  --ram             optimize for low peak RAM (~x^1/3) over speed
  --auto            perf if its peak fits in free RAM, else ram
  -v, --verbose     print every term with its own timing
  -t, --threads N   number of threads (default: all cores)
  --force           run even if estimated peak RSS exceeds free RAM
  -h, --help        show this help
```

### Memory vs speed

By default the fast path is used: peak RSS grows as ~0.14·√x. For very large x
that becomes the binding constraint, so a **low-RAM path** is available whose peak
follows ~O(x^(1/3)) instead, at roughly 1.15–1.2× the time:

- `--perf` (default) — fastest; peak RSS ~√x.
- `--ram` — low-RAM path (no O(√x) π-table, packed Möbius/LPF table, sparse C);
  peak RSS ~x^(1/3), bit-identical result.
- `--auto` — run `--perf` when its estimated peak fits in free RAM, otherwise fall
  back to `--ram` instead of refusing.

These govern the full π(x) run; the per-term flags always use the default build.

Example:

```sh
$ ./primecount 1e18
24739954287740860  [46.410 s]

$ ./primecount 1e18 -v
RAM       available 4.94 GiB, estimated peak ~133.51 MiB, max x ~1.4e+21
build     DIV32 ON, AVX512 OFF, AVX2/BMI2/POPCNT on, threads 8
build   (shared ctx, built once)            [1.517 s]
Phi0  = 64014967544662                       [164 ms]
Sigma = 514634213323316                      [3 ms]
AC    = 9336325709491971                     [19.910 s]
B     = -9158014307746509                    [6.093 s]
D     = 23982993705127420                    [19.323 s]
pi(x) = 24739954287740860                    [45.493 s, +1.517 s build]
```

## Benchmark

Single-threaded fast path below 10¹¹; Gourdon's algorithm (all cores) above.
Wall-clock and **peak resident memory** (`/usr/bin/time -v`, Maximum RSS).

**Machine:** Intel Core i5-9300HF (Coffee Lake, 4C / 8T, AVX2), 8 threads,
DIV32 auto-enabled.

| x     | π(x)                  | time      | peak RAM   |
|-------|-----------------------|-----------|------------|
| 10¹   | 4                     | < 1 ms    | 4.1 MiB    |
| 10²   | 25                    | < 1 ms    | 4.0 MiB    |
| 10³   | 168                   | < 1 ms    | 4.1 MiB    |
| 10⁴   | 1 229                 | < 1 ms    | 4.1 MiB    |
| 10⁵   | 9 592                 | < 1 ms    | 4.0 MiB    |
| 10⁶   | 78 498                | 2 ms      | 6.8 MiB    |
| 10⁷   | 664 579               | 2 ms      | 6.8 MiB    |
| 10⁸   | 5 761 455             | 2 ms      | 6.8 MiB    |
| 10⁹   | 50 847 534            | 3 ms      | 6.8 MiB    |
| 10¹⁰  | 455 052 511           | 4 ms      | 6.8 MiB    |
| 10¹¹  | 4 118 054 813         | 11 ms     | 6.8 MiB    |
| 10¹²  | 37 607 912 018        | 17 ms     | 8.4 MiB    |
| 10¹³  | 346 065 536 839       | 47 ms     | 9.3 MiB    |
| 10¹⁴  | 3 204 941 750 802     | 120 ms    | 10.7 MiB   |
| 10¹⁵  | 29 844 570 422 669    | 0.52 s    | 14.2 MiB   |
| 10¹⁶  | 279 238 341 033 925   | 2.21 s    | 23.5 MiB   |
| 10¹⁷  | 2 623 557 157 654 233 | 10.20 s   | 49.0 MiB   |
| 10¹⁸  | 24 739 954 287 740 860 | 46.41 s  | 115.1 MiB  |
| 10¹⁹  | 234 057 667 276 344 607 | 220.7 s | 321.5 MiB  |

Time scales as **O(x^(2/3) / (log x)²)** (≈ ×4.7 per decade past 10¹⁴); peak RAM
follows the model ≈ 0.14·√x and tracks **O(x^(1/3) (log x)³)** working set.

## Correctness

`./run.sh --selftest` checks, against an exact segmented-sieve oracle:

- each term (Φ₀, Σ, A, B, …) against an independent brute-force implementation;
- the full assembly for x ∈ {10³ … 10⁸};
- **every** x in [1, 5000] against the exact sieve (catches off-by-one edges);
- the known table π(10⁹)=50 847 534, π(10¹⁰)=455 052 511, π(10¹¹)=4 118 054 813.

All values above are bit-exact against published references.

## Limits

Validated exact up to **10²¹** (π(10²¹) matches the reference). Above 10²¹, some
int64 leaf accumulators are untested and may eventually overflow; the program
prints a warning. The 128-bit division path keeps the *divisions* correct beyond
that, but the accumulators are the practical ceiling.

## Algorithm & references

The decomposition, all six terms and the integer-arithmetic subtleties are
documented in **[gourdon.md](gourdon.md)**. Primary sources:

- X. Gourdon, *Computation of π(x): improvements to the Meissel, Lehmer,
  Lagarias, Miller, Odlyzko, Deléglise and Rivat method*, 2001.
- M. Deléglise, J. Rivat, *Computing π(x): The Meissel–Lehmer–Lagarias–Miller–
  Odlyzko Method*, Math. Comp. 65 (1996).
- D. B. Staple, *The combinatorial algorithm for computing π(x)*, 2015.

For the state-of-the-art reference implementation, see Kim Walisch's
[primecount](https://github.com/kimwalisch/primecount).

## License

BSD 2-Clause. See [LICENSE](LICENSE).
