// sieve.hpp — prime generation and an exact π(x) by segmented sieving.
//
// generate_primes(limit) returns every prime <= limit. It is used to build the
// small base of primes the Gourdon terms iterate over. count_primes(x) is the
// slow-but-exact reference used to validate the algorithm at small x.
#ifndef PRIMECOUNT_SIEVE_HPP
#define PRIMECOUNT_SIEVE_HPP

#include <cstdint>
#include <vector>

namespace primecount {

// All primes p with p <= limit, in increasing order (empty if limit < 2).
std::vector<int64_t> generate_primes(int64_t limit);

// Exact π(x) via a segmented sieve of Eratosthenes. Correct for any x >= 0 but
// only practical (seconds) up to ~1e11; used as a correctness oracle.
int64_t count_primes(int64_t x);

} // namespace primecount

#endif // PRIMECOUNT_SIEVE_HPP
