// int_types.hpp — wide integer type and I/O helpers.
//
// x can reach 1e19, which exceeds INT64_MAX (~9.22e18), and several
// intermediate products grow well beyond 64 bits. We therefore use a signed
// 128-bit integer as the main "max integer" type throughout the project.
#ifndef PRIMECOUNT_INT_TYPES_HPP
#define PRIMECOUNT_INT_TYPES_HPP

#include <cstdint>
#include <string>
#include <ostream>

namespace primecount {

using int128_t = __int128;
using uint128_t = unsigned __int128;

// Main wide type used for x and large accumulators.
using maxint_t = int128_t;

// Decimal string for a 128-bit value (std::to_string has no __int128 overload).
std::string to_string(int128_t n);

} // namespace primecount

// Stream operator so we can print maxint_t directly.
std::ostream& operator<<(std::ostream& os, primecount::int128_t n);

#endif // PRIMECOUNT_INT_TYPES_HPP
