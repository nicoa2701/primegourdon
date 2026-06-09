#include "int_types.hpp"

namespace primecount {

std::string to_string(int128_t n) {
  if (n == 0)
    return "0";

  bool negative = n < 0;
  uint128_t u = negative ? -static_cast<uint128_t>(n) : static_cast<uint128_t>(n);

  char buf[40];
  int pos = sizeof(buf);
  while (u > 0) {
    buf[--pos] = static_cast<char>('0' + static_cast<int>(u % 10));
    u /= 10;
  }
  std::string s(buf + pos, sizeof(buf) - pos);
  return negative ? "-" + s : s;
}

} // namespace primecount

std::ostream& operator<<(std::ostream& os, primecount::int128_t n) {
  return os << primecount::to_string(n);
}
