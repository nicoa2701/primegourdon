cmake_minimum_required(VERSION 3.16)
project(primecount CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Portable SIMD baseline (AVX2 + BMI/BMI2 + POPCNT). This targets mainstream
# Haswell+/Zen+ x86-64 CPUs while keeping one binary. It lowers
# std::popcount in d_impl (the Fenwick word counts) to the hardware POPCNT
# instruction; measured ~7-10% faster on D at 1e15-1e16, and more stable thermals
# (vs the libgcc software __popcountdi2 emitted without these flags).
# For a machine you build AND run yourself, -march=native is strictly better
# (e.g. AVX-512 on recent CPUs) — swap it in there. Do NOT ship a
# -march=native binary to an older CPU than it was built on (illegal instruction).
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -funroll-loops -mavx2 -mbmi -mbmi2 -mpopcnt -DNDEBUG")

# 32-bit division (divl) for the A/C easy-leaf terms. Big win on pre-2020 x86
# with a slow 64-bit divider, but a small regression on fast dividers because the
# divisor is a 64-bit runtime value and the fast path costs a per-call guard
# branch. The right choice is not a vendor heuristic but the actual divl-vs-divq
# speed of the build host, so configure runs a small micro-benchmark
# (cmake/div32_probe.cpp) and defaults accordingly. See fast_div_u().
#
# Force it either way (skips the probe):  cmake -DWITH_DIV32=ON ...  (or =OFF)
if(NOT DEFINED WITH_DIV32)
  try_run(DIV32_RUN DIV32_COMPILE
    ${CMAKE_BINARY_DIR}/div32_probe
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/div32_probe.cpp
    CMAKE_FLAGS "-DCMAKE_CXX_FLAGS=-O3"
    RUN_OUTPUT_VARIABLE DIV32_OUT)
  if(DIV32_COMPILE AND (DIV32_RUN EQUAL 0))
    set(_div32_default ON)
  else()
    set(_div32_default OFF)
  endif()
  set(WITH_DIV32 ${_div32_default} CACHE BOOL
      "Emit 32-bit divl for A/C leaves (auto-probed on the build host)")
  message(STATUS "divl probe: ${DIV32_OUT}")
  message(STATUS "WITH_DIV32 (auto) = ${WITH_DIV32}")
else()
  message(STATUS "WITH_DIV32 (forced) = ${WITH_DIV32}")
endif()
if(WITH_DIV32)
  add_compile_definitions(ENABLE_DIV32)
endif()

find_package(OpenMP REQUIRED)

# Core library: everything except the CLI entry point. Shared by the binary and
# the self-test executable so terms can be validated in isolation.
file(GLOB CORE_SOURCES CONFIGURE_DEPENDS src/*.cpp src/gourdon/*.cpp)
list(REMOVE_ITEM CORE_SOURCES ${CMAKE_SOURCE_DIR}/src/main.cpp)

add_library(pc_core STATIC ${CORE_SOURCES})
target_include_directories(pc_core PUBLIC include)
target_link_libraries(pc_core PUBLIC OpenMP::OpenMP_CXX)

# Main CLI binary, placed at the project root so `./primecount` works.
add_executable(primecount src/main.cpp)
target_link_libraries(primecount PRIVATE pc_core)
set_target_properties(primecount PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# Self-tests (kept in the build dir): run with ./build/selftest
add_executable(selftest tests/selftest.cpp)
target_link_libraries(selftest PRIVATE pc_core)
