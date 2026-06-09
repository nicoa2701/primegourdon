#!/usr/bin/env bash
# run.sh — build (robustly) and optionally run primecount.
#
# Robust to the project being moved or to a stale CMakeCache pointing at an old
# path: the script always uses ITS OWN directory as the source tree, and wipes
# build/ if the cache was generated for a different source/build path.
#
# Usage:
#   ./run.sh                      # configure + build only
#   ./run.sh 1e16                 # build, then ./primecount 1e16
#   ./run.sh 1e16 --D -v          # build, then ./primecount 1e16 --D -v
#   ./run.sh --clean              # force a clean reconfigure, then build
#   ./run.sh --selftest           # build, then run ./build/selftest
#   ./run.sh -DWITH_DIV32=ON 1e16 # forward any -D... to the CMake configure
#
# Arg dispatch: --clean / --selftest are handled here; any -D... goes to the
# CMake configure (sticky in the cache -- use --clean to fall back to the
# auto-probed default); everything else is forwarded to ./primecount.
set -euo pipefail

# Absolute path to this script's directory == the project root, regardless of
# the current working directory or where the tree was moved to.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BUILD="$ROOT/build"

CLEAN=0
SELFTEST=0
ARGS=()
CMAKE_ARGS=()
for a in "$@"; do
  case "$a" in
    --clean)    CLEAN=1 ;;
    --selftest) SELFTEST=1 ;;
    -D*)        CMAKE_ARGS+=("$a") ;;  # CMake configure option, e.g. -DWITH_DIV32=ON
    *)          ARGS+=("$a") ;;        # everything else -> ./primecount
  esac
done

# A CMakeCache is stale if it was generated for a different source or build dir
# (happens when the folder is moved/copied). CMake would then error out, so we
# detect it ourselves and wipe the build dir for a clean reconfigure.
cache_is_stale() {
  local cache="$BUILD/CMakeCache.txt"
  [ -f "$cache" ] || return 1  # no cache -> not stale (fresh configure)
  local home bdir
  home="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cache" | head -1)"
  bdir="$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "$cache" | head -1)"
  [ -n "$home" ] && [ "$home" != "$ROOT" ]  && return 0
  [ -n "$bdir" ] && [ "$bdir" != "$BUILD" ] && return 0
  return 1
}

if [ "$CLEAN" -eq 1 ] || cache_is_stale; then
  [ -d "$BUILD" ] && echo ">> wiping stale/old build dir: $BUILD"
  rm -rf "$BUILD"
fi

JOBS="$(nproc 2>/dev/null || echo 4)"
echo ">> configure: $ROOT -> $BUILD ${CMAKE_ARGS[*]:-}"
# -D... args (e.g. -DWITH_DIV32=ON) are forwarded to the configure. They are
# sticky in the CMake cache; pass --clean (or the opposite -D...=OFF) to change
# back, or omit them entirely after a --clean to fall back to the auto-probe.
# Configure output is hidden, but we surface the auto-probe decisions and the
# resulting per-term division backend so it's visible which div is active.
CFG_OUT="$(cmake -S "$ROOT" -B "$BUILD" "${CMAKE_ARGS[@]+"${CMAKE_ARGS[@]}"}" 2>&1)" \
  || { echo "$CFG_OUT"; exit 1; }
echo "$CFG_OUT" | grep -E 'probe:|divider backend:' || true
echo ">> build (-j$JOBS)"
cmake --build "$BUILD" -j"$JOBS"

if [ "$SELFTEST" -eq 1 ]; then
  echo ">> selftest"
  exec "$BUILD/selftest"
fi

if [ "${#ARGS[@]}" -gt 0 ]; then
  echo ">> ./primecount ${ARGS[*]}"
  exec "$ROOT/primecount" "${ARGS[@]}"
fi
