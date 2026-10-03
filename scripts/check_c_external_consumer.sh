#!/bin/bash
# Build and run tests/c_external_consumer (C language only) against an INSTALLED prefix only: check_c_external_consumer.sh <prefix>
set -euo pipefail
PREFIX=${1:?installed TBCCL prefix}
SRC=$(cd "$(dirname "$0")/.." && pwd)/tests/c_external_consumer
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT
cmake -S "$SRC" -B "$B" -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release > "$B/cfg.txt" 2>&1 || { cat "$B/cfg.txt"; exit 1; }
grep "TBCCL " "$B/cfg.txt" || true
cmake --build "$B" -j > "$B/build.txt" 2>&1 || { cat "$B/build.txt"; exit 1; }
# the consumer must not have linked a C++ compiler of its own: the shim's exported link interface carries the C++ runtime
if grep -q "CMAKE_CXX_COMPILER" "$B/CMakeCache.txt" 2>/dev/null; then echo "note: a C++ compiler was enabled by the package"; fi
"$B/c_consumer"
[ -x "$B/c_consumer_cuda" ] && "$B/c_consumer_cuda"
echo "C external consumer ok"
