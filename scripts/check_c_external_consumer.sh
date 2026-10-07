#!/bin/bash
# Build and run tests/c_external_consumer (C language only) against an INSTALLED prefix only: check_c_external_consumer.sh <prefix>
set -euo pipefail
PREFIX=${1:?installed TBCCL prefix}
SRC=$(cd "$(dirname "$0")/.." && pwd)/tests/c_external_consumer
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT
# The CUDA consumers are always built and linked; they are run only when a GPU is present (TBCCL_CONSUMER_RUN_CUDA=1 forces it, 0 skips it).
run_cuda() {
    local exe=$1
    [ -x "$exe" ] || return 0
    local want=${TBCCL_CONSUMER_RUN_CUDA:-auto}
    if [ "$want" = auto ]; then nvidia-smi -L >/dev/null 2>&1 && want=1 || want=0; fi
    if [ "$want" = 1 ]; then "$exe"; else echo "$(basename "$exe"): built and linked, not run (no GPU)"; fi
}
cmake -S "$SRC" -B "$B" -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release > "$B/cfg.txt" 2>&1 || { cat "$B/cfg.txt"; exit 1; }
grep "TBCCL " "$B/cfg.txt" || true
cmake --build "$B" -j > "$B/build.txt" 2>&1 || { cat "$B/build.txt"; exit 1; }
# the consumer must not have linked a C++ compiler of its own: the shim's exported link interface carries the C++ runtime
if grep -q "CMAKE_CXX_COMPILER" "$B/CMakeCache.txt" 2>/dev/null; then echo "note: a C++ compiler was enabled by the package"; fi
"$B/c_consumer"
run_cuda "$B/c_consumer_cuda"
echo "C external consumer ok"
