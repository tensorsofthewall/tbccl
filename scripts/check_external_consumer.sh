#!/bin/bash
# Build and run tests/external_consumer against an INSTALLED prefix only: check_external_consumer.sh <prefix>
# The consumer sees nothing but <prefix>/include and <prefix>/lib (it is configured in a scratch directory with the prefix as its only package path).
set -euo pipefail
PREFIX=${1:?installed TBCCL prefix}
SRC=$(cd "$(dirname "$0")/.." && pwd)/tests/external_consumer
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
"$B/consumer"
run_cuda "$B/consumer_cuda"
# Version policy (SameMajorVersion): a consumer asking for a newer MAJOR version must be rejected, not silently accepted.
if cmake -S "$SRC" -B "$B/reject" -DCMAKE_PREFIX_PATH="$PREFIX" -DTBCCL_REQUIRED_VERSION=1.0 > "$B/reject.txt" 2>&1; then
    echo "package version policy broken: find_package(TBCCL 1.0) was accepted"; exit 1
fi
grep -qi "version" "$B/reject.txt" && echo "package version policy ok (0.x install accepts older 0.y requests, rejects 1.0)"
exit 0
