#!/bin/bash
# Build and run tests/external_consumer against an INSTALLED prefix only: check_external_consumer.sh <prefix>
# The consumer sees nothing but <prefix>/include and <prefix>/lib (it is configured in a scratch directory with the prefix as its only package path).
set -euo pipefail
PREFIX=${1:?installed TBCCL prefix}
SRC=$(cd "$(dirname "$0")/.." && pwd)/tests/external_consumer
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT
cmake -S "$SRC" -B "$B" -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release > "$B/cfg.txt" 2>&1 || { cat "$B/cfg.txt"; exit 1; }
grep "TBCCL " "$B/cfg.txt" || true
cmake --build "$B" -j > "$B/build.txt" 2>&1 || { cat "$B/build.txt"; exit 1; }
"$B/consumer"
[ -x "$B/consumer_cuda" ] && "$B/consumer_cuda"
exit 0
