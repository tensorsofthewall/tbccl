#!/usr/bin/env bash
# Runs INSIDE the manylinux_2_28 container: scripts/ci/linux_archive.sh <host|cuda> <public-version-or-empty> <out-dir>
# Builds the native archive twice from the same tree and requires identical SHA-256 (the archive is deterministic), then leaves one copy in <out-dir>.
set -euo pipefail
VARIANT=${1:?host|cuda}; PUBLIC=${2:-}; OUT=${3:?output directory}
SRC=$(cd "$(dirname "$0")/../.." && pwd)
git config --global --add safe.directory '*'
ARGS=(); [ -z "$PUBLIC" ] || ARGS+=(--public-version "$PUBLIC")
if [ "$VARIANT" = cuda ]; then
    dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
    # CUDA 13.0, the toolkit the Linux PyTorch 2.13 wheel is built with; the packages are pinned and their installed versions are recorded
    dnf install -y -q cuda-nvcc-13-0-13.0.88-1 cuda-cudart-devel-13-0-13.0.96-1 cuda-driver-devel-13-0-13.0.96-1
    export PATH=/usr/local/cuda-13.0/bin:$PATH
    rpm -qa 'cuda*' 'libnv*' | sort > "$OUT/cuda-toolkit-packages.txt"
    ARGS+=(--cuda)
fi
mkdir -p "$OUT/a" "$OUT/b"
"$SRC/scripts/package_native.sh" "${ARGS[@]}" --out "$OUT/a"
"$SRC/scripts/package_native.sh" "${ARGS[@]}" --out "$OUT/b"
A=$(sha256sum "$OUT"/a/*.tar.gz | cut -d' ' -f1); B=$(sha256sum "$OUT"/b/*.tar.gz | cut -d' ' -f1)
echo "build 1: $A"; echo "build 2: $B"
[ "$A" = "$B" ] || { echo "the archive is not reproducible: two builds of the same tree differ" >&2; exit 1; }
echo "reproducible: yes ($VARIANT)"
cp "$OUT"/a/*.tar.gz "$OUT"/
rm -rf "$OUT/a" "$OUT/b"
