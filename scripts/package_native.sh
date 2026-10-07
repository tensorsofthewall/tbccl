#!/usr/bin/env bash
# Build, stage, archive and verify a TBCCL native release archive.
#   scripts/package_native.sh [--cuda] [--cuda-arch "75-real;80-real;...;120"] [--label rc1] [--out DIR] [--jobs N]
# Writes DIR/tbccl-<version>[<label>]-<os>-<arch>[-cuda<major>].tar.gz (+ .sha256). The tarball is deterministic for a given tree and toolchain
# (sorted names, fixed owner, mtime from SOURCE_DATE_EPOCH or the commit time). The archive is then extracted into a fresh directory and
# checked: tbccl-info runs, the out-of-tree C++ and C consumers configure/build/run against it, and no build path leaks into any file.
set -euo pipefail

SRC=$(cd "$(dirname "$0")/.." && pwd)
CUDA=0; CUDA_ARCH="75-real;80-real;86-real;89-real;90-real;100-real;120"; LABEL=""; OUT="$SRC/dist"; JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
while [ $# -gt 0 ]; do
    case "$1" in
        --cuda) CUDA=1 ;;
        --cuda-arch) CUDA_ARCH="$2"; shift ;;
        --label) LABEL="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --jobs) JOBS="$2"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

VERSION=$(sed -n '/^project(/,/)/p' "$SRC/CMakeLists.txt" | sed -n 's/^ *VERSION *\([0-9][0-9.]*\).*/\1/p' | head -1)
[ -n "$VERSION" ] || { echo "cannot read the project version" >&2; exit 1; }
case "$(uname -s)" in Linux) OS=linux ;; Darwin) OS=macos ;; *) echo "unsupported OS" >&2; exit 1 ;; esac
case "$(uname -m)" in x86_64|amd64) ARCH=x86_64 ;; arm64|aarch64) ARCH=arm64 ;; *) echo "unsupported arch" >&2; exit 1 ;; esac
EPOCH=${SOURCE_DATE_EPOCH:-$(git -C "$SRC" log -1 --format=%ct 2>/dev/null || echo 0)}

CFG=(-DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF)
PFX_MAP="-ffile-prefix-map=$SRC=. -fdebug-prefix-map=$SRC=."
SUFFIX=""
if [ "$CUDA" = 1 ]; then
    CUDA_MAJOR=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
    CFG+=(-DTBCCL_ENABLE_CUDA=ON -DTBCCL_CUDA_STATIC_RUNTIME=ON "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH" "-DCMAKE_CUDA_FLAGS=-Xcompiler=-ffile-prefix-map=$SRC=.")
    SUFFIX="-cuda${CUDA_MAJOR}"
fi
if [ "$OS" = macos ]; then CFG+=(-DTBCCL_ENABLE_METAL=ON "-DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-14.0}"); fi
NAME="tbccl-${VERSION}${LABEL}-${OS}-${ARCH}${SUFFIX}"

WORK=$(mktemp -d "${TMPDIR:-/tmp}/tbccl-package.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
BUILD="$WORK/build"; STAGE="$WORK/$NAME"; mkdir -p "$OUT"

cmake -S "$SRC" -B "$BUILD" "${CFG[@]}" "-DCMAKE_CXX_FLAGS=$PFX_MAP" "-DCMAKE_C_FLAGS=$PFX_MAP" >/dev/null
cmake --build "$BUILD" -j "$JOBS" --target tbccl tbccl_c tbccl_info $([ "$CUDA" = 1 ] && echo tbccl_cuda) 2>&1 | tail -2
cmake --install "$BUILD" --prefix "$STAGE" >/dev/null
cp "$SRC/LICENSE" "$STAGE/LICENSE"
cp "$SRC/packaging/README-native.md" "$STAGE/README.md"

( cd "$WORK" && tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$EPOCH" --format=posix --pax-option=delete=atime,delete=ctime \
      -cf - "$NAME" | gzip -n -9 > "$OUT/$NAME.tar.gz" ) 2>/dev/null || {
    # bsdtar (macOS) has no --sort/--pax-option: use an explicit sorted file list
    ( cd "$WORK" && find "$NAME" | LC_ALL=C sort | tar -cf - --no-recursion --uid 0 --gid 0 -T - 2>/dev/null | gzip -n -9 > "$OUT/$NAME.tar.gz" )
}
( cd "$OUT" && { sha256sum "$NAME.tar.gz" 2>/dev/null || shasum -a 256 "$NAME.tar.gz"; } > "$NAME.tar.gz.sha256" )

# ---- verify the ARCHIVE, not the build tree ----
CHECK="$WORK/check"; mkdir -p "$CHECK"; tar -xzf "$OUT/$NAME.tar.gz" -C "$CHECK"
P="$CHECK/$NAME"
for f in include/tbccl/tbccl.h lib/libtbccl.a lib/libtbccl_c.a lib/cmake/TBCCL/TBCCLConfig.cmake lib/pkgconfig/tbccl.pc bin/tbccl-info LICENSE README.md; do
    [ -e "$P/$f" ] || { echo "archive is missing $f" >&2; exit 1; }
done
[ "$CUDA" = 0 ] || [ -e "$P/lib/libtbccl_cuda.a" ] || { echo "CUDA archive has no libtbccl_cuda.a" >&2; exit 1; }
echo "-- tbccl-info:"; "$P/bin/tbccl-info"; "$P/bin/tbccl-info" --json
if ! "$P/bin/tbccl-info" --json | grep -q "\"package_version\": \"$VERSION\""; then echo "tbccl-info reports a different version than $VERSION" >&2; exit 1; fi
echo "-- out-of-tree consumers against the extracted archive:"
"$SRC/scripts/check_external_consumer.sh" "$P" | tail -3
"$SRC/scripts/check_c_external_consumer.sh" "$P" | tail -3
if command -v pkg-config >/dev/null; then
    PKG_CONFIG_PATH="$P/lib/pkgconfig" pkg-config --cflags --libs tbccl
    echo 'int main(void){return 0;}' > "$WORK/pc.c"
    # shellcheck disable=SC2046
    cc "$WORK/pc.c" $(PKG_CONFIG_PATH="$P/lib/pkgconfig" pkg-config --cflags tbccl) -o "$WORK/pc" && echo "pkg-config cflags compile ok"
fi
echo "-- private-path scan:"
if grep -r -a -l -E "$SRC|$WORK|/home/|/Users/" "$P" 2>/dev/null; then echo "a build or home path leaked into the archive" >&2; exit 1; fi
echo "clean"
echo "ARCHIVE $OUT/$NAME.tar.gz"
