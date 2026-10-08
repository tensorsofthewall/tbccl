#!/bin/bash
# Build and run the two documented install examples (examples/version_check.c and examples/two_rank_threads.cpp) against an INSTALLED prefix,
# once through CMake find_package and once through pkg-config: check_doc_examples.sh <prefix>
set -euo pipefail
PREFIX=$(cd "${1:?installed TBCCL prefix}" && pwd)
SRC=$(cd "$(dirname "$0")/.." && pwd)/examples
B=$(mktemp -d)
trap 'rm -rf "$B"' EXIT
cat > "$B/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.20)
project(doc_examples C CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(TBCCL CONFIG REQUIRED)
add_executable(version_check @SRC@/version_check.c)
target_link_libraries(version_check PRIVATE TBCCL::tbccl_c)
add_executable(two_rank_threads @SRC@/two_rank_threads.cpp)
target_link_libraries(two_rank_threads PRIVATE TBCCL::tbccl)
EOF
sed -i "s#@SRC@#$SRC#g" "$B/CMakeLists.txt"
cmake -S "$B" -B "$B/cmake" -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release > "$B/cfg.txt" 2>&1 || { cat "$B/cfg.txt"; exit 1; }
cmake --build "$B/cmake" -j > "$B/build.txt" 2>&1 || { cat "$B/build.txt"; exit 1; }
"$B/cmake/version_check"
"$B/cmake/two_rank_threads"
# pkg-config: the .pc file is relocatable (prefix is derived from its own location). It does not name the C++ runtime that the static
# libraries need, so a C program adds it itself (-lstdc++ on Linux, -lc++ on macOS); a CUDA prefix also needs the CUDA runtime (PKG_EXTRA_LIBS,
# for example "-L/usr/local/cuda/lib64 -lcudart").
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig"
CXXRT=-lstdc++; [ "$(uname)" = Darwin ] && CXXRT=-lc++
cc "$SRC/version_check.c" $(pkg-config --cflags tbccl) $(pkg-config --libs tbccl) $CXXRT ${PKG_EXTRA_LIBS:-} -o "$B/version_check_pc"
c++ -std=c++17 "$SRC/two_rank_threads.cpp" $(pkg-config --cflags tbccl) $(pkg-config --libs tbccl) ${PKG_EXTRA_LIBS:-} -o "$B/two_rank_pc" -pthread
"$B/version_check_pc"
"$B/two_rank_pc"
[ -x "$PREFIX/bin/tbccl-info" ] && "$PREFIX/bin/tbccl-info"
echo "doc examples ok"
