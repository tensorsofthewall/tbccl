# TBCCL native release archive

This archive is an installed TBCCL prefix: headers, static libraries, the CMake package and pkg-config metadata, and the `tbccl-info` tool.

```
include/tbccl/        public headers (C++ API and the stable C ABI, tbccl.h)
lib/libtbccl.a        C++ runtime
lib/libtbccl_c.a      stable C ABI shim
lib/libtbccl_cuda.a   CUDA memory provider (CUDA archives only)
lib/cmake/TBCCL/      CMake package
lib/pkgconfig/        pkg-config file
bin/tbccl-info        prints the version, ABI, wire protocol, platform and memory providers
LICENSE
```

Use it by extracting it anywhere and pointing your build at the extracted directory:

```
cmake -S app -B build -DCMAKE_PREFIX_PATH=/path/to/tbccl-<version>-<platform>
```

```cmake
find_package(TBCCL CONFIG REQUIRED)
target_link_libraries(app PRIVATE TBCCL::tbccl)      # C++ API
target_link_libraries(app PRIVATE TBCCL::tbccl_c)    # stable C ABI
```

or, without CMake, `pkg-config --cflags --libs tbccl` with `PKG_CONFIG_PATH=<prefix>/lib/pkgconfig`.

Run `bin/tbccl-info` (or `bin/tbccl-info --json`) to check what you have. CUDA archives need the CUDA Toolkit (13.x) on the machine that links against them; the macOS archive uses the Metal shared-memory provider. Linux archives are built in a manylinux_2_28 environment and need glibc 2.28 or newer.

Documentation: https://tbccl.tensorsofthewall.com/ Source and issues: https://github.com/tensorsofthewall/tbccl Licence: Apache-2.0.
