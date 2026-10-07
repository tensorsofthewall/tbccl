# License and third-party software

tbccl is licensed under the Apache License, Version 2.0. The full text is in the `LICENSE` file at the repository root. Copyright 2026 Sandesh Bharadwaj.

Contributions are accepted under the same license (Apache-2.0, section 5).

## What TBCCL contains

TBCCL contains no vendored third-party source code and no copied or adapted third-party code. The source tree builds with a C++17 compiler and CMake. CUDA support (optional) uses the CUDA Toolkit's runtime library as a shared library at build and run time; TBCCL does not redistribute any NVIDIA library. Metal support uses system frameworks on macOS.

The installed prefix contains TBCCL's own static libraries and headers. Because the libraries are static, a program or package that links TBCCL embeds TBCCL code; such a package must carry TBCCL's license text (see the torch-tbccl and exo-tbccl pages).

## Documentation build

The documentation toolchain (Sphinx, MyST, Furo, Breathe, Doxygen and their dependencies) is used only to build the documentation and is not redistributed with TBCCL. The pinned versions are in `docs/requirements.txt`.

## Notices

No third-party `NOTICE` file or third-party license bundle is required for TBCCL, because it redistributes no third-party code. A `NOTICE` file is not provided.
