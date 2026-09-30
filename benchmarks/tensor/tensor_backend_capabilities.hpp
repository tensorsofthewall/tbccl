#pragma once

// Phase 32: extends tbccl::local_capabilities() (which is deliberately
// CUDA/Metal-unaware, see tensor_backend.hpp's docstring on the
// core-library/benchmark boundary) with the memory backends and async
// capabilities this specific build actually compiled in
// (TBCCL_ENABLE_CUDA/TBCCL_ENABLE_METAL). Lives under benchmarks/, not
// include/tbccl/, for the same reason tensor_backend.hpp does.

#include <tbccl/peer_capabilities.hpp>

namespace tbccl_bench::tensor
{

// Returns tbccl::local_capabilities() with CudaPageable/CudaPinned
// and/or MetalShared/MetalPrivateStaged (plus the matching
// AsyncCapability) appended when this binary was built with
// TBCCL_ENABLE_CUDA/TBCCL_ENABLE_METAL respectively. A build with
// neither enabled returns exactly tbccl::local_capabilities()
// unchanged (host-only, matching Part AT's requirement that a
// host-only build still works).
tbccl::PeerCapabilities local_tensor_capabilities();

} // namespace tbccl_bench::tensor
