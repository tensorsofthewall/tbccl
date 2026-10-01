#pragma once

// Phase 41 Part P/AJ: the public entry point an external application
// calls to enable tbccl::MemoryKind::Cuda support in the Communicator
// API. This header has zero CUDA dependency (matches every other
// include/tbccl/ header) -- it only declares the registration function;
// the actual CUDA implementation (which does need CUDA types) is
// compiled into the optional device component and linked in by the
// application (see docs/public_api.md's install/link instructions).
// Calling this is the ONLY CUDA-specific step a public-API consumer
// ever needs -- no benchmark header is included, matching Part AE/155-156.

namespace tbccl
{

// Registers the CUDA tbccl::ExternalMemoryProvider factory for
// MemoryKind::Cuda (communicator.hpp's register_memory_provider_factory()
// extension point). Call this once, before constructing any Communicator
// that will move MemoryKind::Cuda buffers. This is declared here (not
// defined) -- linking it requires the optional CUDA-enabled device
// component (see docs/public_api.md); an application that never calls
// it, or never links that component, simply never gets Cuda-kind
// support (Communicator returns ErrorCode::Unsupported for it, same as
// any other unregistered MemoryKind).
void register_cuda_support();

} // namespace tbccl
