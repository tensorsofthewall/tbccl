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

#include <tbccl/communicator.hpp>

#include <cstdint>

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

// Phase 44: per-Communicator persistent CUDA staging counters (diagnostics/tests). Pinned staging is a single
// grow-only block owned by the Communicator; in steady state (capacity already sufficient) no collective allocates
// or frees pinned memory, and the AllReduce root's device scratch is likewise reused.
struct CudaStagingStats
{
    std::uint64_t pinned_alloc_count = 0;
    std::uint64_t pinned_free_count = 0;
    std::uint64_t pinned_allocated_bytes_total = 0;
    std::uint64_t pinned_capacity = 0;
    std::uint64_t pinned_peak_capacity = 0;
    std::uint64_t device_scratch_alloc_count = 0;
    std::uint64_t device_scratch_free_count = 0;
    std::uint64_t device_scratch_capacity = 0;
};

// False if `comm` has not used CUDA memory yet (no resources exist).
bool cuda_staging_stats(const Communicator &comm, CudaStagingStats &out);

// Test seam only: makes the next pinned-staging growth of `comm` fail.
void cuda_staging_test_fail_next_pinned_growth(const Communicator &comm);

} // namespace tbccl
