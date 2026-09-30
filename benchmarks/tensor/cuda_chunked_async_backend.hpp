#pragma once

// Phase 35: true per-chunk CUDA D2H/H2D async staging, as an
// AsyncMemoryBackend implementation. Declared separately from its .cu
// implementation (matching cuda_backend.hpp's own convention) so
// non-CUDA translation units never see a CUDA type. See
// docs/phase35_device_pipeline_design.md for the full design
// rationale.

#include <tbccl/async_transfer.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tbccl_bench::tensor
{

// One instance owns its own device source/destination buffers (NOT
// shared with CudaBackend/TensorBackend -- this class is
// self-contained, per docs/phase35_device_pipeline_design.md's "why a
// new, narrower extension" rationale) and a persistent pinned host
// scratch buffer used as the intermediate hop for every chunk's D2H/
// H2D copy. Always takes TensorCommWorker's staged path
// (supports_direct_transport_access() stays default-false); pass a
// nonzero chunk_hint on the TransferRequest to get real chunk-level
// pipelining, matching the plan's depth1 (chunk_hint == total_bytes,
// effectively one chunk) vs depthN sweep.
class CudaChunkedAsyncBackend final : public tbccl::AsyncMemoryBackend
{
public:
    // Throws std::runtime_error if no CUDA device is available.
    CudaChunkedAsyncBackend();
    ~CudaChunkedAsyncBackend() override;

    CudaChunkedAsyncBackend(const CudaChunkedAsyncBackend &) = delete;
    CudaChunkedAsyncBackend &operator=(const CudaChunkedAsyncBackend &) = delete;

    // (Re)allocates device source/destination buffers for `bytes`
    // bytes each, and the pinned scratch buffer for `max_chunk_bytes`
    // bytes (persistent across iterations -- Part W, never reallocated
    // mid-measured-loop as long as the same (bytes, max_chunk_bytes)
    // is requested again).
    void allocate(std::size_t bytes, std::size_t max_chunk_bytes);

    std::size_t capacity() const noexcept { return capacity_; }

    // Device-side deterministic pattern fill (same pattern_byte()
    // formula as tensor_backend.hpp) + block until it completes.
    void initialize_source(std::uint32_t seed);

    // True byte-for-byte GPU readback verification (never a
    // checksum), for source and destination respectively.
    bool verify_source(std::uint32_t seed) const;
    bool verify_destination(std::uint32_t seed) const;

    // AsyncMemoryBackend overrides -- called only from
    // TensorCommWorker's staging thread (never the enqueuing caller's
    // thread), so blocking inside these is explicitly permitted by
    // Phase 32's own documented contract.
    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override;
    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override;

    // Diagnostic-only (Part W/X item 84/87): counts of pinned-buffer
    // allocations and CUDA stream creations over this instance's
    // lifetime. Expected to stay at 1 each after allocate() -- any
    // growth during a measured loop indicates the "fresh pool per
    // request" class of bug Phase 33 found and fixed for the host
    // path has crept back in for the device path.
    std::size_t diagnostic_pinned_alloc_count() const noexcept;
    std::size_t diagnostic_stream_create_count() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::size_t capacity_ = 0;
};

} // namespace tbccl_bench::tensor
