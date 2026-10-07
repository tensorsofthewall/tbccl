#pragma once

// True per-chunk CUDA D2H/H2D async staging, as an AsyncMemoryBackend
// implementation. Declared separately from its .cu implementation
// (matching cuda_backend.hpp's own convention) so non-CUDA
// translation units never see a CUDA type..

#include <tbccl/async_transfer.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tbccl_bench::tensor
{

// One instance owns its own device source/destination buffers (NOT
// shared with CudaBackend/TensorBackend -- this class is
// self-contained.md's "why a new, narrower extension" rationale) and
// a persistent pinned host scratch buffer used as the intermediate
// hop for every chunk's D2H/ H2D copy. Always takes
// TensorCommWorker's staged path (supports_direct_transport_access()
// stays default-false); pass a nonzero chunk_hint on the
// TransferRequest to get real chunk-level pipelining, matching the
// plan's depth1 (chunk_hint == total_bytes, effectively one chunk) vs
// depthN sweep.
class CudaChunkedAsyncBackend final : public tbccl::AsyncMemoryBackend
{
public:
    // Throws std::runtime_error if no CUDA device is available.
    CudaChunkedAsyncBackend();

    // For the bucket-overlap benchmark, which needs ONE shared copy
    // stream across N bucket backends ("keep streams minimal... do NOT
    // add stream per bucket"), rather than one stream per instance.
    // `shared_copy_stream` must be a cudaStream_t cast to void*, and
    // must outlive this instance; this instance will NOT destroy it
    // (ownership stays with the caller). Passing nullptr (or using the
    // default constructor) preserves the CUDA/Metal device-pipeline
    // work's original behavior: create and own a private stream.
    explicit CudaChunkedAsyncBackend(void *shared_copy_stream);

    ~CudaChunkedAsyncBackend() override;

    CudaChunkedAsyncBackend(const CudaChunkedAsyncBackend &) = delete;
    CudaChunkedAsyncBackend &operator=(const CudaChunkedAsyncBackend &) = delete;

    // (Re)allocates device source/destination buffers for `bytes`
    // bytes each, and the pinned scratch buffer for `max_chunk_bytes`
    // bytes (persistent across iterations, never reallocated
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
    // the documented contract.
    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override;
    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override;

    // Diagnostic-only: counts of pinned-buffer allocations and CUDA
    // stream creations over this instance's lifetime. Expected to
    // stay at 1 each after allocate() -- any growth during a
    // measured loop indicates the "fresh pool per request" class of
    // bug the async fast-path work found and fixed for the host path
    // has crept back in for the device path.
    std::size_t diagnostic_pinned_alloc_count() const noexcept;
    std::size_t diagnostic_stream_create_count() const noexcept;

    // benchmark-only direct device-pointer access, so a compute
    // kernel (benchmarks/tensor/cuda_bucket_compute.hpp) can write
    // this backend's source buffer directly, and so a consumer-side
    // compute step (if ever added) could read the destination buffer.
    // Not used by TensorCommWorker or any generic code --
    // AsyncMemoryBackend's interface never exposes a raw device
    // pointer (generic TBCCL understands ready/ not-ready, never CUDA
    // types).
    void *source_device_ptr() noexcept;
    void *destination_device_ptr() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::size_t capacity_ = 0;
};

} // namespace tbccl_bench::tensor
