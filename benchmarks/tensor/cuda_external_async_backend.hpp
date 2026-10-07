#pragma once

// The external-CUDA-buffer counterpart to
// CudaChunkedAsyncBackend (the CUDA/Metal device-pipeline work). Unlike that class, this one does
// NOT allocate the tensor itself -- it wraps a caller-owned device
// pointer (external_device_ptr != nullptr) and never frees it, OR, when
// constructed with external_device_ptr == nullptr, allocates and owns
// `bytes` of device memory itself (used as the Communicator's own
// AllReduce-root scratch, which IS a TBCCL-owned resource, not the
// caller's tensor -- Part 25). Both modes share the exact same D2H/H2D
// staging mechanism (device <-> persistent pinned scratch <-> the
// TensorCommWorker-owned `staging` buffer) that CudaChunkedAsyncBackend
// already proved correct in CUDA/Metal device-pipeline.
//
// Declared separately from its .cu implementation, matching every other
// CUDA header in this directory, so non-CUDA translation units never see
// a CUDA type.

#include <tbccl/async_transfer.hpp>
#include <tbccl/hetero_allreduce.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace tbccl_bench::tensor
{

// communicator-scoped, persistent CUDA staging resources shared by every CudaExternalAsyncBackend that
// one Communicator creates. Replaces the previous per-collective cudaMallocHost/cudaFreeHost (payload-sized,
// measured at ~4 ms + ~1.3 ms for 25 MiB) and, for the AllReduce root, the per-collective cudaMalloc/cudaFree of the
// device scratch.
//
//  * Pinned host staging: ONE grow-only block. It is only ever used inside a single synchronous
//    stage_source_chunk/commit_destination_chunk call, so use is serialized by a mutex ("lease" == lock scope);
//    that keeps P2P, the AllReduce legs and AllGather's local copy safe even if they ever run concurrently.
//    Growth allocates the new block FIRST and frees the old one only after the swap, under the same lock, so a
//    failed growth leaves the old block usable. It never shrinks; it is freed once, at destruction.
//  * Device scratch (AllReduce root landing zone): one grow-only block handed out as a shared lease; growth never
//    frees a block a live lease still references. A Communicator runs one collective at a time (CollectiveExecutor),
//    so one scratch block is sufficient.
class CudaStagingResources
{
public:
    struct Stats
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

    CudaStagingResources();
    ~CudaStagingResources();
    CudaStagingResources(const CudaStagingResources &) = delete;
    CudaStagingResources &operator=(const CudaStagingResources &) = delete;

    // Calls fn(pinned_pointer) with at least `bytes` of pinned host memory, holding the staging lock.
    void with_pinned(std::size_t bytes, const std::function<void(void *)> &fn);

    // Lease on a device block of at least `bytes` (grow-only).
    std::shared_ptr<void> acquire_device_scratch(std::size_t bytes);

    Stats stats() const;

    // Test seam: the next pinned growth fails (as if cudaMallocHost returned an error).
    void fail_next_pinned_growth();

private:
    struct State;
    std::unique_ptr<State> s_;
};

class CudaExternalAsyncBackend final : public tbccl::AsyncMemoryBackend
{
public:
    // external_device_ptr != nullptr: wraps that caller-owned device pointer non-owning (TBCCL never cudaFree()s
    // it). external_device_ptr == nullptr: allocates and owns `bytes` of device memory itself (for
    // Communicator-owned scratch). `shared_copy_stream` behaves exactly as in CudaChunkedAsyncBackend (nullptr ->
    // create and own a private stream). Throws std::runtime_error if no CUDA device is available. `resources`
    // (optional, the persistent CUDA staging work): communicator-scoped persistent staging; when null the backend
    // owns private per-instance pinned/device storage exactly as before.
    CudaExternalAsyncBackend(
        void *external_device_ptr,
        std::size_t bytes,
        void *shared_copy_stream = nullptr,
        std::shared_ptr<CudaStagingResources> resources = nullptr);

    ~CudaExternalAsyncBackend() override;

    CudaExternalAsyncBackend(const CudaExternalAsyncBackend &) = delete;
    CudaExternalAsyncBackend &operator=(const CudaExternalAsyncBackend &) = delete;

    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override;
    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override;

    // If `producer_stream` is non-null, records an event on it and makes
    // this backend's own copy stream wait on that event before the next
    // stage_source_chunk() -- so a D2H never starts reading the external
    // buffer before the caller's own producer work on `producer_stream`
    // has completed, without a blocking cudaDeviceSynchronize(). Uses
    // one persistent, reused cudaEvent_t (never allocated per call). A
    // no-op if producer_stream is null (the Host-execution-context
    // convention the other backends already use: the buffer is assumed
    // ready).
    void wait_for_producer_stream(void *producer_stream);

    void *device_ptr() const noexcept;
    std::size_t capacity() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// After an H2D into an external CUDA destination completes (Work::wait()
// returned), launches an independent consumer kernel reading directly
// from the device pointer to prove genuine GPU visibility, not merely
// host-readback correctness. out[i] = in[i]*2 for `count` int32 elements
// (matches the bucketed benchmark's existing hash-mix-as-int32
// convention); synchronizes `stream` before returning.
void cuda_external_launch_consumer_double_i32(
    const void *in, void *out, std::size_t count, void *stream);

// Test-only: creates a new CUDA stream, launches a kernel on it that busy-spins for approximately
// `spin_iterations` device clock cycles before writing `value` to every int32 element of `device_ptr`, and
// returns the stream WITHOUT synchronizing it -- the caller immediately passes this stream back as an
// ExecutionContext to prove (or disprove) that TBCCL's stream-dependency wait is actually enforced, rather
// than racing ahead and reading stale/uninitialized data. Caller owns the returned stream and must
// cudaStreamDestroy() it after use (cast back from void*). `non_blocking` creates the stream with
// cudaStreamNonBlocking (like PyTorch's streams): a blocking stream is implicitly ordered against the legacy
// default stream (which TBCCL's reduce kernel uses), which would hide a missing explicit dependency from a
// negative control.
void *cuda_external_test_launch_delayed_write_i32(
    void *device_ptr, std::size_t count, std::int32_t value, std::size_t spin_iterations, bool non_blocking = false);

// Test-only: destroys a stream returned by the function above (keeps
// cuda_runtime.h out of test .cpp files, matching this directory's
// existing convention).
void cuda_external_test_destroy_stream(void *stream);

class CudaExternalReduceBackend final : public tbccl::LocalReduceBackend
{
public:
    // `local_and_output`/`peer` are raw device pointers (NOT host
    // memory); local_and_output += peer, written back into
    // local_and_output. `stream` synchronizes before returning, matching
    // CudaReduceBackend's proven "cudaStreamSynchronize is sufficient"
    // result (the heterogeneous all-reduce work). Passing nullptr uses
    // the default stream.
    CudaExternalReduceBackend(void *local_and_output_device_ptr, const void *peer_device_ptr, void *stream);

    void reduce_sum(std::size_t count, tbccl::DataType datatype) override;

private:
    void *local_and_output_;
    const void *peer_;
    void *stream_;
};

} // namespace tbccl_bench::tensor
