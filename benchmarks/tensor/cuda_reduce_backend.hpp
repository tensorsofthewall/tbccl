#pragma once

// A tiny CUDA SUM kernel wrapped as a LocalReduceBackend
// (tbccl/hetero_allreduce.hpp), operating directly on a
// CudaChunkedAsyncBackend's own source/destination device buffers --
// output[i] = source[i] + destination[i], written back into source[i] so
// the same CudaChunkedAsyncBackend instance's stage_source_chunk() (which
// always reads from its source buffer) sends the combined result for the
// broadcast-back leg with no extra buffer or copy (see
// Part 4).
//
// Declared separately from its .cu implementation (matching every other
// CUDA header in this directory) so non-CUDA translation units never see
// a CUDA type.

#include <tbccl/hetero_allreduce.hpp>

#include "cuda_chunked_async_backend.hpp"

#include <cstddef>

namespace tbccl_bench::tensor
{

class CudaReduceBackend final : public tbccl::LocalReduceBackend
{
public:
    // `backend` must outlive this instance and must already be allocated
    // (allocate() called) with source containing this rank's local input
    // and destination about to receive (or having already received) the
    // peer's contribution. `stream` is an explicit CUDA stream (cast to
    // void*) this reduction launches on and synchronizes before
    // returning -- reusing the proven
    // "cudaStreamSynchronize is sufficient, no cudaEvent_t needed"
    // result. Passing nullptr uses the default stream.
    CudaReduceBackend(CudaChunkedAsyncBackend &backend, void *stream);

    void reduce_sum(std::size_t count, tbccl::DataType datatype) override;

private:
    CudaChunkedAsyncBackend &backend_;
    void *stream_;
};

// Test-only plain memcpy helpers (hardware correctness tests need to seed
// a known peer value into a device buffer and read back a post-reduction
// result) -- kept here rather than in a .cpp test file so no test
// translation unit needs to include <cuda_runtime.h> directly, matching
// this codebase's existing CUDA-type-free-header convention.
void cuda_copy_host_to_device(const void *host_src, void *device_dst, std::size_t bytes);
void cuda_copy_device_to_host(const void *device_src, void *host_dst, std::size_t bytes);

// Proves a just-completed AllReduce's result is genuinely GPU-consumable
// (not merely host-readback-correct) by running a second, independent
// CUDA kernel directly over it: out[i] = in[i] * 2, for `count` float32
// elements. `in`/`out` are device pointers (`out` may equal `in`);
// synchronizes `stream` before returning.
void cuda_launch_consumer_double_f32(
    const void *in, void *out, std::size_t count, void *stream);

} // namespace tbccl_bench::tensor
