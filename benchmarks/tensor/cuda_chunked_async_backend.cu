#include "cuda_chunked_async_backend.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tbccl_bench::tensor
{

namespace
{

    void check_cuda(cudaError_t status, const char *what)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("CUDA error in ") + what + ": " +
                cudaGetErrorString(status));
        }
    }

} // namespace

} // namespace tbccl_bench::tensor

#define TBCCL_CHUNKED_CUDA_CHECK(expr) \
    ::tbccl_bench::tensor::check_cuda((expr), #expr)

namespace tbccl_bench::tensor
{

namespace
{

    // Intentional duplicate of cuda_backend.cu's fill_pattern_kernel
    // (same convention as cuda_sync_bench.cu's own documented
    // duplicate) -- this file must stay self-contained
    __global__ void fill_pattern_kernel(
        std::uint8_t *data, std::size_t count, std::uint32_t seed)
    {
        const std::size_t i =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count) return;

        const std::uint64_t index = static_cast<std::uint64_t>(i);
        const std::uint64_t value =
            index * 131u + (index >> 8) * 17u + static_cast<std::uint64_t>(seed);
        data[i] = static_cast<std::uint8_t>(value & 0xffu);
    }

} // namespace

struct CudaChunkedAsyncBackend::Impl
{
    void *source_device = nullptr;
    void *destination_device = nullptr;
    void *pinned_scratch = nullptr; // max_chunk_bytes, reused every call
    std::size_t max_chunk_bytes = 0;
    cudaStream_t stream = nullptr;

    std::size_t pinned_alloc_count = 0;
    std::size_t stream_create_count = 0;

    ~Impl()
    {
        if (source_device) cudaFree(source_device);
        if (destination_device) cudaFree(destination_device);
        if (pinned_scratch) cudaFreeHost(pinned_scratch);
        if (stream) cudaStreamDestroy(stream);
    }
};

CudaChunkedAsyncBackend::CudaChunkedAsyncBackend() : impl_(std::make_unique<Impl>())
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0)
    {
        throw std::runtime_error(
            "CudaChunkedAsyncBackend: no CUDA device available at runtime: " +
            std::string(cudaGetErrorString(status)));
    }
    TBCCL_CHUNKED_CUDA_CHECK(cudaStreamCreate(&impl_->stream));
    ++impl_->stream_create_count;
}

CudaChunkedAsyncBackend::~CudaChunkedAsyncBackend() = default;

void CudaChunkedAsyncBackend::allocate(std::size_t bytes, std::size_t max_chunk_bytes)
{
    if (bytes == capacity_ && max_chunk_bytes == impl_->max_chunk_bytes &&
        impl_->source_device != nullptr)
    {
        return; // persistent across iterations, no reallocation
    }

    if (impl_->source_device) { cudaFree(impl_->source_device); impl_->source_device = nullptr; }
    if (impl_->destination_device) { cudaFree(impl_->destination_device); impl_->destination_device = nullptr; }
    if (impl_->pinned_scratch) { cudaFreeHost(impl_->pinned_scratch); impl_->pinned_scratch = nullptr; }

    if (bytes > 0)
    {
        TBCCL_CHUNKED_CUDA_CHECK(cudaMalloc(&impl_->source_device, bytes));
        TBCCL_CHUNKED_CUDA_CHECK(cudaMalloc(&impl_->destination_device, bytes));
    }
    if (max_chunk_bytes > 0)
    {
        TBCCL_CHUNKED_CUDA_CHECK(cudaHostAlloc(&impl_->pinned_scratch, max_chunk_bytes, cudaHostAllocDefault));
        ++impl_->pinned_alloc_count;
    }

    capacity_ = bytes;
    impl_->max_chunk_bytes = max_chunk_bytes;
}

void CudaChunkedAsyncBackend::initialize_source(std::uint32_t seed)
{
    if (capacity_ == 0) return;

    constexpr int kThreadsPerBlock = 256;
    const int blocks = static_cast<int>((capacity_ + kThreadsPerBlock - 1) / kThreadsPerBlock);
    fill_pattern_kernel<<<blocks, kThreadsPerBlock, 0, impl_->stream>>>(
        static_cast<std::uint8_t *>(impl_->source_device), capacity_, seed);
    TBCCL_CHUNKED_CUDA_CHECK(cudaGetLastError());
    TBCCL_CHUNKED_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}

namespace
{
    bool verify_device_buffer(void *device_ptr, std::size_t size, std::uint32_t seed)
    {
        if (size == 0) return true;
        std::vector<std::uint8_t> host_copy(size);
        check_cuda(
            cudaMemcpy(host_copy.data(), device_ptr, size, cudaMemcpyDeviceToHost),
            "cudaMemcpy (verify readback)");
        for (std::size_t i = 0; i < host_copy.size(); ++i)
        {
            const std::uint64_t index = static_cast<std::uint64_t>(i);
            const std::uint64_t value =
                index * 131u + (index >> 8) * 17u + static_cast<std::uint64_t>(seed);
            const std::uint8_t expected = static_cast<std::uint8_t>(value & 0xffu);
            if (host_copy[i] != expected) return false;
        }
        return true;
    }
} // namespace

bool CudaChunkedAsyncBackend::verify_source(std::uint32_t seed) const
{
    return verify_device_buffer(impl_->source_device, capacity_, seed);
}

bool CudaChunkedAsyncBackend::verify_destination(std::uint32_t seed) const
{
    return verify_device_buffer(impl_->destination_device, capacity_, seed);
}

void CudaChunkedAsyncBackend::stage_source_chunk(const tbccl::Chunk &chunk, void *staging)
{
    if (chunk.size == 0) return;
    if (chunk.size > impl_->max_chunk_bytes)
    {
        throw std::runtime_error(
            "CudaChunkedAsyncBackend::stage_source_chunk: chunk.size exceeds "
            "the pinned scratch buffer's max_chunk_bytes -- allocate() was "
            "called with too small a chunk bound");
    }

    auto *source = static_cast<const std::uint8_t *>(impl_->source_device) + chunk.offset;
    TBCCL_CHUNKED_CUDA_CHECK(cudaMemcpyAsync(
        impl_->pinned_scratch, source, chunk.size, cudaMemcpyDeviceToHost, impl_->stream));
    TBCCL_CHUNKED_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));

    std::memcpy(staging, impl_->pinned_scratch, chunk.size);
}

void CudaChunkedAsyncBackend::commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging)
{
    if (chunk.size == 0) return;
    if (chunk.size > impl_->max_chunk_bytes)
    {
        throw std::runtime_error(
            "CudaChunkedAsyncBackend::commit_destination_chunk: chunk.size "
            "exceeds the pinned scratch buffer's max_chunk_bytes");
    }

    std::memcpy(impl_->pinned_scratch, staging, chunk.size);

    auto *destination = static_cast<std::uint8_t *>(impl_->destination_device) + chunk.offset;
    TBCCL_CHUNKED_CUDA_CHECK(cudaMemcpyAsync(
        destination, impl_->pinned_scratch, chunk.size, cudaMemcpyHostToDevice, impl_->stream));
    TBCCL_CHUNKED_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}

std::size_t CudaChunkedAsyncBackend::diagnostic_pinned_alloc_count() const noexcept
{
    return impl_->pinned_alloc_count;
}

std::size_t CudaChunkedAsyncBackend::diagnostic_stream_create_count() const noexcept
{
    return impl_->stream_create_count;
}

} // namespace tbccl_bench::tensor
