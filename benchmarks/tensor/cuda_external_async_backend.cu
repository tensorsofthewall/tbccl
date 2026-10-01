#include "cuda_external_async_backend.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace tbccl_bench::tensor
{

namespace
{

    void check_cuda(cudaError_t status, const char *what)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("CUDA error in ") + what + ": " + cudaGetErrorString(status));
        }
    }

} // namespace

#define TBCCL_EXTERNAL_CUDA_CHECK(expr) \
    ::tbccl_bench::tensor::check_cuda((expr), #expr)

struct CudaExternalAsyncBackend::Impl
{
    void *device_ptr = nullptr;
    bool owns_device_ptr = false;
    std::size_t bytes = 0;

    void *pinned_scratch = nullptr; // `bytes`-sized, reused every call
    cudaStream_t stream = nullptr;
    bool owns_stream = true;

    // Part L: one persistent, reused event for producer-stream
    // readiness -- never allocated per chunk/call.
    cudaEvent_t ready_event = nullptr;

    ~Impl()
    {
        if (owns_device_ptr && device_ptr) cudaFree(device_ptr);
        if (pinned_scratch) cudaFreeHost(pinned_scratch);
        if (ready_event) cudaEventDestroy(ready_event);
        if (stream && owns_stream) cudaStreamDestroy(stream);
    }
};

CudaExternalAsyncBackend::CudaExternalAsyncBackend(void *external_device_ptr, std::size_t bytes, void *shared_copy_stream)
    : impl_(std::make_unique<Impl>())
{
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0)
    {
        throw std::runtime_error(
            "CudaExternalAsyncBackend: no CUDA device available at runtime: " +
            std::string(cudaGetErrorString(status)));
    }

    impl_->bytes = bytes;
    if (external_device_ptr != nullptr)
    {
        impl_->device_ptr = external_device_ptr;
        impl_->owns_device_ptr = false;
    }
    else if (bytes > 0)
    {
        TBCCL_EXTERNAL_CUDA_CHECK(cudaMalloc(&impl_->device_ptr, bytes));
        impl_->owns_device_ptr = true;
    }

    if (shared_copy_stream != nullptr)
    {
        impl_->stream = static_cast<cudaStream_t>(shared_copy_stream);
        impl_->owns_stream = false;
    }
    else
    {
        TBCCL_EXTERNAL_CUDA_CHECK(cudaStreamCreate(&impl_->stream));
    }

    if (bytes > 0)
    {
        TBCCL_EXTERNAL_CUDA_CHECK(cudaMallocHost(&impl_->pinned_scratch, bytes));
    }

    TBCCL_EXTERNAL_CUDA_CHECK(cudaEventCreateWithFlags(&impl_->ready_event, cudaEventDisableTiming));
}

CudaExternalAsyncBackend::~CudaExternalAsyncBackend() = default;

void CudaExternalAsyncBackend::wait_for_producer_stream(void *producer_stream)
{
    if (producer_stream == nullptr) return;
    auto producer = static_cast<cudaStream_t>(producer_stream);
    TBCCL_EXTERNAL_CUDA_CHECK(cudaEventRecord(impl_->ready_event, producer));
    TBCCL_EXTERNAL_CUDA_CHECK(cudaStreamWaitEvent(impl_->stream, impl_->ready_event, 0));
}

void CudaExternalAsyncBackend::stage_source_chunk(const tbccl::Chunk &chunk, void *staging)
{
    if (chunk.size == 0) return;
    if (chunk.offset + chunk.size > impl_->bytes)
    {
        throw std::runtime_error("CudaExternalAsyncBackend::stage_source_chunk: chunk exceeds buffer bytes");
    }

    auto *source = static_cast<const std::uint8_t *>(impl_->device_ptr) + chunk.offset;
    TBCCL_EXTERNAL_CUDA_CHECK(cudaMemcpyAsync(
        impl_->pinned_scratch, source, chunk.size, cudaMemcpyDeviceToHost, impl_->stream));
    TBCCL_EXTERNAL_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));

    std::memcpy(staging, impl_->pinned_scratch, chunk.size);
}

void CudaExternalAsyncBackend::commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging)
{
    if (chunk.size == 0) return;
    if (chunk.offset + chunk.size > impl_->bytes)
    {
        throw std::runtime_error("CudaExternalAsyncBackend::commit_destination_chunk: chunk exceeds buffer bytes");
    }

    std::memcpy(impl_->pinned_scratch, staging, chunk.size);

    auto *destination = static_cast<std::uint8_t *>(impl_->device_ptr) + chunk.offset;
    TBCCL_EXTERNAL_CUDA_CHECK(cudaMemcpyAsync(
        destination, impl_->pinned_scratch, chunk.size, cudaMemcpyHostToDevice, impl_->stream));
    TBCCL_EXTERNAL_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
}

void *CudaExternalAsyncBackend::device_ptr() const noexcept { return impl_->device_ptr; }
std::size_t CudaExternalAsyncBackend::capacity() const noexcept { return impl_->bytes; }

namespace
{

    template <typename T>
    __global__ void sum_inplace_kernel(T *dst, const T *src, std::size_t count)
    {
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count) return;
        if constexpr (std::is_integral_v<T>)
        {
            using U = std::make_unsigned_t<T>;
            dst[i] = static_cast<T>(static_cast<U>(dst[i]) + static_cast<U>(src[i]));
        }
        else
        {
            dst[i] = dst[i] + src[i];
        }
    }

    template <typename T>
    void launch_sum(void *dst, const void *src, std::size_t count, cudaStream_t stream)
    {
        if (count == 0) return;
        constexpr int kThreadsPerBlock = 256;
        const int blocks = static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
        sum_inplace_kernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<T *>(dst), static_cast<const T *>(src), count);
        check_cuda(cudaGetLastError(), "sum_inplace_kernel launch");
    }

    __global__ void consumer_double_i32_kernel(const std::int32_t *in, std::int32_t *out, std::size_t count)
    {
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count) return;
        out[i] = static_cast<std::int32_t>(static_cast<std::uint32_t>(in[i]) * 2u);
    }

} // namespace

void cuda_external_launch_consumer_double_i32(
    const void *in, void *out, std::size_t count, void *stream)
{
    if (count == 0) return;
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    constexpr int kThreadsPerBlock = 256;
    const int blocks = static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    consumer_double_i32_kernel<<<blocks, kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const std::int32_t *>(in), static_cast<std::int32_t *>(out), count);
    check_cuda(cudaGetLastError(), "consumer_double_i32_kernel launch");
    check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize after consumer_double_i32_kernel");
}

namespace
{

    __global__ void delayed_write_i32_kernel(std::int32_t *data, std::size_t count, std::int32_t value, std::size_t spin_iterations)
    {
        // Deliberate, data-dependent busy-spin so the compiler cannot
        // optimize it away -- volatile accumulation that influences the
        // actual write below.
        volatile std::uint64_t acc = 0;
        for (std::size_t i = 0; i < spin_iterations; ++i) acc += i;

        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count) return;
        data[i] = value + static_cast<std::int32_t>(acc & 0);
    }

} // namespace

void *cuda_external_test_launch_delayed_write_i32(
    void *device_ptr, std::size_t count, std::int32_t value, std::size_t spin_iterations)
{
    cudaStream_t stream = nullptr;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate (delayed write test stream)");
    if (count > 0)
    {
        constexpr int kThreadsPerBlock = 256;
        const int blocks = static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
        delayed_write_i32_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            static_cast<std::int32_t *>(device_ptr), count, value, spin_iterations);
        check_cuda(cudaGetLastError(), "delayed_write_i32_kernel launch");
    }
    return stream;
}

void cuda_external_test_destroy_stream(void *stream)
{
    if (stream != nullptr) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

CudaExternalReduceBackend::CudaExternalReduceBackend(void *local_and_output_device_ptr, const void *peer_device_ptr, void *stream)
    : local_and_output_(local_and_output_device_ptr), peer_(peer_device_ptr), stream_(stream)
{
}

void CudaExternalReduceBackend::reduce_sum(std::size_t count, tbccl::DataType datatype)
{
    auto stream = static_cast<cudaStream_t>(stream_);
    switch (datatype)
    {
    case tbccl::DataType::Float32: launch_sum<float>(local_and_output_, peer_, count, stream); break;
    case tbccl::DataType::Float64: launch_sum<double>(local_and_output_, peer_, count, stream); break;
    case tbccl::DataType::Int32: launch_sum<std::int32_t>(local_and_output_, peer_, count, stream); break;
    case tbccl::DataType::Int64: launch_sum<std::int64_t>(local_and_output_, peer_, count, stream); break;
    default:
        throw std::runtime_error("CudaExternalReduceBackend: unrecognized DataType");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize after sum_inplace_kernel");
}

} // namespace tbccl_bench::tensor
