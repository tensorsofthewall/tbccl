#include "cuda_external_async_backend.hpp"

#include <cuda_runtime.h>

#include "cuda_reduce_ops.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
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

struct CudaStagingResources::State
{
    mutable std::mutex mutex; // guards the pinned block and its counters; also the "lease" for its use
    void *pinned = nullptr;
    std::size_t pinned_cap = 0;
    Stats stats;
    bool fail_next_growth = false;

    std::mutex device_mutex;
    std::shared_ptr<void> device_block;
    std::size_t device_cap = 0;
    std::atomic<std::uint64_t> device_alloc{0};
    std::atomic<std::uint64_t> device_free{0};
};

CudaStagingResources::CudaStagingResources() : s_(std::make_unique<State>()) {}

CudaStagingResources::~CudaStagingResources()
{
    std::lock_guard<std::mutex> lock(s_->mutex);
    if (s_->pinned)
    {
        cudaFreeHost(s_->pinned);
        ++s_->stats.pinned_free_count;
        s_->pinned = nullptr;
    }
}

void CudaStagingResources::with_pinned(std::size_t bytes, const std::function<void(void *)> &fn)
{
    std::lock_guard<std::mutex> lock(s_->mutex);
    if (s_->pinned_cap < bytes)
    {
        void *fresh = nullptr;
        if (s_->fail_next_growth)
        {
            s_->fail_next_growth = false;
            throw std::runtime_error("device_error: pinned staging growth failed (injected)");
        }
        const cudaError_t status = cudaMallocHost(&fresh, bytes);
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("device_error: pinned staging growth to ") + std::to_string(bytes) +
                " bytes failed: " + cudaGetErrorString(status));
        }
        ++s_->stats.pinned_alloc_count;
        s_->stats.pinned_allocated_bytes_total += bytes;
        void *old = s_->pinned;
        s_->pinned = fresh;
        s_->pinned_cap = bytes;
        s_->stats.pinned_capacity = bytes;
        s_->stats.pinned_peak_capacity = std::max<std::uint64_t>(s_->stats.pinned_peak_capacity, bytes);
        if (old)
        {
            cudaFreeHost(old);
            ++s_->stats.pinned_free_count;
        }
    }
    fn(s_->pinned);
}

std::shared_ptr<void> CudaStagingResources::acquire_device_scratch(std::size_t bytes)
{
    std::lock_guard<std::mutex> lock(s_->device_mutex);
    if (!s_->device_block || s_->device_cap < bytes)
    {
        void *fresh = nullptr;
        TBCCL_EXTERNAL_CUDA_CHECK(cudaMalloc(&fresh, bytes));
        s_->device_alloc.fetch_add(1);
        State *state = s_.get();
        // Freed when the last lease on this block is released; cudaFree is a no-op-safe tail at shutdown.
        s_->device_block = std::shared_ptr<void>(fresh, [state](void *p) {
            cudaFree(p);
            state->device_free.fetch_add(1);
        });
        s_->device_cap = bytes;
    }
    return s_->device_block;
}

CudaStagingResources::Stats CudaStagingResources::stats() const
{
    std::lock_guard<std::mutex> lock(s_->mutex);
    Stats out = s_->stats;
    out.device_scratch_alloc_count = s_->device_alloc.load();
    out.device_scratch_free_count = s_->device_free.load();
    out.device_scratch_capacity = s_->device_cap;
    return out;
}

void CudaStagingResources::fail_next_pinned_growth()
{
    std::lock_guard<std::mutex> lock(s_->mutex);
    s_->fail_next_growth = true;
}

struct CudaExternalAsyncBackend::Impl
{
    std::shared_ptr<CudaStagingResources> resources; // null => private per-instance storage (legacy)
    std::shared_ptr<void> scratch_lease;             // keeps a shared device scratch block alive

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

CudaExternalAsyncBackend::CudaExternalAsyncBackend(
    void *external_device_ptr,
    std::size_t bytes,
    void *shared_copy_stream,
    std::shared_ptr<CudaStagingResources> resources)
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
    impl_->resources = std::move(resources);
    if (external_device_ptr != nullptr)
    {
        impl_->device_ptr = external_device_ptr;
        impl_->owns_device_ptr = false;
    }
    else if (bytes > 0 && impl_->resources)
    {
        impl_->scratch_lease = impl_->resources->acquire_device_scratch(bytes);
        impl_->device_ptr = impl_->scratch_lease.get();
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

    if (bytes > 0 && !impl_->resources)
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
    auto copy_out = [&](void *pinned) {
        TBCCL_EXTERNAL_CUDA_CHECK(cudaMemcpyAsync(pinned, source, chunk.size, cudaMemcpyDeviceToHost, impl_->stream));
        TBCCL_EXTERNAL_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
        std::memcpy(staging, pinned, chunk.size);
    };
    if (impl_->resources) impl_->resources->with_pinned(chunk.size, copy_out);
    else copy_out(impl_->pinned_scratch);
}

void CudaExternalAsyncBackend::commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging)
{
    if (chunk.size == 0) return;
    if (chunk.offset + chunk.size > impl_->bytes)
    {
        throw std::runtime_error("CudaExternalAsyncBackend::commit_destination_chunk: chunk exceeds buffer bytes");
    }

    auto *destination = static_cast<std::uint8_t *>(impl_->device_ptr) + chunk.offset;
    auto copy_in = [&](void *pinned) {
        std::memcpy(pinned, staging, chunk.size);
        TBCCL_EXTERNAL_CUDA_CHECK(cudaMemcpyAsync(destination, pinned, chunk.size, cudaMemcpyHostToDevice, impl_->stream));
        TBCCL_EXTERNAL_CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
    };
    if (impl_->resources) impl_->resources->with_pinned(chunk.size, copy_in);
    else copy_in(impl_->pinned_scratch);
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
        dst[i] = cuda_reduce::sum_elem<T>(dst[i], src[i]);
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
    void *device_ptr, std::size_t count, std::int32_t value, std::size_t spin_iterations, bool non_blocking)
{
    cudaStream_t stream = nullptr;
    check_cuda(
        cudaStreamCreateWithFlags(&stream, non_blocking ? cudaStreamNonBlocking : cudaStreamDefault),
        "cudaStreamCreate (delayed write test stream)");
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
    case tbccl::DataType::Float16: launch_sum<__half>(local_and_output_, peer_, count, stream); break;
    case tbccl::DataType::BFloat16: launch_sum<__nv_bfloat16>(local_and_output_, peer_, count, stream); break;
    default:
        throw std::runtime_error("CudaExternalReduceBackend: unrecognized DataType");
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize after sum_inplace_kernel");
}

} // namespace tbccl_bench::tensor
