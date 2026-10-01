#include "cuda_reduce_backend.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

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

    template <typename T>
    __global__ void sum_inplace_kernel(T *dst, const T *src, std::size_t count)
    {
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count) return;
        dst[i] = dst[i] + src[i];
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
} // namespace

CudaReduceBackend::CudaReduceBackend(CudaChunkedAsyncBackend &backend, void *stream)
    : backend_(backend), stream_(stream)
{
}

void CudaReduceBackend::reduce_sum(std::size_t count, tbccl::DataType datatype)
{
    auto *dst = backend_.source_device_ptr();
    const auto *src = backend_.destination_device_ptr();
    auto stream = static_cast<cudaStream_t>(stream_);

    switch (datatype)
    {
    case tbccl::DataType::Float32:
        launch_sum<float>(dst, src, count, stream);
        break;
    case tbccl::DataType::Float64:
        launch_sum<double>(dst, src, count, stream);
        break;
    case tbccl::DataType::Int32:
        launch_sum<std::int32_t>(dst, src, count, stream);
        break;
    case tbccl::DataType::Int64:
        launch_sum<std::int64_t>(dst, src, count, stream);
        break;
    default:
        throw std::runtime_error("CudaReduceBackend: unrecognized DataType");
    }

    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize after sum_inplace_kernel");
}

void cuda_copy_host_to_device(const void *host_src, void *device_dst, std::size_t bytes)
{
    check_cuda(cudaMemcpy(device_dst, host_src, bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy H2D (cuda_copy_host_to_device)");
}

void cuda_copy_device_to_host(const void *device_src, void *host_dst, std::size_t bytes)
{
    check_cuda(cudaMemcpy(host_dst, device_src, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H (cuda_copy_device_to_host)");
}

namespace
{
    __global__ void consumer_double_kernel(const float *in, float *out, std::size_t count)
    {
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count) return;
        out[i] = in[i] * 2.0f;
    }
} // namespace

void cuda_launch_consumer_double_f32(
    const void *in, void *out, std::size_t count, void *stream)
{
    if (count == 0) return;
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    constexpr int kThreadsPerBlock = 256;
    const int blocks = static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
    consumer_double_kernel<<<blocks, kThreadsPerBlock, 0, cuda_stream>>>(
        static_cast<const float *>(in), static_cast<float *>(out), count);
    check_cuda(cudaGetLastError(), "consumer_double_kernel launch");
    check_cuda(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize after consumer_double_kernel");
}

} // namespace tbccl_bench::tensor
