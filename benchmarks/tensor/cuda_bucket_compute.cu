#include "cuda_bucket_compute.hpp"

#include <cuda_runtime.h>

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

    // Same shared pattern formula as tensor_backend.hpp::pattern_byte(),
    // duplicated per this codebase's established convention for
    // CUDA-only translation units.
    __device__ __host__ std::uint8_t pattern_byte_device(std::size_t i, std::uint32_t seed)
    {
        const std::uint64_t index = static_cast<std::uint64_t>(i);
        const std::uint64_t value =
            index * 131u + (index >> 8) * 17u + static_cast<std::uint64_t>(seed);
        return static_cast<std::uint8_t>(value & 0xffu);
    }

    __global__ void fill_input_kernel(std::uint8_t *data, std::size_t count, std::uint32_t seed)
    {
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count) return;
        data[i] = pattern_byte_device(i, seed);
    }

    // Deterministic per-byte hash-mix: `rounds` iterations of a 32-bit
    // integer avalanche mix (a Murmur3-style finalizer), seeded by the
    // input byte, seed, and byte index -- genuinely depends on real
    // input data (not trivially hoistable/eliminated), uses only
    // integer arithmetic (no float determinism/ denormal concerns),
    // and its cost scales linearly and predictably with `rounds` for
    // calibration.
    __device__ __host__ std::uint8_t bucket_transform(
        std::uint8_t input, std::size_t i, std::uint32_t seed, int rounds)
    {
        std::uint32_t v = static_cast<std::uint32_t>(input) ^ seed ^
                           static_cast<std::uint32_t>(i & 0xffffffffu);
        for (int r = 0; r < rounds; ++r)
        {
            v *= 2654435761u;
            v += 1u;
            v ^= v >> 15;
            v *= 0x85ebca6bu;
            v ^= v >> 13;
        }
        return static_cast<std::uint8_t>(v & 0xffu);
    }

    __global__ void bucket_compute_kernel(
        std::uint8_t *data, std::size_t count, std::uint32_t seed, int rounds)
    {
        const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i >= count) return;
        data[i] = bucket_transform(data[i], i, seed, rounds);
    }

} // namespace

void launch_bucket_fill_input(void *device_ptr, std::size_t bytes, std::uint32_t seed, void *stream)
{
    if (bytes == 0) return;
    constexpr int kThreadsPerBlock = 256;
    const int blocks = static_cast<int>((bytes + kThreadsPerBlock - 1) / kThreadsPerBlock);
    fill_input_kernel<<<blocks, kThreadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<std::uint8_t *>(device_ptr), bytes, seed);
    check_cuda(cudaGetLastError(), "fill_input_kernel launch");
}

void launch_bucket_compute(void *device_ptr, std::size_t bytes, std::uint32_t seed, int rounds, void *stream)
{
    if (bytes == 0) return;
    constexpr int kThreadsPerBlock = 256;
    const int blocks = static_cast<int>((bytes + kThreadsPerBlock - 1) / kThreadsPerBlock);
    bucket_compute_kernel<<<blocks, kThreadsPerBlock, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<std::uint8_t *>(device_ptr), bytes, seed, rounds);
    check_cuda(cudaGetLastError(), "bucket_compute_kernel launch");
}

} // namespace tbccl_bench::tensor
