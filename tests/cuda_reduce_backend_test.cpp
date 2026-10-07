// Hardware correctness tests for CudaReduceBackend
// (benchmarks/tensor/cuda_reduce_backend.{hpp,cu}) on a real CUDA device.
// No networking involved -- this isolates the local SUM kernel itself,
// matching the correctness-ladder step 2 (before any heterogeneous
// cross-machine transfer is attempted).

#include "tensor/cuda_chunked_async_backend.hpp"
#include "tensor/cuda_reduce_backend.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using tbccl_bench::tensor::CudaChunkedAsyncBackend;
    using tbccl_bench::tensor::CudaReduceBackend;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    float value_for(std::size_t i, std::uint32_t seed, std::uint32_t modulus)
    {
        return static_cast<float>((i + seed) % modulus);
    }

    // Seeds `backend`'s source with local-pattern values and destination
    // with peer-pattern values directly (bypassing the network entirely),
    // runs CudaReduceBackend::reduce_sum, and verifies the source buffer
    // now holds the exact elementwise sum via GPU readback.
    void run_one_sum_case(std::size_t count, std::uint32_t local_seed,
                           std::uint32_t local_mod, std::uint32_t peer_seed,
                           std::uint32_t peer_mod)
    {
        const std::size_t bytes = count * sizeof(float);

        CudaChunkedAsyncBackend backend;
        backend.allocate(bytes, bytes);

        std::vector<float> local_values(count);
        std::vector<float> peer_values(count);
        std::vector<float> expected(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            local_values[i] = value_for(i, local_seed, local_mod);
            peer_values[i] = value_for(i, peer_seed, peer_mod);
            expected[i] = local_values[i] + peer_values[i];
        }

        tbccl_bench::tensor::cuda_copy_host_to_device(
            local_values.data(), backend.source_device_ptr(), bytes);
        tbccl_bench::tensor::cuda_copy_host_to_device(
            peer_values.data(), backend.destination_device_ptr(), bytes);

        CudaReduceBackend reduce(backend, /*stream=*/nullptr);
        reduce.reduce_sum(count, tbccl::DataType::Float32);

        std::vector<float> result(count);
        tbccl_bench::tensor::cuda_copy_device_to_host(
            backend.source_device_ptr(), result.data(), bytes);

        for (std::size_t i = 0; i < count; ++i)
        {
            expect(result[i] == expected[i],
                   "element " + std::to_string(i) + " mismatch for count=" +
                       std::to_string(count));
        }
    }

    void test_single_element()
    {
        run_one_sum_case(1, 3, 127, 9, 113);
        std::cout << "[PASS] test_single_element\n";
    }

    void test_odd_count()
    {
        run_one_sum_case(1001, 11, 127, 97, 113);
        std::cout << "[PASS] test_odd_count\n";
    }

    void test_4kib_equivalent_count()
    {
        // 4096 bytes / 4 bytes per float32 = 1024 elements.
        run_one_sum_case(1024, 5, 127, 19, 113);
        std::cout << "[PASS] test_4kib_equivalent_count\n";
    }

    void test_1mib_equivalent_count()
    {
        // 1 MiB / 4 bytes per float32 = 262144 elements.
        run_one_sum_case(262144, 2, 127, 31, 113);
        std::cout << "[PASS] test_1mib_equivalent_count\n";
    }

    void test_repeated_seeds()
    {
        for (std::uint32_t seed = 0; seed < 5; ++seed)
        {
            run_one_sum_case(4096, seed, 127, seed + 50, 113);
        }
        std::cout << "[PASS] test_repeated_seeds\n";
    }

    void test_gpu_consumer_can_use_result()
    {
        constexpr std::size_t kCount = 2048;
        constexpr std::size_t kBytes = kCount * sizeof(float);

        CudaChunkedAsyncBackend backend;
        backend.allocate(kBytes, kBytes);

        std::vector<float> local_values(kCount);
        std::vector<float> peer_values(kCount);
        for (std::size_t i = 0; i < kCount; ++i)
        {
            local_values[i] = value_for(i, 7, 127);
            peer_values[i] = value_for(i, 41, 113);
        }
        tbccl_bench::tensor::cuda_copy_host_to_device(
            local_values.data(), backend.source_device_ptr(), kBytes);
        tbccl_bench::tensor::cuda_copy_host_to_device(
            peer_values.data(), backend.destination_device_ptr(), kBytes);

        CudaReduceBackend reduce(backend, nullptr);
        reduce.reduce_sum(kCount, tbccl::DataType::Float32);

        // Genuine second kernel, independent of the reduction kernel,
        // reading the AllReduce-equivalent result directly from device
        // memory -- proves the result is truly GPU-resident/consumable,
        // not merely correct via host readback.
        tbccl_bench::tensor::cuda_launch_consumer_double_f32(
            backend.source_device_ptr(), backend.destination_device_ptr(),
            kCount, nullptr);

        std::vector<float> doubled(kCount);
        tbccl_bench::tensor::cuda_copy_device_to_host(
            backend.destination_device_ptr(), doubled.data(), kBytes);

        for (std::size_t i = 0; i < kCount; ++i)
        {
            const float expected = (local_values[i] + peer_values[i]) * 2.0f;
            expect(doubled[i] == expected,
                   "consumer kernel result mismatch at element " + std::to_string(i));
        }

        std::cout << "[PASS] test_gpu_consumer_can_use_result\n";
    }

} // namespace

int main()
{
    try
    {
        test_single_element();
        test_odd_count();
        test_4kib_equivalent_count();
        test_1mib_equivalent_count();
        test_repeated_seeds();
        test_gpu_consumer_can_use_result();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";
    return 0;
}
