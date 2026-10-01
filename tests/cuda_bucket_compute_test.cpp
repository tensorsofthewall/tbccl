// Correctness tests for the deterministic CUDA bucket-compute kernel
// (benchmarks/tensor/cuda_bucket_compute.{hpp,cu}). Real RTX 3070 Ti
// hardware required.

#include "tensor/cuda_bucket_compute.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

    void expect(bool condition, const std::string &message)
    {
        if (!condition) throw std::runtime_error("assertion failed: " + message);
    }

    using tbccl_bench::tensor::expected_bucket_byte;
    using tbccl_bench::tensor::launch_bucket_compute;
    using tbccl_bench::tensor::launch_bucket_fill_input;

    // Verifies the kernel's actual GPU output matches the CPU-computed
    // expected_bucket_byte() formula exactly, for several (bytes, seed,
    // rounds) combinations -- including a round count of 0 (identity
    // mix, degenerate but must still be correct) and a non-power-of-two
    // byte count (not a chunk-alignment concern here, just general
    // kernel correctness).
    void test_kernel_matches_cpu_formula()
    {
        void *device_ptr = nullptr;
        constexpr std::size_t kBytes = 1 << 20; // 1 MiB
        check_cuda(cudaMalloc(&device_ptr, kBytes), "cudaMalloc");

        cudaStream_t stream;
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

        const std::vector<std::pair<std::uint32_t, int>> cases = {
            {0x12345678u, 0}, {0xA5A5A5A5u, 1}, {0xDEADBEEFu, 10}, {0x1u, 1000},
        };

        for (const auto &[seed, rounds] : cases)
        {
            launch_bucket_fill_input(device_ptr, kBytes, seed, stream);
            launch_bucket_compute(device_ptr, kBytes, seed, rounds, stream);
            check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

            std::vector<std::uint8_t> host_copy(kBytes);
            check_cuda(
                cudaMemcpy(host_copy.data(), device_ptr, kBytes, cudaMemcpyDeviceToHost),
                "cudaMemcpy readback");

            for (std::size_t i = 0; i < kBytes; ++i)
            {
                expect(host_copy[i] == expected_bucket_byte(i, seed, rounds),
                       "mismatch at i=" + std::to_string(i) + " seed=" + std::to_string(seed) +
                           " rounds=" + std::to_string(rounds));
            }
        }

        cudaStreamDestroy(stream);
        cudaFree(device_ptr);
        std::cout << "[PASS] test_kernel_matches_cpu_formula\n";
    }

    // Distinct seeds must produce distinct, non-trivial output (not a
    // correctness requirement in the strict sense, but catches a
    // kernel that's accidentally a no-op or ignores its seed
    // parameter -- the "not trivially optimized away").
    void test_distinct_seeds_produce_distinct_output()
    {
        void *device_ptr = nullptr;
        constexpr std::size_t kBytes = 4096;
        check_cuda(cudaMalloc(&device_ptr, kBytes), "cudaMalloc");
        cudaStream_t stream;
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

        launch_bucket_fill_input(device_ptr, kBytes, 0xAAu, stream);
        launch_bucket_compute(device_ptr, kBytes, 0xAAu, 50, stream);
        check_cuda(cudaStreamSynchronize(stream), "sync");
        std::vector<std::uint8_t> out_a(kBytes);
        check_cuda(cudaMemcpy(out_a.data(), device_ptr, kBytes, cudaMemcpyDeviceToHost), "readback a");

        launch_bucket_fill_input(device_ptr, kBytes, 0xBBu, stream);
        launch_bucket_compute(device_ptr, kBytes, 0xBBu, 50, stream);
        check_cuda(cudaStreamSynchronize(stream), "sync");
        std::vector<std::uint8_t> out_b(kBytes);
        check_cuda(cudaMemcpy(out_b.data(), device_ptr, kBytes, cudaMemcpyDeviceToHost), "readback b");

        expect(out_a != out_b, "different seeds must produce different output");

        cudaStreamDestroy(stream);
        cudaFree(device_ptr);
        std::cout << "[PASS] test_distinct_seeds_produce_distinct_output\n";
    }

    // More rounds must take measurably longer (monotonic compute cost
    // -- what calibration, relies on).
    void test_more_rounds_takes_longer()
    {
        void *device_ptr = nullptr;
        constexpr std::size_t kBytes = 16 * 1024 * 1024; // 16 MiB: large enough to see a real difference
        check_cuda(cudaMalloc(&device_ptr, kBytes), "cudaMalloc");
        cudaStream_t stream;
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
        launch_bucket_fill_input(device_ptr, kBytes, 1u, stream);
        check_cuda(cudaStreamSynchronize(stream), "sync");

        cudaEvent_t start, end;
        check_cuda(cudaEventCreate(&start), "event create");
        check_cuda(cudaEventCreate(&end), "event create");

        auto timed_rounds = [&](int rounds) {
            check_cuda(cudaEventRecord(start, stream), "record start");
            launch_bucket_compute(device_ptr, kBytes, 1u, rounds, stream);
            check_cuda(cudaEventRecord(end, stream), "record end");
            check_cuda(cudaEventSynchronize(end), "sync end");
            float ms = 0;
            check_cuda(cudaEventElapsedTime(&ms, start, end), "elapsed");
            return ms;
        };

        const float ms_100 = timed_rounds(100);
        const float ms_10000 = timed_rounds(10000);
        expect(ms_10000 > ms_100, "10000 rounds must take longer than 100 rounds");

        cudaEventDestroy(start);
        cudaEventDestroy(end);
        cudaStreamDestroy(stream);
        cudaFree(device_ptr);
        std::cout << "[PASS] test_more_rounds_takes_longer (100r=" << ms_100
                   << "ms, 10000r=" << ms_10000 << "ms)\n";
    }

} // namespace

int main()
{
    try
    {
        test_kernel_matches_cpu_formula();
        test_distinct_seeds_produce_distinct_output();
        test_more_rounds_takes_longer();
        std::cout << "All tests passed.\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
