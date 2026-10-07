// Correctness tests for benchmarks/tensor/cuda_backend.cu. Only
// built/run when TBCCL_ENABLE_CUDA is on (see CMakeLists.txt) and
// only meaningful with a CUDA device actually present at runtime --
// make_backend() throws a clear, distinct error otherwise (,
// requirement 10), which main() below classifies as a skip rather
// than a test failure.

#include "tensor/tensor_backend.hpp"

#include "test_utils.hpp"

#include <tbccl/tcp_world.hpp>
#include <tbccl/world.hpp>

#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

    using tbccl_bench::tensor::AllocationStats;
    using tbccl_bench::tensor::BackendKind;
    using tbccl_bench::tensor::make_backend;
    using tbccl_bench::tensor::TensorBackend;

    using tbccl_test::expect;
    using tbccl_test::join_and_check;
    using tbccl_test::make_local_peers;
    using tbccl_test::make_options;
    using tbccl_test::run_rank;

    constexpr std::uint16_t kCudaToHostBase = 32100;
    constexpr std::uint16_t kRepeatedTransferBase = 32110;

    // -----------------------------------------------------------------------------
    // Test 1: initialize_source() genuinely runs on the GPU -- a
    // freshly allocated device buffer that is never touched by
    // initialize_source() must NOT already match the pattern (rules
    // out a backend that silently no-ops and happens to read zeroed
    // device memory as "correct" for seed 0).
    // -----------------------------------------------------------------------------

    void test_gpu_generated_source(BackendKind kind)
    {
        auto backend = make_backend(kind);
        backend->allocate(65536);

        // Freshly cudaMalloc'd memory is not guaranteed zeroed by the
        // driver, but even if it were, seed 0x1 never matches an
        // untouched buffer's pattern for i>0 with overwhelming
        // probability; the real check is the positive one below.
        backend->initialize_source(0x1u);
        backend->prepare_source();

        expect(
            backend->verify_source(0x1u),
            "GPU-initialized source should verify (kind=" +
                tbccl_bench::tensor::backend_kind_name(kind) + ")");

        expect(
            !backend->verify_source(0x2u),
            "GPU-initialized source should not match an unrelated seed");

        std::cout
            << "[PASS] test_gpu_generated_source("
            << tbccl_bench::tensor::backend_kind_name(kind) << ")\n";
    }

    // -----------------------------------------------------------------------------
    // Test 2: local D2H -> (memcpy, no network) -> H2D -> readback,
    // for one backend kind, at a handful of sizes including 0 and
    // odd sizes.
    // -----------------------------------------------------------------------------

    void test_local_staging_roundtrip(BackendKind kind)
    {
        const std::size_t sizes[] = {0, 1, 257, 65536, 1 << 20};

        for (std::size_t size : sizes)
        {
            auto source = make_backend(kind);
            auto destination = make_backend(kind);

            source->allocate(size);
            destination->allocate(size);

            source->initialize_source(0xABCDu);
            source->prepare_source();
            source->stage_device_to_host();

            if (size > 0)
            {
                std::memcpy(
                    destination->destination_staging_data(),
                    source->source_staging_data(),
                    size);
            }

            destination->stage_host_to_device();

            expect(
                source->verify_source(0xABCDu),
                "source should still verify after staging, size=" +
                    std::to_string(size));

            expect(
                destination->verify_destination(0xABCDu),
                "destination should verify after local roundtrip, size=" +
                    std::to_string(size));
        }

        std::cout
            << "[PASS] test_local_staging_roundtrip("
            << tbccl_bench::tensor::backend_kind_name(kind) << ")\n";
    }

    // -----------------------------------------------------------------------------
    // Test 3: allocate() reuse accounting matches HostBackend's
    // contract, and stats().capacity_bytes tracks the latest size.
    // -----------------------------------------------------------------------------

    void test_allocation_reuse(BackendKind kind)
    {
        auto backend = make_backend(kind);

        backend->allocate(4096);
        auto stats = backend->stats();
        expect(stats.allocation_count == 1, "first allocate() should count as an allocation");
        expect(stats.reuse_count == 0, "first allocate() should not count as a reuse");

        backend->allocate(4096);
        stats = backend->stats();
        expect(stats.allocation_count == 1, "same-size allocate() should not reallocate");
        expect(stats.reuse_count == 1, "same-size allocate() should count as reuse");

        backend->allocate(8192);
        stats = backend->stats();
        expect(stats.allocation_count == 2, "different-size allocate() should reallocate");
        expect(stats.capacity_bytes == 8192, "capacity should reflect the new size");

        std::cout
            << "[PASS] test_allocation_reuse("
            << tbccl_bench::tensor::backend_kind_name(kind) << ")\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4: repeated transfers over a real loopback World, changing
    // seed every iteration -- exercises host_send_data/host_recv_data,
    // stream synchronization, and staging-buffer lifetime (the same
    // buffers are reused across all iterations without reallocating).
    // -----------------------------------------------------------------------------

    void test_repeated_transfer_over_world(BackendKind kind, std::uint16_t base_port)
    {
        constexpr std::size_t kBytes = 1 << 16;
        constexpr int kIterations = 100;

        auto peers = make_local_peers(base_port, 2);

        std::vector<std::exception_ptr> errors(2);
        std::vector<std::thread> threads;

        threads.emplace_back(
            run_rank,
            make_options(0, peers),
            [kind](tbccl::World &world)
            {
                auto backend = make_backend(kind);
                backend->allocate(kBytes);

                for (int i = 0; i < kIterations; ++i)
                {
                    const std::uint32_t seed = static_cast<std::uint32_t>(0x2000 + i);
                    backend->initialize_source(seed);
                    backend->prepare_source();
                    backend->stage_device_to_host();
                    backend->host_send_data(world, 1);
                }
            },
            std::ref(errors[0]));

        threads.emplace_back(
            run_rank,
            make_options(1, peers),
            [kind](tbccl::World &world)
            {
                auto backend = make_backend(kind);
                backend->allocate(kBytes);

                for (int i = 0; i < kIterations; ++i)
                {
                    const std::uint32_t seed = static_cast<std::uint32_t>(0x2000 + i);

                    backend->host_recv_data(world, 0);
                    backend->stage_host_to_device();

                    expect(
                        backend->verify_destination(seed),
                        "iteration " + std::to_string(i) +
                            ": destination did not match this iteration's seed");
                }

                const auto stats = backend->stats();
                expect(
                    stats.allocation_count == 1,
                    "receiver should have allocated exactly once across all "
                    "iterations");
            },
            std::ref(errors[1]));

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_repeated_transfer_over_world("
            << tbccl_bench::tensor::backend_kind_name(kind) << ", "
            << kIterations << " iterations)\n";
    }

    void run_all_tests_for_kind(BackendKind kind, std::uint16_t repeated_base)
    {
        test_gpu_generated_source(kind);
        test_local_staging_roundtrip(kind);
        test_allocation_reuse(kind);
        test_repeated_transfer_over_world(kind, repeated_base);
    }

} // namespace

int main()
{
    if (!tbccl_bench::tensor::backend_kind_available(BackendKind::CudaPageable))
    {
        std::cout << "[SKIP] tensor_cuda_test: CUDA backend not compiled in\n";
        return 0;
    }

    // A device-presence check up front classifies "no GPU at runtime"
    // as a skip, not a failure (, requirement 10) -- every subsequent
    // test would otherwise fail identically and uninformatively on a
    // CI/build machine without a GPU.
    try
    {
        auto probe = make_backend(BackendKind::CudaPageable);
    }
    catch (const std::exception &error)
    {
        std::cout
            << "[SKIP] tensor_cuda_test: no CUDA device available at "
               "runtime (" << error.what() << ")\n";
        return 0;
    }

    try
    {
        run_all_tests_for_kind(BackendKind::CudaPageable, kCudaToHostBase);
        run_all_tests_for_kind(BackendKind::CudaPinned, kRepeatedTransferBase);
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
