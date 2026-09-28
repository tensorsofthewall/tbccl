// Correctness tests for benchmarks/tensor/host_backend.cpp: the
// backend in isolation (deterministic generation, allocation/reuse,
// byte-exact verification, odd sizes) plus host-to-host transfer over
// a real (loopback) World, exercising the full TensorBackend pipeline
// (initialize_source -> prepare_source -> stage_device_to_host ->
// host_send_data / host_recv_data -> stage_host_to_device ->
// verify_destination) exactly as benchmarks/tensor_transfer_bench.cpp
// will.

#include "tensor/host_backend.hpp"
#include "tensor/tensor_backend.hpp"

#include "test_utils.hpp"

#include <tbccl/tcp_world.hpp>
#include <tbccl/world.hpp>

#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

    using tbccl_bench::tensor::BackendKind;
    using tbccl_bench::tensor::make_host_backend;

    using tbccl_test::expect;
    using tbccl_test::join_and_check;
    using tbccl_test::make_local_peers;
    using tbccl_test::make_options;
    using tbccl_test::run_rank;

    // Fixed high port ranges, well clear of every range used by the
    // pre-existing test files (which top out in the 29500s-29600s).
    constexpr std::uint16_t kHostToHostBase = 32000;
    constexpr std::uint16_t kRepeatedTransferBase = 32010;
    constexpr std::uint16_t kBidirectionalBase = 32020;

    // -----------------------------------------------------------------------------
    // Test 1: deterministic generation is reproducible for a given
    // seed, and differs (with overwhelming probability) for a
    // different seed -- the property changing seeds in later tests
    // relies on to catch stale-buffer bugs.
    // -----------------------------------------------------------------------------

    void test_deterministic_generation()
    {
        auto backend = make_host_backend();
        backend->allocate(4096);

        backend->initialize_source(0x1234u);
        backend->prepare_source();

        expect(
            backend->verify_source(0x1234u),
            "source should match the seed it was initialized with");

        expect(
            !backend->verify_source(0x5678u),
            "source should not match an unrelated seed");

        std::cout << "[PASS] test_deterministic_generation\n";
    }

    // -----------------------------------------------------------------------------
    // Test 2: allocate() with the same size reuses storage
    // (reuse_count increments, allocation_count does not); a
    // different size reallocates.
    // -----------------------------------------------------------------------------

    void test_allocation_reuse()
    {
        auto backend = make_host_backend();

        backend->allocate(1024);
        auto stats = backend->stats();
        expect(stats.allocation_count == 1, "first allocate() should count as an allocation");
        expect(stats.reuse_count == 0, "first allocate() should not count as a reuse");
        expect(stats.capacity_bytes == 1024, "capacity should match the requested size");

        backend->allocate(1024);
        backend->allocate(1024);
        stats = backend->stats();
        expect(stats.allocation_count == 1, "same-size allocate() should not reallocate");
        expect(stats.reuse_count == 2, "same-size allocate() should count as reuse");

        backend->allocate(2048);
        stats = backend->stats();
        expect(stats.allocation_count == 2, "different-size allocate() should reallocate");
        expect(stats.capacity_bytes == 2048, "capacity should reflect the new size");

        expect(backend->capacity() == 2048, "capacity() should match the latest allocate()");

        std::cout << "[PASS] test_allocation_reuse\n";
    }

    // -----------------------------------------------------------------------------
    // Test 3: odd, non-power-of-two byte sizes generate and verify
    // correctly, including single-byte and zero-byte buffers.
    // -----------------------------------------------------------------------------

    void test_odd_byte_sizes()
    {
        const std::size_t sizes[] = {0, 1, 3, 7, 257, 1003, 65537};

        for (std::size_t size : sizes)
        {
            auto backend = make_host_backend();
            backend->allocate(size);
            backend->initialize_source(0xABCDu);
            backend->prepare_source();

            expect(
                backend->verify_source(0xABCDu),
                "odd-size source should verify for size " +
                    std::to_string(size));
        }

        std::cout << "[PASS] test_odd_byte_sizes\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4: unknown backend name and an unavailable-on-this-build
    // backend kind both reject clearly, without touching any World.
    // -----------------------------------------------------------------------------

    void test_configuration_mismatch_rejection()
    {
        bool threw = false;

        try
        {
            tbccl_bench::tensor::parse_backend_kind("not-a-real-backend");
        }
        catch (const std::exception &)
        {
            threw = true;
        }

        expect(threw, "parse_backend_kind should reject an unknown name");

        // metal-* is never available on a non-Apple build; cuda-* is
        // conditionally available depending on TBCCL_ENABLE_CUDA, so
        // only the Metal kinds are unconditionally testable here.
#if !defined(__APPLE__)
        expect(
            !tbccl_bench::tensor::backend_kind_available(
                BackendKind::MetalShared),
            "metal-shared should be unavailable on a non-Apple build");

        bool make_threw = false;

        try
        {
            auto backend =
                tbccl_bench::tensor::make_backend(BackendKind::MetalShared);
        }
        catch (const std::exception &)
        {
            make_threw = true;
        }

        expect(
            make_threw,
            "make_backend(MetalShared) should throw on a non-Apple build");
#endif

        std::cout << "[PASS] test_configuration_mismatch_rejection\n";
    }

    // -----------------------------------------------------------------------------
    // Test 5: full host -> host pipeline over a real (loopback) World:
    // rank 0 generates a source tensor and sends it; rank 1 receives
    // it and stages it into its destination buffer. Every stage
    // method is exercised, not just an opaque end-to-end call.
    // -----------------------------------------------------------------------------

    void test_host_to_host_transfer()
    {
        constexpr std::size_t kBytes = 1 << 20; // 1 MiB
        constexpr std::uint32_t kSeed = 0x9E3779B9u;

        auto peers = make_local_peers(kHostToHostBase, 2);

        std::vector<std::exception_ptr> errors(2);
        std::vector<std::thread> threads;

        threads.emplace_back(
            run_rank,
            make_options(0, peers),
            [](tbccl::World &world)
            {
                auto backend = make_host_backend();
                backend->allocate(kBytes);
                backend->initialize_source(kSeed);
                backend->prepare_source();
                backend->stage_device_to_host();
                backend->host_send_data(world, 1);
            },
            std::ref(errors[0]));

        threads.emplace_back(
            run_rank,
            make_options(1, peers),
            [](tbccl::World &world)
            {
                auto backend = make_host_backend();
                backend->allocate(kBytes);
                backend->host_recv_data(world, 0);
                backend->stage_host_to_device();
                backend->synchronize();

                expect(
                    backend->verify_destination(kSeed),
                    "destination should match the source's seed after transfer");
            },
            std::ref(errors[1]));

        join_and_check(threads, errors);

        std::cout << "[PASS] test_host_to_host_transfer\n";
    }

    // -----------------------------------------------------------------------------
    // Test 6: repeated transfers on reused buffers, changing the seed
    // every iteration. Catches stale-buffer bugs (destination still
    // holding a previous iteration's pattern) and missed-transfer
    // bugs (destination never actually overwritten).
    // -----------------------------------------------------------------------------

    void test_repeated_transfers_changing_seeds()
    {
        constexpr std::size_t kBytes = 4096;
        constexpr int kIterations = 200;

        auto peers = make_local_peers(kRepeatedTransferBase, 2);

        std::vector<std::exception_ptr> errors(2);
        std::vector<std::thread> threads;

        threads.emplace_back(
            run_rank,
            make_options(0, peers),
            [](tbccl::World &world)
            {
                auto backend = make_host_backend();
                backend->allocate(kBytes);

                for (int i = 0; i < kIterations; ++i)
                {
                    const std::uint32_t seed = static_cast<std::uint32_t>(0x1000 + i);
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
            [](tbccl::World &world)
            {
                auto backend = make_host_backend();
                backend->allocate(kBytes);

                for (int i = 0; i < kIterations; ++i)
                {
                    const std::uint32_t seed = static_cast<std::uint32_t>(0x1000 + i);

                    backend->host_recv_data(world, 0);
                    backend->stage_host_to_device();

                    expect(
                        backend->verify_destination(seed),
                        "iteration " + std::to_string(i) +
                            ": destination did not match this iteration's seed "
                            "(stale buffer or missed transfer)");
                }
            },
            std::ref(errors[1]));

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_repeated_transfers_changing_seeds ("
            << kIterations << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 7: both physical directions in the same process (each rank
    // both sends its own tensor and receives the other's), verifying
    // TCP's full-duplex send()/recv() and the backend abstraction
    // compose correctly.
    // -----------------------------------------------------------------------------

    void test_bidirectional_transfer()
    {
        constexpr std::size_t kBytes = 65536;

        auto peers = make_local_peers(kBidirectionalBase, 2);

        std::vector<std::exception_ptr> errors(2);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < 2; ++rank)
        {
            const std::size_t peer = 1 - rank;
            const std::uint32_t own_seed = static_cast<std::uint32_t>(0x4000 + rank);
            const std::uint32_t peer_seed = static_cast<std::uint32_t>(0x4000 + peer);

            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [peer, own_seed, peer_seed](tbccl::World &world)
                {
                    auto send_backend = make_host_backend();
                    send_backend->allocate(kBytes);
                    send_backend->initialize_source(own_seed);
                    send_backend->prepare_source();
                    send_backend->stage_device_to_host();

                    auto recv_backend = make_host_backend();
                    recv_backend->allocate(kBytes);

                    send_backend->host_send_data(world, peer);
                    recv_backend->host_recv_data(world, peer);
                    recv_backend->stage_host_to_device();

                    expect(
                        recv_backend->verify_destination(peer_seed),
                        "received tensor should match the peer's seed");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_bidirectional_transfer\n";
    }

} // namespace

int main()
{
    try
    {
        test_deterministic_generation();
        test_allocation_reuse();
        test_odd_byte_sizes();
        test_configuration_mismatch_rejection();
        test_host_to_host_transfer();
        test_repeated_transfers_changing_seeds();
        test_bidirectional_transfer();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
