#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include <chrono>
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

    // Fixed high port ranges, one block per test, well away from
    // production (18515) and from the other test files' ranges
    // (tcp_transport_test: 28515+, world_test: 29500-29569,
    // barrier_test: 29570-29659).
    constexpr std::uint16_t kSingleRankBase = 29660;
    constexpr std::uint16_t kTwoRankBase = 29662;
    constexpr std::uint16_t kArbitraryRootBase = 29670;
    constexpr std::uint16_t kEveryRankAsRootBase = 29680;
    constexpr std::uint16_t kRepeatedBase = 29690;
    constexpr std::uint16_t kZeroBytesBase = 29700;
    constexpr std::uint16_t kInvalidRootBase = 29710;
    constexpr std::uint16_t kNullBufferBase = 29720;
    constexpr std::uint16_t kLargePayloadBase = 29730;
    constexpr std::uint16_t kPhaseBase = 29740;
    constexpr std::uint16_t kBroadcastThenBarrierBase = 29750;
    constexpr std::uint16_t kBarrierThenBroadcastBase = 29760;
    constexpr std::uint16_t kFourRankBase = 29770;
    constexpr std::uint16_t kEightRankBase = 29780;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    std::vector<tbccl::PeerEndpoint> make_local_peers(
        std::uint16_t base_port,
        std::size_t count)
    {
        std::vector<tbccl::PeerEndpoint> peers;

        for (std::size_t i = 0; i < count; ++i)
        {
            peers.push_back(
                {"127.0.0.1",
                 static_cast<std::uint16_t>(base_port + i)});
        }

        return peers;
    }

    tbccl::TcpWorldOptions make_options(
        std::size_t rank,
        const std::vector<tbccl::PeerEndpoint> &peers,
        int timeout_ms = 5000)
    {
        tbccl::TcpWorldOptions options;

        options.rank = rank;
        options.peers = peers;
        options.bootstrap_timeout = std::chrono::milliseconds(timeout_ms);

        return options;
    }

    void run_rank(
        const tbccl::TcpWorldOptions &options,
        const std::function<void(tbccl::World &)> &body,
        std::exception_ptr &out_exception)
    {
        try
        {
            auto world = tbccl::create_tcp_world(options);
            body(*world);
        }
        catch (...)
        {
            out_exception = std::current_exception();
        }
    }

    std::string what_or_empty(const std::exception_ptr &ptr)
    {
        if (!ptr)
        {
            return "";
        }

        try
        {
            std::rethrow_exception(ptr);
        }
        catch (const std::exception &error)
        {
            return error.what();
        }
        catch (...)
        {
            return "non-std::exception";
        }
    }

    void join_and_check(
        std::vector<std::thread> &threads,
        const std::vector<std::exception_ptr> &errors)
    {
        for (auto &thread : threads)
        {
            thread.join();
        }

        for (std::size_t rank = 0; rank < errors.size(); ++rank)
        {
            if (errors[rank])
            {
                throw std::runtime_error(
                    "rank " + std::to_string(rank) +
                    " failed: " + what_or_empty(errors[rank]));
            }
        }
    }

    std::vector<std::uint8_t> deterministic_buffer(
        std::size_t size,
        std::uint32_t seed)
    {
        std::vector<std::uint8_t> buffer(size);

        for (std::size_t i = 0; i < size; ++i)
        {
            buffer[i] = static_cast<std::uint8_t>(
                (static_cast<std::uint32_t>(i) * 2654435761u + seed) &
                0xFFu);
        }

        return buffer;
    }

    // -----------------------------------------------------------------------------
    // Test 1: a 1-rank World — broadcast must return immediately, with
    // the (already-correct, since root == self) buffer left unchanged.
    // -----------------------------------------------------------------------------

    void test_single_rank_broadcast()
    {
        auto peers = make_local_peers(kSingleRankBase, 1);

        auto world = tbccl::create_tcp_world(make_options(0, peers));

        auto buffer = deterministic_buffer(64, 0xAAu);
        const auto original = buffer;

        tbccl::broadcast(*world, buffer.data(), buffer.size(), 0);

        expect(buffer == original, "single-rank broadcast changed the buffer");

        std::cout << "[PASS] test_single_rank_broadcast\n";
    }

    // -----------------------------------------------------------------------------
    // Test 2: 2 ranks, root = 0, 64 B payload.
    // -----------------------------------------------------------------------------

    void test_two_rank_root0()
    {
        auto peers = make_local_peers(kTwoRankBase, 2);

        const auto source = deterministic_buffer(64, 0x1000u);

        std::vector<std::exception_ptr> errors(2);
        std::vector<std::thread> threads;

        threads.emplace_back(
            run_rank,
            make_options(0, peers),
            [&](tbccl::World &world)
            {
                auto buffer = source;
                tbccl::broadcast(world, buffer.data(), buffer.size(), 0);
                expect(buffer == source, "root buffer changed");
            },
            std::ref(errors[0]));

        threads.emplace_back(
            run_rank,
            make_options(1, peers),
            [&](tbccl::World &world)
            {
                auto buffer = deterministic_buffer(64, 0x2000u); // different pattern
                tbccl::broadcast(world, buffer.data(), buffer.size(), 0);
                expect(buffer == source, "non-root did not receive root's data");
            },
            std::ref(errors[1]));

        join_and_check(threads, errors);

        std::cout << "[PASS] test_two_rank_root0\n";
    }

    // -----------------------------------------------------------------------------
    // Test 3: root is NOT rank 0 (3 ranks, root = 2) — catches any
    // accidental hard-coding of rank 0 as the sender.
    // -----------------------------------------------------------------------------

    void test_arbitrary_root()
    {
        constexpr std::size_t kRoot = 2;
        auto peers = make_local_peers(kArbitraryRootBase, 3);

        const auto source = deterministic_buffer(64, 0x3000u);

        std::vector<std::exception_ptr> errors(3);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < 3; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank, &source](tbccl::World &world)
                {
                    auto buffer = (rank == kRoot)
                                      ? source
                                      : deterministic_buffer(64, 0x9999u);

                    tbccl::broadcast(world, buffer.data(), buffer.size(), kRoot);

                    expect(
                        buffer == source,
                        "rank " + std::to_string(rank) +
                            " did not end up with root's data");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_arbitrary_root\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4: every rank takes a turn as root, sequentially, on one
    // bootstrapped 4-rank World. Before each round only the current
    // root has the expected pattern; after, every rank must have it.
    // -----------------------------------------------------------------------------

    void test_every_rank_as_root()
    {
        constexpr std::size_t kSize = 4;
        auto peers = make_local_peers(kEveryRankAsRootBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    for (std::size_t root = 0; root < kSize; ++root)
                    {
                        const auto pattern =
                            deterministic_buffer(
                                64, static_cast<std::uint32_t>(0x4000 + root));

                        auto buffer = (rank == root)
                                          ? pattern
                                          : deterministic_buffer(64, 0xDEADu);

                        tbccl::broadcast(
                            world, buffer.data(), buffer.size(), root);

                        expect(
                            buffer == pattern,
                            "rank " + std::to_string(rank) +
                                " wrong data for root " +
                                std::to_string(root));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_every_rank_as_root\n";
    }

    // -----------------------------------------------------------------------------
    // Test 5: 1000 repeated broadcasts on a 3-rank World, root rotating
    // every iteration. Checks for stale bytes / protocol drift.
    // -----------------------------------------------------------------------------

    void test_repeated_broadcasts()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 1000;

        auto peers = make_local_peers(kRepeatedBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < kIterations; ++iteration)
                    {
                        const std::size_t root =
                            static_cast<std::size_t>(iteration) % kSize;

                        const auto pattern =
                            deterministic_buffer(
                                64, static_cast<std::uint32_t>(iteration));

                        auto buffer = (rank == root)
                                          ? pattern
                                          : deterministic_buffer(64, 0xFFFFu);

                        tbccl::broadcast(
                            world, buffer.data(), buffer.size(), root);

                        if (buffer != pattern)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": wrong data");
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_repeated_broadcasts (" << kIterations
            << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 6: bytes == 0 must succeed with no crash/throw/deadlock,
    // buffer may be nullptr. Exercised for root == 0 and a non-zero
    // root, sequentially on one bootstrapped World.
    // -----------------------------------------------------------------------------

    void test_zero_bytes()
    {
        auto peers = make_local_peers(kZeroBytesBase, 3);

        std::vector<std::exception_ptr> errors(3);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < 3; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [](tbccl::World &world)
                {
                    tbccl::broadcast(world, nullptr, 0, 0);
                    tbccl::broadcast(world, nullptr, 0, 2);
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_zero_bytes\n";
    }

    // -----------------------------------------------------------------------------
    // Test 7: an invalid root must be rejected. broadcast() validates
    // root before any communication on every rank independently, so
    // every rank calling it with the same bad root is safe (no rank
    // ever blocks waiting on a peer that already threw).
    // -----------------------------------------------------------------------------

    void test_invalid_root()
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(kInvalidRootBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [](tbccl::World &world)
                {
                    std::uint8_t buffer = 0;
                    bool threw = false;

                    try
                    {
                        tbccl::broadcast(world, &buffer, sizeof(buffer), 3);
                    }
                    catch (const std::exception &error)
                    {
                        threw = true;

                        const std::string message = error.what();

                        expect(
                            message.find("root rank 3") != std::string::npos,
                            "error should name the invalid root: " + message);

                        expect(
                            message.find("world size 3") != std::string::npos,
                            "error should name the world size: " + message);
                    }

                    expect(threw, "broadcast() should reject root >= size");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_invalid_root\n";
    }

    // -----------------------------------------------------------------------------
    // Test 8: nullptr + bytes > 0 must be rejected cleanly. Every rank
    // calls the same invalid operation, so — like test 7 — no rank
    // depends on another actually sending/receiving.
    // -----------------------------------------------------------------------------

    void test_null_buffer_nonzero_bytes()
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(kNullBufferBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [](tbccl::World &world)
                {
                    bool threw = false;

                    try
                    {
                        tbccl::broadcast(world, nullptr, 64, 0);
                    }
                    catch (const std::exception &error)
                    {
                        threw = true;

                        const std::string message = error.what();

                        expect(
                            message.find("buffer is null") != std::string::npos,
                            "wrong error for null buffer: " + message);
                    }

                    expect(threw, "broadcast() should reject nullptr with bytes > 0");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_null_buffer_nonzero_bytes\n";
    }

    // -----------------------------------------------------------------------------
    // Test 9: large payloads (1 MiB, 16 MiB), byte-for-byte, across 3
    // ranks — confirms correctness well beyond a single TCP segment or
    // socket buffer.
    // -----------------------------------------------------------------------------

    void test_large_payload(std::size_t size, std::uint16_t base_port)
    {
        auto peers = make_local_peers(base_port, 3);

        const auto source = deterministic_buffer(size, 0xC0FFEEu);

        std::vector<std::exception_ptr> errors(3);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < 3; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers, /*timeout_ms=*/10000),
                [rank, &source](tbccl::World &world)
                {
                    std::vector<std::uint8_t> buffer;

                    if (rank == 0)
                    {
                        buffer = source;
                    }
                    else
                    {
                        buffer.assign(source.size(), 0);
                    }

                    tbccl::broadcast(world, buffer.data(), buffer.size(), 0);

                    expect(buffer == source, "large payload mismatch");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_large_payload (" << size << " bytes)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 10: point-to-point phase A, then broadcast, then
    // point-to-point phase B — byte streams must stay aligned across
    // the collective boundary.
    // -----------------------------------------------------------------------------

    void test_communication_phases()
    {
        auto peers = make_local_peers(kPhaseBase, 3);

        std::vector<std::exception_ptr> errors(3);
        std::vector<std::thread> threads;

        constexpr std::uint32_t kPhaseAValue = 0x11111111u;
        constexpr std::uint32_t kPhaseBValue = 0x22222222u;
        const auto broadcastPattern = deterministic_buffer(64, 0x5000u);

        threads.emplace_back(
            run_rank,
            make_options(0, peers),
            [&](tbccl::World &world)
            {
                std::uint32_t received = 0;
                world.recv(1, &received, sizeof(received));
                expect(received == kPhaseAValue, "phase A: wrong value from rank 1");

                auto buffer = deterministic_buffer(64, 0x9999u);
                tbccl::broadcast(world, buffer.data(), buffer.size(), 2);
                expect(buffer == broadcastPattern, "rank 0: wrong broadcast data");
            },
            std::ref(errors[0]));

        threads.emplace_back(
            run_rank,
            make_options(1, peers),
            [&](tbccl::World &world)
            {
                std::uint32_t v = kPhaseAValue;
                world.send(0, &v, sizeof(v));

                auto buffer = deterministic_buffer(64, 0x9999u);
                tbccl::broadcast(world, buffer.data(), buffer.size(), 2);
                expect(buffer == broadcastPattern, "rank 1: wrong broadcast data");

                std::uint32_t received = 0;
                world.recv(2, &received, sizeof(received));
                expect(received == kPhaseBValue, "phase B: wrong value from rank 2");
            },
            std::ref(errors[1]));

        threads.emplace_back(
            run_rank,
            make_options(2, peers),
            [&](tbccl::World &world)
            {
                auto buffer = broadcastPattern;
                tbccl::broadcast(world, buffer.data(), buffer.size(), 2);

                std::uint32_t v = kPhaseBValue;
                world.send(1, &v, sizeof(v));
            },
            std::ref(errors[2]));

        join_and_check(threads, errors);

        std::cout << "[PASS] test_communication_phases\n";
    }

    // -----------------------------------------------------------------------------
    // Test 11 / 12: broadcast and barrier interleaved, in both orders,
    // repeated — the project now has two collectives sharing the same
    // World connections, so this checks neither one consumes bytes
    // belonging to the other.
    // -----------------------------------------------------------------------------

    void test_broadcast_then_barrier()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;

        auto peers = make_local_peers(kBroadcastThenBarrierBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < kIterations; ++iteration)
                    {
                        const std::size_t root =
                            static_cast<std::size_t>(iteration) % kSize;

                        const auto pattern =
                            deterministic_buffer(
                                32, static_cast<std::uint32_t>(iteration));

                        auto buffer = (rank == root)
                                          ? pattern
                                          : deterministic_buffer(32, 0xAAAAu);

                        tbccl::broadcast(
                            world, buffer.data(), buffer.size(), root);

                        tbccl::barrier(world);

                        if (buffer != pattern)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": wrong data after broadcast+barrier");
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_broadcast_then_barrier (" << kIterations
            << " iterations)\n";
    }

    void test_barrier_then_broadcast()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;

        auto peers = make_local_peers(kBarrierThenBroadcastBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < kIterations; ++iteration)
                    {
                        const std::size_t root =
                            static_cast<std::size_t>(iteration) % kSize;

                        const auto pattern =
                            deterministic_buffer(
                                32, static_cast<std::uint32_t>(iteration));

                        auto buffer = (rank == root)
                                          ? pattern
                                          : deterministic_buffer(32, 0xBBBBu);

                        tbccl::barrier(world);

                        tbccl::broadcast(
                            world, buffer.data(), buffer.size(), root);

                        if (buffer != pattern)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": wrong data after barrier+broadcast");
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_barrier_then_broadcast (" << kIterations
            << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 13: more local ranks, rotating root, repeated. Correctness
    // at larger N, not performance.
    // -----------------------------------------------------------------------------

    void test_n_rank_repeated_broadcasts(
        std::size_t n,
        std::uint16_t base_port,
        int iterations)
    {
        auto peers = make_local_peers(base_port, n);

        std::vector<std::exception_ptr> errors(n);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < n; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers, /*timeout_ms=*/8000),
                [rank, n, iterations](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < iterations; ++iteration)
                    {
                        const std::size_t root =
                            static_cast<std::size_t>(iteration) % n;

                        const auto pattern =
                            deterministic_buffer(
                                64, static_cast<std::uint32_t>(iteration));

                        auto buffer = (rank == root)
                                          ? pattern
                                          : deterministic_buffer(64, 0xCCCCu);

                        tbccl::broadcast(
                            world, buffer.data(), buffer.size(), root);

                        if (buffer != pattern)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": wrong data");
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_n_rank_repeated_broadcasts (n=" << n
            << ", iterations=" << iterations << ")\n";
    }

} // namespace

int main()
{
    try
    {
        test_single_rank_broadcast();
        test_two_rank_root0();
        test_arbitrary_root();
        test_every_rank_as_root();
        test_repeated_broadcasts();
        test_zero_bytes();
        test_invalid_root();
        test_null_buffer_nonzero_bytes();
        test_large_payload(1ULL * 1024 * 1024, kLargePayloadBase);
        test_large_payload(16ULL * 1024 * 1024, kLargePayloadBase + 3);
        test_communication_phases();
        test_broadcast_then_barrier();
        test_barrier_then_broadcast();
        test_n_rank_repeated_broadcasts(4, kFourRankBase, 100);
        test_n_rank_repeated_broadcasts(8, kEightRankBase, 100);
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
