#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include <chrono>
#include <cstdint>
#include <cstring>
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
    // barrier_test: 29570-29659, broadcast_test: 29660-29790).
    constexpr std::uint16_t kSingleRankBase = 29800;
    constexpr std::uint16_t kTwoRankBase = 29802;
    constexpr std::uint16_t kThreeRankBase = 29810;
    constexpr std::uint16_t kRankOrderBase = 29820;
    constexpr std::uint16_t kOddSizeBase = 29830;
    constexpr std::uint16_t kZeroBytesSingleBase = 29840;
    constexpr std::uint16_t kZeroBytesBase = 29842;
    constexpr std::uint16_t kNullSendBase = 29850;
    constexpr std::uint16_t kNullRecvBase = 29860;
    constexpr std::uint16_t kLargePayloadBase = 29870;
    constexpr std::uint16_t kRepeatedBase = 29880;
    constexpr std::uint16_t kAllGatherBarrierBase = 29890;
    constexpr std::uint16_t kBarrierAllGatherBase = 29900;
    constexpr std::uint16_t kBroadcastAllGatherBase = 29910;
    constexpr std::uint16_t kAllGatherBroadcastBase = 29920;
    constexpr std::uint16_t kPhaseBase = 29930;
    constexpr std::uint16_t kFourRankBase = 29940;
    constexpr std::uint16_t kEightRankBase = 29950;

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
    // Test 1: 1-rank World — recv_buffer must equal send_buffer.
    // -----------------------------------------------------------------------------

    void test_single_rank()
    {
        auto peers = make_local_peers(kSingleRankBase, 1);

        auto world = tbccl::create_tcp_world(make_options(0, peers));

        const auto send = deterministic_buffer(64, 0xAAu);
        std::vector<std::uint8_t> recv(64, 0);

        tbccl::all_gather(*world, send.data(), recv.data(), send.size());

        expect(recv == send, "single-rank all_gather: recv != send");

        std::cout << "[PASS] test_single_rank\n";
    }

    // -----------------------------------------------------------------------------
    // Test 2: 2 ranks — result must be exactly [A][B].
    // -----------------------------------------------------------------------------

    void test_two_ranks()
    {
        constexpr std::size_t kBytesPerRank = 64;

        auto peers = make_local_peers(kTwoRankBase, 2);

        const auto patternA = deterministic_buffer(kBytesPerRank, 0x1000u);
        const auto patternB = deterministic_buffer(kBytesPerRank, 0x2000u);

        std::vector<std::exception_ptr> errors(2);
        std::vector<std::thread> threads;

        auto body = [&](std::size_t rank) -> std::function<void(tbccl::World &)>
        {
            return [&, rank](tbccl::World &world)
            {
                const auto &send = (rank == 0) ? patternA : patternB;

                std::vector<std::uint8_t> recv(2 * kBytesPerRank, 0);

                tbccl::all_gather(world, send.data(), recv.data(), send.size());

                expect(
                    std::memcmp(recv.data(), patternA.data(), kBytesPerRank) == 0,
                    "rank " + std::to_string(rank) + ": slot 0 != A");

                expect(
                    std::memcmp(
                        recv.data() + kBytesPerRank, patternB.data(),
                        kBytesPerRank) == 0,
                    "rank " + std::to_string(rank) + ": slot 1 != B");
            };
        };

        for (std::size_t rank = 0; rank < 2; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers), body(rank),
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_two_ranks\n";
    }

    // -----------------------------------------------------------------------------
    // Test 3: 3 ranks, [A][B][C] — the core correctness test.
    // -----------------------------------------------------------------------------

    void test_three_ranks()
    {
        constexpr std::size_t kBytesPerRank = 64;
        constexpr std::size_t kSize = 3;

        auto peers = make_local_peers(kThreeRankBase, kSize);

        std::vector<std::vector<std::uint8_t>> patterns;

        for (std::size_t i = 0; i < kSize; ++i)
        {
            patterns.push_back(
                deterministic_buffer(
                    kBytesPerRank, static_cast<std::uint32_t>(0x3000 + i)));
        }

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [&, rank](tbccl::World &world)
                {
                    std::vector<std::uint8_t> recv(kSize * kBytesPerRank, 0);

                    tbccl::all_gather(
                        world, patterns[rank].data(), recv.data(),
                        kBytesPerRank);

                    for (std::size_t slot = 0; slot < kSize; ++slot)
                    {
                        expect(
                            std::memcmp(
                                recv.data() + slot * kBytesPerRank,
                                patterns[slot].data(), kBytesPerRank) == 0,
                            "rank " + std::to_string(rank) + ": slot " +
                                std::to_string(slot) + " mismatch");
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_three_ranks\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4: explicit rank-order verification — each rank contributes
    // its own rank number; slot i must equal i, not merely "some
    // permutation of the contributions".
    // -----------------------------------------------------------------------------

    void test_rank_order()
    {
        constexpr std::size_t kSize = 5;

        auto peers = make_local_peers(kRankOrderBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    const std::uint32_t value = static_cast<std::uint32_t>(rank);
                    std::vector<std::uint32_t> recv(kSize, 0xFFFFFFFFu);

                    tbccl::all_gather(
                        world, &value, recv.data(), sizeof(value));

                    for (std::size_t slot = 0; slot < kSize; ++slot)
                    {
                        expect(
                            recv[slot] == static_cast<std::uint32_t>(slot),
                            "rank " + std::to_string(rank) + ": slot " +
                                std::to_string(slot) + " == " +
                                std::to_string(recv[slot]) + ", expected " +
                                std::to_string(slot));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_rank_order\n";
    }

    // -----------------------------------------------------------------------------
    // Test 5: odd, non-power-of-two payload size (257 bytes) — catches
    // alignment/word-size assumptions.
    // -----------------------------------------------------------------------------

    void test_odd_size_payload()
    {
        constexpr std::size_t kBytesPerRank = 257;
        constexpr std::size_t kSize = 3;

        auto peers = make_local_peers(kOddSizeBase, kSize);

        std::vector<std::vector<std::uint8_t>> patterns;

        for (std::size_t i = 0; i < kSize; ++i)
        {
            patterns.push_back(
                deterministic_buffer(
                    kBytesPerRank, static_cast<std::uint32_t>(0x7000 + i)));
        }

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [&, rank](tbccl::World &world)
                {
                    std::vector<std::uint8_t> recv(kSize * kBytesPerRank, 0);

                    tbccl::all_gather(
                        world, patterns[rank].data(), recv.data(),
                        kBytesPerRank);

                    for (std::size_t slot = 0; slot < kSize; ++slot)
                    {
                        expect(
                            std::memcmp(
                                recv.data() + slot * kBytesPerRank,
                                patterns[slot].data(), kBytesPerRank) == 0,
                            "rank " + std::to_string(rank) + ": slot " +
                                std::to_string(slot) + " mismatch");
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_odd_size_payload\n";
    }

    // -----------------------------------------------------------------------------
    // Test 6: bytes_per_rank == 0 must succeed with no throw/deadlock,
    // for both a 1-rank and a 3-rank World.
    // -----------------------------------------------------------------------------

    void test_zero_bytes()
    {
        {
            auto peers = make_local_peers(kZeroBytesSingleBase, 1);
            auto world = tbccl::create_tcp_world(make_options(0, peers));
            tbccl::all_gather(*world, nullptr, nullptr, 0);
        }

        {
            constexpr std::size_t kSize = 3;
            auto peers = make_local_peers(kZeroBytesBase, kSize);

            std::vector<std::exception_ptr> errors(kSize);
            std::vector<std::thread> threads;

            for (std::size_t rank = 0; rank < kSize; ++rank)
            {
                threads.emplace_back(
                    run_rank,
                    make_options(rank, peers),
                    [](tbccl::World &world)
                    { tbccl::all_gather(world, nullptr, nullptr, 0); },
                    std::ref(errors[rank]));
            }

            join_and_check(threads, errors);
        }

        std::cout << "[PASS] test_zero_bytes\n";
    }

    // -----------------------------------------------------------------------------
    // Test 7: null send buffer with bytes_per_rank > 0 must be
    // rejected. Every rank calls the same invalid operation, so
    // validation-before-any-communication means no rank can hang
    // waiting on one that already threw.
    // -----------------------------------------------------------------------------

    void test_null_send_buffer()
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(kNullSendBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [](tbccl::World &world)
                {
                    std::vector<std::uint8_t> recv(3 * 64, 0);
                    bool threw = false;

                    try
                    {
                        tbccl::all_gather(world, nullptr, recv.data(), 64);
                    }
                    catch (const std::exception &error)
                    {
                        threw = true;

                        const std::string message = error.what();

                        expect(
                            message.find("send buffer is null") !=
                                std::string::npos,
                            "wrong error for null send buffer: " + message);
                    }

                    expect(
                        threw,
                        "all_gather() should reject null send_buffer");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_null_send_buffer\n";
    }

    // -----------------------------------------------------------------------------
    // Test 8: null receive buffer with bytes_per_rank > 0 must be
    // rejected, same rationale as test 7.
    // -----------------------------------------------------------------------------

    void test_null_recv_buffer()
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(kNullRecvBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [](tbccl::World &world)
                {
                    std::vector<std::uint8_t> send(64, 0);
                    bool threw = false;

                    try
                    {
                        tbccl::all_gather(world, send.data(), nullptr, 64);
                    }
                    catch (const std::exception &error)
                    {
                        threw = true;

                        const std::string message = error.what();

                        expect(
                            message.find("receive buffer is null") !=
                                std::string::npos,
                            "wrong error for null receive buffer: " + message);
                    }

                    expect(
                        threw,
                        "all_gather() should reject null recv_buffer");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_null_recv_buffer\n";
    }

    // -----------------------------------------------------------------------------
    // Test 9: large per-rank payloads (1 MiB, 4 MiB) across 3 ranks,
    // byte-for-byte.
    // -----------------------------------------------------------------------------

    void test_large_payload(std::size_t bytes_per_rank, std::uint16_t base_port)
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(base_port, kSize);

        std::vector<std::vector<std::uint8_t>> patterns;

        for (std::size_t i = 0; i < kSize; ++i)
        {
            patterns.push_back(
                deterministic_buffer(
                    bytes_per_rank, static_cast<std::uint32_t>(0xC0FFEE + i)));
        }

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers, /*timeout_ms=*/15000),
                [&, rank](tbccl::World &world)
                {
                    std::vector<std::uint8_t> recv(kSize * bytes_per_rank, 0);

                    tbccl::all_gather(
                        world, patterns[rank].data(), recv.data(),
                        bytes_per_rank);

                    for (std::size_t slot = 0; slot < kSize; ++slot)
                    {
                        expect(
                            std::memcmp(
                                recv.data() + slot * bytes_per_rank,
                                patterns[slot].data(), bytes_per_rank) == 0,
                            "rank " + std::to_string(rank) + ": slot " +
                                std::to_string(slot) + " mismatch");
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_large_payload (" << bytes_per_rank
            << " bytes/rank)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 10: 1000 repeated all_gathers on a 3-rank World, changing
    // content every iteration.
    // -----------------------------------------------------------------------------

    void test_repeated_all_gather()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 1000;
        constexpr std::size_t kBytesPerRank = 64;

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
                        const auto send =
                            deterministic_buffer(
                                kBytesPerRank,
                                static_cast<std::uint32_t>(
                                    iteration * 100 + static_cast<int>(rank)));

                        std::vector<std::uint8_t> recv(
                            kSize * kBytesPerRank, 0);

                        tbccl::all_gather(
                            world, send.data(), recv.data(), kBytesPerRank);

                        for (std::size_t slot = 0; slot < kSize; ++slot)
                        {
                            const auto expected =
                                deterministic_buffer(
                                    kBytesPerRank,
                                    static_cast<std::uint32_t>(
                                        iteration * 100 +
                                        static_cast<int>(slot)));

                            if (std::memcmp(
                                    recv.data() + slot * kBytesPerRank,
                                    expected.data(), kBytesPerRank) != 0)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": slot " + std::to_string(slot) +
                                    " mismatch");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_repeated_all_gather (" << kIterations
            << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Tests 11-14: all_gather interleaved with barrier() and
    // broadcast(), both orders, repeated — three collectives now share
    // the same World connections.
    // -----------------------------------------------------------------------------

    void run_interleaved(
        const std::string &name,
        std::uint16_t base_port,
        const std::function<void(tbccl::World &, std::size_t, int)> &per_iteration)
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;

        auto peers = make_local_peers(base_port, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank, &per_iteration](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < kIterations; ++iteration)
                    {
                        per_iteration(world, rank, iteration);
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] " << name << " (" << kIterations << " iterations)\n";
    }

    void test_all_gather_then_barrier()
    {
        constexpr std::size_t kBytesPerRank = 32;

        run_interleaved(
            "test_all_gather_then_barrier",
            kAllGatherBarrierBase,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const auto send =
                    deterministic_buffer(
                        kBytesPerRank, static_cast<std::uint32_t>(iteration));

                std::vector<std::uint8_t> recv(world.size() * kBytesPerRank, 0);

                tbccl::all_gather(world, send.data(), recv.data(), kBytesPerRank);

                tbccl::barrier(world);

                const auto expected_slot =
                    deterministic_buffer(
                        kBytesPerRank, static_cast<std::uint32_t>(iteration));

                if (std::memcmp(
                        recv.data() + rank * kBytesPerRank,
                        expected_slot.data(), kBytesPerRank) != 0)
                {
                    throw std::runtime_error("own slot corrupted");
                }
            });
    }

    void test_barrier_then_all_gather()
    {
        constexpr std::size_t kBytesPerRank = 32;

        run_interleaved(
            "test_barrier_then_all_gather",
            kBarrierAllGatherBase,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                tbccl::barrier(world);

                const auto send =
                    deterministic_buffer(
                        kBytesPerRank, static_cast<std::uint32_t>(iteration));

                std::vector<std::uint8_t> recv(world.size() * kBytesPerRank, 0);

                tbccl::all_gather(world, send.data(), recv.data(), kBytesPerRank);

                const auto expected_slot =
                    deterministic_buffer(
                        kBytesPerRank, static_cast<std::uint32_t>(iteration));

                if (std::memcmp(
                        recv.data() + rank * kBytesPerRank,
                        expected_slot.data(), kBytesPerRank) != 0)
                {
                    throw std::runtime_error("own slot corrupted");
                }
            });
    }

    void test_broadcast_then_all_gather()
    {
        constexpr std::size_t kBytesPerRank = 32;

        run_interleaved(
            "test_broadcast_then_all_gather",
            kBroadcastAllGatherBase,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const std::size_t root =
                    static_cast<std::size_t>(iteration) % world.size();

                const auto broadcastPattern =
                    deterministic_buffer(
                        16, static_cast<std::uint32_t>(0x8000 + iteration));

                auto bbuf = (rank == root)
                                ? broadcastPattern
                                : deterministic_buffer(16, 0x1234u);

                tbccl::broadcast(world, bbuf.data(), bbuf.size(), root);

                if (bbuf != broadcastPattern)
                {
                    throw std::runtime_error("broadcast payload corrupted");
                }

                const auto send =
                    deterministic_buffer(
                        kBytesPerRank, static_cast<std::uint32_t>(iteration));

                std::vector<std::uint8_t> recv(world.size() * kBytesPerRank, 0);

                tbccl::all_gather(world, send.data(), recv.data(), kBytesPerRank);

                const auto expected_slot =
                    deterministic_buffer(
                        kBytesPerRank, static_cast<std::uint32_t>(iteration));

                if (std::memcmp(
                        recv.data() + rank * kBytesPerRank,
                        expected_slot.data(), kBytesPerRank) != 0)
                {
                    throw std::runtime_error("own slot corrupted");
                }
            });
    }

    void test_all_gather_then_broadcast()
    {
        constexpr std::size_t kBytesPerRank = 32;

        run_interleaved(
            "test_all_gather_then_broadcast",
            kAllGatherBroadcastBase,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const auto send =
                    deterministic_buffer(
                        kBytesPerRank, static_cast<std::uint32_t>(iteration));

                std::vector<std::uint8_t> recv(world.size() * kBytesPerRank, 0);

                tbccl::all_gather(world, send.data(), recv.data(), kBytesPerRank);

                const auto expected_slot =
                    deterministic_buffer(
                        kBytesPerRank, static_cast<std::uint32_t>(iteration));

                if (std::memcmp(
                        recv.data() + rank * kBytesPerRank,
                        expected_slot.data(), kBytesPerRank) != 0)
                {
                    throw std::runtime_error("own slot corrupted");
                }

                const std::size_t root =
                    static_cast<std::size_t>(iteration) % world.size();

                const auto broadcastPattern =
                    deterministic_buffer(
                        16, static_cast<std::uint32_t>(0x9000 + iteration));

                auto bbuf = (rank == root)
                                ? broadcastPattern
                                : deterministic_buffer(16, 0x4321u);

                tbccl::broadcast(world, bbuf.data(), bbuf.size(), root);

                if (bbuf != broadcastPattern)
                {
                    throw std::runtime_error("broadcast payload corrupted");
                }
            });
    }

    // -----------------------------------------------------------------------------
    // Test 15: point-to-point traffic before and after all_gather —
    // byte streams must stay aligned across the collective boundary.
    // -----------------------------------------------------------------------------

    void test_communication_phases()
    {
        constexpr std::size_t kBytesPerRank = 64;
        auto peers = make_local_peers(kPhaseBase, 3);

        std::vector<std::exception_ptr> errors(3);
        std::vector<std::thread> threads;

        constexpr std::uint32_t kPhaseAValue = 0x11111111u;
        constexpr std::uint32_t kPhaseBValue = 0x22222222u;

        threads.emplace_back(
            run_rank,
            make_options(0, peers),
            [&](tbccl::World &world)
            {
                std::uint32_t received = 0;
                world.recv(1, &received, sizeof(received));
                expect(received == kPhaseAValue, "phase A: wrong value from rank 1");

                const auto send = deterministic_buffer(kBytesPerRank, 0);
                std::vector<std::uint8_t> recv(3 * kBytesPerRank, 0);
                tbccl::all_gather(world, send.data(), recv.data(), kBytesPerRank);

                std::uint32_t v = kPhaseBValue;
                world.send(1, &v, sizeof(v));
            },
            std::ref(errors[0]));

        threads.emplace_back(
            run_rank,
            make_options(1, peers),
            [&](tbccl::World &world)
            {
                std::uint32_t v = kPhaseAValue;
                world.send(0, &v, sizeof(v));

                const auto send = deterministic_buffer(kBytesPerRank, 1);
                std::vector<std::uint8_t> recv(3 * kBytesPerRank, 0);
                tbccl::all_gather(world, send.data(), recv.data(), kBytesPerRank);

                std::uint32_t received = 0;
                world.recv(0, &received, sizeof(received));
                expect(received == kPhaseBValue, "phase B: wrong value from rank 0");
            },
            std::ref(errors[1]));

        threads.emplace_back(
            run_rank,
            make_options(2, peers),
            [&](tbccl::World &world)
            {
                const auto send = deterministic_buffer(kBytesPerRank, 2);
                std::vector<std::uint8_t> recv(3 * kBytesPerRank, 0);
                tbccl::all_gather(world, send.data(), recv.data(), kBytesPerRank);

                for (std::size_t slot = 0; slot < 3; ++slot)
                {
                    const auto expected =
                        deterministic_buffer(
                            kBytesPerRank, static_cast<std::uint32_t>(slot));

                    expect(
                        std::memcmp(
                            recv.data() + slot * kBytesPerRank,
                            expected.data(), kBytesPerRank) == 0,
                        "rank 2: slot " + std::to_string(slot) + " mismatch");
                }
            },
            std::ref(errors[2]));

        join_and_check(threads, errors);

        std::cout << "[PASS] test_communication_phases\n";
    }

    // -----------------------------------------------------------------------------
    // Tests 16/17: more local ranks, repeated. Correctness at larger N.
    // -----------------------------------------------------------------------------

    void test_n_rank_repeated_all_gather(
        std::size_t n,
        std::uint16_t base_port,
        int iterations)
    {
        constexpr std::size_t kBytesPerRank = 64;

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
                        const auto send =
                            deterministic_buffer(
                                kBytesPerRank,
                                static_cast<std::uint32_t>(
                                    iteration * 100 + static_cast<int>(rank)));

                        std::vector<std::uint8_t> recv(n * kBytesPerRank, 0);

                        tbccl::all_gather(
                            world, send.data(), recv.data(), kBytesPerRank);

                        for (std::size_t slot = 0; slot < n; ++slot)
                        {
                            const auto expected =
                                deterministic_buffer(
                                    kBytesPerRank,
                                    static_cast<std::uint32_t>(
                                        iteration * 100 +
                                        static_cast<int>(slot)));

                            if (std::memcmp(
                                    recv.data() + slot * kBytesPerRank,
                                    expected.data(), kBytesPerRank) != 0)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": slot " + std::to_string(slot) +
                                    " mismatch");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_n_rank_repeated_all_gather (n=" << n
            << ", iterations=" << iterations << ")\n";
    }

} // namespace

int main()
{
    try
    {
        test_single_rank();
        test_two_ranks();
        test_three_ranks();
        test_rank_order();
        test_odd_size_payload();
        test_zero_bytes();
        test_null_send_buffer();
        test_null_recv_buffer();
        test_large_payload(1ULL * 1024 * 1024, kLargePayloadBase);
        test_large_payload(4ULL * 1024 * 1024, kLargePayloadBase + 3);
        test_repeated_all_gather();
        test_all_gather_then_barrier();
        test_barrier_then_all_gather();
        test_broadcast_then_all_gather();
        test_all_gather_then_broadcast();
        test_communication_phases();
        test_n_rank_repeated_all_gather(4, kFourRankBase, 100);
        test_n_rank_repeated_all_gather(8, kEightRankBase, 100);
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
