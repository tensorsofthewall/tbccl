#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include <atomic>
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

    // Fixed high port ranges, one block per test, well away from the
    // production default (18515) and from tcp_transport_test's and
    // world_test's ranges.
    constexpr std::uint16_t kSingleRankBase = 29570;
    constexpr std::uint16_t kTwoRankBase = 29572;
    constexpr std::uint16_t kSyncBase = 29580;
    constexpr std::uint16_t kThreeRankBase = 29590;
    constexpr std::uint16_t kDelayedAtomicsBase = 29600;
    constexpr std::uint16_t kRepeatedBase = 29610;
    constexpr std::uint16_t kPhaseBase = 29620;
    constexpr std::uint16_t kFourRankBase = 29630;
    constexpr std::uint16_t kEightRankBase = 29640;
    constexpr std::uint16_t kInvalidTokenBase = 29650;

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

    // -----------------------------------------------------------------------------
    // Test 1: a 1-rank World has nothing to synchronize with; barrier()
    // must return immediately, with no network operations.
    // -----------------------------------------------------------------------------

    void test_single_rank_barrier()
    {
        auto peers = make_local_peers(kSingleRankBase, 1);

        auto world = tbccl::create_tcp_world(make_options(0, peers));
        tbccl::barrier(*world);

        std::cout << "[PASS] test_single_rank_barrier\n";
    }

    // -----------------------------------------------------------------------------
    // Test 2: basic 2-rank smoke test.
    // -----------------------------------------------------------------------------

    void test_two_rank_barrier()
    {
        auto peers = make_local_peers(kTwoRankBase, 2);

        std::vector<std::exception_ptr> errors(2);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < 2; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [](tbccl::World &world)
                { tbccl::barrier(world); },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_two_rank_barrier\n";
    }

    // -----------------------------------------------------------------------------
    // Test 3: basic 3-rank smoke test.
    // -----------------------------------------------------------------------------

    void test_three_rank_barrier()
    {
        auto peers = make_local_peers(kThreeRankBase, 3);

        std::vector<std::exception_ptr> errors(3);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < 3; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [](tbccl::World &world)
                { tbccl::barrier(world); },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_three_rank_barrier\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4a: the real correctness property, proven via wall-clock
    // ordering. rank 1 sleeps before calling barrier(); every rank
    // timestamps itself right after barrier() returns; no rank's
    // departure may precede the late rank's arrival. A no-op or
    // half-implemented barrier would pass test_two/three_rank_barrier
    // above but fail this one.
    // -----------------------------------------------------------------------------

    void test_barrier_actually_synchronizes()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kLateRank = 1;
        constexpr auto kLateSleep = std::chrono::milliseconds(300);

        auto peers = make_local_peers(kSyncBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::chrono::steady_clock::time_point> arrival(kSize);
        std::vector<std::chrono::steady_clock::time_point> departure(kSize);

        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank, &arrival, &departure, kLateSleep](tbccl::World &world)
                {
                    if (rank == kLateRank)
                    {
                        std::this_thread::sleep_for(kLateSleep);
                    }

                    arrival[rank] = std::chrono::steady_clock::now();

                    tbccl::barrier(world);

                    departure[rank] = std::chrono::steady_clock::now();
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            expect(
                departure[rank] >= arrival[kLateRank],
                "rank " + std::to_string(rank) +
                    " left the barrier before the late rank arrived");
        }

        std::cout << "[PASS] test_barrier_actually_synchronizes\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4b: the same property, proven the way the plan asks for it —
    // explicit atomic "finished" flags that the late rank checks are
    // still false immediately before it enters the barrier itself.
    // -----------------------------------------------------------------------------

    void test_delayed_arrival_atomics()
    {
        auto peers = make_local_peers(kDelayedAtomicsBase, 3);

        std::vector<std::exception_ptr> errors(3);
        std::atomic<bool> rank0_finished{false};
        std::atomic<bool> rank1_finished{false};
        std::atomic<bool> premature{false};

        std::vector<std::thread> threads;

        threads.emplace_back(
            run_rank,
            make_options(0, peers),
            [&](tbccl::World &world)
            {
                tbccl::barrier(world);
                rank0_finished.store(true);
            },
            std::ref(errors[0]));

        threads.emplace_back(
            run_rank,
            make_options(1, peers),
            [&](tbccl::World &world)
            {
                tbccl::barrier(world);
                rank1_finished.store(true);
            },
            std::ref(errors[1]));

        threads.emplace_back(
            run_rank,
            make_options(2, peers),
            [&](tbccl::World &world)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                if (rank0_finished.load() || rank1_finished.load())
                {
                    premature.store(true);
                }

                tbccl::barrier(world);
            },
            std::ref(errors[2]));

        join_and_check(threads, errors);

        expect(
            !premature.load(),
            "rank 0 or rank 1 exited barrier before rank 2 arrived");
        expect(rank0_finished.load(), "rank 0 never finished barrier");
        expect(rank1_finished.load(), "rank 1 never finished barrier");

        std::cout << "[PASS] test_delayed_arrival_atomics\n";
    }

    // -----------------------------------------------------------------------------
    // Test 5: 1000 consecutive barriers on a 3-rank World. Validates no
    // stale bytes, no unread ARRIVED/RELEASE tokens, and no
    // barrier-to-barrier desynchronization or intermittent deadlock.
    // -----------------------------------------------------------------------------

    void test_repeated_barriers()
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
                [](tbccl::World &world)
                {
                    for (int i = 0; i < kIterations; ++i)
                    {
                        tbccl::barrier(world);
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_repeated_barriers (" << kIterations
            << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 6: barrier() correctly separates two sequential application
    // communication phases — ranks 1 and 2 send values to rank 0 before
    // the barrier, rank 0 sends different values back after it.
    // -----------------------------------------------------------------------------

    void test_communication_phases()
    {
        auto peers = make_local_peers(kPhaseBase, 3);

        std::vector<std::exception_ptr> errors(3);
        std::vector<std::thread> threads;

        constexpr std::uint32_t kBeforeValue1 = 0x11111111u;
        constexpr std::uint32_t kBeforeValue2 = 0x22222222u;
        constexpr std::uint32_t kAfterValue1 = 0x33333333u;
        constexpr std::uint32_t kAfterValue2 = 0x44444444u;

        threads.emplace_back(
            run_rank,
            make_options(0, peers),
            [&](tbccl::World &world)
            {
                std::uint32_t v1 = 0;
                std::uint32_t v2 = 0;

                world.recv(1, &v1, sizeof(v1));
                world.recv(2, &v2, sizeof(v2));

                expect(v1 == kBeforeValue1, "phase A: wrong value from rank 1");
                expect(v2 == kBeforeValue2, "phase A: wrong value from rank 2");

                tbccl::barrier(world);

                std::uint32_t a1 = kAfterValue1;
                std::uint32_t a2 = kAfterValue2;

                world.send(1, &a1, sizeof(a1));
                world.send(2, &a2, sizeof(a2));
            },
            std::ref(errors[0]));

        threads.emplace_back(
            run_rank,
            make_options(1, peers),
            [&](tbccl::World &world)
            {
                std::uint32_t v = kBeforeValue1;
                world.send(0, &v, sizeof(v));

                tbccl::barrier(world);

                std::uint32_t received = 0;
                world.recv(0, &received, sizeof(received));

                expect(
                    received == kAfterValue1, "phase B: wrong value at rank 1");
            },
            std::ref(errors[1]));

        threads.emplace_back(
            run_rank,
            make_options(2, peers),
            [&](tbccl::World &world)
            {
                std::uint32_t v = kBeforeValue2;
                world.send(0, &v, sizeof(v));

                tbccl::barrier(world);

                std::uint32_t received = 0;
                world.recv(0, &received, sizeof(received));

                expect(
                    received == kAfterValue2, "phase B: wrong value at rank 2");
            },
            std::ref(errors[2]));

        join_and_check(threads, errors);

        std::cout << "[PASS] test_communication_phases\n";
    }

    // -----------------------------------------------------------------------------
    // Not in the numbered plan, but acceptance criteria requires
    // "control tokens are validated" — none of the tests above exercise
    // that path, since they all send well-formed tokens by construction
    // (they call tbccl::barrier() on both sides). Here rank 1 bypasses
    // barrier() and sends a bare garbage byte directly on the same
    // connection barrier() would use, simulating a misbehaving/corrupt
    // peer; rank 0's barrier() call must reject it rather than silently
    // accepting arbitrary bytes as an arrival signal.
    // -----------------------------------------------------------------------------

    void test_invalid_token_rejected()
    {
        auto peers = make_local_peers(kInvalidTokenBase, 2);

        std::vector<std::exception_ptr> errors(2);
        std::vector<std::thread> threads;

        threads.emplace_back(
            run_rank,
            make_options(0, peers),
            [](tbccl::World &world)
            {
                bool threw = false;

                try
                {
                    tbccl::barrier(world);
                }
                catch (const std::exception &error)
                {
                    threw = true;

                    const std::string message = error.what();

                    expect(
                        message.find("invalid arrival token") !=
                            std::string::npos,
                        "wrong error for invalid arrival token: " + message);

                    expect(
                        message.find("rank 1") != std::string::npos,
                        "error should name the offending rank: " + message);
                }

                expect(threw, "barrier() should reject a garbage token");
            },
            std::ref(errors[0]));

        threads.emplace_back(
            run_rank,
            make_options(1, peers),
            [](tbccl::World &world)
            {
                const std::uint8_t garbage = 0xFF;
                world.send(0, &garbage, sizeof(garbage));
            },
            std::ref(errors[1]));

        join_and_check(threads, errors);

        std::cout << "[PASS] test_invalid_token_rejected\n";
    }

    // -----------------------------------------------------------------------------
    // Test 7: more local ranks, exercised with repeated barriers.
    // Correctness at larger N, not performance.
    // -----------------------------------------------------------------------------

    void test_n_rank_repeated_barriers(
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
                [iterations](tbccl::World &world)
                {
                    for (int i = 0; i < iterations; ++i)
                    {
                        tbccl::barrier(world);
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_n_rank_repeated_barriers (n=" << n
            << ", iterations=" << iterations << ")\n";
    }

} // namespace

int main()
{
    try
    {
        test_single_rank_barrier();
        test_two_rank_barrier();
        test_three_rank_barrier();
        test_barrier_actually_synchronizes();
        test_delayed_arrival_atomics();
        test_repeated_barriers();
        test_communication_phases();
        test_invalid_token_rejected();
        test_n_rank_repeated_barriers(4, kFourRankBase, 100);
        test_n_rank_repeated_barriers(8, kEightRankBase, 100);
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
