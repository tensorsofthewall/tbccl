#include <tbccl/tcp_world.hpp>

#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace
{

    // Fixed high port ranges, one block per test, well away from the
    // production default (18515) and from tcp_transport_test's range.
    constexpr std::uint16_t kTwoRankBase = 29500;
    constexpr std::uint16_t kThreeRankBase = 29510;
    constexpr std::uint16_t kDelayedStartBase = 29520;
    constexpr std::uint16_t kLargePayloadBase = 29530;
    constexpr std::uint16_t kInvalidRankBase = 29540;
    constexpr std::uint16_t kSizeMismatchBase = 29550;
    constexpr std::uint16_t kMissingRankBase = 29560;

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

    // Runs `body(world)` after bootstrapping, capturing any exception
    // (bootstrap or body) rather than letting it escape the thread and
    // terminate the whole test process.
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

    // -----------------------------------------------------------------------------
    // Test: 2-rank bootstrap + bidirectional exchange
    // -----------------------------------------------------------------------------

    void test_two_rank_bootstrap()
    {
        const auto peers = make_local_peers(kTwoRankBase, 2);

        const auto buffer_a = deterministic_buffer(256, 0xA0A0A0A0u);
        const auto buffer_b = deterministic_buffer(300, 0xB0B0B0B0u);

        std::exception_ptr error0;
        std::exception_ptr error1;

        tbccl::TcpWorldOptions options0;
        options0.rank = 0;
        options0.peers = peers;
        options0.bootstrap_timeout = std::chrono::milliseconds(5000);

        tbccl::TcpWorldOptions options1;
        options1.rank = 1;
        options1.peers = peers;
        options1.bootstrap_timeout = std::chrono::milliseconds(5000);

        std::thread thread0(
            run_rank,
            options0,
            [&](tbccl::World &world)
            {
                expect(world.rank() == 0, "rank 0 reported wrong rank()");
                expect(world.size() == 2, "rank 0 reported wrong size()");

                world.send(1, buffer_a.data(), buffer_a.size());

                std::vector<std::uint8_t> received(buffer_b.size());
                world.recv(1, received.data(), received.size());

                expect(received == buffer_b, "rank 0 received wrong buffer_b");
            },
            std::ref(error0));

        std::thread thread1(
            run_rank,
            options1,
            [&](tbccl::World &world)
            {
                expect(world.rank() == 1, "rank 1 reported wrong rank()");
                expect(world.size() == 2, "rank 1 reported wrong size()");

                std::vector<std::uint8_t> received(buffer_a.size());
                world.recv(0, received.data(), received.size());

                expect(received == buffer_a, "rank 1 received wrong buffer_a");

                world.send(0, buffer_b.data(), buffer_b.size());
            },
            std::ref(error1));

        thread0.join();
        thread1.join();

        if (error0)
        {
            throw std::runtime_error("rank 0 failed: " + what_or_empty(error0));
        }

        if (error1)
        {
            throw std::runtime_error("rank 1 failed: " + what_or_empty(error1));
        }

        std::cout << "[PASS] test_two_rank_bootstrap\n";
    }

    // -----------------------------------------------------------------------------
    // Test: 3-rank bootstrap + pairwise communication in a fixed global
    // pair order, so no rank can deadlock waiting on another.
    // -----------------------------------------------------------------------------

    void test_three_rank_bootstrap()
    {
        const auto peers = make_local_peers(kThreeRankBase, 3);

        const std::vector<std::pair<std::size_t, std::size_t>> pair_order = {
            {0, 1}, {0, 2}, {1, 2}};

        std::vector<std::exception_ptr> errors(3);

        auto rank_body = [&](std::size_t self) -> std::function<void(tbccl::World &)>
        {
            return [&pair_order, self](tbccl::World &world)
            {
                expect(world.rank() == self, "wrong rank() in 3-rank test");
                expect(world.size() == 3, "wrong size() in 3-rank test");

                for (std::size_t pair_index = 0;
                     pair_index < pair_order.size();
                     ++pair_index)
                {
                    const auto [lo, hi] = pair_order[pair_index];

                    if (self != lo && self != hi)
                    {
                        continue;
                    }

                    const std::size_t peer = (self == lo) ? hi : lo;

                    const auto request =
                        deterministic_buffer(
                            64, static_cast<std::uint32_t>(0x1000 + pair_index));

                    const auto reply =
                        deterministic_buffer(
                            48, static_cast<std::uint32_t>(0x2000 + pair_index));

                    if (self == lo)
                    {
                        world.send(peer, request.data(), request.size());

                        std::vector<std::uint8_t> received(reply.size());
                        world.recv(peer, received.data(), received.size());

                        expect(
                            received == reply,
                            "pair " + std::to_string(pair_index) +
                                ": lower rank got wrong reply");
                    }
                    else
                    {
                        std::vector<std::uint8_t> received(request.size());
                        world.recv(peer, received.data(), received.size());

                        expect(
                            received == request,
                            "pair " + std::to_string(pair_index) +
                                ": higher rank got wrong request");

                        world.send(peer, reply.data(), reply.size());
                    }
                }
            };
        };

        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < 3; ++rank)
        {
            tbccl::TcpWorldOptions options;
            options.rank = rank;
            options.peers = peers;
            options.bootstrap_timeout = std::chrono::milliseconds(5000);

            threads.emplace_back(
                run_rank, options, rank_body(rank), std::ref(errors[rank]));
        }

        for (auto &thread : threads)
        {
            thread.join();
        }

        for (std::size_t rank = 0; rank < 3; ++rank)
        {
            if (errors[rank])
            {
                throw std::runtime_error(
                    "rank " + std::to_string(rank) +
                    " failed: " + what_or_empty(errors[rank]));
            }
        }

        std::cout << "[PASS] test_three_rank_bootstrap\n";
    }

    // -----------------------------------------------------------------------------
    // Test: ranks start out of order; retry/backoff must recover.
    // -----------------------------------------------------------------------------

    void test_delayed_start_retry()
    {
        const auto peers = make_local_peers(kDelayedStartBase, 3);

        std::vector<std::exception_ptr> errors(3);
        std::vector<int> observed_rank(3, -1);

        const std::vector<int> start_delay_ms = {300, 100, 0}; // rank 0, 1, 2

        auto rank_body = [&](std::size_t self) -> std::function<void(tbccl::World &)>
        {
            return [&observed_rank, self](tbccl::World &world)
            {
                observed_rank[self] = static_cast<int>(world.rank());
                expect(world.size() == 3, "wrong size() in delayed-start test");
            };
        };

        std::vector<std::thread> threads;

        const auto test_start = std::chrono::steady_clock::now();

        for (std::size_t rank = 0; rank < 3; ++rank)
        {
            tbccl::TcpWorldOptions options;
            options.rank = rank;
            options.peers = peers;
            options.bootstrap_timeout = std::chrono::milliseconds(4000);
            options.retry_delay = std::chrono::milliseconds(50);

            const int delay_ms = start_delay_ms[rank];

            threads.emplace_back(
                [&, options, delay_ms, rank]()
                {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(delay_ms));

                    run_rank(options, rank_body(rank), errors[rank]);
                });
        }

        for (auto &thread : threads)
        {
            thread.join();
        }

        const auto elapsed = std::chrono::steady_clock::now() - test_start;

        for (std::size_t rank = 0; rank < 3; ++rank)
        {
            if (errors[rank])
            {
                throw std::runtime_error(
                    "rank " + std::to_string(rank) +
                    " failed: " + what_or_empty(errors[rank]));
            }

            expect(
                observed_rank[rank] == static_cast<int>(rank),
                "rank mismatch after delayed start");
        }

        expect(
            elapsed < std::chrono::milliseconds(4000),
            "delayed-start bootstrap took implausibly long");

        std::cout << "[PASS] test_delayed_start_retry\n";
    }

    // -----------------------------------------------------------------------------
    // Test: >= 1 MiB payload round-trips byte-for-byte through the World.
    // -----------------------------------------------------------------------------

    void test_large_payload()
    {
        const auto peers = make_local_peers(kLargePayloadBase, 2);

        constexpr std::size_t kSize = 1ULL * 1024 * 1024;

        const auto payload = deterministic_buffer(kSize, 0xC0FFEEu);

        std::exception_ptr error0;
        std::exception_ptr error1;

        tbccl::TcpWorldOptions options0;
        options0.rank = 0;
        options0.peers = peers;
        options0.bootstrap_timeout = std::chrono::milliseconds(5000);

        tbccl::TcpWorldOptions options1;
        options1.rank = 1;
        options1.peers = peers;
        options1.bootstrap_timeout = std::chrono::milliseconds(5000);

        std::thread thread0(
            run_rank,
            options0,
            [&](tbccl::World &world)
            {
                world.send(1, payload.data(), payload.size());
            },
            std::ref(error0));

        std::thread thread1(
            run_rank,
            options1,
            [&](tbccl::World &world)
            {
                std::vector<std::uint8_t> received(payload.size());
                world.recv(0, received.data(), received.size());

                expect(received == payload, "large World payload mismatch");
            },
            std::ref(error1));

        thread0.join();
        thread1.join();

        if (error0)
        {
            throw std::runtime_error("rank 0 failed: " + what_or_empty(error0));
        }

        if (error1)
        {
            throw std::runtime_error("rank 1 failed: " + what_or_empty(error1));
        }

        std::cout << "[PASS] test_large_payload\n";
    }

    // -----------------------------------------------------------------------------
    // Test: invalid peer rank rejected by send()/recv()
    // -----------------------------------------------------------------------------

    void test_invalid_rank()
    {
        const auto peers = make_local_peers(kInvalidRankBase, 2);

        std::exception_ptr error0;
        std::exception_ptr error1;

        tbccl::TcpWorldOptions options0;
        options0.rank = 0;
        options0.peers = peers;
        options0.bootstrap_timeout = std::chrono::milliseconds(5000);

        tbccl::TcpWorldOptions options1;
        options1.rank = 1;
        options1.peers = peers;
        options1.bootstrap_timeout = std::chrono::milliseconds(5000);

        std::thread thread0(
            run_rank,
            options0,
            [](tbccl::World &world)
            {
                std::uint8_t value = 0;

                bool threw_out_of_range = false;

                try
                {
                    world.send(world.size(), &value, sizeof(value));
                }
                catch (const std::exception &error)
                {
                    threw_out_of_range = true;

                    expect(
                        std::string(error.what()).find(
                            std::to_string(world.size())) !=
                            std::string::npos,
                        "out-of-range error should mention offending rank");
                }

                expect(
                    threw_out_of_range,
                    "send() to world.size() should throw");

                bool threw_self = false;

                try
                {
                    world.recv(world.rank(), &value, sizeof(value));
                }
                catch (const std::exception &error)
                {
                    threw_self = true;

                    expect(
                        std::string(error.what()).find(
                            std::to_string(world.rank())) !=
                            std::string::npos,
                        "self-rank error should mention offending rank");
                }

                expect(threw_self, "recv() from local rank should throw");
            },
            std::ref(error0));

        // rank 1 just needs to exist so rank 0's bootstrap completes.
        std::thread thread1(
            run_rank,
            options1,
            [](tbccl::World &) {},
            std::ref(error1));

        thread0.join();
        thread1.join();

        if (error0)
        {
            throw std::runtime_error("rank 0 failed: " + what_or_empty(error0));
        }

        if (error1)
        {
            throw std::runtime_error("rank 1 failed: " + what_or_empty(error1));
        }

        std::cout << "[PASS] test_invalid_rank\n";
    }

    // -----------------------------------------------------------------------------
    // Test: incompatible world-size configs must fail bootstrap cleanly,
    // never silently construct mismatched Worlds, and never hang.
    // -----------------------------------------------------------------------------

    void test_world_size_mismatch()
    {
        const std::uint16_t port0 = kSizeMismatchBase;
        const std::uint16_t port1 = kSizeMismatchBase + 1;
        const std::uint16_t port2 = kSizeMismatchBase + 2;

        // rank 0 believes the world has 2 members.
        tbccl::TcpWorldOptions options0;
        options0.rank = 0;
        options0.peers = {{"127.0.0.1", port0}, {"127.0.0.1", port1}};
        options0.bootstrap_timeout = std::chrono::milliseconds(3000);
        options0.retry_delay = std::chrono::milliseconds(50);

        // rank 1 believes the world has 3 members (peer 2 is never
        // actually launched; the mismatch must be caught before that
        // would even matter).
        tbccl::TcpWorldOptions options1;
        options1.rank = 1;
        options1.peers = {
            {"127.0.0.1", port0}, {"127.0.0.1", port1}, {"127.0.0.1", port2}};
        options1.bootstrap_timeout = std::chrono::milliseconds(3000);
        options1.retry_delay = std::chrono::milliseconds(50);

        std::exception_ptr error0;
        std::exception_ptr error1;

        const auto test_start = std::chrono::steady_clock::now();

        std::thread thread0(
            run_rank,
            options0,
            [](tbccl::World &) {},
            std::ref(error0));

        std::thread thread1(
            run_rank,
            options1,
            [](tbccl::World &) {},
            std::ref(error1));

        thread0.join();
        thread1.join();

        const auto elapsed = std::chrono::steady_clock::now() - test_start;

        expect(
            static_cast<bool>(error0),
            "rank 0 should have rejected the world-size mismatch");

        expect(
            static_cast<bool>(error1),
            "rank 1 should have failed too (peer closed on mismatch)");

        expect(
            what_or_empty(error0).find("world size") != std::string::npos,
            "rank 0's error should mention world size");

        expect(
            elapsed < std::chrono::milliseconds(3000),
            "world-size mismatch test took implausibly long");

        std::cout << "[PASS] test_world_size_mismatch\n";
    }

    // -----------------------------------------------------------------------------
    // Test: a permanently missing rank must time out, not hang.
    // -----------------------------------------------------------------------------

    void test_missing_rank_timeout()
    {
        // 3-rank world; rank 2 is never launched.
        const auto peers = make_local_peers(kMissingRankBase, 3);

        tbccl::TcpWorldOptions options0;
        options0.rank = 0;
        options0.peers = peers;
        options0.bootstrap_timeout = std::chrono::milliseconds(500);
        options0.retry_delay = std::chrono::milliseconds(50);

        tbccl::TcpWorldOptions options1;
        options1.rank = 1;
        options1.peers = peers;
        options1.bootstrap_timeout = std::chrono::milliseconds(500);
        options1.retry_delay = std::chrono::milliseconds(50);

        std::exception_ptr error0;
        std::exception_ptr error1;

        const auto test_start = std::chrono::steady_clock::now();

        std::thread thread0(
            run_rank,
            options0,
            [](tbccl::World &) {},
            std::ref(error0));

        std::thread thread1(
            run_rank,
            options1,
            [](tbccl::World &) {},
            std::ref(error1));

        thread0.join();
        thread1.join();

        const auto elapsed = std::chrono::steady_clock::now() - test_start;

        expect(
            static_cast<bool>(error0),
            "rank 0 should time out waiting for missing rank 2");

        expect(
            static_cast<bool>(error1),
            "rank 1 should time out waiting for missing rank 2");

        expect(
            elapsed < std::chrono::milliseconds(3000),
            "missing-rank bootstrap did not fail within a bounded time "
            "(possible hang)");

        std::cout << "[PASS] test_missing_rank_timeout\n";
    }

} // namespace

int main()
{
    try
    {
        test_two_rank_bootstrap();
        test_three_rank_bootstrap();
        test_delayed_start_retry();
        test_large_payload();
        test_invalid_rank();
        test_world_size_mismatch();
        test_missing_rank_timeout();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
