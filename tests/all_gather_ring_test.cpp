#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "all_gather_internal.hpp"
#include "test_utils.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using tbccl_test::deterministic_buffer;
using tbccl_test::expect;
using tbccl_test::join_and_check;
using tbccl_test::make_local_peers;
using tbccl_test::make_options;
using tbccl_test::run_rank;

namespace
{

    // Fixed high port ranges, one block per test, well away from
    // production (18515) and every other test file's range (the
    // highest other file, algorithm_selector_test, runs through
    // ~32710). Moved from an original 31000 base after that range was
    // found to collide with an unrelated process on a developer
    // machine (port 31052 held by an unrelated 'code' process) --
    // 33000 is clear of every other test file's range.
    // Sizes swept by test_rank_counts (1,2,3,4,5,7,8) need up to 8
    // ports per offset, so kRankCountsBase's offsets (0,10,...,60) run
    // through 33000-33067; every later block starts well past that.
    constexpr std::uint16_t kRankCountsBase = 33000;
    constexpr std::uint16_t kOddSizesBase = 33100;
    constexpr std::uint16_t kLargeMessagesBase = 33150;
    constexpr std::uint16_t kRepeated3RankBase = 33200;
    constexpr std::uint16_t kRepeated5RankBase = 33210;
    constexpr std::uint16_t kRepeated8RankBase = 33220;
    constexpr std::uint16_t kRingThenBarrierBase = 33240;
    constexpr std::uint16_t kBarrierThenRingBase = 33250;
    constexpr std::uint16_t kBroadcastThenRingBase = 33260;
    constexpr std::uint16_t kRingThenBroadcastBase = 33270;
    constexpr std::uint16_t kAllReduceThenRingBase = 33280;
    constexpr std::uint16_t kRingThenReduceScatterBase = 33290;

    // Runs both all_gather algorithm variants on a fresh World and
    // checks each against the directly-computed expected layout
    // (`[rank0's contribution][rank1's][...]`), not merely against
    // each other — this simultaneously exercises "reference vs ring
    // byte-equivalence" and "chunk ordering is exactly rank-ordered".
    void run_gather_correctness(
        std::uint16_t base_port,
        std::size_t size,
        std::size_t bytes_per_rank,
        int timeout_ms = 5000)
    {
        auto peers = make_local_peers(base_port, size);

        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers, timeout_ms),
                [rank, size, bytes_per_rank](tbccl::World &world)
                {
                    const auto send = deterministic_buffer(
                        bytes_per_rank,
                        static_cast<std::uint32_t>(rank) + 0x1000u);

                    std::vector<std::uint8_t> expected(size * bytes_per_rank);

                    for (std::size_t r = 0; r < size; ++r)
                    {
                        const auto contribution = deterministic_buffer(
                            bytes_per_rank,
                            static_cast<std::uint32_t>(r) + 0x1000u);

                        std::memcpy(
                            expected.data() + r * bytes_per_rank,
                            contribution.data(), bytes_per_rank);
                    }

                    std::vector<std::uint8_t> reference_recv(
                        size * bytes_per_rank, 0);

                    tbccl::detail::all_gather_reference(
                        world, send.data(), reference_recv.data(),
                        bytes_per_rank);

                    expect(
                        reference_recv == expected,
                        "reference all_gather mismatch on rank " +
                            std::to_string(rank));

                    std::vector<std::uint8_t> ring_recv(
                        size * bytes_per_rank, 0);

                    tbccl::detail::all_gather_ring(
                        world, send.data(), ring_recv.data(), bytes_per_rank);

                    expect(
                        ring_recv == expected,
                        "ring all_gather mismatch on rank " +
                            std::to_string(rank));
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    // -----------------------------------------------------------------------------
    // Tests 1-7 (acceptance list): 1/2/3/4/5/7/8-rank ring correctness,
    // each directly against the expected rank-ordered layout and
    // against the reference implementation.
    // -----------------------------------------------------------------------------

    void test_rank_counts()
    {
        for (auto [size, offset] :
             {std::pair<std::size_t, std::uint16_t>{1, 0},
              {2, 10}, {3, 20}, {4, 30}, {5, 40}, {7, 50}, {8, 60}})
        {
            run_gather_correctness(
                static_cast<std::uint16_t>(kRankCountsBase + offset), size,
                64);

            std::cout
                << "[PASS] test_rank_counts (size=" << size << ")\n";
        }
    }

    // -----------------------------------------------------------------------------
    // Test: odd, non-power-of-two message sizes — ring must make no
    // alignment/word-size assumptions.
    // -----------------------------------------------------------------------------

    void test_odd_sizes()
    {
        constexpr std::size_t kSize = 3;
        std::uint16_t base = kOddSizesBase;

        for (std::size_t bytes : {1, 3, 7, 257, 1003})
        {
            run_gather_correctness(base, kSize, bytes);
            base = static_cast<std::uint16_t>(base + kSize + 1);

            std::cout
                << "[PASS] test_odd_sizes (bytes=" << bytes << ")\n";
        }
    }

    // -----------------------------------------------------------------------------
    // Test: large payloads — 1 MiB per rank on 2 and 3 ranks, catching
    // blocking-send deadlocks that small socket-buffer-sized messages
    // would not exercise.
    // -----------------------------------------------------------------------------

    void test_large_messages()
    {
        constexpr std::size_t kBytesPerRank = 1024 * 1024;

        run_gather_correctness(
            kLargeMessagesBase, 2, kBytesPerRank, /*timeout_ms=*/10000);
        std::cout << "[PASS] test_large_messages (size=2, 1 MiB)\n";

        run_gather_correctness(
            kLargeMessagesBase + 3, 3, kBytesPerRank, /*timeout_ms=*/10000);
        std::cout << "[PASS] test_large_messages (size=3, 1 MiB)\n";
    }

    // -----------------------------------------------------------------------------
    // Test: repeated ring all_gather calls, changing content every
    // iteration, verified every iteration. 3 ranks x 1000, plus 5 x
    // 100 and 8 x 100 for broader N coverage.
    // -----------------------------------------------------------------------------

    void run_repeated_ring(
        std::uint16_t base_port,
        std::size_t size,
        int iterations,
        std::size_t bytes_per_rank)
    {
        auto peers = make_local_peers(base_port, size);

        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers, /*timeout_ms=*/8000),
                [rank, size, iterations, bytes_per_rank](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < iterations;
                         ++iteration)
                    {
                        const auto seed =
                            static_cast<std::uint32_t>(
                                rank * 1000 +
                                static_cast<std::size_t>(iteration));

                        const auto send =
                            deterministic_buffer(bytes_per_rank, seed);

                        std::vector<std::uint8_t> recv(
                            size * bytes_per_rank, 0);

                        tbccl::detail::all_gather_ring(
                            world, send.data(), recv.data(), bytes_per_rank);

                        for (std::size_t r = 0; r < size; ++r)
                        {
                            const auto expected_seed =
                                static_cast<std::uint32_t>(
                                    r * 1000 +
                                    static_cast<std::size_t>(iteration));
                            const auto expected = deterministic_buffer(
                                bytes_per_rank, expected_seed);

                            if (std::memcmp(
                                    recv.data() + r * bytes_per_rank,
                                    expected.data(), bytes_per_rank) != 0)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": slot " + std::to_string(r) +
                                    " mismatch on rank " +
                                    std::to_string(rank));
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] run_repeated_ring (size=" << size
            << ", iterations=" << iterations << ")\n";
    }

    // -----------------------------------------------------------------------------
    // Test: ring composed sequentially with the existing collectives,
    // both orders, separate buffers.
    // -----------------------------------------------------------------------------

    void test_ring_then_barrier()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        auto peers = make_local_peers(kRingThenBarrierBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < kIterations;
                         ++iteration)
                    {
                        const auto send = deterministic_buffer(
                            16, static_cast<std::uint32_t>(rank));
                        std::vector<std::uint8_t> recv(kSize * 16, 0);

                        tbccl::detail::all_gather_ring(
                            world, send.data(), recv.data(), 16);

                        tbccl::barrier(world);

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            const auto expected = deterministic_buffer(
                                16, static_cast<std::uint32_t>(r));

                            if (std::memcmp(
                                    recv.data() + r * 16, expected.data(),
                                    16) != 0)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": mismatch");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
        std::cout << "[PASS] test_ring_then_barrier\n";
    }

    void test_barrier_then_ring()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        auto peers = make_local_peers(kBarrierThenRingBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < kIterations;
                         ++iteration)
                    {
                        tbccl::barrier(world);

                        const auto send = deterministic_buffer(
                            16, static_cast<std::uint32_t>(rank));
                        std::vector<std::uint8_t> recv(kSize * 16, 0);

                        tbccl::detail::all_gather_ring(
                            world, send.data(), recv.data(), 16);

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            const auto expected = deterministic_buffer(
                                16, static_cast<std::uint32_t>(r));

                            if (std::memcmp(
                                    recv.data() + r * 16, expected.data(),
                                    16) != 0)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": mismatch");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
        std::cout << "[PASS] test_barrier_then_ring\n";
    }

    void test_broadcast_then_ring()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        auto peers = make_local_peers(kBroadcastThenRingBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < kIterations;
                         ++iteration)
                    {
                        const std::size_t root =
                            static_cast<std::size_t>(iteration) % kSize;
                        const auto pattern = deterministic_buffer(
                            16, static_cast<std::uint32_t>(iteration));
                        auto bcast_buffer =
                            (rank == root)
                                ? pattern
                                : deterministic_buffer(16, 0xEEEEu);

                        tbccl::broadcast(
                            world, bcast_buffer.data(), bcast_buffer.size(),
                            root);

                        if (bcast_buffer != pattern)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": broadcast mismatch");
                        }

                        const auto send = deterministic_buffer(
                            16, static_cast<std::uint32_t>(rank) + 0x2000u);
                        std::vector<std::uint8_t> recv(kSize * 16, 0);

                        tbccl::detail::all_gather_ring(
                            world, send.data(), recv.data(), 16);

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            const auto expected = deterministic_buffer(
                                16, static_cast<std::uint32_t>(r) + 0x2000u);

                            if (std::memcmp(
                                    recv.data() + r * 16, expected.data(),
                                    16) != 0)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": ring mismatch");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
        std::cout << "[PASS] test_broadcast_then_ring\n";
    }

    void test_ring_then_broadcast()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        auto peers = make_local_peers(kRingThenBroadcastBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < kIterations;
                         ++iteration)
                    {
                        const auto send = deterministic_buffer(
                            16, static_cast<std::uint32_t>(rank) + 0x3000u);
                        std::vector<std::uint8_t> recv(kSize * 16, 0);

                        tbccl::detail::all_gather_ring(
                            world, send.data(), recv.data(), 16);

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            const auto expected = deterministic_buffer(
                                16, static_cast<std::uint32_t>(r) + 0x3000u);

                            if (std::memcmp(
                                    recv.data() + r * 16, expected.data(),
                                    16) != 0)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": ring mismatch");
                            }
                        }

                        const std::size_t root =
                            static_cast<std::size_t>(iteration) % kSize;
                        const auto pattern = deterministic_buffer(
                            16, static_cast<std::uint32_t>(iteration));
                        auto bcast_buffer =
                            (rank == root)
                                ? pattern
                                : deterministic_buffer(16, 0xEEEEu);

                        tbccl::broadcast(
                            world, bcast_buffer.data(), bcast_buffer.size(),
                            root);

                        if (bcast_buffer != pattern)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": broadcast mismatch");
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
        std::cout << "[PASS] test_ring_then_broadcast\n";
    }

    void test_all_reduce_then_ring()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        auto peers = make_local_peers(kAllReduceThenRingBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < kIterations;
                         ++iteration)
                    {
                        const std::int32_t send_ar =
                            static_cast<std::int32_t>(rank) + 1;
                        std::int32_t recv_ar = 0;

                        tbccl::all_reduce(
                            world, &send_ar, &recv_ar, 1,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                        if (recv_ar != 6)
                        {
                            throw std::runtime_error("all_reduce mismatch");
                        }

                        const auto send = deterministic_buffer(
                            16, static_cast<std::uint32_t>(rank) + 0x4000u);
                        std::vector<std::uint8_t> recv(kSize * 16, 0);

                        tbccl::detail::all_gather_ring(
                            world, send.data(), recv.data(), 16);

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            const auto expected = deterministic_buffer(
                                16, static_cast<std::uint32_t>(r) + 0x4000u);

                            if (std::memcmp(
                                    recv.data() + r * 16, expected.data(),
                                    16) != 0)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": ring mismatch");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
        std::cout << "[PASS] test_all_reduce_then_ring\n";
    }

    void test_ring_then_reduce_scatter()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        auto peers = make_local_peers(kRingThenReduceScatterBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < kIterations;
                         ++iteration)
                    {
                        const auto send = deterministic_buffer(
                            16, static_cast<std::uint32_t>(rank) + 0x5000u);
                        std::vector<std::uint8_t> recv(kSize * 16, 0);

                        tbccl::detail::all_gather_ring(
                            world, send.data(), recv.data(), 16);

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            const auto expected = deterministic_buffer(
                                16, static_cast<std::uint32_t>(r) + 0x5000u);

                            if (std::memcmp(
                                    recv.data() + r * 16, expected.data(),
                                    16) != 0)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": ring mismatch");
                            }
                        }

                        std::vector<std::int32_t> send_rs = {1, 2, 3};
                        std::int32_t recv_rs = 0;

                        tbccl::reduce_scatter(
                            world, send_rs.data(), &recv_rs, 1,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                        const std::int32_t expected_rs =
                            (rank == 0) ? 3 : (rank == 1 ? 6 : 9);

                        if (recv_rs != expected_rs)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": reduce_scatter mismatch");
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
        std::cout << "[PASS] test_ring_then_reduce_scatter\n";
    }

} // namespace

int main()
{
    try
    {
        test_rank_counts();
        test_odd_sizes();
        test_large_messages();
        run_repeated_ring(kRepeated3RankBase, 3, 1000, 64);
        run_repeated_ring(kRepeated5RankBase, 5, 100, 64);
        run_repeated_ring(kRepeated8RankBase, 8, 100, 64);
        test_ring_then_barrier();
        test_barrier_then_ring();
        test_broadcast_then_ring();
        test_ring_then_broadcast();
        test_all_reduce_then_ring();
        test_ring_then_reduce_scatter();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
