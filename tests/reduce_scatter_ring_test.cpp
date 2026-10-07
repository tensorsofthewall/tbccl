#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "all_gather_internal.hpp"
#include "reduce_scatter_internal.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using tbccl_test::deterministic_buffer;
using tbccl_test::expect;
using tbccl_test::join_and_check;
using tbccl_test::make_local_peers;
using tbccl_test::make_options;
using tbccl_test::run_rank;

namespace
{

    // Fixed high port ranges, one block per test, well clear of every
    // other test file's range (up through all_gather_ring_test's
    // 31000-31292 and various one-off local port usages during
    // benchmark smoke testing).
    constexpr std::uint16_t kRankCountsBase = 31500;
    constexpr std::uint16_t kSegmentOwnershipBase = 31560;
    constexpr std::uint16_t kOddCountsBase = 31570;
    constexpr std::uint16_t kEveryDatatypeBase = 31600;
    constexpr std::uint16_t kEveryOperationBase = 31610;
    constexpr std::uint16_t kNegativeValuesBase = 31620;
    constexpr std::uint16_t kLargePayload3RankBase = 31630;
    constexpr std::uint16_t kLargePayload2RankBase = 31640;
    constexpr std::uint16_t kRepeated3RankBase = 31650;
    constexpr std::uint16_t kRepeated5RankBase = 31660;
    constexpr std::uint16_t kRepeated8RankBase = 31670;
    constexpr std::uint16_t kRingRsThenBarrierBase = 31690;
    constexpr std::uint16_t kBarrierThenRingRsBase = 31700;
    constexpr std::uint16_t kBroadcastThenRingRsBase = 31710;
    constexpr std::uint16_t kRingRsThenBroadcastBase = 31720;
    constexpr std::uint16_t kAllGatherThenRingRsBase = 31730;
    constexpr std::uint16_t kRingRsThenAllGatherBase = 31740;
    constexpr std::uint16_t kAllReduceThenRingRsBase = 31750;
    constexpr std::uint16_t kRingRsThenAllReduceBase = 31760;
    constexpr std::uint16_t kRingAgThenRingRsBase = 31770;
    constexpr std::uint16_t kRingRsThenRingAgBase = 31780;

    // Deterministic values kept small ([1, 7]) so Product cannot
    // explode numerically even at 8 ranks.
    template <typename T>
    T generate_value(std::size_t rank, std::size_t global_index, std::uint32_t seed)
    {
        const long value =
            (static_cast<long>(rank) * 131 +
             static_cast<long>(global_index) * 7 +
             static_cast<long>(seed)) %
                7 +
            1;

        return static_cast<T>(value);
    }

    template <typename T>
    void combine(T &acc, T value, tbccl::ReduceOp op)
    {
        switch (op)
        {
        case tbccl::ReduceOp::Sum:
            acc = acc + value;
            break;
        case tbccl::ReduceOp::Product:
            acc = acc * value;
            break;
        case tbccl::ReduceOp::Min:
            acc = acc < value ? acc : value;
            break;
        case tbccl::ReduceOp::Max:
            acc = acc > value ? acc : value;
            break;
        }
    }

    // Reference vs ring comparison policy (see the reduce-scatter test plan):
    // exact for integers and for Float Min/Max, tolerance-based for
    // Float Sum/Product since ring and reference reduce in different
    // orders and floating-point Sum/Product are not associative in
    // general. (In practice, with our bounded small-integer test
    // values every intermediate stays exactly representable, so exact
    // equality would also pass — but the tolerance comparison is the
    // policy-correct approach regardless of that coincidence.)
    template <typename T>
    bool approximately_equal(T a, T b)
    {
        if constexpr (std::is_integral_v<T>)
        {
            return a == b;
        }
        else
        {
            const double da = static_cast<double>(a);
            const double db = static_cast<double>(b);
            const double diff = std::fabs(da - db);
            const double scale = std::max({1.0, std::fabs(da), std::fabs(db)});

            return diff <= 1e-5 * scale;
        }
    }

    // Bootstraps a World and runs both reference and ring
    // reduce_scatter on every rank, each checked against an
    // independently-computed expected segment (not merely against each
    // other), so correctness is not circular.
    template <typename T>
    void run_correctness(
        std::uint16_t base_port,
        std::size_t size,
        std::size_t recv_count,
        tbccl::DataType datatype,
        tbccl::ReduceOp op,
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
                [rank, size, recv_count, datatype, op](tbccl::World &world)
                {
                    const std::size_t total_count = size * recv_count;
                    constexpr std::uint32_t kSeed = 0x1234u;

                    std::vector<T> send(total_count);

                    for (std::size_t i = 0; i < total_count; ++i)
                    {
                        send[i] = generate_value<T>(rank, i, kSeed);
                    }

                    std::vector<T> expected(recv_count);

                    for (std::size_t i = 0; i < recv_count; ++i)
                    {
                        const std::size_t global_index = rank * recv_count + i;
                        T acc = generate_value<T>(0, global_index, kSeed);

                        for (std::size_t p = 1; p < size; ++p)
                        {
                            combine(
                                acc, generate_value<T>(p, global_index, kSeed),
                                op);
                        }

                        expected[i] = acc;
                    }

                    std::vector<T> reference_recv(recv_count, T{});

                    tbccl::detail::reduce_scatter_reference(
                        world, send.data(), reference_recv.data(),
                        recv_count, datatype, op);

                    for (std::size_t i = 0; i < recv_count; ++i)
                    {
                        expect(
                            approximately_equal(reference_recv[i], expected[i]),
                            "reference reduce_scatter mismatch on rank " +
                                std::to_string(rank) + " element " +
                                std::to_string(i));
                    }

                    std::vector<T> ring_recv(recv_count, T{});

                    tbccl::detail::reduce_scatter_ring(
                        world, send.data(), ring_recv.data(), recv_count,
                        datatype, op);

                    for (std::size_t i = 0; i < recv_count; ++i)
                    {
                        expect(
                            approximately_equal(ring_recv[i], expected[i]),
                            "ring reduce_scatter mismatch on rank " +
                                std::to_string(rank) + " element " +
                                std::to_string(i));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    // -----------------------------------------------------------------------------
    // Tests: 1/2/3/4/5/7/8-rank ring correctness against an
    // independently-computed expected result and against reference.
    // -----------------------------------------------------------------------------

    void test_rank_counts()
    {
        for (auto [size, offset] :
             {std::pair<std::size_t, std::uint16_t>{1, 0},
              {2, 8}, {3, 16}, {4, 24}, {5, 32}, {7, 40}, {8, 48}})
        {
            run_correctness<std::int32_t>(
                static_cast<std::uint16_t>(kRankCountsBase + offset), size,
                8, tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

            std::cout << "[PASS] test_rank_counts (size=" << size << ")\n";
        }
    }

    // -----------------------------------------------------------------------------
    // Test: explicit segment-ownership check at N=4, recv_count=1 — a
    // worked example shape. If the shifted ring
    // formula were wrong (e.g. rank ended up with a neighbor's segment
    // instead of its own), comparing against the independently-indexed
    // `expected[i]` in run_correctness would catch it immediately.
    // -----------------------------------------------------------------------------

    void test_segment_ownership()
    {
        run_correctness<std::int32_t>(
            kSegmentOwnershipBase, 4, 1, tbccl::DataType::Int32,
            tbccl::ReduceOp::Sum);

        std::cout << "[PASS] test_segment_ownership\n";
    }

    // -----------------------------------------------------------------------------
    // Test: odd, non-power-of-two recv_count.
    // -----------------------------------------------------------------------------

    void test_odd_counts()
    {
        std::uint16_t base = kOddCountsBase;

        for (std::size_t recv_count : {1, 3, 7, 257, 1003})
        {
            run_correctness<std::int64_t>(
                base, 3, recv_count, tbccl::DataType::Int64,
                tbccl::ReduceOp::Sum);

            base = static_cast<std::uint16_t>(base + 4);

            std::cout
                << "[PASS] test_odd_counts (recv_count=" << recv_count
                << ")\n";
        }
    }

    // -----------------------------------------------------------------------------
    // Test: every datatype.
    // -----------------------------------------------------------------------------

    void test_every_datatype()
    {
        run_correctness<std::int32_t>(
            kEveryDatatypeBase, 3, 4, tbccl::DataType::Int32,
            tbccl::ReduceOp::Sum);
        run_correctness<std::int64_t>(
            kEveryDatatypeBase + 4, 3, 4, tbccl::DataType::Int64,
            tbccl::ReduceOp::Sum);
        run_correctness<float>(
            kEveryDatatypeBase + 8, 3, 4, tbccl::DataType::Float32,
            tbccl::ReduceOp::Sum);
        run_correctness<double>(
            kEveryDatatypeBase + 12, 3, 4, tbccl::DataType::Float64,
            tbccl::ReduceOp::Sum);

        std::cout << "[PASS] test_every_datatype\n";
    }

    // -----------------------------------------------------------------------------
    // Test: every operation. Product values are kept numerically
    // bounded by generate_value()'s [1, 7] range.
    // -----------------------------------------------------------------------------

    void test_every_operation()
    {
        std::uint16_t base = kEveryOperationBase;

        for (auto op :
             {tbccl::ReduceOp::Sum, tbccl::ReduceOp::Product,
              tbccl::ReduceOp::Min, tbccl::ReduceOp::Max})
        {
            run_correctness<std::int32_t>(
                base, 3, 4, tbccl::DataType::Int32, op);
            base = static_cast<std::uint16_t>(base + 4);
        }

        std::cout << "[PASS] test_every_operation\n";
    }

    // -----------------------------------------------------------------------------
    // Test: negative values, Int32, Sum/Min/Max, recv_count=1 so each
    // rank's single output element is its own dedicated column.
    // -----------------------------------------------------------------------------

    void test_negative_values()
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(kNegativeValuesBase, kSize);

        // columns[col][rank]
        const std::array<std::array<std::int32_t, kSize>, kSize> columns = {{
            {-5, 3, -1},
            {2, -4, 9},
            {-2, 6, -3},
        }};

        struct Case
        {
            tbccl::ReduceOp op;
            std::array<std::int32_t, kSize> expected;
        };

        const std::array<Case, 3> cases = {{
            {tbccl::ReduceOp::Sum, {-3, 7, 1}},
            {tbccl::ReduceOp::Min, {-5, -4, -3}},
            {tbccl::ReduceOp::Max, {3, 9, 6}},
        }};

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank, &columns, &cases](tbccl::World &world)
                {
                    std::vector<std::int32_t> send(kSize);

                    for (std::size_t col = 0; col < kSize; ++col)
                    {
                        send[col] = columns[col][rank];
                    }

                    for (const auto &c : cases)
                    {
                        std::vector<std::int32_t> recv(1, 0);

                        tbccl::detail::reduce_scatter_ring(
                            world, send.data(), recv.data(), 1,
                            tbccl::DataType::Int32, c.op);

                        expect(
                            recv[0] == c.expected[rank],
                            "negative-value ring result mismatch on rank " +
                                std::to_string(rank));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_negative_values\n";
    }

    // -----------------------------------------------------------------------------
    // Test: large payloads — must not deadlock. 3 ranks with a ~1 MiB
    // output segment (~3 MiB input per rank), and 2 ranks with a ~1
    // MiB output segment each (exercises simultaneous full-duplex
    // send/recv on the same connection).
    // -----------------------------------------------------------------------------

    void test_large_payload()
    {
        run_correctness<float>(
            kLargePayload3RankBase, 3, 262144, tbccl::DataType::Float32,
            tbccl::ReduceOp::Sum, /*timeout_ms=*/10000);
        std::cout << "[PASS] test_large_payload (size=3)\n";

        run_correctness<float>(
            kLargePayload2RankBase, 2, 262144, tbccl::DataType::Float32,
            tbccl::ReduceOp::Sum, /*timeout_ms=*/10000);
        std::cout << "[PASS] test_large_payload (size=2)\n";
    }

    // -----------------------------------------------------------------------------
    // Test: repeated ring reduce_scatter calls, changing content every
    // iteration, rotating operation, verified every iteration.
    // -----------------------------------------------------------------------------

    void run_repeated(
        std::uint16_t base_port,
        std::size_t size,
        int iterations)
    {
        constexpr std::size_t kRecvCount = 4;
        const std::size_t total_count = size * kRecvCount;

        auto peers = make_local_peers(base_port, size);

        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        const std::array<tbccl::ReduceOp, 4> ops = {
            tbccl::ReduceOp::Sum, tbccl::ReduceOp::Min, tbccl::ReduceOp::Max,
            tbccl::ReduceOp::Product};

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers, /*timeout_ms=*/8000),
                [rank, size, iterations, total_count, &ops](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < iterations;
                         ++iteration)
                    {
                        const tbccl::ReduceOp op =
                            ops[static_cast<std::size_t>(iteration) %
                                ops.size()];
                        const auto seed =
                            static_cast<std::uint32_t>(iteration);

                        std::vector<std::int32_t> send(total_count);

                        for (std::size_t i = 0; i < total_count; ++i)
                        {
                            send[i] =
                                generate_value<std::int32_t>(rank, i, seed);
                        }

                        std::vector<std::int32_t> recv(kRecvCount, 0);

                        tbccl::detail::reduce_scatter_ring(
                            world, send.data(), recv.data(), kRecvCount,
                            tbccl::DataType::Int32, op);

                        for (std::size_t i = 0; i < kRecvCount; ++i)
                        {
                            const std::size_t global_index =
                                rank * kRecvCount + i;
                            std::int32_t acc =
                                generate_value<std::int32_t>(
                                    0, global_index, seed);

                            for (std::size_t p = 1; p < size; ++p)
                            {
                                combine(
                                    acc,
                                    generate_value<std::int32_t>(
                                        p, global_index, seed),
                                    op);
                            }

                            if (recv[i] != acc)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": element " + std::to_string(i) +
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
            << "[PASS] run_repeated (size=" << size
            << ", iterations=" << iterations << ")\n";
    }

    // -----------------------------------------------------------------------------
    // Cross-collective sequencing: ring reduce_scatter composed with
    // barrier/broadcast/all_gather(reference)/all_reduce and, crucially for the
    // optimized ring all-reduce work, ring all_gather — both orders, separate
    // buffers.
    // -----------------------------------------------------------------------------

    template <typename Before, typename After>
    void run_interleaved(
        std::uint16_t base_port,
        int iterations,
        Before before,
        After after,
        const char *name)
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(base_port, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank, iterations, before, after](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < iterations;
                         ++iteration)
                    {
                        before(world, rank, iteration);

                        std::vector<std::int32_t> send = {
                            static_cast<std::int32_t>(rank) + 1,
                            static_cast<std::int32_t>(rank) + 4,
                            static_cast<std::int32_t>(rank) + 7};
                        std::int32_t recv = 0;

                        tbccl::detail::reduce_scatter_ring(
                            world, send.data(), &recv, 1,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                        const std::int32_t base_value =
                            (rank == 0) ? 1 : (rank == 1 ? 4 : 7);
                        std::int32_t expected = 0;

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            expected +=
                                base_value + static_cast<std::int32_t>(r);
                        }

                        if (recv != expected)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": wrong ring reduce_scatter segment on "
                                "rank " +
                                std::to_string(rank));
                        }

                        after(world, rank, iteration);
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] " << name << " (" << iterations << " iterations)\n";
    }

    auto noop = [](tbccl::World &, std::size_t, int) {};

    void test_ring_rs_then_barrier()
    {
        run_interleaved(
            kRingRsThenBarrierBase, 100, noop,
            [](tbccl::World &world, std::size_t, int) { tbccl::barrier(world); },
            "test_ring_rs_then_barrier");
    }

    void test_barrier_then_ring_rs()
    {
        run_interleaved(
            kBarrierThenRingRsBase, 100,
            [](tbccl::World &world, std::size_t, int) { tbccl::barrier(world); },
            noop, "test_barrier_then_ring_rs");
    }

    void test_broadcast_then_ring_rs()
    {
        run_interleaved(
            kBroadcastThenRingRsBase, 100,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const std::size_t root =
                    static_cast<std::size_t>(iteration) % 3;
                const auto pattern = deterministic_buffer(
                    16, static_cast<std::uint32_t>(iteration));
                auto buffer = (rank == root)
                                  ? pattern
                                  : deterministic_buffer(16, 0xEEEEu);

                tbccl::broadcast(
                    world, buffer.data(), buffer.size(), root);

                if (buffer != pattern)
                {
                    throw std::runtime_error("broadcast mismatch");
                }
            },
            noop, "test_broadcast_then_ring_rs");
    }

    void test_ring_rs_then_broadcast()
    {
        run_interleaved(
            kRingRsThenBroadcastBase, 100, noop,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const std::size_t root =
                    static_cast<std::size_t>(iteration) % 3;
                const auto pattern = deterministic_buffer(
                    16, static_cast<std::uint32_t>(iteration));
                auto buffer = (rank == root)
                                  ? pattern
                                  : deterministic_buffer(16, 0xEEEEu);

                tbccl::broadcast(
                    world, buffer.data(), buffer.size(), root);

                if (buffer != pattern)
                {
                    throw std::runtime_error("broadcast mismatch");
                }
            },
            "test_ring_rs_then_broadcast");
    }

    void test_all_gather_then_ring_rs()
    {
        run_interleaved(
            kAllGatherThenRingRsBase, 100,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const std::uint8_t send = static_cast<std::uint8_t>(
                    (rank + static_cast<std::size_t>(iteration)) & 0xFF);
                std::vector<std::uint8_t> recv(3, 0);

                tbccl::all_gather(world, &send, recv.data(), 1);

                for (std::size_t r = 0; r < 3; ++r)
                {
                    const std::uint8_t expected = static_cast<std::uint8_t>(
                        (r + static_cast<std::size_t>(iteration)) & 0xFF);

                    if (recv[r] != expected)
                    {
                        throw std::runtime_error("all_gather mismatch");
                    }
                }
            },
            noop, "test_all_gather_then_ring_rs");
    }

    void test_ring_rs_then_all_gather()
    {
        run_interleaved(
            kRingRsThenAllGatherBase, 100, noop,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const std::uint8_t send = static_cast<std::uint8_t>(
                    (rank + static_cast<std::size_t>(iteration) + 1) & 0xFF);
                std::vector<std::uint8_t> recv(3, 0);

                tbccl::all_gather(world, &send, recv.data(), 1);

                for (std::size_t r = 0; r < 3; ++r)
                {
                    const std::uint8_t expected = static_cast<std::uint8_t>(
                        (r + static_cast<std::size_t>(iteration) + 1) &
                        0xFF);

                    if (recv[r] != expected)
                    {
                        throw std::runtime_error("all_gather mismatch");
                    }
                }
            },
            "test_ring_rs_then_all_gather");
    }

    void test_all_reduce_then_ring_rs()
    {
        run_interleaved(
            kAllReduceThenRingRsBase, 100,
            [](tbccl::World &world, std::size_t rank, int)
            {
                const std::int32_t send = static_cast<std::int32_t>(rank) + 1;
                std::int32_t recv = 0;

                tbccl::all_reduce(
                    world, &send, &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);

                if (recv != 6)
                {
                    throw std::runtime_error("all_reduce mismatch");
                }
            },
            noop, "test_all_reduce_then_ring_rs");
    }

    void test_ring_rs_then_all_reduce()
    {
        run_interleaved(
            kRingRsThenAllReduceBase, 100, noop,
            [](tbccl::World &world, std::size_t rank, int)
            {
                const std::int32_t send = static_cast<std::int32_t>(rank) + 1;
                std::int32_t recv = 0;

                tbccl::all_reduce(
                    world, &send, &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);

                if (recv != 6)
                {
                    throw std::runtime_error("all_reduce mismatch");
                }
            },
            "test_ring_rs_then_all_reduce");
    }

    // Ring AllGather <-> ring ReduceScatter, both orders — the pair
    // the optimized AllReduce will
    // compose directly.
    void test_ring_allgather_then_ring_rs()
    {
        run_interleaved(
            kRingAgThenRingRsBase, 100,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const auto send = deterministic_buffer(
                    16, static_cast<std::uint32_t>(rank) + 0x7000u);
                std::vector<std::uint8_t> recv(3 * 16, 0);

                tbccl::detail::all_gather_ring(
                    world, send.data(), recv.data(), 16);

                for (std::size_t r = 0; r < 3; ++r)
                {
                    const auto expected = deterministic_buffer(
                        16, static_cast<std::uint32_t>(r) + 0x7000u);

                    if (std::memcmp(
                            recv.data() + r * 16, expected.data(), 16) != 0)
                    {
                        throw std::runtime_error(
                            "iteration " + std::to_string(iteration) +
                            ": ring all_gather mismatch");
                    }
                }
            },
            noop, "test_ring_allgather_then_ring_rs");
    }

    void test_ring_rs_then_ring_allgather()
    {
        run_interleaved(
            kRingRsThenRingAgBase, 100, noop,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const auto send = deterministic_buffer(
                    16, static_cast<std::uint32_t>(rank) + 0x8000u);
                std::vector<std::uint8_t> recv(3 * 16, 0);

                tbccl::detail::all_gather_ring(
                    world, send.data(), recv.data(), 16);

                for (std::size_t r = 0; r < 3; ++r)
                {
                    const auto expected = deterministic_buffer(
                        16, static_cast<std::uint32_t>(r) + 0x8000u);

                    if (std::memcmp(
                            recv.data() + r * 16, expected.data(), 16) != 0)
                    {
                        throw std::runtime_error(
                            "iteration " + std::to_string(iteration) +
                            ": ring all_gather mismatch");
                    }
                }
            },
            "test_ring_rs_then_ring_allgather");
    }

} // namespace

int main()
{
    try
    {
        test_rank_counts();
        test_segment_ownership();
        test_odd_counts();
        test_every_datatype();
        test_every_operation();
        test_negative_values();
        test_large_payload();
        run_repeated(kRepeated3RankBase, 3, 1000);
        run_repeated(kRepeated5RankBase, 5, 100);
        run_repeated(kRepeated8RankBase, 8, 100);
        test_ring_rs_then_barrier();
        test_barrier_then_ring_rs();
        test_broadcast_then_ring_rs();
        test_ring_rs_then_broadcast();
        test_all_gather_then_ring_rs();
        test_ring_rs_then_all_gather();
        test_all_reduce_then_ring_rs();
        test_ring_rs_then_all_reduce();
        test_ring_allgather_then_ring_rs();
        test_ring_rs_then_ring_allgather();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
