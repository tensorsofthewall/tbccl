#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "all_gather_internal.hpp"
#include "all_reduce_internal.hpp"
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
    // other test file's range (up through reduce_scatter_ring_test's
    // 31500-31790).
    constexpr std::uint16_t kRankCountsBase = 32100;
    constexpr std::uint16_t kDivisibilityRejectionBase = 32200;
    constexpr std::uint16_t kSingleRankBase = 32220;
    constexpr std::uint16_t kEveryDatatypeBase = 32230;
    constexpr std::uint16_t kEveryOperationBase = 32250;
    constexpr std::uint16_t kNegativeValuesBase = 32270;
    constexpr std::uint16_t kInPlaceBase = 32280;
    constexpr std::uint16_t kUnusualDivisibleBase = 32300;
    constexpr std::uint16_t kLargePayloadBase = 32340;
    constexpr std::uint16_t kZeroCountBase = 32400;
    constexpr std::uint16_t kRepeated3RankBase = 32410;
    constexpr std::uint16_t kRepeated5RankBase = 32420;
    constexpr std::uint16_t kRepeated8RankBase = 32430;
    constexpr std::uint16_t kRingArThenBarrierBase = 32450;
    constexpr std::uint16_t kBarrierThenRingArBase = 32460;
    constexpr std::uint16_t kBroadcastThenRingArBase = 32470;
    constexpr std::uint16_t kRingArThenBroadcastBase = 32480;
    constexpr std::uint16_t kAllGatherThenRingArBase = 32490;
    constexpr std::uint16_t kRingArThenAllGatherBase = 32500;
    constexpr std::uint16_t kReduceScatterThenRingArBase = 32510;
    constexpr std::uint16_t kRingArThenReduceScatterBase = 32520;
    constexpr std::uint16_t kRingAgThenRingArBase = 32530;
    constexpr std::uint16_t kRingArThenRingAgBase = 32540;
    constexpr std::uint16_t kRingRsThenRingArBase = 32550;
    constexpr std::uint16_t kRingArThenRingRsBase = 32560;

    // Deterministic values kept small ([1, 7]) so Product cannot
    // explode numerically even at 8 ranks.
    template <typename T>
    T generate_value(std::size_t rank, std::size_t index, std::uint32_t seed)
    {
        const long value =
            (static_cast<long>(rank) * 131 + static_cast<long>(index) * 7 +
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

    // Reference vs ring comparison policy: exact for integers and for
    // Float Min/Max, tolerance-based for Float Sum/Product (ring and
    // reference reduce in different orders).
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

    // Bootstraps a World and runs both reference and ring all_reduce
    // on every rank, each checked against an independently-computed
    // expected full output (not merely against each other). `count`
    // must be divisible by `size` (ring's requirement).
    template <typename T>
    void run_correctness(
        std::uint16_t base_port,
        std::size_t size,
        std::size_t count,
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
                [rank, count, datatype, op](tbccl::World &world)
                {
                    constexpr std::uint32_t kSeed = 0x2345u;
                    const std::size_t size2 = world.size();

                    std::vector<T> send(count);

                    for (std::size_t i = 0; i < count; ++i)
                    {
                        send[i] = generate_value<T>(rank, i, kSeed);
                    }

                    std::vector<T> expected(count);

                    for (std::size_t i = 0; i < count; ++i)
                    {
                        T acc = generate_value<T>(0, i, kSeed);

                        for (std::size_t p = 1; p < size2; ++p)
                        {
                            combine(acc, generate_value<T>(p, i, kSeed), op);
                        }

                        expected[i] = acc;
                    }

                    std::vector<T> reference_recv(count, T{});

                    tbccl::detail::all_reduce_reference(
                        world, send.data(), reference_recv.data(), count,
                        datatype, op);

                    for (std::size_t i = 0; i < count; ++i)
                    {
                        expect(
                            approximately_equal(
                                reference_recv[i], expected[i]),
                            "reference all_reduce mismatch on rank " +
                                std::to_string(rank) + " element " +
                                std::to_string(i));
                    }

                    std::vector<T> ring_recv(count, T{});

                    tbccl::detail::all_reduce_ring(
                        world, send.data(), ring_recv.data(), count,
                        datatype, op);

                    for (std::size_t i = 0; i < count; ++i)
                    {
                        expect(
                            approximately_equal(ring_recv[i], expected[i]),
                            "ring all_reduce mismatch on rank " +
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
    // count chosen divisible by each world size.
    // -----------------------------------------------------------------------------

    void test_rank_counts()
    {
        for (auto [size, offset] :
             {std::pair<std::size_t, std::uint16_t>{1, 0},
              {2, 8}, {3, 16}, {4, 24}, {5, 32}, {7, 40}, {8, 48}})
        {
            const std::size_t count = size * 4; // divisible by construction

            run_correctness<std::int32_t>(
                static_cast<std::uint16_t>(kRankCountsBase + offset), size,
                count, tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

            std::cout << "[PASS] test_rank_counts (size=" << size << ")\n";
        }
    }

    // -----------------------------------------------------------------------------
    // Test: ring rejects non-divisible counts before communication.
    // Every rank calls the same invalid request so none strands its
    // peers.
    // -----------------------------------------------------------------------------

    void run_divisibility_rejection(
        std::uint16_t base_port, std::size_t size, std::size_t count)
    {
        auto peers = make_local_peers(base_port, size);

        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [count](tbccl::World &world)
                {
                    std::vector<std::int32_t> send(count, 1);
                    std::vector<std::int32_t> recv(count, 0);
                    bool threw = false;

                    try
                    {
                        tbccl::detail::all_reduce_ring(
                            world, send.data(), recv.data(), count,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);
                    }
                    catch (const std::exception &error)
                    {
                        threw = true;

                        const std::string message = error.what();

                        expect(
                            message.find("not divisible") !=
                                std::string::npos,
                            "wrong error for non-divisible count: " +
                                message);
                    }

                    expect(
                        threw,
                        "all_reduce_ring() should reject a non-divisible "
                        "count");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    void test_divisibility_rejection()
    {
        run_divisibility_rejection(kDivisibilityRejectionBase, 3, 10);
        run_divisibility_rejection(kDivisibilityRejectionBase + 4, 4, 257);

        std::cout << "[PASS] test_divisibility_rejection\n";
    }

    // -----------------------------------------------------------------------------
    // Test: 1-rank ring — trivially divisible, result must equal
    // input, no ring traffic.
    // -----------------------------------------------------------------------------

    void test_single_rank()
    {
        auto peers = make_local_peers(kSingleRankBase, 1);
        auto world = tbccl::create_tcp_world(make_options(0, peers));

        {
            const std::vector<std::int32_t> send = {1, -2, 3, 0};
            std::vector<std::int32_t> recv(send.size(), 0);

            tbccl::detail::all_reduce_ring(
                *world, send.data(), recv.data(), send.size(),
                tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

            expect(recv == send, "single-rank ring all_reduce should equal input");
        }

        {
            const std::vector<float> send = {1.5f, -2.25f, 3.0f};
            std::vector<float> recv(send.size(), 0.0f);

            tbccl::detail::all_reduce_ring(
                *world, send.data(), recv.data(), send.size(),
                tbccl::DataType::Float32, tbccl::ReduceOp::Min);

            expect(
                recv == send,
                "single-rank Float32 ring all_reduce should equal input");
        }

        std::cout << "[PASS] test_single_rank\n";
    }

    // -----------------------------------------------------------------------------
    // Test: every datatype.
    // -----------------------------------------------------------------------------

    void test_every_datatype()
    {
        run_correctness<std::int32_t>(
            kEveryDatatypeBase, 3, 6, tbccl::DataType::Int32,
            tbccl::ReduceOp::Sum);
        run_correctness<std::int64_t>(
            kEveryDatatypeBase + 4, 3, 6, tbccl::DataType::Int64,
            tbccl::ReduceOp::Sum);
        run_correctness<float>(
            kEveryDatatypeBase + 8, 3, 6, tbccl::DataType::Float32,
            tbccl::ReduceOp::Sum);
        run_correctness<double>(
            kEveryDatatypeBase + 12, 3, 6, tbccl::DataType::Float64,
            tbccl::ReduceOp::Sum);

        std::cout << "[PASS] test_every_datatype\n";
    }

    // -----------------------------------------------------------------------------
    // Test: every operation.
    // -----------------------------------------------------------------------------

    void test_every_operation()
    {
        std::uint16_t base = kEveryOperationBase;

        for (auto op :
             {tbccl::ReduceOp::Sum, tbccl::ReduceOp::Product,
              tbccl::ReduceOp::Min, tbccl::ReduceOp::Max})
        {
            run_correctness<std::int32_t>(
                base, 3, 6, tbccl::DataType::Int32, op);
            base = static_cast<std::uint16_t>(base + 4);
        }

        std::cout << "[PASS] test_every_operation\n";
    }

    // -----------------------------------------------------------------------------
    // Test: negative values, Int32, Sum/Min/Max. count = 3 so it
    // divides the 3-rank world evenly, as ring all_reduce requires.
    // -----------------------------------------------------------------------------

    void test_negative_values()
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(kNegativeValuesBase, kSize);

        const std::vector<std::vector<std::int32_t>> send = {
            {-5, 2, -2}, {3, -4, 6}, {-1, 9, -3}};

        struct Case
        {
            tbccl::ReduceOp op;
            std::vector<std::int32_t> expected;
        };

        const std::vector<Case> cases = {
            {tbccl::ReduceOp::Sum, {-3, 7, 1}},
            {tbccl::ReduceOp::Min, {-5, -4, -3}},
            {tbccl::ReduceOp::Max, {3, 9, 6}},
        };

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank, &send, &cases](tbccl::World &world)
                {
                    for (const auto &c : cases)
                    {
                        std::vector<std::int32_t> recv(3, 0);

                        tbccl::detail::all_reduce_ring(
                            world, send[rank].data(), recv.data(), 3,
                            tbccl::DataType::Int32, c.op);

                        expect(
                            recv == c.expected,
                            "negative-value ring all_reduce mismatch on "
                            "rank " +
                                std::to_string(rank));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_negative_values\n";
    }

    // -----------------------------------------------------------------------------
    // Test: exact send_buffer == recv_buffer aliasing, 2 and 3 ranks.
    // Uses a `size`-element buffer (trivially divisible by `size`) —
    // element i is (rank + i*10 + 1), so every element's expected sum
    // is independently computable.
    // -----------------------------------------------------------------------------

    void run_in_place(std::uint16_t base_port, std::size_t size)
    {
        auto peers = make_local_peers(base_port, size);

        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank, size](tbccl::World &world)
                {
                    std::vector<std::int32_t> buffer(size);

                    for (std::size_t i = 0; i < size; ++i)
                    {
                        buffer[i] = static_cast<std::int32_t>(rank) +
                                    static_cast<std::int32_t>(i) * 10 + 1;
                    }

                    tbccl::detail::all_reduce_ring(
                        world, buffer.data(), buffer.data(), buffer.size(),
                        tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                    for (std::size_t i = 0; i < size; ++i)
                    {
                        std::int32_t expected = 0;

                        for (std::size_t r = 0; r < size; ++r)
                        {
                            expected += static_cast<std::int32_t>(r) +
                                         static_cast<std::int32_t>(i) * 10 +
                                         1;
                        }

                        expect(
                            buffer[i] == expected,
                            "in-place ring all_reduce produced wrong result "
                            "on rank " +
                                std::to_string(rank) + " element " +
                                std::to_string(i));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    void test_exact_in_place_aliasing()
    {
        run_in_place(kInPlaceBase, 2);
        run_in_place(kInPlaceBase + 4, 3);

        std::cout << "[PASS] test_exact_in_place_aliasing\n";
    }

    // -----------------------------------------------------------------------------
    // Test: unusual but divisible counts — count/N == 257 in every
    // case, a non-power-of-two segment size.
    // -----------------------------------------------------------------------------

    void test_unusual_divisible_counts()
    {
        run_correctness<std::int64_t>(
            kUnusualDivisibleBase, 3, 771, tbccl::DataType::Int64,
            tbccl::ReduceOp::Sum);
        run_correctness<std::int64_t>(
            kUnusualDivisibleBase + 8, 5, 1285, tbccl::DataType::Int64,
            tbccl::ReduceOp::Sum);
        run_correctness<std::int64_t>(
            kUnusualDivisibleBase + 16, 7, 1799, tbccl::DataType::Int64,
            tbccl::ReduceOp::Sum);

        std::cout << "[PASS] test_unusual_divisible_counts\n";
    }

    // -----------------------------------------------------------------------------
    // Test: large payloads — must not deadlock. ~1 MiB total tensor per
    // rank on 4 ranks (segment ~256 KiB) and on 2 ranks (segment ~512
    // KiB, exercises simultaneous full-duplex send/recv in both ring
    // phases), plus a ~4 MiB tensor on 4 ranks.
    // -----------------------------------------------------------------------------

    void test_large_payload()
    {
        constexpr std::size_t kOneMiBCount = 262144; // Float32 elements

        run_correctness<float>(
            kLargePayloadBase, 4, kOneMiBCount, tbccl::DataType::Float32,
            tbccl::ReduceOp::Sum, /*timeout_ms=*/10000);
        std::cout << "[PASS] test_large_payload (size=4, ~1 MiB)\n";

        run_correctness<float>(
            kLargePayloadBase + 8, 2, kOneMiBCount, tbccl::DataType::Float32,
            tbccl::ReduceOp::Sum, /*timeout_ms=*/10000);
        std::cout << "[PASS] test_large_payload (size=2, ~1 MiB)\n";

        run_correctness<float>(
            kLargePayloadBase + 20, 4, kOneMiBCount * 4,
            tbccl::DataType::Float32, tbccl::ReduceOp::Sum,
            /*timeout_ms=*/15000);
        std::cout << "[PASS] test_large_payload (size=4, ~4 MiB)\n";
    }

    // -----------------------------------------------------------------------------
    // Test: count == 0 must succeed with no throw/deadlock, buffers
    // may be nullptr, and the divisibility check must not apply.
    // -----------------------------------------------------------------------------

    void test_zero_count()
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(kZeroCountBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [](tbccl::World &world)
                {
                    tbccl::detail::all_reduce_ring(
                        world, nullptr, nullptr, 0, tbccl::DataType::Float32,
                        tbccl::ReduceOp::Sum);
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_zero_count\n";
    }

    // -----------------------------------------------------------------------------
    // Test: 1000 repeated ring all_reduce calls, changing content every
    // iteration, verified every iteration. Also 5x100 and 8x100.
    // -----------------------------------------------------------------------------

    void run_repeated(std::uint16_t base_port, std::size_t size, int iterations)
    {
        const std::size_t count = size * 2;

        auto peers = make_local_peers(base_port, size);

        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers, /*timeout_ms=*/8000),
                [rank, size, count, iterations](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < iterations;
                         ++iteration)
                    {
                        const auto seed = static_cast<std::uint32_t>(iteration);

                        std::vector<std::int32_t> send(count);

                        for (std::size_t i = 0; i < count; ++i)
                        {
                            send[i] =
                                generate_value<std::int32_t>(rank, i, seed);
                        }

                        std::vector<std::int32_t> recv(count, 0);

                        tbccl::detail::all_reduce_ring(
                            world, send.data(), recv.data(), count,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                        for (std::size_t i = 0; i < count; ++i)
                        {
                            std::int32_t acc =
                                generate_value<std::int32_t>(0, i, seed);

                            for (std::size_t p = 1; p < size; ++p)
                            {
                                combine(
                                    acc,
                                    generate_value<std::int32_t>(p, i, seed),
                                    tbccl::ReduceOp::Sum);
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
    // Cross-collective sequencing: ring all_reduce composed with
    // barrier/broadcast/all_gather(reference)/reduce_scatter(reference)
    // /ring all_gather/ring reduce_scatter — both orders, separate
    // buffers. Uses a fixed 3-rank world size and a small ring
    // all_reduce kernel (count=3, one element per rank) whose expected
    // result depends only on rank count, not iteration.
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
                            static_cast<std::int32_t>(rank) + 1,
                            static_cast<std::int32_t>(rank) + 1};
                        std::vector<std::int32_t> recv(3, 0);

                        tbccl::detail::all_reduce_ring(
                            world, send.data(), recv.data(), 3,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                        constexpr std::int32_t kExpected = 1 + 2 + 3;

                        if (recv[0] != kExpected || recv[1] != kExpected ||
                            recv[2] != kExpected)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": wrong ring all_reduce result on rank " +
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

    void test_ring_ar_then_barrier()
    {
        run_interleaved(
            kRingArThenBarrierBase, 100, noop,
            [](tbccl::World &world, std::size_t, int) { tbccl::barrier(world); },
            "test_ring_ar_then_barrier");
    }

    void test_barrier_then_ring_ar()
    {
        run_interleaved(
            kBarrierThenRingArBase, 100,
            [](tbccl::World &world, std::size_t, int) { tbccl::barrier(world); },
            noop, "test_barrier_then_ring_ar");
    }

    void test_broadcast_then_ring_ar()
    {
        run_interleaved(
            kBroadcastThenRingArBase, 100,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const std::size_t root =
                    static_cast<std::size_t>(iteration) % 3;
                const auto pattern = deterministic_buffer(
                    16, static_cast<std::uint32_t>(iteration));
                auto buffer = (rank == root)
                                  ? pattern
                                  : deterministic_buffer(16, 0xEEEEu);

                tbccl::broadcast(world, buffer.data(), buffer.size(), root);

                if (buffer != pattern)
                {
                    throw std::runtime_error("broadcast mismatch");
                }
            },
            noop, "test_broadcast_then_ring_ar");
    }

    void test_ring_ar_then_broadcast()
    {
        run_interleaved(
            kRingArThenBroadcastBase, 100, noop,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const std::size_t root =
                    static_cast<std::size_t>(iteration) % 3;
                const auto pattern = deterministic_buffer(
                    16, static_cast<std::uint32_t>(iteration));
                auto buffer = (rank == root)
                                  ? pattern
                                  : deterministic_buffer(16, 0xEEEEu);

                tbccl::broadcast(world, buffer.data(), buffer.size(), root);

                if (buffer != pattern)
                {
                    throw std::runtime_error("broadcast mismatch");
                }
            },
            "test_ring_ar_then_broadcast");
    }

    void test_all_gather_then_ring_ar()
    {
        run_interleaved(
            kAllGatherThenRingArBase, 100,
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
            noop, "test_all_gather_then_ring_ar");
    }

    void test_ring_ar_then_all_gather()
    {
        run_interleaved(
            kRingArThenAllGatherBase, 100, noop,
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
            "test_ring_ar_then_all_gather");
    }

    void test_reduce_scatter_then_ring_ar()
    {
        run_interleaved(
            kReduceScatterThenRingArBase, 100,
            [](tbccl::World &world, std::size_t, int)
            {
                std::vector<std::int32_t> send = {1, 2, 3};
                std::int32_t recv = 0;

                tbccl::reduce_scatter(
                    world, send.data(), &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);
            },
            noop, "test_reduce_scatter_then_ring_ar");
    }

    void test_ring_ar_then_reduce_scatter()
    {
        run_interleaved(
            kRingArThenReduceScatterBase, 100, noop,
            [](tbccl::World &world, std::size_t, int)
            {
                std::vector<std::int32_t> send = {1, 2, 3};
                std::int32_t recv = 0;

                tbccl::reduce_scatter(
                    world, send.data(), &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);
            },
            "test_ring_ar_then_reduce_scatter");
    }

    void test_ring_allgather_then_ring_ar()
    {
        run_interleaved(
            kRingAgThenRingArBase, 100,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const auto send = deterministic_buffer(
                    16, static_cast<std::uint32_t>(rank) + 0x9000u);
                std::vector<std::uint8_t> recv(3 * 16, 0);

                tbccl::detail::all_gather_ring(
                    world, send.data(), recv.data(), 16);

                for (std::size_t r = 0; r < 3; ++r)
                {
                    const auto expected = deterministic_buffer(
                        16, static_cast<std::uint32_t>(r) + 0x9000u);

                    if (std::memcmp(
                            recv.data() + r * 16, expected.data(), 16) != 0)
                    {
                        throw std::runtime_error(
                            "iteration " + std::to_string(iteration) +
                            ": ring all_gather mismatch");
                    }
                }
            },
            noop, "test_ring_allgather_then_ring_ar");
    }

    void test_ring_ar_then_ring_allgather()
    {
        run_interleaved(
            kRingArThenRingAgBase, 100, noop,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const auto send = deterministic_buffer(
                    16, static_cast<std::uint32_t>(rank) + 0xA000u);
                std::vector<std::uint8_t> recv(3 * 16, 0);

                tbccl::detail::all_gather_ring(
                    world, send.data(), recv.data(), 16);

                for (std::size_t r = 0; r < 3; ++r)
                {
                    const auto expected = deterministic_buffer(
                        16, static_cast<std::uint32_t>(r) + 0xA000u);

                    if (std::memcmp(
                            recv.data() + r * 16, expected.data(), 16) != 0)
                    {
                        throw std::runtime_error(
                            "iteration " + std::to_string(iteration) +
                            ": ring all_gather mismatch");
                    }
                }
            },
            "test_ring_ar_then_ring_allgather");
    }

    void test_ring_reduce_scatter_then_ring_ar()
    {
        run_interleaved(
            kRingRsThenRingArBase, 100,
            [](tbccl::World &world, std::size_t, int)
            {
                std::vector<std::int32_t> send = {1, 2, 3};
                std::int32_t recv = 0;

                tbccl::detail::reduce_scatter_ring(
                    world, send.data(), &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);
            },
            noop, "test_ring_reduce_scatter_then_ring_ar");
    }

    void test_ring_ar_then_ring_reduce_scatter()
    {
        run_interleaved(
            kRingArThenRingRsBase, 100, noop,
            [](tbccl::World &world, std::size_t, int)
            {
                std::vector<std::int32_t> send = {1, 2, 3};
                std::int32_t recv = 0;

                tbccl::detail::reduce_scatter_ring(
                    world, send.data(), &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);
            },
            "test_ring_ar_then_ring_reduce_scatter");
    }

} // namespace

int main()
{
    try
    {
        test_rank_counts();
        test_divisibility_rejection();
        test_single_rank();
        test_every_datatype();
        test_every_operation();
        test_negative_values();
        test_exact_in_place_aliasing();
        test_unusual_divisible_counts();
        test_large_payload();
        test_zero_count();
        run_repeated(kRepeated3RankBase, 3, 1000);
        run_repeated(kRepeated5RankBase, 5, 100);
        run_repeated(kRepeated8RankBase, 8, 100);
        test_ring_ar_then_barrier();
        test_barrier_then_ring_ar();
        test_broadcast_then_ring_ar();
        test_ring_ar_then_broadcast();
        test_all_gather_then_ring_ar();
        test_ring_ar_then_all_gather();
        test_reduce_scatter_then_ring_ar();
        test_ring_ar_then_reduce_scatter();
        test_ring_allgather_then_ring_ar();
        test_ring_ar_then_ring_allgather();
        test_ring_reduce_scatter_then_ring_ar();
        test_ring_ar_then_ring_reduce_scatter();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
