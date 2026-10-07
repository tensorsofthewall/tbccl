#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "test_utils.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
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
    // production (18515) and every other test file's range (up through
    // all_reduce_test's 30570-30577).
    constexpr std::uint16_t kSingleRankBase = 30600;
    constexpr std::uint16_t kBasicTwoRankBase = 30610;
    constexpr std::uint16_t kThreeRankOrderedBase = 30620;
    constexpr std::uint16_t kAllDatatypesBase = 30630;
    constexpr std::uint16_t kNegativeValuesBase = 30650;
    constexpr std::uint16_t kFloatingPointF32Base = 30660;
    constexpr std::uint16_t kFloatingPointF64Base = 30670;
    constexpr std::uint16_t kZeroCountSingleBase = 30680;
    constexpr std::uint16_t kZeroCountThreeBase = 30690;
    constexpr std::uint16_t kZeroCountEightBase = 30700;
    constexpr std::uint16_t kNullSendBase = 30710;
    constexpr std::uint16_t kNullRecvBase = 30720;
    constexpr std::uint16_t kOddCountBase = 30730;
    constexpr std::uint16_t kLargePayloadBase = 30740;
    constexpr std::uint16_t kRepeatedBase = 30760;
    constexpr std::uint16_t kReduceScatterThenBarrierBase = 30770;
    constexpr std::uint16_t kBarrierThenReduceScatterBase = 30780;
    constexpr std::uint16_t kBroadcastThenReduceScatterBase = 30790;
    constexpr std::uint16_t kReduceScatterThenBroadcastBase = 30800;
    constexpr std::uint16_t kAllGatherThenReduceScatterBase = 30810;
    constexpr std::uint16_t kReduceScatterThenAllGatherBase = 30820;
    constexpr std::uint16_t kAllReduceThenReduceScatterBase = 30830;
    constexpr std::uint16_t kReduceScatterThenAllReduceBase = 30840;
    constexpr std::uint16_t kReduceThenReduceScatterBase = 30850;
    constexpr std::uint16_t kReduceScatterThenReduceBase = 30860;
    constexpr std::uint16_t kP2pAroundBase = 30870;
    constexpr std::uint16_t kFourRankBase = 30880;
    constexpr std::uint16_t kEightRankBase = 30890;

    // Bootstraps a World from `peers` and runs one reduce_scatter() on
    // every rank, checking each rank's own segment against
    // `expected_segments[rank]`. `per_rank_send` holds each rank's full
    // N*recv_count contribution.
    template <typename T>
    void run_reduce_scatter_case(
        const std::vector<tbccl::PeerEndpoint> &peers,
        const std::vector<std::vector<T>> &per_rank_send,
        tbccl::DataType datatype,
        tbccl::ReduceOp op,
        std::size_t recv_count,
        const std::vector<std::vector<T>> &expected_segments)
    {
        const std::size_t size = per_rank_send.size();

        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank, &per_rank_send, datatype, op, recv_count,
                 &expected_segments](tbccl::World &world)
                {
                    const auto &send = per_rank_send[rank];
                    std::vector<T> recv(recv_count);

                    tbccl::reduce_scatter(
                        world, send.data(), recv.data(), recv_count,
                        datatype, op);

                    expect(
                        recv == expected_segments[rank],
                        "reduce_scatter: wrong segment on rank " +
                            std::to_string(rank));
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    // -----------------------------------------------------------------------------
    // Test 1: a 1-rank World — output must equal the local input (the
    // full contribution and the received segment are the same size),
    // for Int32 and Float32.
    // -----------------------------------------------------------------------------

    void test_single_rank()
    {
        auto peers = make_local_peers(kSingleRankBase, 1);
        auto world = tbccl::create_tcp_world(make_options(0, peers));

        {
            const std::vector<std::int32_t> send = {1, -2, 3, 0};
            std::vector<std::int32_t> recv(send.size(), 0);

            tbccl::reduce_scatter(
                *world, send.data(), recv.data(), send.size(),
                tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

            expect(
                recv == send,
                "single-rank Int32 reduce_scatter should equal input");
        }

        {
            const std::vector<float> send = {1.5f, -2.25f, 3.0f};
            std::vector<float> recv(send.size(), 0.0f);

            tbccl::reduce_scatter(
                *world, send.data(), recv.data(), send.size(),
                tbccl::DataType::Float32, tbccl::ReduceOp::Min);

            expect(
                recv == send,
                "single-rank Float32 reduce_scatter should equal input");
        }

        std::cout << "[PASS] test_single_rank\n";
    }

    // -----------------------------------------------------------------------------
    // Test 2: basic 2-rank Sum, recv_count = 2.
    // -----------------------------------------------------------------------------

    void test_basic_two_rank_sum()
    {
        auto peers = make_local_peers(kBasicTwoRankBase, 2);

        const std::vector<std::vector<std::int32_t>> send = {
            {1, 2, 3, 4}, {10, 20, 30, 40}};
        const std::vector<std::vector<std::int32_t>> expected = {
            {11, 22}, {33, 44}};

        run_reduce_scatter_case<std::int32_t>(
            peers, send, tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 2,
            expected);

        std::cout << "[PASS] test_basic_two_rank_sum\n";
    }

    // -----------------------------------------------------------------------------
    // Test 3: three-rank rank-ordered segments — a
    // worked example. Explicitly checks rank r receives segment r, not
    // merely "some correct data".
    // -----------------------------------------------------------------------------

    void test_three_rank_ordered_segments()
    {
        auto peers = make_local_peers(kThreeRankOrderedBase, 3);

        const std::vector<std::vector<std::int32_t>> send = {
            {1, 2, 3, 4, 5, 6},
            {10, 20, 30, 40, 50, 60},
            {7, 8, 9, 10, 11, 12}};
        const std::vector<std::vector<std::int32_t>> expected = {
            {18, 30}, {42, 54}, {66, 78}};

        run_reduce_scatter_case<std::int32_t>(
            peers, send, tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 2,
            expected);

        std::cout << "[PASS] test_three_rank_ordered_segments\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4 / 5: all four datatypes, all four operations, recv_count =
    // 1 so each rank's single output element is its own dedicated
    // "column" — this simultaneously exercises rank-to-segment mapping
    // and per-op correctness. Values are small integers exactly
    // representable in every supported type.
    // -----------------------------------------------------------------------------

    void test_all_datatypes_all_operations()
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(kAllDatatypesBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    // column r (segment for rank r) contributed by each
                    // rank; contributions[rank][col].
                    const std::array<std::array<double, kSize>, kSize>
                        contributions = {{
                            {1.0, 5.0, 10.0},
                            {2.0, 4.0, 10.0},
                            {3.0, 3.0, 10.0},
                        }};

                    struct Case
                    {
                        tbccl::ReduceOp op;
                        std::array<double, kSize> expected;
                    };

                    const std::array<Case, 4> cases = {{
                        {tbccl::ReduceOp::Sum, {6.0, 12.0, 30.0}},
                        {tbccl::ReduceOp::Product, {6.0, 60.0, 1000.0}},
                        {tbccl::ReduceOp::Min, {1.0, 3.0, 10.0}},
                        {tbccl::ReduceOp::Max, {3.0, 5.0, 10.0}},
                    }};

                    for (const auto &c : cases)
                    {
                        {
                            std::vector<std::int32_t> send(kSize);
                            for (std::size_t col = 0; col < kSize; ++col)
                            {
                                send[col] = static_cast<std::int32_t>(
                                    contributions[rank][col]);
                            }
                            std::vector<std::int32_t> recv(1, 0);

                            tbccl::reduce_scatter(
                                world, send.data(), recv.data(), 1,
                                tbccl::DataType::Int32, c.op);

                            expect(
                                recv[0] ==
                                    static_cast<std::int32_t>(
                                        c.expected[rank]),
                                "Int32 result mismatch on rank " +
                                    std::to_string(rank));
                        }

                        {
                            std::vector<std::int64_t> send(kSize);
                            for (std::size_t col = 0; col < kSize; ++col)
                            {
                                send[col] = static_cast<std::int64_t>(
                                    contributions[rank][col]);
                            }
                            std::vector<std::int64_t> recv(1, 0);

                            tbccl::reduce_scatter(
                                world, send.data(), recv.data(), 1,
                                tbccl::DataType::Int64, c.op);

                            expect(
                                recv[0] ==
                                    static_cast<std::int64_t>(
                                        c.expected[rank]),
                                "Int64 result mismatch on rank " +
                                    std::to_string(rank));
                        }

                        {
                            std::vector<float> send(kSize);
                            for (std::size_t col = 0; col < kSize; ++col)
                            {
                                send[col] =
                                    static_cast<float>(contributions[rank][col]);
                            }
                            std::vector<float> recv(1, 0.0f);

                            tbccl::reduce_scatter(
                                world, send.data(), recv.data(), 1,
                                tbccl::DataType::Float32, c.op);

                            expect(
                                recv[0] ==
                                    static_cast<float>(c.expected[rank]),
                                "Float32 result mismatch on rank " +
                                    std::to_string(rank));
                        }

                        {
                            std::vector<double> send(kSize);
                            for (std::size_t col = 0; col < kSize; ++col)
                            {
                                send[col] = contributions[rank][col];
                            }
                            std::vector<double> recv(1, 0.0);

                            tbccl::reduce_scatter(
                                world, send.data(), recv.data(), 1,
                                tbccl::DataType::Float64, c.op);

                            expect(
                                recv[0] == c.expected[rank],
                                "Float64 result mismatch on rank " +
                                    std::to_string(rank));
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_all_datatypes_all_operations\n";
    }

    // -----------------------------------------------------------------------------
    // Test 6: negative values, Int32, Sum/Min/Max. recv_count = 1, so
    // each rank's send is [column0, column1, column2] — one value per
    // segment — and each column is reduced independently across ranks.
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
            std::array<std::int32_t, kSize> expected; // by segment/rank
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

                        tbccl::reduce_scatter(
                            world, send.data(), recv.data(), 1,
                            tbccl::DataType::Int32, c.op);

                        expect(
                            recv[0] == c.expected[rank],
                            "negative-value result mismatch on rank " +
                                std::to_string(rank));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_negative_values\n";
    }

    // -----------------------------------------------------------------------------
    // Test 7: floating point, non-integer values, exactly representable
    // in binary floating point so equality (not tolerance) is valid.
    // recv_count = 1, one column per rank/segment.
    // -----------------------------------------------------------------------------

    void test_floating_point()
    {
        {
            auto peers = make_local_peers(kFloatingPointF32Base, 3);

            // column r is rank r's segment; each rank contributes its
            // own value into every column so every segment reduces the
            // same {0.5, 1.25, -2.0} set.
            const std::vector<std::vector<float>> send = {
                {0.5f, 0.5f, 0.5f},
                {1.25f, 1.25f, 1.25f},
                {-2.0f, -2.0f, -2.0f}};

            run_reduce_scatter_case<float>(
                peers, send, tbccl::DataType::Float32, tbccl::ReduceOp::Sum,
                1, {{-0.25f}, {-0.25f}, {-0.25f}});
        }

        {
            auto peers = make_local_peers(kFloatingPointF64Base, 3);

            const std::vector<std::vector<double>> send = {
                {0.5, 0.5, 0.5}, {1.25, 1.25, 1.25}, {-2.0, -2.0, -2.0}};

            run_reduce_scatter_case<double>(
                peers, send, tbccl::DataType::Float64, tbccl::ReduceOp::Product,
                1, {{-1.25}, {-1.25}, {-1.25}});
        }

        std::cout << "[PASS] test_floating_point\n";
    }

    // -----------------------------------------------------------------------------
    // Test 8: recv_count == 0 must succeed with no throw/deadlock,
    // buffers may be nullptr — on 1-, 3-, and 8-rank Worlds.
    // -----------------------------------------------------------------------------

    void test_zero_count()
    {
        {
            auto peers = make_local_peers(kZeroCountSingleBase, 1);
            auto world = tbccl::create_tcp_world(make_options(0, peers));

            tbccl::reduce_scatter(
                *world, nullptr, nullptr, 0, tbccl::DataType::Float32,
                tbccl::ReduceOp::Sum);
        }

        for (auto [size, base] :
             {std::pair<std::size_t, std::uint16_t>{3, kZeroCountThreeBase},
              std::pair<std::size_t, std::uint16_t>{8, kZeroCountEightBase}})
        {
            auto peers = make_local_peers(base, size);

            std::vector<std::exception_ptr> errors(size);
            std::vector<std::thread> threads;

            for (std::size_t rank = 0; rank < size; ++rank)
            {
                threads.emplace_back(
                    run_rank,
                    make_options(rank, peers),
                    [](tbccl::World &world)
                    {
                        tbccl::reduce_scatter(
                            world, nullptr, nullptr, 0, tbccl::DataType::Int64,
                            tbccl::ReduceOp::Max);
                    },
                    std::ref(errors[rank]));
            }

            join_and_check(threads, errors);
        }

        std::cout << "[PASS] test_zero_count\n";
    }

    // -----------------------------------------------------------------------------
    // Test 9: nullptr send_buffer with recv_count > 0 must be rejected
    // on every rank without communication.
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
                    std::int32_t recv = 0;
                    bool threw = false;

                    try
                    {
                        tbccl::reduce_scatter(
                            world, nullptr, &recv, 1, tbccl::DataType::Int32,
                            tbccl::ReduceOp::Sum);
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
                        "reduce_scatter() should reject null send buffer "
                        "for recv_count > 0");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_null_send_buffer\n";
    }

    // -----------------------------------------------------------------------------
    // Test 10: nullptr recv_buffer with recv_count > 0 must be rejected
    // on every rank without communication.
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
                    std::vector<std::int32_t> send(3, 1);
                    bool threw = false;

                    try
                    {
                        tbccl::reduce_scatter(
                            world, send.data(), nullptr, 1,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);
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
                        "reduce_scatter() should reject null receive buffer "
                        "for recv_count > 0");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_null_recv_buffer\n";
    }

    // -----------------------------------------------------------------------------
    // Test 11: an odd recv_count (257), Int64, 3 ranks.
    // -----------------------------------------------------------------------------

    void test_odd_count()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kRecvCount = 257;
        auto peers = make_local_peers(kOddCountBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    const std::size_t total_count = kSize * kRecvCount;
                    std::vector<std::int64_t> send(total_count);

                    for (std::size_t i = 0; i < total_count; ++i)
                    {
                        send[i] =
                            static_cast<std::int64_t>(i) +
                            static_cast<std::int64_t>(rank) * 1000000;
                    }

                    std::vector<std::int64_t> recv(kRecvCount, 0);

                    tbccl::reduce_scatter(
                        world, send.data(), recv.data(), kRecvCount,
                        tbccl::DataType::Int64, tbccl::ReduceOp::Sum);

                    for (std::size_t i = 0; i < kRecvCount; ++i)
                    {
                        const std::size_t global_index =
                            rank * kRecvCount + i;
                        std::int64_t expected = 0;

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            expected +=
                                static_cast<std::int64_t>(global_index) +
                                static_cast<std::int64_t>(r) * 1000000;
                        }

                        if (recv[i] != expected)
                        {
                            throw std::runtime_error(
                                "element " + std::to_string(i) +
                                " mismatch on rank " + std::to_string(rank));
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_odd_count\n";
    }

    // -----------------------------------------------------------------------------
    // Test 12: large payload — recv_count = 262144 Float32 elements
    // (~1 MiB output per rank), 3 ranks, Sum.
    // -----------------------------------------------------------------------------

    void test_large_payload()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kRecvCount = 262144;
        auto peers = make_local_peers(kLargePayloadBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers, /*timeout_ms=*/10000),
                [rank](tbccl::World &world)
                {
                    const std::size_t total_count = kSize * kRecvCount;
                    std::vector<float> send(total_count);

                    for (std::size_t i = 0; i < total_count; ++i)
                    {
                        send[i] = static_cast<float>((i % 13) + rank + 1);
                    }

                    std::vector<float> recv(kRecvCount, 0.0f);

                    tbccl::reduce_scatter(
                        world, send.data(), recv.data(), kRecvCount,
                        tbccl::DataType::Float32, tbccl::ReduceOp::Sum);

                    for (std::size_t i = 0; i < kRecvCount; ++i)
                    {
                        const std::size_t global_index = rank * kRecvCount + i;
                        float expected = 0.0f;

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            expected +=
                                static_cast<float>((global_index % 13) + r + 1);
                        }

                        if (recv[i] != expected)
                        {
                            throw std::runtime_error(
                                "element " + std::to_string(i) +
                                " mismatch on rank " + std::to_string(rank));
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_large_payload\n";
    }

    // -----------------------------------------------------------------------------
    // Test 13: 1000 repeated reduce_scatter calls, small recv_count,
    // rotating operation. Values kept in [1, 4] so Product cannot
    // explode.
    // -----------------------------------------------------------------------------

    void test_repeated_reduce_scatter()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 1000;
        constexpr std::size_t kRecvCount = 2;
        constexpr std::size_t kTotalCount = kSize * kRecvCount;

        auto peers = make_local_peers(kRepeatedBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        const std::array<tbccl::ReduceOp, 4> ops = {
            tbccl::ReduceOp::Sum, tbccl::ReduceOp::Min, tbccl::ReduceOp::Max,
            tbccl::ReduceOp::Product};

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank, &ops](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < kIterations;
                         ++iteration)
                    {
                        const tbccl::ReduceOp op =
                            ops[static_cast<std::size_t>(iteration) %
                                ops.size()];

                        std::vector<std::int32_t> send(kTotalCount);

                        for (std::size_t i = 0; i < kTotalCount; ++i)
                        {
                            send[i] = static_cast<std::int32_t>(
                                ((iteration + static_cast<int>(rank) +
                                  static_cast<int>(i)) %
                                 4) +
                                1);
                        }

                        std::vector<std::int32_t> recv(kRecvCount, 0);

                        tbccl::reduce_scatter(
                            world, send.data(), recv.data(), kRecvCount,
                            tbccl::DataType::Int32, op);

                        for (std::size_t i = 0; i < kRecvCount; ++i)
                        {
                            const std::size_t global_index =
                                rank * kRecvCount + i;

                            std::int32_t acc = 0;
                            bool first = true;

                            for (std::size_t r = 0; r < kSize; ++r)
                            {
                                const std::int32_t v =
                                    static_cast<std::int32_t>(
                                        ((iteration + static_cast<int>(r) +
                                          static_cast<int>(global_index)) %
                                         4) +
                                        1);

                                if (first)
                                {
                                    acc = v;
                                    first = false;
                                    continue;
                                }

                                switch (op)
                                {
                                case tbccl::ReduceOp::Sum:
                                    acc += v;
                                    break;
                                case tbccl::ReduceOp::Product:
                                    acc *= v;
                                    break;
                                case tbccl::ReduceOp::Min:
                                    acc = std::min(acc, v);
                                    break;
                                case tbccl::ReduceOp::Max:
                                    acc = std::max(acc, v);
                                    break;
                                }
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
            << "[PASS] test_repeated_reduce_scatter (" << kIterations
            << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Shared small Int32-Sum kernel for the interleaving tests below:
    // recv_count = 1, world size 3, rank r's segment is always
    // (r+1) + (r+1+3) + (r+1+6) — see the send layout below — so the
    // expected result depends only on the fixed rank, not the
    // iteration, keeping the interleaving tests focused on stream
    // alignment rather than re-deriving reduce_scatter's own math.
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

                        tbccl::reduce_scatter(
                            world, send.data(), &recv, 1,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                        // segment `rank` sums column `rank` across all
                        // 3 ranks: (rank+1)+(rank+2)+(rank+3) for
                        // column 0 style offsets — expand explicitly.
                        std::int32_t expected = 0;

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            const std::int32_t base =
                                (rank == 0) ? 1 : (rank == 1 ? 4 : 7);
                            expected += base + static_cast<std::int32_t>(r);
                        }

                        if (recv != expected)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": wrong reduce_scatter segment on rank " +
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

    // -----------------------------------------------------------------------------
    // Test 14 / 15: reduce_scatter and barrier interleaved, both
    // orders.
    // -----------------------------------------------------------------------------

    void test_reduce_scatter_then_barrier()
    {
        run_interleaved(
            kReduceScatterThenBarrierBase, 100, noop,
            [](tbccl::World &world, std::size_t, int) { tbccl::barrier(world); },
            "test_reduce_scatter_then_barrier");
    }

    void test_barrier_then_reduce_scatter()
    {
        run_interleaved(
            kBarrierThenReduceScatterBase, 100,
            [](tbccl::World &world, std::size_t, int) { tbccl::barrier(world); },
            noop, "test_barrier_then_reduce_scatter");
    }

    // -----------------------------------------------------------------------------
    // Test 16: broadcast and reduce_scatter interleaved, both orders,
    // separate buffers.
    // -----------------------------------------------------------------------------

    void test_broadcast_then_reduce_scatter()
    {
        run_interleaved(
            kBroadcastThenReduceScatterBase, 100,
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
                    throw std::runtime_error(
                        "iteration " + std::to_string(iteration) +
                        ": broadcast mismatch");
                }
            },
            noop, "test_broadcast_then_reduce_scatter");
    }

    void test_reduce_scatter_then_broadcast()
    {
        run_interleaved(
            kReduceScatterThenBroadcastBase, 100, noop,
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
                    throw std::runtime_error(
                        "iteration " + std::to_string(iteration) +
                        ": broadcast mismatch");
                }
            },
            "test_reduce_scatter_then_broadcast");
    }

    // -----------------------------------------------------------------------------
    // Test 17: all_gather and reduce_scatter interleaved, both orders,
    // separate buffers.
    // -----------------------------------------------------------------------------

    void test_all_gather_then_reduce_scatter()
    {
        run_interleaved(
            kAllGatherThenReduceScatterBase, 100,
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
                        throw std::runtime_error(
                            "iteration " + std::to_string(iteration) +
                            ": all_gather mismatch");
                    }
                }
            },
            noop, "test_all_gather_then_reduce_scatter");
    }

    void test_reduce_scatter_then_all_gather()
    {
        run_interleaved(
            kReduceScatterThenAllGatherBase, 100, noop,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const std::uint8_t send = static_cast<std::uint8_t>(
                    (rank + static_cast<std::size_t>(iteration) + 1) & 0xFF);
                std::vector<std::uint8_t> recv(3, 0);

                tbccl::all_gather(world, &send, recv.data(), 1);

                for (std::size_t r = 0; r < 3; ++r)
                {
                    const std::uint8_t expected = static_cast<std::uint8_t>(
                        (r + static_cast<std::size_t>(iteration) + 1) & 0xFF);

                    if (recv[r] != expected)
                    {
                        throw std::runtime_error(
                            "iteration " + std::to_string(iteration) +
                            ": all_gather mismatch");
                    }
                }
            },
            "test_reduce_scatter_then_all_gather");
    }

    // -----------------------------------------------------------------------------
    // Test 18: all_reduce and reduce_scatter interleaved, both orders —
    // both use typed reductions but different output semantics.
    // -----------------------------------------------------------------------------

    void test_all_reduce_then_reduce_scatter()
    {
        run_interleaved(
            kAllReduceThenReduceScatterBase, 100,
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
            noop, "test_all_reduce_then_reduce_scatter");
    }

    void test_reduce_scatter_then_all_reduce()
    {
        run_interleaved(
            kReduceScatterThenAllReduceBase, 100, noop,
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
            "test_reduce_scatter_then_all_reduce");
    }

    // -----------------------------------------------------------------------------
    // Test 19: reduce() and reduce_scatter() interleaved, both orders —
    // validates repeated use of the existing root reduction path.
    // -----------------------------------------------------------------------------

    void test_reduce_then_reduce_scatter()
    {
        run_interleaved(
            kReduceThenReduceScatterBase, 100,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const std::size_t root =
                    static_cast<std::size_t>(iteration) % 3;
                const std::int32_t send =
                    static_cast<std::int32_t>(rank) + 10;
                std::int32_t recv = 0;

                tbccl::reduce(
                    world, &send, rank == root ? &recv : nullptr, 1,
                    tbccl::DataType::Int32, tbccl::ReduceOp::Sum, root);

                if (rank == root && recv != 10 + 11 + 12)
                {
                    throw std::runtime_error("reduce mismatch");
                }
            },
            noop, "test_reduce_then_reduce_scatter");
    }

    void test_reduce_scatter_then_reduce()
    {
        run_interleaved(
            kReduceScatterThenReduceBase, 100, noop,
            [](tbccl::World &world, std::size_t rank, int iteration)
            {
                const std::size_t root =
                    static_cast<std::size_t>(iteration) % 3;
                const std::int32_t send =
                    static_cast<std::int32_t>(rank) + 10;
                std::int32_t recv = 0;

                tbccl::reduce(
                    world, &send, rank == root ? &recv : nullptr, 1,
                    tbccl::DataType::Int32, tbccl::ReduceOp::Sum, root);

                if (rank == root && recv != 10 + 11 + 12)
                {
                    throw std::runtime_error("reduce mismatch");
                }
            },
            "test_reduce_scatter_then_reduce");
    }

    // -----------------------------------------------------------------------------
    // Test 20: point-to-point traffic before and after a
    // reduce_scatter — stream alignment must survive the collective
    // boundary.
    // -----------------------------------------------------------------------------

    void test_p2p_around_reduce_scatter()
    {
        auto peers = make_local_peers(kP2pAroundBase, 3);

        std::vector<std::exception_ptr> errors(3);
        std::vector<std::thread> threads;

        constexpr std::uint32_t kPhaseAValue = 0x11111111u;
        constexpr std::uint32_t kPhaseBValue = 0x22222222u;

        threads.emplace_back(
            run_rank,
            make_options(0, peers),
            [](tbccl::World &world)
            {
                std::uint32_t received = 0;
                world.recv(1, &received, sizeof(received));
                expect(received == kPhaseAValue, "phase A: wrong value from rank 1");

                const std::vector<std::int32_t> send = {1, 2, 3};
                std::int32_t recv = 0;
                tbccl::reduce_scatter(
                    world, send.data(), &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);
                expect(recv == 1 + 1 + 1, "rank 0: wrong reduce_scatter result");
            },
            std::ref(errors[0]));

        threads.emplace_back(
            run_rank,
            make_options(1, peers),
            [](tbccl::World &world)
            {
                std::uint32_t v = kPhaseAValue;
                world.send(0, &v, sizeof(v));

                const std::vector<std::int32_t> send = {1, 2, 3};
                std::int32_t recv = 0;
                tbccl::reduce_scatter(
                    world, send.data(), &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);
                expect(recv == 2 + 2 + 2, "rank 1: wrong reduce_scatter result");

                std::uint32_t received = 0;
                world.recv(2, &received, sizeof(received));
                expect(received == kPhaseBValue, "phase B: wrong value from rank 2");
            },
            std::ref(errors[1]));

        threads.emplace_back(
            run_rank,
            make_options(2, peers),
            [](tbccl::World &world)
            {
                const std::vector<std::int32_t> send = {1, 2, 3};
                std::int32_t recv = 0;
                tbccl::reduce_scatter(
                    world, send.data(), &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);
                expect(recv == 3 + 3 + 3, "rank 2: wrong reduce_scatter result");

                std::uint32_t v = kPhaseBValue;
                world.send(1, &v, sizeof(v));
            },
            std::ref(errors[2]));

        join_and_check(threads, errors);

        std::cout << "[PASS] test_p2p_around_reduce_scatter\n";
    }

    // -----------------------------------------------------------------------------
    // Test 21 / 22: more local ranks, repeated, moderate recv_count.
    // -----------------------------------------------------------------------------

    void test_n_rank_repeated_reduce_scatter(
        std::size_t n,
        std::uint16_t base_port,
        int iterations)
    {
        constexpr std::size_t kRecvCount = 4;
        const std::size_t total_count = n * kRecvCount;
        auto peers = make_local_peers(base_port, n);

        std::vector<std::exception_ptr> errors(n);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < n; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers, /*timeout_ms=*/8000),
                [rank, n, iterations, total_count](tbccl::World &world)
                {
                    for (int iteration = 0; iteration < iterations;
                         ++iteration)
                    {
                        std::vector<std::int32_t> send(total_count);

                        for (std::size_t i = 0; i < total_count; ++i)
                        {
                            send[i] = static_cast<std::int32_t>(rank + i + 1);
                        }

                        std::vector<std::int32_t> recv(kRecvCount, 0);

                        tbccl::reduce_scatter(
                            world, send.data(), recv.data(), kRecvCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                        for (std::size_t i = 0; i < kRecvCount; ++i)
                        {
                            const std::size_t global_index =
                                rank * kRecvCount + i;
                            std::int32_t expected = 0;

                            for (std::size_t r = 0; r < n; ++r)
                            {
                                expected += static_cast<std::int32_t>(
                                    r + global_index + 1);
                            }

                            if (recv[i] != expected)
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
            << "[PASS] test_n_rank_repeated_reduce_scatter (n=" << n
            << ", iterations=" << iterations << ")\n";
    }

} // namespace

int main()
{
    try
    {
        test_single_rank();
        test_basic_two_rank_sum();
        test_three_rank_ordered_segments();
        test_all_datatypes_all_operations();
        test_negative_values();
        test_floating_point();
        test_zero_count();
        test_null_send_buffer();
        test_null_recv_buffer();
        test_odd_count();
        test_large_payload();
        test_repeated_reduce_scatter();
        test_reduce_scatter_then_barrier();
        test_barrier_then_reduce_scatter();
        test_broadcast_then_reduce_scatter();
        test_reduce_scatter_then_broadcast();
        test_all_gather_then_reduce_scatter();
        test_reduce_scatter_then_all_gather();
        test_all_reduce_then_reduce_scatter();
        test_reduce_scatter_then_all_reduce();
        test_reduce_then_reduce_scatter();
        test_reduce_scatter_then_reduce();
        test_p2p_around_reduce_scatter();
        test_n_rank_repeated_reduce_scatter(4, kFourRankBase, 100);
        test_n_rank_repeated_reduce_scatter(8, kEightRankBase, 100);
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
