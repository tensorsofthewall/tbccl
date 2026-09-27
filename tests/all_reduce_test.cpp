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
#include <utility>
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
    // production (18515) and every other test file's range
    // (tcp_transport_test: 28515+, world_test: 29500-29569,
    // barrier_test: 29570-29659, broadcast_test: 29660-29790,
    // all_gather_test: 29800-29958, reduce_test: 30000-30217).
    constexpr std::uint16_t kSingleRankBase = 30300;
    constexpr std::uint16_t kBasicTwoRankBase = 30310;
    constexpr std::uint16_t kThreeRankSumBase = 30320;
    constexpr std::uint16_t kAllDatatypesBase = 30330;
    constexpr std::uint16_t kNegativeValuesBase = 30350;
    constexpr std::uint16_t kFloatingPointF32Base = 30360;
    constexpr std::uint16_t kFloatingPointF64Base = 30370;
    constexpr std::uint16_t kZeroCountSingleBase = 30380;
    constexpr std::uint16_t kZeroCountMultiBase = 30390;
    constexpr std::uint16_t kNullSendBase = 30400;
    constexpr std::uint16_t kNullRecvBase = 30410;
    constexpr std::uint16_t kAliasingBase = 30420;
    constexpr std::uint16_t kOddCountBase = 30430;
    constexpr std::uint16_t kLargePayloadBase = 30440;
    constexpr std::uint16_t kRepeatedBase = 30460;
    constexpr std::uint16_t kAllReduceThenBarrierBase = 30470;
    constexpr std::uint16_t kBarrierThenAllReduceBase = 30480;
    constexpr std::uint16_t kBroadcastThenAllReduceBase = 30490;
    constexpr std::uint16_t kAllReduceThenBroadcastBase = 30500;
    constexpr std::uint16_t kAllGatherThenAllReduceBase = 30510;
    constexpr std::uint16_t kAllReduceThenAllGatherBase = 30520;
    constexpr std::uint16_t kReduceThenAllReduceBase = 30530;
    constexpr std::uint16_t kAllReduceThenReduceBase = 30540;
    constexpr std::uint16_t kP2pAroundBase = 30550;
    constexpr std::uint16_t kFourRankBase = 30560;
    constexpr std::uint16_t kEightRankBase = 30570;

    // Bootstraps a World from `peers` and runs each (op, expected) case
    // in `cases` sequentially against it on every rank, checking every
    // rank's own result after each case (all_reduce has no root — every
    // rank must see the identical reduced result).
    template <typename T>
    void run_all_reduce_cases(
        const std::vector<tbccl::PeerEndpoint> &peers,
        const std::vector<std::vector<T>> &per_rank_values,
        tbccl::DataType datatype,
        const std::vector<std::pair<tbccl::ReduceOp, std::vector<T>>> &cases)
    {
        const std::size_t size = per_rank_values.size();

        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank, &per_rank_values, datatype, &cases](tbccl::World &world)
                {
                    const auto &send = per_rank_values[rank];

                    for (const auto &reduce_case : cases)
                    {
                        const auto &op = reduce_case.first;
                        const auto &expected = reduce_case.second;

                        std::vector<T> recv(expected.size());

                        tbccl::all_reduce(
                            world, send.data(), recv.data(), send.size(),
                            datatype, op);

                        expect(
                            recv == expected,
                            "all_reduce: wrong result on rank " +
                                std::to_string(rank));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    // -----------------------------------------------------------------------------
    // Test 1: a 1-rank World — result must equal the local input, for
    // Int32 and one floating datatype.
    // -----------------------------------------------------------------------------

    void test_single_rank()
    {
        auto peers = make_local_peers(kSingleRankBase, 1);
        auto world = tbccl::create_tcp_world(make_options(0, peers));

        {
            const std::vector<std::int32_t> send = {1, -2, 3, 0};
            std::vector<std::int32_t> recv(send.size(), 0);

            tbccl::all_reduce(
                *world, send.data(), recv.data(), send.size(),
                tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

            expect(recv == send, "single-rank Int32 all_reduce should equal input");
        }

        {
            const std::vector<float> send = {1.5f, -2.25f, 3.0f};
            std::vector<float> recv(send.size(), 0.0f);

            tbccl::all_reduce(
                *world, send.data(), recv.data(), send.size(),
                tbccl::DataType::Float32, tbccl::ReduceOp::Min);

            expect(
                recv == send, "single-rank Float32 all_reduce should equal input");
        }

        std::cout << "[PASS] test_single_rank\n";
    }

    // -----------------------------------------------------------------------------
    // Test 2: basic 2-rank Sum.
    // -----------------------------------------------------------------------------

    void test_basic_two_rank_sum()
    {
        auto peers = make_local_peers(kBasicTwoRankBase, 2);

        const std::vector<std::vector<std::int32_t>> values = {
            {1, 2, 3}, {4, 5, 6}};

        run_all_reduce_cases<std::int32_t>(
            peers, values, tbccl::DataType::Int32,
            {{tbccl::ReduceOp::Sum, {5, 7, 9}}});

        std::cout << "[PASS] test_basic_two_rank_sum\n";
    }

    // -----------------------------------------------------------------------------
    // Test 3: three-rank Sum, the phase plan's own worked example.
    // -----------------------------------------------------------------------------

    void test_three_rank_sum()
    {
        auto peers = make_local_peers(kThreeRankSumBase, 3);

        const std::vector<std::vector<std::int32_t>> values = {
            {1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

        run_all_reduce_cases<std::int32_t>(
            peers, values, tbccl::DataType::Int32,
            {{tbccl::ReduceOp::Sum, {12, 15, 18}}});

        std::cout << "[PASS] test_three_rank_sum\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4 / 5: all four datatypes, all four operations, on one
    // bootstrapped 3-rank World, verified on every rank. Values are
    // small integers exactly representable in every supported type, so
    // exact equality is valid even for the floating-point cases.
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
                    const std::array<double, kSize> values0 = {1.0, 2.0, 3.0};
                    const std::array<double, kSize> values1 = {5.0, 4.0, 3.0};

                    struct Case
                    {
                        tbccl::ReduceOp op;
                        double expected0;
                        double expected1;
                    };

                    const std::array<Case, 4> cases = {{
                        {tbccl::ReduceOp::Sum, 6.0, 12.0},
                        {tbccl::ReduceOp::Product, 6.0, 60.0},
                        {tbccl::ReduceOp::Min, 1.0, 3.0},
                        {tbccl::ReduceOp::Max, 3.0, 5.0},
                    }};

                    for (const auto &c : cases)
                    {
                        {
                            const std::vector<std::int32_t> send = {
                                static_cast<std::int32_t>(values0[rank]),
                                static_cast<std::int32_t>(values1[rank])};
                            std::vector<std::int32_t> recv(2, 0);

                            tbccl::all_reduce(
                                world, send.data(), recv.data(), send.size(),
                                tbccl::DataType::Int32, c.op);

                            expect(
                                recv[0] ==
                                        static_cast<std::int32_t>(c.expected0) &&
                                    recv[1] ==
                                        static_cast<std::int32_t>(c.expected1),
                                "Int32 result mismatch");
                        }

                        {
                            const std::vector<std::int64_t> send = {
                                static_cast<std::int64_t>(values0[rank]),
                                static_cast<std::int64_t>(values1[rank])};
                            std::vector<std::int64_t> recv(2, 0);

                            tbccl::all_reduce(
                                world, send.data(), recv.data(), send.size(),
                                tbccl::DataType::Int64, c.op);

                            expect(
                                recv[0] ==
                                        static_cast<std::int64_t>(c.expected0) &&
                                    recv[1] ==
                                        static_cast<std::int64_t>(c.expected1),
                                "Int64 result mismatch");
                        }

                        {
                            const std::vector<float> send = {
                                static_cast<float>(values0[rank]),
                                static_cast<float>(values1[rank])};
                            std::vector<float> recv(2, 0.0f);

                            tbccl::all_reduce(
                                world, send.data(), recv.data(), send.size(),
                                tbccl::DataType::Float32, c.op);

                            expect(
                                recv[0] == static_cast<float>(c.expected0) &&
                                    recv[1] == static_cast<float>(c.expected1),
                                "Float32 result mismatch");
                        }

                        {
                            const std::vector<double> send = {
                                values0[rank], values1[rank]};
                            std::vector<double> recv(2, 0.0);

                            tbccl::all_reduce(
                                world, send.data(), recv.data(), send.size(),
                                tbccl::DataType::Float64, c.op);

                            expect(
                                recv[0] == c.expected0 && recv[1] == c.expected1,
                                "Float64 result mismatch");
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_all_datatypes_all_operations\n";
    }

    // -----------------------------------------------------------------------------
    // Test 6: negative values, Int32, Sum/Min/Max.
    // -----------------------------------------------------------------------------

    void test_negative_values()
    {
        auto peers = make_local_peers(kNegativeValuesBase, 3);

        const std::vector<std::vector<std::int32_t>> values = {
            {-5, 2}, {3, -4}, {-1, 9}};

        run_all_reduce_cases<std::int32_t>(
            peers, values, tbccl::DataType::Int32,
            {
                {tbccl::ReduceOp::Sum, {-3, 7}},
                {tbccl::ReduceOp::Min, {-5, -4}},
                {tbccl::ReduceOp::Max, {3, 9}},
            });

        std::cout << "[PASS] test_negative_values\n";
    }

    // -----------------------------------------------------------------------------
    // Test 7: floating point, non-integer values, exactly representable
    // in binary floating point so equality (not tolerance) is valid.
    // -----------------------------------------------------------------------------

    void test_floating_point()
    {
        {
            auto peers = make_local_peers(kFloatingPointF32Base, 3);
            const std::vector<std::vector<float>> values = {
                {0.5f}, {1.25f}, {-2.0f}};

            run_all_reduce_cases<float>(
                peers, values, tbccl::DataType::Float32,
                {
                    {tbccl::ReduceOp::Sum, {-0.25f}},
                    {tbccl::ReduceOp::Product, {-1.25f}},
                    {tbccl::ReduceOp::Min, {-2.0f}},
                    {tbccl::ReduceOp::Max, {1.25f}},
                });
        }

        {
            auto peers = make_local_peers(kFloatingPointF64Base, 3);
            const std::vector<std::vector<double>> values = {
                {0.5}, {1.25}, {-2.0}};

            run_all_reduce_cases<double>(
                peers, values, tbccl::DataType::Float64,
                {
                    {tbccl::ReduceOp::Sum, {-0.25}},
                    {tbccl::ReduceOp::Product, {-1.25}},
                    {tbccl::ReduceOp::Min, {-2.0}},
                    {tbccl::ReduceOp::Max, {1.25}},
                });
        }

        std::cout << "[PASS] test_floating_point\n";
    }

    // -----------------------------------------------------------------------------
    // Test 8: count == 0 must succeed with no throw/deadlock, buffers
    // may be nullptr — on both a 1-rank and a 3-rank World.
    // -----------------------------------------------------------------------------

    void test_zero_count()
    {
        {
            auto peers = make_local_peers(kZeroCountSingleBase, 1);
            auto world = tbccl::create_tcp_world(make_options(0, peers));

            tbccl::all_reduce(
                *world, nullptr, nullptr, 0, tbccl::DataType::Float32,
                tbccl::ReduceOp::Sum);
        }

        {
            constexpr std::size_t kSize = 3;
            auto peers = make_local_peers(kZeroCountMultiBase, kSize);

            std::vector<std::exception_ptr> errors(kSize);
            std::vector<std::thread> threads;

            for (std::size_t rank = 0; rank < kSize; ++rank)
            {
                threads.emplace_back(
                    run_rank,
                    make_options(rank, peers),
                    [](tbccl::World &world)
                    {
                        tbccl::all_reduce(
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
    // Test 9: nullptr send_buffer with count > 0 must be rejected on
    // every rank without communication.
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
                        tbccl::all_reduce(
                            world, nullptr, &recv, 4, tbccl::DataType::Int32,
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
                        "all_reduce() should reject null send buffer for "
                        "count > 0");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_null_send_buffer\n";
    }

    // -----------------------------------------------------------------------------
    // Test 10: nullptr recv_buffer with count > 0 must be rejected on
    // every rank without communication — unlike reduce(), this is
    // invalid on every rank (every rank receives the result), so a real
    // multi-rank World is safe here: no rank depends on a peer that has
    // already started communicating.
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
                    std::int32_t send = 1;
                    bool threw = false;

                    try
                    {
                        tbccl::all_reduce(
                            world, &send, nullptr, 1, tbccl::DataType::Int32,
                            tbccl::ReduceOp::Sum);
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
                        "all_reduce() should reject null receive buffer for "
                        "count > 0");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_null_recv_buffer\n";
    }

    // -----------------------------------------------------------------------------
    // Test 11: exact send_buffer == recv_buffer aliasing, on every
    // rank, real multi-rank World. Documented in collectives.hpp as
    // supported.
    // -----------------------------------------------------------------------------

    void test_exact_in_place_aliasing()
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(kAliasingBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    std::vector<std::int32_t> buffer = {
                        static_cast<std::int32_t>(rank) + 1,
                        static_cast<std::int32_t>(rank) * 2 + 1};

                    tbccl::all_reduce(
                        world, buffer.data(), buffer.data(), buffer.size(),
                        tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                    // sum(1..kSize) and sum(1,3,5) for the two elements.
                    const std::int32_t expected0 =
                        static_cast<std::int32_t>(kSize * (kSize + 1) / 2);
                    std::int32_t expected1 = 0;

                    for (std::size_t r = 0; r < kSize; ++r)
                    {
                        expected1 += static_cast<std::int32_t>(r) * 2 + 1;
                    }

                    expect(
                        buffer[0] == expected0 && buffer[1] == expected1,
                        "in-place all_reduce produced wrong result on rank " +
                            std::to_string(rank));
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_exact_in_place_aliasing\n";
    }

    // -----------------------------------------------------------------------------
    // Test 12: an odd element count (257), Int64.
    // -----------------------------------------------------------------------------

    void test_odd_count()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kCount = 257;
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
                    std::vector<std::int64_t> send(kCount);

                    for (std::size_t i = 0; i < kCount; ++i)
                    {
                        send[i] =
                            static_cast<std::int64_t>(i) +
                            static_cast<std::int64_t>(rank) * 1000;
                    }

                    std::vector<std::int64_t> recv(kCount, 0);

                    tbccl::all_reduce(
                        world, send.data(), recv.data(), kCount,
                        tbccl::DataType::Int64, tbccl::ReduceOp::Sum);

                    for (std::size_t i = 0; i < kCount; ++i)
                    {
                        std::int64_t expected = 0;

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            expected +=
                                static_cast<std::int64_t>(i) +
                                static_cast<std::int64_t>(r) * 1000;
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
    // Test 13: 1 MiB- and 4 MiB-class Float32 all_reduce, 3 ranks, Sum,
    // verified on every rank.
    // -----------------------------------------------------------------------------

    void test_large_payload(std::size_t count, std::uint16_t base_port)
    {
        constexpr std::size_t kSize = 3;
        auto peers = make_local_peers(base_port, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers, /*timeout_ms=*/10000),
                [rank, count](tbccl::World &world)
                {
                    std::vector<float> send(count);

                    for (std::size_t i = 0; i < count; ++i)
                    {
                        send[i] = static_cast<float>((i % 13) + rank + 1);
                    }

                    std::vector<float> recv(count, 0.0f);

                    tbccl::all_reduce(
                        world, send.data(), recv.data(), count,
                        tbccl::DataType::Float32, tbccl::ReduceOp::Sum);

                    for (std::size_t i = 0; i < count; ++i)
                    {
                        float expected = 0.0f;

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            expected += static_cast<float>((i % 13) + r + 1);
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

        std::cout
            << "[PASS] test_large_payload (" << count << " float32 elements)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 14: 1000 repeated all_reduces, small element count, rotating
    // operation. Values kept in [1, 4] so Product cannot explode.
    // -----------------------------------------------------------------------------

    void test_repeated_all_reduces()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 1000;
        constexpr std::size_t kCount = 4;

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

                        std::vector<std::int32_t> send(kCount);

                        for (std::size_t i = 0; i < kCount; ++i)
                        {
                            send[i] = static_cast<std::int32_t>(
                                ((iteration + static_cast<int>(rank) +
                                  static_cast<int>(i)) %
                                 4) +
                                1);
                        }

                        std::vector<std::int32_t> recv(kCount, 0);

                        tbccl::all_reduce(
                            world, send.data(), recv.data(), kCount,
                            tbccl::DataType::Int32, op);

                        for (std::size_t i = 0; i < kCount; ++i)
                        {
                            std::int32_t acc = 0;
                            bool first = true;

                            for (std::size_t r = 0; r < kSize; ++r)
                            {
                                const std::int32_t v =
                                    static_cast<std::int32_t>(
                                        ((iteration + static_cast<int>(r) +
                                          static_cast<int>(i)) %
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
            << "[PASS] test_repeated_all_reduces (" << kIterations
            << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Shared small Int32-Sum kernel for the interleaving tests below:
    // every rank contributes rank+1, so the all_reduce result is always
    // kSize*(kSize+1)/2 regardless of which collective ran around it.
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

                        const std::int32_t send =
                            static_cast<std::int32_t>(rank) + 1;
                        std::int32_t recv = 0;

                        tbccl::all_reduce(
                            world, &send, &recv, 1, tbccl::DataType::Int32,
                            tbccl::ReduceOp::Sum);

                        constexpr std::int32_t kExpected =
                            static_cast<std::int32_t>(kSize * (kSize + 1) / 2);

                        if (recv != kExpected)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": wrong all_reduce sum on rank " +
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
    // Test 15 / 16: all_reduce and barrier interleaved, both orders.
    // -----------------------------------------------------------------------------

    void test_all_reduce_then_barrier()
    {
        run_interleaved(
            kAllReduceThenBarrierBase, 100, noop,
            [](tbccl::World &world, std::size_t, int) { tbccl::barrier(world); },
            "test_all_reduce_then_barrier");
    }

    void test_barrier_then_all_reduce()
    {
        run_interleaved(
            kBarrierThenAllReduceBase, 100,
            [](tbccl::World &world, std::size_t, int) { tbccl::barrier(world); },
            noop, "test_barrier_then_all_reduce");
    }

    // -----------------------------------------------------------------------------
    // Test 17 / 18: broadcast and all_reduce interleaved, both orders,
    // using a separate buffer for the broadcast.
    // -----------------------------------------------------------------------------

    void test_broadcast_then_all_reduce()
    {
        run_interleaved(
            kBroadcastThenAllReduceBase, 100,
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
            noop, "test_broadcast_then_all_reduce");
    }

    void test_all_reduce_then_broadcast()
    {
        run_interleaved(
            kAllReduceThenBroadcastBase, 100, noop,
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
            "test_all_reduce_then_broadcast");
    }

    // -----------------------------------------------------------------------------
    // Test 19: all_gather and all_reduce interleaved, both orders,
    // using a separate buffer for the all_gather.
    // -----------------------------------------------------------------------------

    void test_all_gather_then_all_reduce()
    {
        run_interleaved(
            kAllGatherThenAllReduceBase, 100,
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
            noop, "test_all_gather_then_all_reduce");
    }

    void test_all_reduce_then_all_gather()
    {
        run_interleaved(
            kAllReduceThenAllGatherBase, 100, noop,
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
            "test_all_reduce_then_all_gather");
    }

    // -----------------------------------------------------------------------------
    // Test 20: reduce and all_reduce interleaved, both orders, using a
    // separate buffer for the reduce. Validates the typed reduction
    // path is reused cleanly across consecutive collective calls.
    // -----------------------------------------------------------------------------

    void test_reduce_then_all_reduce()
    {
        run_interleaved(
            kReduceThenAllReduceBase, 100,
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

                if (rank == root)
                {
                    constexpr std::int32_t kExpected = 10 + 11 + 12;

                    if (recv != kExpected)
                    {
                        throw std::runtime_error(
                            "iteration " + std::to_string(iteration) +
                            ": reduce mismatch");
                    }
                }
            },
            noop, "test_reduce_then_all_reduce");
    }

    void test_all_reduce_then_reduce()
    {
        run_interleaved(
            kAllReduceThenReduceBase, 100, noop,
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

                if (rank == root)
                {
                    constexpr std::int32_t kExpected = 10 + 11 + 12;

                    if (recv != kExpected)
                    {
                        throw std::runtime_error(
                            "iteration " + std::to_string(iteration) +
                            ": reduce mismatch");
                    }
                }
            },
            "test_all_reduce_then_reduce");
    }

    // -----------------------------------------------------------------------------
    // Test 21: point-to-point traffic before and after an all_reduce —
    // stream alignment must survive the collective boundary.
    // -----------------------------------------------------------------------------

    void test_p2p_around_all_reduce()
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

                const std::int32_t send = 1;
                std::int32_t recv = 0;
                tbccl::all_reduce(
                    world, &send, &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);
                expect(recv == 6, "rank 0: wrong all_reduce result");
            },
            std::ref(errors[0]));

        threads.emplace_back(
            run_rank,
            make_options(1, peers),
            [](tbccl::World &world)
            {
                std::uint32_t v = kPhaseAValue;
                world.send(0, &v, sizeof(v));

                const std::int32_t send = 2;
                std::int32_t recv = 0;
                tbccl::all_reduce(
                    world, &send, &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);
                expect(recv == 6, "rank 1: wrong all_reduce result");

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
                const std::int32_t send = 3;
                std::int32_t recv = 0;
                tbccl::all_reduce(
                    world, &send, &recv, 1, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);
                expect(recv == 6, "rank 2: wrong all_reduce result");

                std::uint32_t v = kPhaseBValue;
                world.send(1, &v, sizeof(v));
            },
            std::ref(errors[2]));

        join_and_check(threads, errors);

        std::cout << "[PASS] test_p2p_around_all_reduce\n";
    }

    // -----------------------------------------------------------------------------
    // Test 22 / 23: more local ranks, repeated, moderate buffers.
    // -----------------------------------------------------------------------------

    void test_n_rank_repeated_all_reduce(
        std::size_t n,
        std::uint16_t base_port,
        int iterations)
    {
        constexpr std::size_t kCount = 8;
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
                    for (int iteration = 0; iteration < iterations;
                         ++iteration)
                    {
                        std::vector<std::int32_t> send(kCount);

                        for (std::size_t i = 0; i < kCount; ++i)
                        {
                            send[i] = static_cast<std::int32_t>(rank + i + 1);
                        }

                        std::vector<std::int32_t> recv(kCount, 0);

                        tbccl::all_reduce(
                            world, send.data(), recv.data(), kCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                        for (std::size_t i = 0; i < kCount; ++i)
                        {
                            std::int32_t expected = 0;

                            for (std::size_t r = 0; r < n; ++r)
                            {
                                expected += static_cast<std::int32_t>(r + i + 1);
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
            << "[PASS] test_n_rank_repeated_all_reduce (n=" << n
            << ", iterations=" << iterations << ")\n";
    }

} // namespace

int main()
{
    try
    {
        test_single_rank();
        test_basic_two_rank_sum();
        test_three_rank_sum();
        test_all_datatypes_all_operations();
        test_negative_values();
        test_floating_point();
        test_zero_count();
        test_null_send_buffer();
        test_null_recv_buffer();
        test_exact_in_place_aliasing();
        test_odd_count();
        test_large_payload(262144, kLargePayloadBase);
        test_large_payload(1048576, kLargePayloadBase + 3);
        test_repeated_all_reduces();
        test_all_reduce_then_barrier();
        test_barrier_then_all_reduce();
        test_broadcast_then_all_reduce();
        test_all_reduce_then_broadcast();
        test_all_gather_then_all_reduce();
        test_all_reduce_then_all_gather();
        test_reduce_then_all_reduce();
        test_all_reduce_then_reduce();
        test_p2p_around_all_reduce();
        test_n_rank_repeated_all_reduce(4, kFourRankBase, 100);
        test_n_rank_repeated_all_reduce(8, kEightRankBase, 100);
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
