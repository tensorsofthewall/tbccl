#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace
{

    // Fixed high port ranges, one block per test, well away from
    // production (18515) and from the other test files' ranges
    // (tcp_transport_test: 28515+, world_test: 29500-29569,
    // barrier_test: 29570-29659, broadcast_test: 29660-29790,
    // all_gather_test: 29800-29958).
    constexpr std::uint16_t kSingleRankBase = 30000;
    constexpr std::uint16_t kInt32SumBase = 30010;
    constexpr std::uint16_t kArbitraryRootBase = 30020;
    constexpr std::uint16_t kEveryRootBase = 30030;
    constexpr std::uint16_t kAllDatatypesBase = 30040;
    constexpr std::uint16_t kNegativeValuesBase = 30050;
    constexpr std::uint16_t kFloatingPointF32Base = 30060;
    constexpr std::uint16_t kFloatingPointF64Base = 30070;
    constexpr std::uint16_t kZeroCountBase = 30080;
    constexpr std::uint16_t kNullSendBase = 30090;
    constexpr std::uint16_t kInvalidRootBase = 30100;
    constexpr std::uint16_t kNonRootNullRecvBase = 30110;
    constexpr std::uint16_t kLargeReductionBase = 30120;
    constexpr std::uint16_t kOddCountBase = 30130;
    constexpr std::uint16_t kRepeatedBase = 30140;
    constexpr std::uint16_t kReduceThenBarrierBase = 30150;
    constexpr std::uint16_t kBarrierThenReduceBase = 30160;
    constexpr std::uint16_t kBroadcastThenReduceBase = 30170;
    constexpr std::uint16_t kAllGatherThenReduceBase = 30180;
    constexpr std::uint16_t kReduceThenAllGatherBase = 30190;
    constexpr std::uint16_t kFourRankBase = 30200;
    constexpr std::uint16_t kEightRankBase = 30210;

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

    // A World whose send()/recv() throw a distinctive message if ever
    // called. Used only to test that reduce()'s argument validation
    // happens before any communication is attempted, without stranding
    // real peer ranks (see test_null_root_recv_buffer below and item
    // #35 of the phase plan).
    class FailingWorld : public tbccl::World
    {
    public:
        FailingWorld(std::size_t rank, std::size_t size)
            : rank_(rank), size_(size)
        {
        }

        std::size_t rank() const noexcept override { return rank_; }

        std::size_t size() const noexcept override { return size_; }

        void send(std::size_t, const void *, std::size_t) override
        {
            throw std::runtime_error("FailingWorld::send should not be called");
        }

        void recv(std::size_t, void *, std::size_t) override
        {
            throw std::runtime_error("FailingWorld::recv should not be called");
        }

    private:
        std::size_t rank_;
        std::size_t size_;
    };

    // Bootstraps a World from `peers` and runs each (op, root, expected)
    // case in `cases` sequentially against it on every rank, checking
    // the root's result after each case. Reused across most of the
    // small correctness tests below to avoid re-bootstrapping a World
    // per case.
    template <typename T>
    void run_reduce_cases(
        const std::vector<tbccl::PeerEndpoint> &peers,
        const std::vector<std::vector<T>> &per_rank_values,
        tbccl::DataType datatype,
        const std::vector<
            std::tuple<tbccl::ReduceOp, std::size_t, std::vector<T>>> &cases)
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
                        const auto &op = std::get<0>(reduce_case);
                        const auto &root = std::get<1>(reduce_case);
                        const auto &expected = std::get<2>(reduce_case);

                        std::vector<T> recv(rank == root ? expected.size() : 0);

                        tbccl::reduce(
                            world,
                            send.data(),
                            rank == root ? recv.data() : nullptr,
                            send.size(),
                            datatype,
                            op,
                            root);

                        if (rank == root)
                        {
                            expect(
                                recv == expected,
                                "reduce: wrong result on root " +
                                    std::to_string(root));
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    // -----------------------------------------------------------------------------
    // Test 1: a 1-rank World — root necessarily equals self, so the
    // result must equal the local input exactly, for every supported
    // datatype (checked here for Int32 and Float32).
    // -----------------------------------------------------------------------------

    void test_single_rank()
    {
        auto peers = make_local_peers(kSingleRankBase, 1);
        auto world = tbccl::create_tcp_world(make_options(0, peers));

        {
            const std::vector<std::int32_t> send = {1, -2, 3, 0};
            std::vector<std::int32_t> recv(send.size(), 0);

            tbccl::reduce(
                *world, send.data(), recv.data(), send.size(),
                tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 0);

            expect(recv == send, "single-rank Int32 reduce should equal input");
        }

        {
            const std::vector<float> send = {1.5f, -2.25f, 3.0f};
            std::vector<float> recv(send.size(), 0.0f);

            tbccl::reduce(
                *world, send.data(), recv.data(), send.size(),
                tbccl::DataType::Float32, tbccl::ReduceOp::Min, 0);

            expect(recv == send, "single-rank Float32 reduce should equal input");
        }

        std::cout << "[PASS] test_single_rank\n";
    }

    // -----------------------------------------------------------------------------
    // Test 2: Int32 Sum across 3 ranks with the phase plan's own worked
    // example.
    // -----------------------------------------------------------------------------

    void test_int32_sum()
    {
        auto peers = make_local_peers(kInt32SumBase, 3);

        const std::vector<std::vector<std::int32_t>> values = {
            {1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

        run_reduce_cases<std::int32_t>(
            peers, values, tbccl::DataType::Int32,
            {{tbccl::ReduceOp::Sum, 0, {12, 15, 18}}});

        std::cout << "[PASS] test_int32_sum\n";
    }

    // -----------------------------------------------------------------------------
    // Test 3: root is NOT rank 0 (3 ranks, root = 2) — catches any
    // accidental hard-coding of rank 0 as the receiver.
    // -----------------------------------------------------------------------------

    void test_arbitrary_root()
    {
        constexpr std::size_t kRoot = 2;
        auto peers = make_local_peers(kArbitraryRootBase, 3);

        const std::vector<std::vector<std::int32_t>> values = {
            {10, 20}, {30, 40}, {50, 60}};

        run_reduce_cases<std::int32_t>(
            peers, values, tbccl::DataType::Int32,
            {{tbccl::ReduceOp::Sum, kRoot, {90, 120}}});

        std::cout << "[PASS] test_arbitrary_root\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4: every rank takes a turn as root on one bootstrapped
    // 4-rank World.
    // -----------------------------------------------------------------------------

    void test_every_root()
    {
        constexpr std::size_t kSize = 4;
        auto peers = make_local_peers(kEveryRootBase, kSize);

        std::vector<std::vector<std::int32_t>> values(kSize);
        std::vector<std::tuple<
            tbccl::ReduceOp, std::size_t, std::vector<std::int32_t>>>
            cases;

        std::int32_t expected0 = 0;
        std::int32_t expected1 = 0;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            values[rank] = {
                static_cast<std::int32_t>(rank) + 1,
                static_cast<std::int32_t>(rank) * 2 + 1};

            expected0 += values[rank][0];
            expected1 += values[rank][1];
        }

        for (std::size_t root = 0; root < kSize; ++root)
        {
            cases.push_back(
                {tbccl::ReduceOp::Sum, root, {expected0, expected1}});
        }

        run_reduce_cases<std::int32_t>(
            peers, values, tbccl::DataType::Int32, cases);

        std::cout << "[PASS] test_every_root\n";
    }

    // -----------------------------------------------------------------------------
    // Test 5 / 6: all four datatypes, all four operations, on one
    // bootstrapped 3-rank World. Values are small integers exactly
    // representable in every supported type, so exact equality (no
    // tolerance) is valid even for the floating-point cases.
    // -----------------------------------------------------------------------------

    void test_all_datatypes_all_operations()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kRoot = 0;
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

                            tbccl::reduce(
                                world, send.data(),
                                rank == kRoot ? recv.data() : nullptr,
                                send.size(), tbccl::DataType::Int32, c.op,
                                kRoot);

                            if (rank == kRoot)
                            {
                                expect(
                                    recv[0] ==
                                            static_cast<std::int32_t>(
                                                c.expected0) &&
                                        recv[1] ==
                                            static_cast<std::int32_t>(
                                                c.expected1),
                                    "Int32 result mismatch");
                            }
                        }

                        {
                            const std::vector<std::int64_t> send = {
                                static_cast<std::int64_t>(values0[rank]),
                                static_cast<std::int64_t>(values1[rank])};
                            std::vector<std::int64_t> recv(2, 0);

                            tbccl::reduce(
                                world, send.data(),
                                rank == kRoot ? recv.data() : nullptr,
                                send.size(), tbccl::DataType::Int64, c.op,
                                kRoot);

                            if (rank == kRoot)
                            {
                                expect(
                                    recv[0] ==
                                            static_cast<std::int64_t>(
                                                c.expected0) &&
                                        recv[1] ==
                                            static_cast<std::int64_t>(
                                                c.expected1),
                                    "Int64 result mismatch");
                            }
                        }

                        {
                            const std::vector<float> send = {
                                static_cast<float>(values0[rank]),
                                static_cast<float>(values1[rank])};
                            std::vector<float> recv(2, 0.0f);

                            tbccl::reduce(
                                world, send.data(),
                                rank == kRoot ? recv.data() : nullptr,
                                send.size(), tbccl::DataType::Float32, c.op,
                                kRoot);

                            if (rank == kRoot)
                            {
                                expect(
                                    recv[0] == static_cast<float>(c.expected0) &&
                                        recv[1] ==
                                            static_cast<float>(c.expected1),
                                    "Float32 result mismatch");
                            }
                        }

                        {
                            const std::vector<double> send = {
                                values0[rank], values1[rank]};
                            std::vector<double> recv(2, 0.0);

                            tbccl::reduce(
                                world, send.data(),
                                rank == kRoot ? recv.data() : nullptr,
                                send.size(), tbccl::DataType::Float64, c.op,
                                kRoot);

                            if (rank == kRoot)
                            {
                                expect(
                                    recv[0] == c.expected0 &&
                                        recv[1] == c.expected1,
                                    "Float64 result mismatch");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_all_datatypes_all_operations\n";
    }

    // -----------------------------------------------------------------------------
    // Test 7: negative values, Int32, Sum/Min/Max on one bootstrapped
    // 3-rank World.
    // -----------------------------------------------------------------------------

    void test_negative_values()
    {
        auto peers = make_local_peers(kNegativeValuesBase, 3);

        const std::vector<std::vector<std::int32_t>> values = {
            {-5, 2}, {3, -4}, {-1, 9}};

        run_reduce_cases<std::int32_t>(
            peers, values, tbccl::DataType::Int32,
            {
                {tbccl::ReduceOp::Sum, 0, {-3, 7}},
                {tbccl::ReduceOp::Min, 0, {-5, -4}},
                {tbccl::ReduceOp::Max, 0, {3, 9}},
            });

        std::cout << "[PASS] test_negative_values\n";
    }

    // -----------------------------------------------------------------------------
    // Test 8: floating point, non-integer values. Chosen exactly
    // representable in binary floating point, so equality (not
    // tolerance) is valid here.
    // -----------------------------------------------------------------------------

    void test_floating_point()
    {
        {
            auto peers = make_local_peers(kFloatingPointF32Base, 3);
            const std::vector<std::vector<float>> values = {
                {0.5f}, {1.25f}, {-2.0f}};

            run_reduce_cases<float>(
                peers, values, tbccl::DataType::Float32,
                {
                    {tbccl::ReduceOp::Sum, 0, {-0.25f}},
                    {tbccl::ReduceOp::Product, 0, {-1.25f}},
                    {tbccl::ReduceOp::Min, 0, {-2.0f}},
                    {tbccl::ReduceOp::Max, 0, {1.25f}},
                });
        }

        {
            auto peers = make_local_peers(kFloatingPointF64Base, 3);
            const std::vector<std::vector<double>> values = {
                {0.5}, {1.25}, {-2.0}};

            run_reduce_cases<double>(
                peers, values, tbccl::DataType::Float64,
                {
                    {tbccl::ReduceOp::Sum, 0, {-0.25}},
                    {tbccl::ReduceOp::Product, 0, {-1.25}},
                    {tbccl::ReduceOp::Min, 0, {-2.0}},
                    {tbccl::ReduceOp::Max, 0, {1.25}},
                });
        }

        std::cout << "[PASS] test_floating_point\n";
    }

    // -----------------------------------------------------------------------------
    // Test 9: count == 0 must succeed with no throw/deadlock, buffers
    // may be nullptr, for any valid root.
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
                    tbccl::reduce(
                        world, nullptr, nullptr, 0,
                        tbccl::DataType::Float32, tbccl::ReduceOp::Sum, 0);

                    tbccl::reduce(
                        world, nullptr, nullptr, 0,
                        tbccl::DataType::Int64, tbccl::ReduceOp::Max, 2);
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_zero_count\n";
    }

    // -----------------------------------------------------------------------------
    // Test 10: nullptr send_buffer with count > 0 must be rejected on
    // every rank without communication — every rank calls the same
    // invalid operation, so none depends on a peer that already threw.
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
                        tbccl::reduce(
                            world, nullptr, &recv, 4,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 0);
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
                        "reduce() should reject null send buffer for "
                        "count > 0");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_null_send_buffer\n";
    }

    // -----------------------------------------------------------------------------
    // Test 11: root's null recv_buffer must be rejected before any
    // communication is attempted. Using a real multi-rank World here
    // would strand the non-root ranks in send() once root throws before
    // ever calling recv() — see FailingWorld's comment above — so this
    // uses a single-process fake World whose send()/recv() throw a
    // distinctive message if reduce() ever reaches them.
    // -----------------------------------------------------------------------------

    void test_null_root_recv_buffer()
    {
        FailingWorld world(/*rank=*/0, /*size=*/3);

        const std::vector<std::int32_t> send = {1, 2, 3};
        bool threw = false;

        try
        {
            tbccl::reduce(
                world, send.data(), nullptr, send.size(),
                tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 0);
        }
        catch (const std::exception &error)
        {
            threw = true;

            const std::string message = error.what();

            expect(
                message.find("receive buffer is null") != std::string::npos,
                "wrong error for null root receive buffer: " + message);
        }

        expect(
            threw,
            "reduce() should reject null root receive buffer for "
            "count > 0");

        std::cout << "[PASS] test_null_root_recv_buffer\n";
    }

    // -----------------------------------------------------------------------------
    // Test 12: non-root ranks passing recv_buffer == nullptr is valid
    // and must succeed, on a real multi-rank World.
    // -----------------------------------------------------------------------------

    void test_non_root_null_recv_buffer()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kRoot = 1;
        auto peers = make_local_peers(kNonRootNullRecvBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    const std::int32_t send = static_cast<std::int32_t>(rank) + 1;
                    std::int32_t recv = 0;

                    tbccl::reduce(
                        world, &send, rank == kRoot ? &recv : nullptr,
                        1, tbccl::DataType::Int32, tbccl::ReduceOp::Sum,
                        kRoot);

                    if (rank == kRoot)
                    {
                        expect(recv == 1 + 2 + 3, "wrong sum on root");
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_non_root_null_recv_buffer\n";
    }

    // -----------------------------------------------------------------------------
    // Test 13: an invalid root must be rejected on every rank before
    // any communication, same reasoning as test_null_send_buffer.
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
                    std::int32_t send = 1;
                    std::int32_t recv = 0;
                    bool threw = false;

                    try
                    {
                        tbccl::reduce(
                            world, &send, &recv, 1,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 3);
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

                    expect(threw, "reduce() should reject root >= size");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_invalid_root\n";
    }

    // -----------------------------------------------------------------------------
    // Test 14: 1 MiB-class Float32 reduction (262144 elements), 3
    // ranks, Sum. Deterministic small-integer data keeps the expected
    // result exact.
    // -----------------------------------------------------------------------------

    void test_large_reduction()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kCount = 262144; // 1 MiB of float32
        auto peers = make_local_peers(kLargeReductionBase, kSize);

        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank,
                make_options(rank, peers, /*timeout_ms=*/10000),
                [rank](tbccl::World &world)
                {
                    std::vector<float> send(kCount);

                    for (std::size_t i = 0; i < kCount; ++i)
                    {
                        send[i] = static_cast<float>((i % 13) + rank + 1);
                    }

                    std::vector<float> recv(rank == 0 ? kCount : 0);

                    tbccl::reduce(
                        world, send.data(),
                        rank == 0 ? recv.data() : nullptr, kCount,
                        tbccl::DataType::Float32, tbccl::ReduceOp::Sum, 0);

                    if (rank == 0)
                    {
                        for (std::size_t i = 0; i < kCount; ++i)
                        {
                            float expected = 0.0f;

                            for (std::size_t r = 0; r < kSize; ++r)
                            {
                                expected +=
                                    static_cast<float>((i % 13) + r + 1);
                            }

                            if (recv[i] != expected)
                            {
                                throw std::runtime_error(
                                    "element " + std::to_string(i) +
                                    " mismatch");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_large_reduction\n";
    }

    // -----------------------------------------------------------------------------
    // Test 15: an odd element count (257), Int64, root != 0.
    // -----------------------------------------------------------------------------

    void test_odd_element_count()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kCount = 257;
        constexpr std::size_t kRoot = 1;
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

                    std::vector<std::int64_t> recv(rank == kRoot ? kCount : 0);

                    tbccl::reduce(
                        world, send.data(),
                        rank == kRoot ? recv.data() : nullptr, kCount,
                        tbccl::DataType::Int64, tbccl::ReduceOp::Sum, kRoot);

                    if (rank == kRoot)
                    {
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
                                    " mismatch");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_odd_element_count\n";
    }

    // -----------------------------------------------------------------------------
    // Test 16: 1000 repeated reductions, small element count, rotating
    // root and operation. Values are kept in [1, 4] so Product cannot
    // explode numerically. Primary goal: no protocol drift, no stale
    // bytes, no deadlock.
    // -----------------------------------------------------------------------------

    void test_repeated_reductions()
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
                        const std::size_t root =
                            static_cast<std::size_t>(iteration) % kSize;
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

                        std::vector<std::int32_t> recv(
                            rank == root ? kCount : 0);

                        tbccl::reduce(
                            world, send.data(),
                            rank == root ? recv.data() : nullptr, kCount,
                            tbccl::DataType::Int32, op, root);

                        if (rank == root)
                        {
                            for (std::size_t i = 0; i < kCount; ++i)
                            {
                                std::int32_t acc = 0;
                                bool first = true;

                                for (std::size_t r = 0; r < kSize; ++r)
                                {
                                    const std::int32_t v =
                                        static_cast<std::int32_t>(
                                            ((iteration +
                                              static_cast<int>(r) +
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
                                        "iteration " +
                                        std::to_string(iteration) +
                                        ": element " + std::to_string(i) +
                                        " mismatch");
                                }
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_repeated_reductions (" << kIterations
            << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 17 / 18: reduce and barrier interleaved, in both orders,
    // repeated — checks neither collective consumes bytes belonging to
    // the other.
    // -----------------------------------------------------------------------------

    void test_reduce_then_barrier()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        auto peers = make_local_peers(kReduceThenBarrierBase, kSize);

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

                        const std::int32_t send =
                            static_cast<std::int32_t>(rank) + 1;
                        std::int32_t recv = 0;

                        tbccl::reduce(
                            world, &send, rank == root ? &recv : nullptr, 1,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum,
                            root);

                        tbccl::barrier(world);

                        if (rank == root)
                        {
                            constexpr std::int32_t kExpected =
                                static_cast<std::int32_t>(
                                    kSize * (kSize + 1) / 2);

                            if (recv != kExpected)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": wrong sum after reduce+barrier");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_reduce_then_barrier (" << kIterations
            << " iterations)\n";
    }

    void test_barrier_then_reduce()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        auto peers = make_local_peers(kBarrierThenReduceBase, kSize);

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

                        tbccl::barrier(world);

                        const std::int32_t send =
                            static_cast<std::int32_t>(rank) + 1;
                        std::int32_t recv = 0;

                        tbccl::reduce(
                            world, &send, rank == root ? &recv : nullptr, 1,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum,
                            root);

                        if (rank == root)
                        {
                            constexpr std::int32_t kExpected =
                                static_cast<std::int32_t>(
                                    kSize * (kSize + 1) / 2);

                            if (recv != kExpected)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": wrong sum after barrier+reduce");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_barrier_then_reduce (" << kIterations
            << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 19: broadcast and reduce interleaved, repeated, using
    // separate buffers for each.
    // -----------------------------------------------------------------------------

    void test_broadcast_then_reduce()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        auto peers = make_local_peers(kBroadcastThenReduceBase, kSize);

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
                        const std::size_t broadcast_root =
                            static_cast<std::size_t>(iteration) % kSize;
                        const std::size_t reduce_root =
                            static_cast<std::size_t>(iteration + 1) % kSize;

                        const auto pattern = deterministic_buffer(
                            16, static_cast<std::uint32_t>(iteration));
                        auto bcast_buffer =
                            (rank == broadcast_root)
                                ? pattern
                                : deterministic_buffer(16, 0xEEEEu);

                        tbccl::broadcast(
                            world, bcast_buffer.data(), bcast_buffer.size(),
                            broadcast_root);

                        if (bcast_buffer != pattern)
                        {
                            throw std::runtime_error(
                                "iteration " + std::to_string(iteration) +
                                ": broadcast mismatch");
                        }

                        const std::int32_t send =
                            static_cast<std::int32_t>(rank) + 1;
                        std::int32_t recv = 0;

                        tbccl::reduce(
                            world, &send,
                            rank == reduce_root ? &recv : nullptr, 1,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum,
                            reduce_root);

                        if (rank == reduce_root)
                        {
                            constexpr std::int32_t kExpected =
                                static_cast<std::int32_t>(
                                    kSize * (kSize + 1) / 2);

                            if (recv != kExpected)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": reduce mismatch");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_broadcast_then_reduce (" << kIterations
            << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 20 / 21: all_gather and reduce interleaved, both orders,
    // repeated, using separate buffers for each.
    // -----------------------------------------------------------------------------

    void test_all_gather_then_reduce()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        auto peers = make_local_peers(kAllGatherThenReduceBase, kSize);

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
                        const std::size_t reduce_root =
                            static_cast<std::size_t>(iteration) % kSize;

                        const std::uint8_t send_ag = static_cast<std::uint8_t>(
                            (rank + static_cast<std::size_t>(iteration)) &
                            0xFF);
                        std::vector<std::uint8_t> recv_ag(kSize, 0);

                        tbccl::all_gather(world, &send_ag, recv_ag.data(), 1);

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            const std::uint8_t expected =
                                static_cast<std::uint8_t>(
                                    (r + static_cast<std::size_t>(iteration)) &
                                    0xFF);

                            if (recv_ag[r] != expected)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": all_gather mismatch");
                            }
                        }

                        const std::int32_t send_r =
                            static_cast<std::int32_t>(rank) + 1;
                        std::int32_t recv_r = 0;

                        tbccl::reduce(
                            world, &send_r,
                            rank == reduce_root ? &recv_r : nullptr, 1,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum,
                            reduce_root);

                        if (rank == reduce_root)
                        {
                            constexpr std::int32_t kExpected =
                                static_cast<std::int32_t>(
                                    kSize * (kSize + 1) / 2);

                            if (recv_r != kExpected)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": reduce mismatch after all_gather");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_all_gather_then_reduce (" << kIterations
            << " iterations)\n";
    }

    void test_reduce_then_all_gather()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        auto peers = make_local_peers(kReduceThenAllGatherBase, kSize);

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
                        const std::size_t reduce_root =
                            static_cast<std::size_t>(iteration) % kSize;

                        const std::int32_t send_r =
                            static_cast<std::int32_t>(rank) + 10;
                        std::int32_t recv_r = 0;

                        tbccl::reduce(
                            world, &send_r,
                            rank == reduce_root ? &recv_r : nullptr, 1,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum,
                            reduce_root);

                        if (rank == reduce_root)
                        {
                            std::int32_t expected = 0;

                            for (std::size_t r = 0; r < kSize; ++r)
                            {
                                expected += static_cast<std::int32_t>(r) + 10;
                            }

                            if (recv_r != expected)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": reduce mismatch before all_gather");
                            }
                        }

                        const std::uint8_t send_ag = static_cast<std::uint8_t>(
                            (rank + static_cast<std::size_t>(iteration) + 1) &
                            0xFF);
                        std::vector<std::uint8_t> recv_ag(kSize, 0);

                        tbccl::all_gather(world, &send_ag, recv_ag.data(), 1);

                        for (std::size_t r = 0; r < kSize; ++r)
                        {
                            const std::uint8_t expected =
                                static_cast<std::uint8_t>(
                                    (r + static_cast<std::size_t>(iteration) +
                                     1) &
                                    0xFF);

                            if (recv_ag[r] != expected)
                            {
                                throw std::runtime_error(
                                    "iteration " + std::to_string(iteration) +
                                    ": all_gather mismatch after reduce");
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_reduce_then_all_gather (" << kIterations
            << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 22 / 23: more local ranks, rotating root, repeated.
    // Correctness at larger N, not performance.
    // -----------------------------------------------------------------------------

    void test_n_rank_repeated_reduce(
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
                        const std::size_t root =
                            static_cast<std::size_t>(iteration) % n;

                        std::vector<std::int32_t> send(kCount);

                        for (std::size_t i = 0; i < kCount; ++i)
                        {
                            send[i] = static_cast<std::int32_t>(rank + i + 1);
                        }

                        std::vector<std::int32_t> recv(
                            rank == root ? kCount : 0);

                        tbccl::reduce(
                            world, send.data(),
                            rank == root ? recv.data() : nullptr, kCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum,
                            root);

                        if (rank == root)
                        {
                            for (std::size_t i = 0; i < kCount; ++i)
                            {
                                std::int32_t expected = 0;

                                for (std::size_t r = 0; r < n; ++r)
                                {
                                    expected +=
                                        static_cast<std::int32_t>(r + i + 1);
                                }

                                if (recv[i] != expected)
                                {
                                    throw std::runtime_error(
                                        "iteration " +
                                        std::to_string(iteration) +
                                        ": element " + std::to_string(i) +
                                        " mismatch");
                                }
                            }
                        }
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout
            << "[PASS] test_n_rank_repeated_reduce (n=" << n
            << ", iterations=" << iterations << ")\n";
    }

} // namespace

int main()
{
    try
    {
        test_single_rank();
        test_int32_sum();
        test_arbitrary_root();
        test_every_root();
        test_all_datatypes_all_operations();
        test_negative_values();
        test_floating_point();
        test_zero_count();
        test_null_send_buffer();
        test_null_root_recv_buffer();
        test_non_root_null_recv_buffer();
        test_invalid_root();
        test_large_reduction();
        test_odd_element_count();
        test_repeated_reductions();
        test_reduce_then_barrier();
        test_barrier_then_reduce();
        test_broadcast_then_reduce();
        test_all_gather_then_reduce();
        test_reduce_then_all_gather();
        test_n_rank_repeated_reduce(4, kFourRankBase, 100);
        test_n_rank_repeated_reduce(8, kEightRankBase, 100);
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
