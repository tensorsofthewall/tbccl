#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "all_gather_internal.hpp"
#include "all_reduce_internal.hpp"
#include "reduce_scatter_internal.hpp"
#include "ring_executor.hpp"
#include "test_utils.hpp"

#include <algorithm>
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

using tbccl::detail::all_reduce_pipelined;
using tbccl::detail::all_reduce_reference;
using tbccl::detail::all_reduce_ring;
using tbccl::detail::RingExecutorAccess;
using tbccl::detail::RingExecutorStats;

namespace
{

    constexpr std::uint16_t kRankCountsBase = 25400;
    constexpr std::uint16_t kChunkSizeBase = 25420;
    constexpr std::uint16_t kEveryDatatypeBase = 25440;
    constexpr std::uint16_t kEveryOperationBase = 25460;
    constexpr std::uint16_t kInvalidChunkBase = 25480;
    constexpr std::uint16_t kDivisibilityBase = 25490;
    constexpr std::uint16_t kSingleRankBase = 25500;
    constexpr std::uint16_t kLargePayloadBase = 25510;
    constexpr std::uint16_t kRepeatedBase = 25520;
    constexpr std::uint16_t kMixedSequencingBase = 25530;
    constexpr std::uint16_t kWorkerReuseBase = 25540;
    constexpr std::uint16_t kExactAliasBase = 25550;

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

    template <typename T>
    void run_correctness(
        std::uint16_t base_port,
        std::size_t size,
        std::size_t count,
        std::size_t chunk_bytes,
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
                [rank, count, chunk_bytes, datatype, op](tbccl::World &world)
                {
                    constexpr std::uint32_t kSeed = 0x9abcu;
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
                    all_reduce_reference(
                        world, send.data(), reference_recv.data(), count,
                        datatype, op);

                    std::vector<T> ring_recv(count, T{});
                    all_reduce_ring(
                        world, send.data(), ring_recv.data(), count,
                        datatype, op);

                    std::vector<T> pipelined_recv(count, T{});
                    all_reduce_pipelined(
                        world, send.data(), pipelined_recv.data(), count,
                        datatype, op, chunk_bytes);

                    for (std::size_t i = 0; i < count; ++i)
                    {
                        expect(
                            approximately_equal(reference_recv[i], expected[i]),
                            "reference mismatch on rank " +
                                std::to_string(rank) + " element " +
                                std::to_string(i));
                        expect(
                            approximately_equal(ring_recv[i], expected[i]),
                            "existing-ring mismatch on rank " +
                                std::to_string(rank) + " element " +
                                std::to_string(i));
                        expect(
                            approximately_equal(pipelined_recv[i], expected[i]),
                            "pipelined mismatch on rank " +
                                std::to_string(rank) + " element " +
                                std::to_string(i) + " (chunk_bytes=" +
                                std::to_string(chunk_bytes) + ")");
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    void test_rank_counts()
    {
        for (auto [size, offset] :
             {std::pair<std::size_t, std::uint16_t>{1, 0},
              {2, 8}, {3, 16}, {4, 24}, {5, 32}, {7, 40}, {8, 48}})
        {
            const std::size_t count = size * 4; // divisible by construction

            run_correctness<std::int32_t>(
                static_cast<std::uint16_t>(kRankCountsBase + offset), size,
                count, 3 * sizeof(std::int32_t), tbccl::DataType::Int32,
                tbccl::ReduceOp::Sum);

            std::cout << "[PASS] test_rank_counts (size=" << size << ")\n";
        }
    }

    void test_chunk_size_relationships()
    {
        constexpr std::size_t kSize = 4;
        constexpr std::size_t kCount = 4096; // -> 1024 elements/rank, Float32

        struct Case
        {
            std::size_t chunk_bytes;
            const char *label;
        };

        std::uint16_t offset = 0;

        for (Case c :
             {Case{8192, "chunk > segment (1 chunk)"},
              Case{1024, "chunk == segment (1 chunk)"},
              Case{512, "2 chunks"}, Case{256, "4 chunks"},
              Case{60, "uneven chunks"}})
        {
            run_correctness<float>(
                static_cast<std::uint16_t>(kChunkSizeBase + offset), kSize,
                kCount, c.chunk_bytes, tbccl::DataType::Float32,
                tbccl::ReduceOp::Sum);

            std::cout << "[PASS] test_chunk_size_relationships (" << c.label
                       << ", chunk_bytes=" << c.chunk_bytes << ")\n";

            offset += 2;
        }
    }

    void test_every_datatype()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kCount = 3 * 257; // divisible, non-power-of-two

        std::uint16_t offset = 0;

        run_correctness<std::int32_t>(
            static_cast<std::uint16_t>(kEveryDatatypeBase + offset), kSize,
            kCount, 37 * sizeof(std::int32_t), tbccl::DataType::Int32,
            tbccl::ReduceOp::Sum);
        offset += 4;

        run_correctness<std::int64_t>(
            static_cast<std::uint16_t>(kEveryDatatypeBase + offset), kSize,
            kCount, 37 * sizeof(std::int64_t), tbccl::DataType::Int64,
            tbccl::ReduceOp::Sum);
        offset += 4;

        run_correctness<float>(
            static_cast<std::uint16_t>(kEveryDatatypeBase + offset), kSize,
            kCount, 37 * sizeof(float), tbccl::DataType::Float32,
            tbccl::ReduceOp::Sum);
        offset += 4;

        run_correctness<double>(
            static_cast<std::uint16_t>(kEveryDatatypeBase + offset), kSize,
            kCount, 37 * sizeof(double), tbccl::DataType::Float64,
            tbccl::ReduceOp::Sum);

        std::cout << "[PASS] test_every_datatype\n";
    }

    void test_every_operation()
    {
        constexpr std::size_t kSize = 4;
        constexpr std::size_t kCount = 4 * 33;

        std::uint16_t offset = 0;

        for (tbccl::ReduceOp op :
             {tbccl::ReduceOp::Sum, tbccl::ReduceOp::Product,
              tbccl::ReduceOp::Min, tbccl::ReduceOp::Max})
        {
            run_correctness<std::int32_t>(
                static_cast<std::uint16_t>(kEveryOperationBase + offset),
                kSize, kCount, 11 * sizeof(std::int32_t),
                tbccl::DataType::Int32, op);

            offset += 5;
        }

        std::cout << "[PASS] test_every_operation\n";
    }

    void test_invalid_chunk_size()
    {
        constexpr std::size_t kSize = 3;

        auto peers = make_local_peers(kInvalidChunkBase, kSize);
        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers),
                [](tbccl::World &world)
                {
                    std::vector<std::int32_t> send(9, 1);
                    std::vector<std::int32_t> recv(9, 0);

                    bool threw = false;
                    try
                    {
                        all_reduce_pipelined(
                            world, send.data(), recv.data(), 9,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 0);
                    }
                    catch (const std::exception &error)
                    {
                        threw = true;
                        expect(
                            std::string(error.what())
                                    .find("chunk_bytes must be") !=
                                std::string::npos,
                            "wrong error for zero chunk_bytes");
                    }
                    expect(threw, "should reject chunk_bytes == 0");

                    threw = false;
                    try
                    {
                        all_reduce_pipelined(
                            world, send.data(), recv.data(), 9,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 6);
                    }
                    catch (const std::exception &error)
                    {
                        threw = true;
                        expect(
                            std::string(error.what())
                                    .find("multiple of the datatype "
                                          "size") != std::string::npos,
                            "wrong error for misaligned chunk_bytes: " +
                                std::string(error.what()));
                    }
                    expect(threw, "should reject misaligned chunk_bytes");

                    std::int32_t dummy = 0;
                    all_reduce_pipelined(
                        world, &dummy, &dummy, 0, tbccl::DataType::Int32,
                        tbccl::ReduceOp::Sum, 0);

                    expect(
                        RingExecutorAccess::stats(world)
                                .worker_start_count == 0,
                        "zero-count pipelined all_reduce must not create "
                        "a ring executor");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_invalid_chunk_size\n";
    }

    void test_divisibility_rejection()
    {
        constexpr std::size_t kSize = 4;
        constexpr std::size_t kCount = 257; // not divisible by 4

        auto peers = make_local_peers(kDivisibilityBase, kSize);
        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers),
                [](tbccl::World &world)
                {
                    std::vector<std::int32_t> send(kCount, 1);
                    std::vector<std::int32_t> recv(kCount, 0);
                    bool threw = false;

                    try
                    {
                        all_reduce_pipelined(
                            world, send.data(), recv.data(), kCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 4);
                    }
                    catch (const std::exception &error)
                    {
                        threw = true;
                        expect(
                            std::string(error.what()).find(
                                "not divisible") != std::string::npos,
                            "wrong error for non-divisible count: " +
                                std::string(error.what()));
                    }

                    expect(
                        threw,
                        "all_reduce_pipelined should reject a "
                        "non-divisible count");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_divisibility_rejection\n";
    }

    void test_single_rank()
    {
        auto peers = make_local_peers(kSingleRankBase, 1);
        std::vector<std::exception_ptr> errors(1);
        std::vector<std::thread> threads;

        threads.emplace_back(
            run_rank, make_options(0, peers),
            [](tbccl::World &world)
            {
                std::vector<std::int32_t> send = {1, 2, 3, 4};
                std::vector<std::int32_t> recv(4, 0);

                all_reduce_pipelined(
                    world, send.data(), recv.data(), 4,
                    tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 4);

                expect(
                    recv == send,
                    "single-rank pipelined all_reduce should just copy "
                    "the contribution");
            },
            std::ref(errors[0]));

        join_and_check(threads, errors);

        std::cout << "[PASS] test_single_rank\n";
    }

    void test_exact_in_place_aliasing()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kCount = 6;

        auto peers = make_local_peers(kExactAliasBase, kSize);
        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    std::vector<std::int32_t> buffer(kCount);
                    for (std::size_t i = 0; i < kCount; ++i)
                    {
                        buffer[i] =
                            static_cast<std::int32_t>(rank + i * 10 + 1);
                    }

                    all_reduce_pipelined(
                        world, buffer.data(), buffer.data(), kCount,
                        tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 4);

                    const std::size_t size2 = world.size();

                    for (std::size_t i = 0; i < kCount; ++i)
                    {
                        std::int32_t expected_sum = 0;
                        for (std::size_t p = 0; p < size2; ++p)
                        {
                            expected_sum +=
                                static_cast<std::int32_t>(p + i * 10 + 1);
                        }

                        expect(
                            buffer[i] == expected_sum,
                            "in-place pipelined all_reduce mismatch at "
                            "element " +
                                std::to_string(i));
                    }
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_exact_in_place_aliasing\n";
    }

    void test_large_payload()
    {
        // 1 MiB tensor per rank, Float32.
        run_correctness<float>(
            kLargePayloadBase, 2, 262144, 256 * 1024,
            tbccl::DataType::Float32, tbccl::ReduceOp::Sum, 15000);
        std::cout << "[PASS] test_large_payload (size=2, 1 MiB tensor)\n";

        run_correctness<float>(
            static_cast<std::uint16_t>(kLargePayloadBase + 4), 4, 262144,
            1024 * 1024, tbccl::DataType::Float32, tbccl::ReduceOp::Sum,
            20000);
        std::cout << "[PASS] test_large_payload (size=4, 1 MiB tensor)\n";
    }

    void test_repeated_invocation()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 1000;
        constexpr std::size_t kCount = 3;
        constexpr std::size_t kChunkBytes = 4; // 1 element per chunk

        auto peers = make_local_peers(kRepeatedBase, kSize);
        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers, 30000),
                [rank](tbccl::World &world)
                {
                    const std::size_t size2 = world.size();

                    std::vector<std::int32_t> send(
                        kCount, static_cast<std::int32_t>(rank + 1));
                    std::vector<std::int32_t> recv(kCount, 0);

                    const auto expected = static_cast<std::int32_t>(
                        size2 * (size2 + 1) / 2);

                    for (int i = 0; i < kIterations; ++i)
                    {
                        all_reduce_pipelined(
                            world, send.data(), recv.data(), kCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum,
                            kChunkBytes);

                        for (std::size_t j = 0; j < kCount; ++j)
                        {
                            expect(
                                recv[j] == expected,
                                "repeated pipelined all_reduce mismatch, "
                                "iteration " +
                                    std::to_string(i));
                        }
                    }

                    const RingExecutorStats stats =
                        RingExecutorAccess::stats(world);
                    expect(
                        stats.worker_start_count == 1,
                        "one worker across 1000 repeated pipelined "
                        "all_reduce calls");
                    expect(
                        stats.submitted_jobs ==
                            static_cast<std::size_t>(kIterations) * 2,
                        "expected 2 submitted jobs per pipelined "
                        "all_reduce call (reduce-scatter phase + "
                        "all-gather phase), got " +
                            std::to_string(stats.submitted_jobs));
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_repeated_invocation (" << kIterations
                   << " iterations)\n";
    }

    void test_mixed_sequencing()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        constexpr std::size_t kCount = 3;

        auto peers = make_local_peers(kMixedSequencingBase, kSize);
        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers, 20000),
                [rank](tbccl::World &world)
                {
                    const std::size_t size2 = world.size();
                    std::vector<std::int32_t> send(
                        kCount, static_cast<std::int32_t>(rank + 1));

                    const auto expected = static_cast<std::int32_t>(
                        size2 * (size2 + 1) / 2);

                    std::size_t expected_jobs = 0;

                    for (int i = 0; i < kIterations; ++i)
                    {
                        std::vector<std::int32_t> recv_a(kCount, 0);
                        all_reduce_pipelined(
                            world, send.data(), recv_a.data(), kCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 4);
                        expected_jobs += 2;

                        std::vector<std::int32_t> recv_b(kCount, 0);
                        all_reduce_ring(
                            world, send.data(), recv_b.data(), kCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);
                        expected_jobs += 2;

                        std::vector<std::int32_t> recv_c(kCount, 0);
                        all_reduce_reference(
                            world, send.data(), recv_c.data(), kCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                        for (std::size_t j = 0; j < kCount; ++j)
                        {
                            expect(
                                recv_a[j] == expected && recv_b[j] == expected &&
                                    recv_c[j] == expected,
                                "mixed-sequencing output mismatch, "
                                "iteration " +
                                    std::to_string(i));
                        }

                        tbccl::barrier(world);
                    }

                    const RingExecutorStats stats =
                        RingExecutorAccess::stats(world);
                    expect(
                        stats.worker_start_count == 1,
                        "one worker across mixed pipelined/ring/reference "
                        "all_reduce sequencing");
                    expect(
                        stats.submitted_jobs == expected_jobs,
                        "submitted_jobs mismatch in mixed sequencing");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_mixed_sequencing (" << kIterations
                   << " iterations)\n";
    }

    void test_worker_reuse()
    {
        auto peers = make_local_peers(kWorkerReuseBase, 2);
        std::vector<std::exception_ptr> errors(2);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < 2; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers),
                [rank](tbccl::World &world)
                {
                    expect(
                        RingExecutorAccess::stats(world)
                                .worker_start_count == 0,
                        "no executor before the first pipelined call");

                    std::vector<std::int32_t> send(
                        4, static_cast<std::int32_t>(rank + 1));
                    std::vector<std::int32_t> recv(4, 0);

                    for (int i = 0; i < 3; ++i)
                    {
                        all_reduce_pipelined(
                            world, send.data(), recv.data(), 4,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 4);
                    }

                    const RingExecutorStats stats =
                        RingExecutorAccess::stats(world);
                    expect(
                        stats.worker_start_count == 1,
                        "one worker across 3 pipelined all_reduce calls");
                    expect(
                        stats.submitted_jobs == 6,
                        "6 submitted jobs (2 per call x 3 calls)");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_worker_reuse\n";
    }

} // namespace

int main()
{
    try
    {
        test_rank_counts();
        test_chunk_size_relationships();
        test_every_datatype();
        test_every_operation();
        test_invalid_chunk_size();
        test_divisibility_rejection();
        test_single_rank();
        test_exact_in_place_aliasing();
        test_large_payload();
        test_repeated_invocation();
        test_mixed_sequencing();
        test_worker_reuse();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
