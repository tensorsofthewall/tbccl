#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "ring_executor.hpp"
#include "reduce_scatter_internal.hpp"
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

using tbccl::detail::reduce_scatter_pipelined;
using tbccl::detail::reduce_scatter_reference;
using tbccl::detail::reduce_scatter_ring;
using tbccl::detail::RingExecutorAccess;
using tbccl::detail::RingExecutorStats;

namespace
{

    constexpr std::uint16_t kRankCountsBase = 25200;
    constexpr std::uint16_t kChunkSizeBase = 25220;
    constexpr std::uint16_t kEveryDatatypeBase = 25240;
    constexpr std::uint16_t kEveryOperationBase = 25260;
    constexpr std::uint16_t kInvalidChunkBase = 25280;
    constexpr std::uint16_t kSingleRankBase = 25290;
    constexpr std::uint16_t kLargePayloadBase = 25300;
    constexpr std::uint16_t kRepeatedBase = 25310;
    constexpr std::uint16_t kMixedSequencingBase = 25320;
    constexpr std::uint16_t kWorkerReuseBase = 25330;

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

    // Bootstraps a World and compares reference, existing ring, and
    // pipelined reduce_scatter against an independently-computed
    // expected segment.
    template <typename T>
    void run_correctness(
        std::uint16_t base_port,
        std::size_t size,
        std::size_t recv_count,
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
                [rank, recv_count, chunk_bytes, datatype,
                 op](tbccl::World &world)
                {
                    constexpr std::uint32_t kSeed = 0x5678u;
                    const std::size_t size2 = world.size();
                    const std::size_t total_count = size2 * recv_count;

                    std::vector<T> send(total_count);
                    for (std::size_t i = 0; i < total_count; ++i)
                    {
                        send[i] = generate_value<T>(rank, i, kSeed);
                    }

                    std::vector<T> expected(recv_count);
                    for (std::size_t i = 0; i < recv_count; ++i)
                    {
                        const std::size_t global_index =
                            rank * recv_count + i;
                        T acc = generate_value<T>(0, global_index, kSeed);

                        for (std::size_t p = 1; p < size2; ++p)
                        {
                            combine(
                                acc, generate_value<T>(p, global_index, kSeed),
                                op);
                        }

                        expected[i] = acc;
                    }

                    std::vector<T> reference_recv(recv_count, T{});
                    reduce_scatter_reference(
                        world, send.data(), reference_recv.data(),
                        recv_count, datatype, op);

                    std::vector<T> ring_recv(recv_count, T{});
                    reduce_scatter_ring(
                        world, send.data(), ring_recv.data(), recv_count,
                        datatype, op);

                    std::vector<T> pipelined_recv(recv_count, T{});
                    reduce_scatter_pipelined(
                        world, send.data(), pipelined_recv.data(),
                        recv_count, datatype, op, chunk_bytes);

                    for (std::size_t i = 0; i < recv_count; ++i)
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
            const std::size_t recv_count = 37; // non-power-of-two

            run_correctness<std::int32_t>(
                static_cast<std::uint16_t>(kRankCountsBase + offset), size,
                recv_count, 17 * sizeof(std::int32_t),
                tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

            std::cout << "[PASS] test_rank_counts (size=" << size << ")\n";
        }
    }

    void test_chunk_size_relationships()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kRecvCount = 1024; // Float32 -> 4096 B segment

        struct Case
        {
            std::size_t chunk_bytes;
            const char *label;
        };

        std::uint16_t offset = 0;

        for (Case c :
             {Case{8192, "chunk > segment (1 chunk)"},
              Case{4096, "chunk == segment (1 chunk)"},
              Case{2048, "2 chunks"}, Case{1024, "4 chunks"},
              Case{4, "many single-element chunks"}})
        {
            run_correctness<float>(
                static_cast<std::uint16_t>(kChunkSizeBase + offset), kSize,
                kRecvCount, c.chunk_bytes, tbccl::DataType::Float32,
                tbccl::ReduceOp::Sum);

            std::cout << "[PASS] test_chunk_size_relationships (" << c.label
                       << ", chunk_bytes=" << c.chunk_bytes << ")\n";

            offset += 2;
        }

        // 7 chunks, uneven: 1024 * 4 = 4096 B / 600 B -> ceil = 7
        // chunks, last one shorter.
        run_correctness<float>(
            static_cast<std::uint16_t>(kChunkSizeBase + offset), kSize,
            kRecvCount, 600, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
        std::cout << "[PASS] test_chunk_size_relationships (7 chunks "
                     "(uneven), chunk_bytes=600)\n";
    }

    void test_every_datatype()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kRecvCount = 257; // non-power-of-two

        std::uint16_t offset = 0;

        run_correctness<std::int32_t>(
            static_cast<std::uint16_t>(kEveryDatatypeBase + offset), kSize,
            kRecvCount, 37 * sizeof(std::int32_t), tbccl::DataType::Int32,
            tbccl::ReduceOp::Sum);
        offset += 4;

        run_correctness<std::int64_t>(
            static_cast<std::uint16_t>(kEveryDatatypeBase + offset), kSize,
            kRecvCount, 37 * sizeof(std::int64_t), tbccl::DataType::Int64,
            tbccl::ReduceOp::Sum);
        offset += 4;

        run_correctness<float>(
            static_cast<std::uint16_t>(kEveryDatatypeBase + offset), kSize,
            kRecvCount, 37 * sizeof(float), tbccl::DataType::Float32,
            tbccl::ReduceOp::Sum);
        offset += 4;

        run_correctness<double>(
            static_cast<std::uint16_t>(kEveryDatatypeBase + offset), kSize,
            kRecvCount, 37 * sizeof(double), tbccl::DataType::Float64,
            tbccl::ReduceOp::Sum);

        std::cout << "[PASS] test_every_datatype\n";
    }

    void test_every_operation()
    {
        constexpr std::size_t kSize = 4;
        constexpr std::size_t kRecvCount = 129;

        std::uint16_t offset = 0;

        for (tbccl::ReduceOp op :
             {tbccl::ReduceOp::Sum, tbccl::ReduceOp::Product,
              tbccl::ReduceOp::Min, tbccl::ReduceOp::Max})
        {
            run_correctness<std::int32_t>(
                static_cast<std::uint16_t>(kEveryOperationBase + offset),
                kSize, kRecvCount, 23 * sizeof(std::int32_t),
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
                    std::vector<std::int32_t> send(3 * 16, 1);
                    std::vector<std::int32_t> recv(16, 0);

                    // chunk_bytes == 0.
                    bool threw = false;
                    try
                    {
                        reduce_scatter_pipelined(
                            world, send.data(), recv.data(), 16,
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

                    // Misaligned chunk_bytes (not a multiple of
                    // sizeof(int32_t) == 4).
                    threw = false;
                    try
                    {
                        reduce_scatter_pipelined(
                            world, send.data(), recv.data(), 16,
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

                    // recv_count == 0 must remain valid regardless of
                    // chunk_bytes and must not create an executor.
                    std::int32_t dummy = 0;
                    reduce_scatter_pipelined(
                        world, &dummy, &dummy, 0, tbccl::DataType::Int32,
                        tbccl::ReduceOp::Sum, 0);

                    expect(
                        RingExecutorAccess::stats(world)
                                .worker_start_count == 0,
                        "zero-count pipelined reduce_scatter must not "
                        "create a ring executor");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_invalid_chunk_size\n";
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

                reduce_scatter_pipelined(
                    world, send.data(), recv.data(), 4,
                    tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 4);

                expect(
                    recv == send,
                    "single-rank pipelined reduce_scatter should just "
                    "copy the contribution");
                expect(
                    RingExecutorAccess::stats(world).worker_start_count ==
                        0,
                    "single-rank pipelined reduce_scatter must not "
                    "create a ring executor");
            },
            std::ref(errors[0]));

        join_and_check(threads, errors);

        std::cout << "[PASS] test_single_rank\n";
    }

    void test_large_payload()
    {
        // 1 MiB output segment, Float32 -> 262144 elements.
        run_correctness<float>(
            kLargePayloadBase, 2, 262144, 256 * 1024,
            tbccl::DataType::Float32, tbccl::ReduceOp::Sum, 15000);
        std::cout << "[PASS] test_large_payload (size=2, 1 MiB output)\n";

        run_correctness<float>(
            static_cast<std::uint16_t>(kLargePayloadBase + 4), 4, 262144,
            1024 * 1024, tbccl::DataType::Float32, tbccl::ReduceOp::Sum,
            20000);
        std::cout << "[PASS] test_large_payload (size=4, 1 MiB output)\n";
    }

    void test_repeated_invocation()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 1000;
        constexpr std::size_t kRecvCount = 16;
        constexpr std::size_t kChunkBytes = 9 * sizeof(std::int32_t);

        auto peers = make_local_peers(kRepeatedBase, kSize);
        std::vector<std::exception_ptr> errors(kSize);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < kSize; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers, 20000),
                [rank](tbccl::World &world)
                {
                    const std::size_t size2 = world.size();
                    const std::size_t total_count = size2 * kRecvCount;

                    for (int i = 0; i < kIterations; ++i)
                    {
                        std::vector<std::int32_t> send(total_count);
                        for (std::size_t j = 0; j < total_count; ++j)
                        {
                            send[j] = static_cast<std::int32_t>(
                                (rank + 1) + (i % 5));
                        }

                        std::vector<std::int32_t> recv(kRecvCount, 0);
                        reduce_scatter_pipelined(
                            world, send.data(), recv.data(), kRecvCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum,
                            kChunkBytes);

                        std::int32_t expected_sum = 0;
                        for (std::size_t p = 0; p < size2; ++p)
                        {
                            expected_sum += static_cast<std::int32_t>(
                                (p + 1) + (i % 5));
                        }

                        for (std::size_t j = 0; j < kRecvCount; ++j)
                        {
                            expect(
                                recv[j] == expected_sum,
                                "repeated pipelined reduce_scatter "
                                "mismatch, iteration " +
                                    std::to_string(i));
                        }
                    }

                    const RingExecutorStats stats =
                        RingExecutorAccess::stats(world);
                    expect(
                        stats.worker_start_count == 1,
                        "one worker across 1000 repeated pipelined calls");
                    expect(
                        stats.submitted_jobs ==
                            static_cast<std::size_t>(kIterations),
                        "expected " + std::to_string(kIterations) +
                            " submitted jobs, got " +
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
        constexpr std::size_t kRecvCount = 8;

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
                    const std::size_t total_count = size2 * kRecvCount;
                    std::vector<std::int32_t> send(
                        total_count, static_cast<std::int32_t>(rank + 1));

                    std::size_t expected_jobs = 0;

                    for (int i = 0; i < kIterations; ++i)
                    {
                        std::vector<std::int32_t> recv_a(kRecvCount, 0);
                        reduce_scatter_pipelined(
                            world, send.data(), recv_a.data(), kRecvCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 8);
                        ++expected_jobs;

                        std::vector<std::int32_t> recv_b(kRecvCount, 0);
                        reduce_scatter_ring(
                            world, send.data(), recv_b.data(), kRecvCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);
                        ++expected_jobs;

                        std::vector<std::int32_t> recv_c(kRecvCount, 0);
                        reduce_scatter_reference(
                            world, send.data(), recv_c.data(), kRecvCount,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                        expect(
                            recv_a == recv_b && recv_b == recv_c,
                            "mixed-sequencing output mismatch, iteration " +
                                std::to_string(i));

                        tbccl::barrier(world);
                    }

                    const RingExecutorStats stats =
                        RingExecutorAccess::stats(world);
                    expect(
                        stats.worker_start_count == 1,
                        "one worker across mixed pipelined/ring/reference "
                        "sequencing");
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
                        16, static_cast<std::int32_t>(rank + 1));
                    std::vector<std::int32_t> recv(8, 0);

                    for (int i = 0; i < 3; ++i)
                    {
                        reduce_scatter_pipelined(
                            world, send.data(), recv.data(), 8,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum, 4);
                    }

                    const RingExecutorStats stats =
                        RingExecutorAccess::stats(world);
                    expect(
                        stats.worker_start_count == 1,
                        "one worker across 3 pipelined calls");
                    expect(
                        stats.submitted_jobs == 3,
                        "three submitted jobs");
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
        test_single_rank();
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
