#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "all_gather_internal.hpp"
#include "ring_executor.hpp"
#include "test_utils.hpp"

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

using tbccl::detail::all_gather_pipelined;
using tbccl::detail::all_gather_reference;
using tbccl::detail::all_gather_ring;
using tbccl::detail::RingExecutorAccess;
using tbccl::detail::RingExecutorStats;

namespace
{

    // Fixed port ranges, one block per test, kept below 32768 (see
    // algorithm_selector_test's port comment) and clear of every other
    // test file's range.
    constexpr std::uint16_t kRankCountsBase = 25000;
    constexpr std::uint16_t kChunkSizeBase = 25020;
    constexpr std::uint16_t kOddSizeBase = 25040;
    constexpr std::uint16_t kInvalidChunkBase = 25060;
    constexpr std::uint16_t kLargePayloadBase = 25070;
    constexpr std::uint16_t kRepeatedBase = 25080;
    constexpr std::uint16_t kMixedSequencingBase = 25090;
    constexpr std::uint16_t kWorkerReuseBase = 25100;
    constexpr std::uint16_t kSingleRankBase = 25110;

    // Bootstraps a World and runs reference, existing ring, and
    // pipelined all_gather, checking all three produce byte-identical
    // output.
    void run_correctness(
        std::uint16_t base_port,
        std::size_t size,
        std::size_t bytes_per_rank,
        std::size_t chunk_bytes,
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
                [rank, bytes_per_rank, chunk_bytes](tbccl::World &world)
                {
                    const std::size_t size2 = world.size();
                    const auto send = deterministic_buffer(
                        bytes_per_rank,
                        static_cast<std::uint32_t>(rank) + 0x1000u);
                    const std::size_t total = size2 * bytes_per_rank;

                    std::vector<std::uint8_t> reference_recv(total, 0);
                    all_gather_reference(
                        world, send.data(), reference_recv.data(),
                        bytes_per_rank);

                    std::vector<std::uint8_t> ring_recv(total, 0);
                    all_gather_ring(
                        world, send.data(), ring_recv.data(),
                        bytes_per_rank);

                    expect(
                        reference_recv == ring_recv,
                        "reference vs existing-ring mismatch on rank " +
                            std::to_string(rank));

                    std::vector<std::uint8_t> pipelined_recv(total, 0);
                    all_gather_pipelined(
                        world, send.data(), pipelined_recv.data(),
                        bytes_per_rank, chunk_bytes);

                    expect(
                        reference_recv == pipelined_recv,
                        "reference vs pipelined mismatch on rank " +
                            std::to_string(rank) + " (bytes_per_rank=" +
                            std::to_string(bytes_per_rank) +
                            ", chunk_bytes=" + std::to_string(chunk_bytes) +
                            ")");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    // -----------------------------------------------------------------------------
    // Rank counts: 1, 2, 3, 4, 5, 7, 8.
    // -----------------------------------------------------------------------------

    void test_rank_counts()
    {
        for (auto [size, offset] :
             {std::pair<std::size_t, std::uint16_t>{1, 0},
              {2, 8}, {3, 16}, {4, 24}, {5, 32}, {7, 40}, {8, 48}})
        {
            // 1000-byte segment, 256-byte chunks -> 4 chunks (256,
            // 256, 256, 232), exercising a non-power-of-two rank count
            // together with a non-evenly-dividing chunk size.
            run_correctness(
                static_cast<std::uint16_t>(kRankCountsBase + offset), size,
                1000, 256);

            std::cout << "[PASS] test_rank_counts (size=" << size << ")\n";
        }
    }

    // -----------------------------------------------------------------------------
    // Chunk-size relationships: larger than / equal to / smaller than
    // the segment; 1/2/3/7 chunks.
    // -----------------------------------------------------------------------------

    void test_chunk_size_relationships()
    {
        constexpr std::size_t kSize = 3;
        constexpr std::size_t kSegment = 4096;

        struct Case
        {
            std::size_t chunk_bytes;
            const char *label;
        };

        std::uint16_t offset = 0;

        for (Case c :
             {Case{8192, "chunk > segment (1 chunk)"},
              Case{4096, "chunk == segment (1 chunk)"},
              Case{2048, "2 chunks"}, Case{1366, "3 chunks (uneven)"},
              Case{586, "7 chunks (uneven)"}, Case{1, "many tiny chunks"}})
        {
            run_correctness(
                static_cast<std::uint16_t>(kChunkSizeBase + offset), kSize,
                kSegment, c.chunk_bytes);

            std::cout << "[PASS] test_chunk_size_relationships (" << c.label
                       << ", chunk_bytes=" << c.chunk_bytes << ")\n";

            offset += 2;
        }
    }

    // -----------------------------------------------------------------------------
    // Odd byte sizes.
    // -----------------------------------------------------------------------------

    void test_odd_sizes()
    {
        constexpr std::size_t kSize = 4;

        std::uint16_t offset = 0;

        for (std::size_t bytes_per_rank : {1, 3, 7, 257, 1003})
        {
            // Chunk size deliberately does not evenly divide most of
            // these.
            run_correctness(
                static_cast<std::uint16_t>(kOddSizeBase + offset), kSize,
                bytes_per_rank, 3);

            std::cout << "[PASS] test_odd_sizes (bytes_per_rank="
                       << bytes_per_rank << ")\n";

            offset += 2;
        }
    }

    // -----------------------------------------------------------------------------
    // Invalid chunk-size rejection — before any communication, so no
    // rank strands its peers.
    // -----------------------------------------------------------------------------

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
                    std::vector<std::uint8_t> send(64, 1);
                    std::vector<std::uint8_t> recv(64 * 3, 0);
                    bool threw = false;

                    try
                    {
                        all_gather_pipelined(
                            world, send.data(), recv.data(), 64, 0);
                    }
                    catch (const std::exception &error)
                    {
                        threw = true;
                        expect(
                            std::string(error.what())
                                    .find("chunk_bytes must be") !=
                                std::string::npos,
                            "wrong error for zero chunk_bytes: " +
                                std::string(error.what()));
                    }

                    expect(
                        threw,
                        "all_gather_pipelined should reject chunk_bytes "
                        "== 0 for a nonzero payload");

                    // Zero-payload + zero-chunk-size must remain valid
                    // (the zero-size fast path returns before chunk
                    // validation ever runs) and must not create a ring
                    // executor.
                    all_gather_pipelined(
                        world, nullptr, nullptr, 0, 0);

                    expect(
                        RingExecutorAccess::stats(world)
                                .worker_start_count == 0,
                        "zero-size pipelined all_gather must not create "
                        "a ring executor");
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_invalid_chunk_size\n";
    }

    // -----------------------------------------------------------------------------
    // Single rank: must short-circuit without creating an executor.
    // -----------------------------------------------------------------------------

    void test_single_rank()
    {
        auto peers = make_local_peers(kSingleRankBase, 1);
        std::vector<std::exception_ptr> errors(1);
        std::vector<std::thread> threads;

        threads.emplace_back(
            run_rank, make_options(0, peers),
            [](tbccl::World &world)
            {
                std::vector<std::uint8_t> send(64, 9);
                std::vector<std::uint8_t> recv(64, 0);

                all_gather_pipelined(
                    world, send.data(), recv.data(), 64, 16);

                expect(recv == send, "single-rank pipelined all_gather "
                                      "should just copy the contribution");
                expect(
                    RingExecutorAccess::stats(world).worker_start_count ==
                        0,
                    "single-rank pipelined all_gather must not create a "
                    "ring executor");
            },
            std::ref(errors[0]));

        join_and_check(threads, errors);

        std::cout << "[PASS] test_single_rank\n";
    }

    // -----------------------------------------------------------------------------
    // Large payload.
    // -----------------------------------------------------------------------------

    void test_large_payload()
    {
        run_correctness(kLargePayloadBase, 2, 1u << 20, 256u * 1024, 15000);
        std::cout << "[PASS] test_large_payload (size=2, 1 MiB)\n";

        run_correctness(
            static_cast<std::uint16_t>(kLargePayloadBase + 4), 3, 4u << 20,
            1024u * 1024, 20000);
        std::cout << "[PASS] test_large_payload (size=3, 4 MiB)\n";
    }

    // -----------------------------------------------------------------------------
    // Repeated invocation: 1000 iterations, changing contents,
    // verifying worker reuse.
    // -----------------------------------------------------------------------------

    void test_repeated_invocation()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 1000;
        constexpr std::size_t kBytesPerRank = 64;
        constexpr std::size_t kChunkBytes = 17; // deliberately uneven

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
                    const std::size_t total = size2 * kBytesPerRank;

                    for (int i = 0; i < kIterations; ++i)
                    {
                        const auto send = deterministic_buffer(
                            kBytesPerRank,
                            static_cast<std::uint32_t>(rank * 1000 + i));

                        std::vector<std::uint8_t> recv(total, 0);
                        all_gather_pipelined(
                            world, send.data(), recv.data(), kBytesPerRank,
                            kChunkBytes);

                        for (std::size_t r = 0; r < size2; ++r)
                        {
                            const auto expected = deterministic_buffer(
                                kBytesPerRank,
                                static_cast<std::uint32_t>(r * 1000 + i));

                            expect(
                                std::memcmp(
                                    recv.data() + r * kBytesPerRank,
                                    expected.data(), kBytesPerRank) == 0,
                                "repeated pipelined all_gather mismatch, "
                                "iteration " +
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

    // -----------------------------------------------------------------------------
    // Mixed sequencing: pipelined interleaved with existing ring and
    // reference, on the same World, verifying no stale session state
    // and worker reuse.
    // -----------------------------------------------------------------------------

    void test_mixed_sequencing()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 100;
        constexpr std::size_t kBytesPerRank = 32;

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
                    const std::size_t total = size2 * kBytesPerRank;
                    const auto send = deterministic_buffer(
                        kBytesPerRank, static_cast<std::uint32_t>(rank));

                    std::size_t expected_jobs = 0;

                    for (int i = 0; i < kIterations; ++i)
                    {
                        std::vector<std::uint8_t> recv_a(total, 0);
                        all_gather_pipelined(
                            world, send.data(), recv_a.data(),
                            kBytesPerRank, 5);
                        ++expected_jobs;

                        std::vector<std::uint8_t> recv_b(total, 0);
                        all_gather_ring(
                            world, send.data(), recv_b.data(),
                            kBytesPerRank);
                        ++expected_jobs;

                        std::vector<std::uint8_t> recv_c(total, 0);
                        all_gather_reference(
                            world, send.data(), recv_c.data(),
                            kBytesPerRank);
                        // reference does not touch the ring executor.

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
                        "submitted_jobs mismatch in mixed sequencing: "
                        "expected " +
                            std::to_string(expected_jobs) + ", got " +
                            std::to_string(stats.submitted_jobs));
                },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);

        std::cout << "[PASS] test_mixed_sequencing (" << kIterations
                   << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Persistent worker reuse, checked directly (not just inferred
    // from the repeated/mixed tests above).
    // -----------------------------------------------------------------------------

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

                    const auto send = deterministic_buffer(
                        64, static_cast<std::uint32_t>(rank));
                    std::vector<std::uint8_t> recv(128, 0);

                    all_gather_pipelined(
                        world, send.data(), recv.data(), 64, 16);
                    all_gather_pipelined(
                        world, send.data(), recv.data(), 64, 16);
                    all_gather_pipelined(
                        world, send.data(), recv.data(), 64, 16);

                    const RingExecutorStats stats =
                        RingExecutorAccess::stats(world);
                    expect(
                        stats.worker_start_count == 1,
                        "one worker across 3 pipelined calls");
                    expect(
                        stats.submitted_jobs == 3,
                        "three submitted jobs (one send job per pipelined "
                        "call — chunking happens within that one job, not "
                        "as separate submissions)");
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
        test_odd_sizes();
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
