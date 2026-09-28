// Tests for the chunk arithmetic and chunk-level progress tracking in
// ring_pipeline_session.hpp, independent of any actual collective or
// networking — see all_gather_pipelined_test.cpp /
// reduce_scatter_pipelined_test.cpp / all_reduce_pipelined_test.cpp
// for end-to-end pipelined-collective correctness.

#include "ring_pipeline_session.hpp"
#include "test_utils.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

using tbccl_test::expect;

namespace
{

    // -------------------------------------------------------------------
    // Chunk arithmetic.
    // -------------------------------------------------------------------

    void test_compute_chunk_count()
    {
        using tbccl::detail::compute_chunk_count;

        expect(compute_chunk_count(0, 64) == 0, "zero total -> zero chunks");
        expect(
            compute_chunk_count(1048576, 262144) == 4,
            "1 MiB / 256 KiB -> 4 chunks (exact division)");
        expect(
            compute_chunk_count(1000, 256) == 4,
            "1000 / 256 -> 4 chunks (inexact division, per the plan's "
            "worked example)");
        expect(
            compute_chunk_count(100, 1000) == 1,
            "chunk_bytes > total_bytes -> 1 chunk (no separate clamping "
            "needed)");
        expect(
            compute_chunk_count(1000, 1000) == 1,
            "chunk_bytes == total_bytes -> 1 chunk");
        expect(
            compute_chunk_count(1000, 999) == 2,
            "just-over-half chunk_bytes -> 2 chunks");
        expect(compute_chunk_count(7, 1) == 7, "1-byte chunks -> 7 chunks");

        std::cout << "[PASS] test_compute_chunk_count\n";
    }

    void test_chunk_length()
    {
        using tbccl::detail::chunk_length;

        // 1000 bytes split into 256-byte chunks: 256, 256, 256, 232.
        expect(chunk_length(1000, 256, 0) == 256, "chunk 0 length");
        expect(chunk_length(1000, 256, 1) == 256, "chunk 1 length");
        expect(chunk_length(1000, 256, 2) == 256, "chunk 2 length");
        expect(
            chunk_length(1000, 256, 3) == 232,
            "final (shorter) chunk length");

        // Exact division: every chunk is full-sized.
        expect(chunk_length(1048576, 262144, 0) == 262144, "exact chunk 0");
        expect(chunk_length(1048576, 262144, 3) == 262144, "exact chunk 3");

        // chunk_bytes > total_bytes: the single chunk is the whole
        // region.
        expect(
            chunk_length(100, 1000, 0) == 100,
            "oversized chunk_bytes -> chunk covers the whole region");

        std::cout << "[PASS] test_chunk_length\n";
    }

    void test_validate_chunk_bytes()
    {
        using tbccl::detail::validate_chunk_bytes;

        // AllGather-style: element_size == 1, any positive size ok.
        try
        {
            validate_chunk_bytes("ctx", 1, 1);
            validate_chunk_bytes("ctx", 3, 1);
            validate_chunk_bytes("ctx", 1048576, 1);
        }
        catch (const std::exception &error)
        {
            expect(false, std::string("unexpected throw: ") + error.what());
        }

        bool threw = false;
        try
        {
            validate_chunk_bytes("all_gather_pipelined", 0, 1);
        }
        catch (const std::exception &error)
        {
            threw = true;
            expect(
                std::string(error.what()).find("chunk_bytes must be") !=
                    std::string::npos,
                "wrong message for zero chunk_bytes");
        }
        expect(threw, "chunk_bytes == 0 should be rejected");

        // ReduceScatter-style: element_size > 1, must divide evenly.
        try
        {
            validate_chunk_bytes("reduce_scatter_pipelined", 4, 4);
            validate_chunk_bytes("reduce_scatter_pipelined", 4096, 4);
            validate_chunk_bytes("reduce_scatter_pipelined", 8, 8);
        }
        catch (const std::exception &error)
        {
            expect(
                false,
                std::string("unexpected throw for aligned size: ") +
                    error.what());
        }

        threw = false;
        try
        {
            validate_chunk_bytes("reduce_scatter_pipelined", 6, 4);
        }
        catch (const std::exception &error)
        {
            threw = true;
            expect(
                std::string(error.what()).find("multiple of the datatype "
                                                 "size") != std::string::npos,
                "wrong message for misaligned chunk_bytes: " +
                    std::string(error.what()));
        }
        expect(threw, "chunk_bytes not a multiple of element_size should "
                       "be rejected");

        std::cout << "[PASS] test_validate_chunk_bytes\n";
    }

    // -------------------------------------------------------------------
    // RingPipelineSession synchronization.
    // -------------------------------------------------------------------

    void test_session_chunk_progress()
    {
        using tbccl::detail::RingPipelineSession;

        RingPipelineSession session(2); // 2 steps

        std::atomic<bool> unblocked{false};
        std::atomic<bool> result{false};

        std::thread waiter(
            [&]()
            {
                result = session.wait_for_chunks(0, 3);
                unblocked = true;
            });

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        expect(
            !unblocked.load(),
            "waiter should still be blocked before 3 chunks of step 0 "
            "complete");

        session.mark_chunk_complete(0); // chunk 0 of step 0
        session.mark_chunk_complete(0); // chunk 1 of step 0
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        expect(
            !unblocked.load(),
            "waiter should still be blocked after only 2 of 3 chunks");

        session.mark_chunk_complete(0); // chunk 2 of step 0
        waiter.join();

        expect(unblocked.load(), "waiter should unblock after 3 chunks");
        expect(result.load(), "wait_for_chunks should return true on "
                               "normal completion");

        std::cout << "[PASS] test_session_chunk_progress\n";
    }

    void test_session_steps_are_independent()
    {
        using tbccl::detail::RingPipelineSession;

        RingPipelineSession session(2);

        session.mark_chunk_complete(1); // only step 1 progresses
        session.mark_chunk_complete(1);

        // Step 1 has 2 completions, so a wait for 2 must return
        // immediately (bounded by the join below, never truly
        // blocking).
        std::thread step1_waiter(
            [&]() { expect(
                        session.wait_for_chunks(1, 2),
                        "step 1 should already have 2 completions"); });
        step1_waiter.join();

        // Step 0 has zero completions and is never marked in this
        // test, so a real (unbounded) wait on it would hang forever —
        // instead, prove its counter is still untouched by step 1's
        // progress via a bounded check: a waiter for step 0 must still
        // be blocked after a short sleep, then get released once step
        // 0 (and only step 0) is marked complete.
        std::atomic<bool> step0_unblocked{false};
        std::thread step0_waiter(
            [&]()
            {
                session.wait_for_chunks(0, 1);
                step0_unblocked = true;
            });

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        expect(
            !step0_unblocked.load(),
            "step 0 must still be blocked — step 1's progress must not "
            "leak into step 0's counter");

        session.mark_chunk_complete(0);
        step0_waiter.join();
        expect(
            step0_unblocked.load(),
            "step 0 should unblock once step 0 itself is marked complete");

        std::cout << "[PASS] test_session_steps_are_independent\n";
    }

    void test_session_failure_unblocks_waiter()
    {
        using tbccl::detail::RingPipelineSession;

        RingPipelineSession session(1);
        std::atomic<bool> result{true};

        std::thread waiter(
            [&]() { result = session.wait_for_chunks(0, 100); });

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        session.report_failure();
        waiter.join();

        expect(
            !result.load(),
            "wait_for_chunks should return false once report_failure() "
            "has been called");

        std::cout << "[PASS] test_session_failure_unblocks_waiter\n";
    }

    void test_session_multiple_chunks_per_step()
    {
        using tbccl::detail::RingPipelineSession;

        // Simulates the real usage pattern: a sender waiting on
        // successive chunks of the same step, one at a time, as the
        // receiver completes them in order.
        RingPipelineSession session(1);

        std::thread receiver(
            [&]()
            {
                for (int i = 0; i < 5; ++i)
                {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(5));
                    session.mark_chunk_complete(0);
                }
            });

        for (std::size_t required = 1; required <= 5; ++required)
        {
            const bool ok = session.wait_for_chunks(0, required);
            expect(ok, "wait_for_chunks should succeed for chunk " +
                           std::to_string(required));
        }

        receiver.join();

        std::cout << "[PASS] test_session_multiple_chunks_per_step\n";
    }

} // namespace

int main()
{
    try
    {
        test_compute_chunk_count();
        test_chunk_length();
        test_validate_chunk_bytes();
        test_session_chunk_progress();
        test_session_steps_are_independent();
        test_session_failure_unblocks_waiter();
        test_session_multiple_chunks_per_step();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
