// Tests for the persistent ring execution infrastructure
// (ring_executor.hpp) introduced in Phase 13, independent from the
// large existing collective correctness suites
// (all_gather_ring_test.cpp / reduce_scatter_ring_test.cpp /
// all_reduce_ring_test.cpp, all left untouched — see CMakeLists.txt).
//
// Part 1 exercises RingExecutor/RingSession directly, with no
// networking at all, for fast and deterministic coverage of job
// execution/ordering, exception propagation, and lifecycle.
//
// Part 2 exercises the World-level wiring (lazy construction, worker
// reuse across many/mixed ring collectives, reference-vs-ring
// transitions, forced algorithms, zero-count/single-rank fast paths,
// World destruction/recreation, and independent Worlds) over real
// local TCP loopback Worlds, using RingExecutorAccess::stats() as
// test-only observability rather than comparing OS thread IDs (which
// may be reused after a thread terminates).

#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "all_gather_internal.hpp"
#include "all_reduce_internal.hpp"
#include "reduce_scatter_internal.hpp"
#include "ring_executor.hpp"
#include "test_utils.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <functional>
#include <iostream>
#include <mutex>
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

using tbccl::detail::all_gather_reference;
using tbccl::detail::all_gather_ring;
using tbccl::detail::all_reduce_reference;
using tbccl::detail::all_reduce_ring;
using tbccl::detail::reduce_scatter_reference;
using tbccl::detail::reduce_scatter_ring;
using tbccl::detail::RingExecutor;
using tbccl::detail::RingExecutorAccess;
using tbccl::detail::RingExecutorStats;
using tbccl::detail::RingSession;

namespace
{

    // Fixed port ranges, one block per distributed test, kept below
    // 32768 (the default Linux ephemeral port range floor) and clear
    // of every other test file's range — see algorithm_selector_test's
    // port comment for why that matters.
    constexpr std::uint16_t kLazyWorldBase = 24000;
    constexpr std::uint16_t kZeroCountBase = 24010;
    constexpr std::uint16_t kSingleRankBase = 24020;
    constexpr std::uint16_t kMixedReuseBase = 24030;
    constexpr std::uint16_t kRefToRingBase = 24040;
    constexpr std::uint16_t kForcedAlgoBase = 24050;
    constexpr std::uint16_t kRepeatedAllGather3Base = 24060;
    constexpr std::uint16_t kRepeatedAllGather5Base = 24070;
    constexpr std::uint16_t kRepeatedAllGather8Base = 24080;
    constexpr std::uint16_t kRepeatedReduceScatter3Base = 24090;
    constexpr std::uint16_t kRepeatedReduceScatter5Base = 24100;
    constexpr std::uint16_t kRepeatedReduceScatter8Base = 24110;
    constexpr std::uint16_t kRepeatedAllReduceBase = 24120;
    constexpr std::array<std::uint16_t, 4> kWorldDestructionBases = {
        24130, 24140, 24150, 24160};
    constexpr std::uint16_t kIndependentWorldABase = 24200;
    constexpr std::uint16_t kIndependentWorldBBase = 24210;

    // -------------------------------------------------------------------
    // Generic multi-rank runner, shared by every distributed test below.
    // -------------------------------------------------------------------

    void run_on_ranks(
        std::uint16_t base_port,
        std::size_t size,
        const std::function<void(tbccl::World &, std::size_t)> &body,
        int timeout_ms = 5000)
    {
        auto peers = make_local_peers(base_port, size);
        std::vector<std::exception_ptr> errors(size);
        std::vector<std::thread> threads;

        for (std::size_t rank = 0; rank < size; ++rank)
        {
            threads.emplace_back(
                run_rank, make_options(rank, peers, timeout_ms),
                [rank, &body](tbccl::World &world) { body(world, rank); },
                std::ref(errors[rank]));
        }

        join_and_check(threads, errors);
    }

    // Small RAII environment-variable override, scoped to this file
    // (mirrors algorithm_selector_test.cpp's, kept local rather than
    // shared to avoid coupling these two test binaries together).
    class EnvOverride
    {
    public:
        EnvOverride(std::string name, const std::string &value)
            : name_(std::move(name))
        {
            const char *existing = std::getenv(name_.c_str());
            had_previous_ = existing != nullptr;

            if (had_previous_)
            {
                previous_ = existing;
            }

            apply(value);
        }

        ~EnvOverride()
        {
            if (had_previous_)
            {
                apply(previous_);
            }
            else
            {
#if defined(_WIN32)
                _putenv_s(name_.c_str(), "");
#else
                unsetenv(name_.c_str());
#endif
            }
        }

        EnvOverride(const EnvOverride &) = delete;
        EnvOverride &operator=(const EnvOverride &) = delete;

    private:
        void apply(const std::string &value)
        {
#if defined(_WIN32)
            _putenv_s(name_.c_str(), value.c_str());
#else
            setenv(name_.c_str(), value.c_str(), 1);
#endif
        }

        std::string name_;
        std::string previous_;
        bool had_previous_ = false;
    };

    // ===================================================================
    // Part 1: standalone RingExecutor / RingSession tests (no
    // networking).
    // ===================================================================

    void test_first_job_executes()
    {
        RingExecutor executor;
        bool sender_ran = false;
        bool receiver_ran = false;

        executor.execute(
            [&]() { sender_ran = true; }, [&]() { receiver_ran = true; });

        expect(sender_ran, "sender should have run");
        expect(receiver_ran, "receiver should have run");

        const RingExecutorStats stats = executor.stats();
        expect(
            stats.worker_start_count == 1,
            "worker_start_count should be 1 after the first job");
        expect(stats.submitted_jobs == 1, "submitted_jobs should be 1");
        expect(stats.completed_jobs == 1, "completed_jobs should be 1");

        std::cout << "[PASS] test_first_job_executes\n";
    }

    void test_jobs_execute_in_order()
    {
        RingExecutor executor;
        std::vector<int> order;
        std::mutex order_mutex;

        constexpr int kJobs = 20;

        for (int i = 0; i < kJobs; ++i)
        {
            executor.execute(
                [&, i]()
                {
                    std::lock_guard<std::mutex> lock(order_mutex);
                    order.push_back(i);
                },
                []() {});
        }

        expect(
            order.size() == static_cast<std::size_t>(kJobs),
            "expected " + std::to_string(kJobs) + " recorded executions");

        for (int i = 0; i < kJobs; ++i)
        {
            expect(
                order[static_cast<std::size_t>(i)] == i,
                "jobs must execute in submission order");
        }

        const RingExecutorStats stats = executor.stats();
        expect(
            stats.worker_start_count == 1,
            "one worker should back the whole sequence");
        expect(
            stats.submitted_jobs == static_cast<std::size_t>(kJobs),
            "submitted_jobs mismatch");
        expect(
            stats.completed_jobs == static_cast<std::size_t>(kJobs),
            "completed_jobs mismatch");

        std::cout << "[PASS] test_jobs_execute_in_order\n";
    }

    void test_exception_from_sender_propagates()
    {
        RingExecutor executor;
        bool threw = false;

        try
        {
            executor.execute(
                []() { throw std::runtime_error("sender failure"); },
                []() {});
        }
        catch (const std::exception &error)
        {
            threw = true;
            expect(
                std::string(error.what()) == "sender failure",
                "wrong exception propagated from sender: " +
                    std::string(error.what()));
        }

        expect(threw, "sender exception should propagate to the caller");

        bool ran = false;
        executor.execute([&]() { ran = true; }, []() {});
        expect(
            ran,
            "the executor must remain usable for a later job after a "
            "sender failure");

        std::cout << "[PASS] test_exception_from_sender_propagates\n";
    }

    void test_exception_from_receiver_propagates()
    {
        RingExecutor executor;
        bool threw = false;
        bool sender_ran = false;

        try
        {
            executor.execute(
                [&]() { sender_ran = true; },
                []() { throw std::runtime_error("receiver failure"); });
        }
        catch (const std::exception &error)
        {
            threw = true;
            expect(
                std::string(error.what()) == "receiver failure",
                "wrong exception propagated from receiver: " +
                    std::string(error.what()));
        }

        expect(threw, "receiver exception should propagate to the caller");
        expect(
            sender_ran,
            "sender should still have run concurrently with the failing "
            "receiver");

        bool ran = false;
        executor.execute([&]() { ran = true; }, []() {});
        expect(
            ran,
            "the executor must remain usable for a later job after a "
            "receiver failure");

        std::cout << "[PASS] test_exception_from_receiver_propagates\n";
    }

    void test_receiver_exception_preferred_when_both_fail()
    {
        RingExecutor executor;
        std::string caught;

        try
        {
            executor.execute(
                []() { throw std::runtime_error("sender failure"); },
                []() { throw std::runtime_error("receiver failure"); });
        }
        catch (const std::exception &error)
        {
            caught = error.what();
        }

        expect(
            caught == "receiver failure",
            "receiver's exception should be preferred when both sides "
            "fail, got: " +
                caught);

        std::cout
            << "[PASS] test_receiver_exception_preferred_when_both_fail\n";
    }

    void test_ring_session_step_progress()
    {
        RingSession session;
        std::atomic<bool> unblocked{false};
        std::atomic<bool> result{false};

        std::thread waiter(
            [&]()
            {
                result = session.wait_for_completed_steps(3);
                unblocked = true;
            });

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        expect(
            !unblocked.load(),
            "waiter should still be blocked before step 3 completes");

        session.complete_receive_step(); // step 1
        session.complete_receive_step(); // step 2
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        expect(
            !unblocked.load(),
            "waiter should still be blocked after only 2 of 3 steps");

        session.complete_receive_step(); // step 3
        waiter.join();

        expect(
            unblocked.load(),
            "waiter should have unblocked once step 3 completed");
        expect(
            result.load(),
            "wait_for_completed_steps should return true on normal "
            "completion");

        std::cout << "[PASS] test_ring_session_step_progress\n";
    }

    void test_ring_session_failure_unblocks_waiter()
    {
        RingSession session;
        std::atomic<bool> result{true};

        std::thread waiter(
            [&]() { result = session.wait_for_completed_steps(100); });

        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        session.report_failure();
        waiter.join();

        expect(
            !result.load(),
            "wait_for_completed_steps should return false once "
            "report_failure() has been called");

        std::cout << "[PASS] test_ring_session_failure_unblocks_waiter\n";
    }

    void test_operation_scoped_state_resets_between_jobs()
    {
        // Two independent RingSessions used across two separate
        // execute() calls on the SAME executor: the first fails, and
        // the second — a fresh session — must not be affected by it.
        RingExecutor executor;

        RingSession failing_session;
        bool threw = false;

        try
        {
            executor.execute(
                [&]()
                {
                    if (!failing_session.wait_for_completed_steps(1))
                    {
                        throw std::runtime_error("propagated failure");
                    }
                },
                [&]() { failing_session.report_failure(); });
        }
        catch (const std::exception &)
        {
            threw = true;
        }

        expect(threw, "expected the first job's session to have failed");

        RingSession healthy_session;
        bool sender_ok = false;

        executor.execute(
            [&]() { sender_ok = healthy_session.wait_for_completed_steps(1); },
            [&]() { healthy_session.complete_receive_step(); });

        expect(
            sender_ok,
            "a fresh RingSession on a later job must not inherit an "
            "earlier, unrelated session's failure");

        std::cout
            << "[PASS] test_operation_scoped_state_resets_between_jobs\n";
    }

    void test_idle_worker_no_busy_spin()
    {
        RingExecutor executor;
        executor.execute([]() {}, []() {});

        const std::clock_t cpu_before = std::clock();
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const std::clock_t cpu_after = std::clock();

        const double cpu_seconds =
            static_cast<double>(cpu_after - cpu_before) / CLOCKS_PER_SEC;

        // A busy-spinning worker would burn close to a full core's
        // worth of CPU time during this window; a condition_variable-
        // blocked worker burns effectively none. 100ms of process CPU
        // time during a 300ms idle window is a generous bound that
        // would only be exceeded by genuine spinning, not scheduler
        // noise.
        expect(
            cpu_seconds < 0.1,
            "idle worker appears to be busy-spinning: consumed " +
                std::to_string(cpu_seconds) +
                "s of process CPU time during a 0.3s idle window");

        std::cout << "[PASS] test_idle_worker_no_busy_spin\n";
    }

    void test_destroying_idle_executor_exits_cleanly()
    {
        const auto start = std::chrono::steady_clock::now();

        {
            RingExecutor executor; // never given a job
            (void)executor;
        }

        const auto elapsed = std::chrono::steady_clock::now() - start;

        expect(
            elapsed < std::chrono::seconds(2),
            "destroying an idle executor should be fast, not hang on a "
            "missed shutdown notification");

        std::cout << "[PASS] test_destroying_idle_executor_exits_cleanly\n";
    }

    void test_multiple_executors_independent()
    {
        RingExecutor a;
        RingExecutor b;

        a.execute([]() {}, []() {});
        a.execute([]() {}, []() {});
        b.execute([]() {}, []() {});

        const RingExecutorStats stats_a = a.stats();
        const RingExecutorStats stats_b = b.stats();

        expect(stats_a.worker_start_count == 1, "executor a: one worker");
        expect(stats_b.worker_start_count == 1, "executor b: one worker");
        expect(
            stats_a.submitted_jobs == 2, "executor a: two submitted jobs");
        expect(stats_b.submitted_jobs == 1, "executor b: one submitted job");
        expect(
            stats_a.completed_jobs == 2, "executor a: two completed jobs");
        expect(stats_b.completed_jobs == 1, "executor b: one completed job");

        std::cout << "[PASS] test_multiple_executors_independent\n";
    }

    // ===================================================================
    // Part 2: World-level lifecycle and reuse tests (real local TCP
    // Worlds).
    // ===================================================================

    void test_world_worker_lazy_and_reused()
    {
        run_on_ranks(
            kLazyWorldBase, 2,
            [](tbccl::World &world, std::size_t rank)
            {
                const RingExecutorStats before = RingExecutorAccess::stats(world);
                expect(
                    before.worker_start_count == 0,
                    "no ring executor should exist before any ring "
                    "collective runs");

                std::vector<std::uint8_t> send(
                    64, static_cast<std::uint8_t>(rank + 1));
                std::vector<std::uint8_t> recv(64 * 2, 0);

                all_gather_ring(world, send.data(), recv.data(), 64);

                const RingExecutorStats after_one =
                    RingExecutorAccess::stats(world);
                expect(
                    after_one.worker_start_count == 1,
                    "one worker after the first ring call");
                expect(
                    after_one.submitted_jobs == 1,
                    "one submitted job after the first ring call");
                expect(
                    after_one.completed_jobs == 1,
                    "one completed job after the first ring call");

                all_gather_ring(world, send.data(), recv.data(), 64);
                all_gather_ring(world, send.data(), recv.data(), 64);

                const RingExecutorStats after_three =
                    RingExecutorAccess::stats(world);
                expect(
                    after_three.worker_start_count == 1,
                    "still one worker after 3 ring calls");
                expect(
                    after_three.submitted_jobs == 3,
                    "three submitted jobs after 3 ring calls");
                expect(
                    after_three.completed_jobs == 3,
                    "three completed jobs after 3 ring calls");
            });

        std::cout << "[PASS] test_world_worker_lazy_and_reused\n";
    }

    void test_zero_count_no_executor()
    {
        run_on_ranks(
            kZeroCountBase, 2,
            [](tbccl::World &, std::size_t) {},
            5000);

        // Re-run with real (zero-count) collective calls, checked per
        // rank inside the body this time.
        run_on_ranks(
            kZeroCountBase, 2,
            [](tbccl::World &world, std::size_t)
            {
                std::uint8_t dummy8 = 0;
                all_gather_ring(world, &dummy8, &dummy8, 0);

                std::int32_t dummy32 = 0;
                reduce_scatter_ring(
                    world, &dummy32, &dummy32, 0, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);

                all_reduce_ring(
                    world, &dummy32, &dummy32, 0, tbccl::DataType::Int32,
                    tbccl::ReduceOp::Sum);

                expect(
                    RingExecutorAccess::stats(world).worker_start_count == 0,
                    "zero-count ring collectives must not create a ring "
                    "executor");
            });

        std::cout << "[PASS] test_zero_count_no_executor\n";
    }

    void test_single_rank_no_executor()
    {
        run_on_ranks(
            kSingleRankBase, 1,
            [](tbccl::World &world, std::size_t)
            {
                std::vector<std::uint8_t> send(64, 7);
                std::vector<std::uint8_t> recv(64, 0);

                all_gather_ring(world, send.data(), recv.data(), 64);

                expect(
                    recv == send,
                    "single-rank all_gather_ring should just copy the "
                    "local contribution");
                expect(
                    RingExecutorAccess::stats(world).worker_start_count ==
                        0,
                    "single-rank ring all_gather must short-circuit "
                    "without creating an executor");
            });

        std::cout << "[PASS] test_single_rank_no_executor\n";
    }

    void test_mixed_collective_reuse()
    {
        run_on_ranks(
            kMixedReuseBase, 3,
            [](tbccl::World &world, std::size_t rank)
            {
                constexpr std::size_t kSize = 3;
                constexpr int kIterations = 100;

                std::size_t expected_jobs = 0;

                for (int i = 0; i < kIterations; ++i)
                {
                    {
                        std::vector<std::uint8_t> send(
                            8, static_cast<std::uint8_t>(rank + 1));
                        std::vector<std::uint8_t> recv(8 * kSize, 0);
                        all_gather_ring(world, send.data(), recv.data(), 8);
                        ++expected_jobs;
                    }
                    {
                        std::vector<std::int32_t> send(
                            kSize * 2, static_cast<std::int32_t>(rank + 1));
                        std::vector<std::int32_t> recv(2, 0);
                        reduce_scatter_ring(
                            world, send.data(), recv.data(), 2,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);
                        ++expected_jobs;
                    }
                    {
                        std::vector<std::int32_t> send(
                            3, static_cast<std::int32_t>(rank + 1));
                        std::vector<std::int32_t> recv(3, 0);
                        all_reduce_ring(
                            world, send.data(), recv.data(), 3,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);
                        expected_jobs += 2;
                    }
                    {
                        std::vector<std::uint8_t> send(
                            8, static_cast<std::uint8_t>(rank + 1));
                        std::vector<std::uint8_t> recv(8 * kSize, 0);
                        all_gather_ring(world, send.data(), recv.data(), 8);
                        ++expected_jobs;
                    }
                    {
                        std::vector<std::int32_t> send(
                            3, static_cast<std::int32_t>(rank + 1));
                        std::vector<std::int32_t> recv(3, 0);
                        all_reduce_ring(
                            world, send.data(), recv.data(), 3,
                            tbccl::DataType::Int32, tbccl::ReduceOp::Sum);
                        expected_jobs += 2;
                    }
                }

                const RingExecutorStats stats =
                    RingExecutorAccess::stats(world);
                expect(
                    stats.worker_start_count == 1,
                    "one worker across the whole mixed-collective reuse "
                    "loop");
                expect(
                    stats.submitted_jobs == expected_jobs,
                    "submitted_jobs mismatch in mixed-collective reuse: "
                    "expected " +
                        std::to_string(expected_jobs) + ", got " +
                        std::to_string(stats.submitted_jobs));
                expect(
                    stats.completed_jobs == expected_jobs,
                    "completed_jobs mismatch in mixed-collective reuse");
            },
            20000);

        std::cout << "[PASS] test_mixed_collective_reuse\n";
    }

    void test_reference_to_ring_transitions()
    {
        run_on_ranks(
            kRefToRingBase, 3,
            [](tbccl::World &world, std::size_t rank)
            {
                constexpr std::size_t kSize = 3;

                std::vector<std::uint8_t> ag_send(
                    8, static_cast<std::uint8_t>(rank + 1));
                std::vector<std::uint8_t> ag_recv(8 * kSize, 0);

                all_gather_reference(
                    world, ag_send.data(), ag_recv.data(), 8);

                expect(
                    RingExecutorAccess::stats(world).worker_start_count ==
                        0,
                    "reference all_gather must not create a ring "
                    "executor");

                all_gather_ring(world, ag_send.data(), ag_recv.data(), 8);

                RingExecutorStats stats = RingExecutorAccess::stats(world);
                expect(
                    stats.worker_start_count == 1,
                    "ring all_gather should create the executor");
                expect(
                    stats.submitted_jobs == 1,
                    "one job after the first ring call");

                std::vector<std::int32_t> rs_send(
                    kSize * 2, static_cast<std::int32_t>(rank + 1));
                std::vector<std::int32_t> rs_recv(2, 0);

                reduce_scatter_reference(
                    world, rs_send.data(), rs_recv.data(), 2,
                    tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                stats = RingExecutorAccess::stats(world);
                expect(
                    stats.submitted_jobs == 1,
                    "reference reduce_scatter must not submit a job to "
                    "the ring executor");

                reduce_scatter_ring(
                    world, rs_send.data(), rs_recv.data(), 2,
                    tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                stats = RingExecutorAccess::stats(world);
                expect(
                    stats.submitted_jobs == 2,
                    "ring reduce_scatter should submit exactly one more "
                    "job");

                std::vector<std::int32_t> ar_send(
                    3, static_cast<std::int32_t>(rank + 1));
                std::vector<std::int32_t> ar_recv(3, 0);

                all_reduce_reference(
                    world, ar_send.data(), ar_recv.data(), 3,
                    tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                stats = RingExecutorAccess::stats(world);
                expect(
                    stats.submitted_jobs == 2,
                    "reference all_reduce must not submit any jobs to "
                    "the ring executor");

                all_reduce_ring(
                    world, ar_send.data(), ar_recv.data(), 3,
                    tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                stats = RingExecutorAccess::stats(world);
                expect(
                    stats.submitted_jobs == 4,
                    "ring all_reduce should submit exactly two more jobs "
                    "(reduce_scatter + all_gather phases)");
                expect(
                    stats.worker_start_count == 1,
                    "still exactly one worker throughout");

                all_gather_reference(
                    world, ag_send.data(), ag_recv.data(), 8);

                stats = RingExecutorAccess::stats(world);
                expect(
                    stats.submitted_jobs == 4,
                    "a later reference call must not submit any further "
                    "jobs");
            });

        std::cout << "[PASS] test_reference_to_ring_transitions\n";
    }

    // Env vars are process-wide state, and every rank in run_on_ranks()
    // runs as its own std::thread within this one test process —
    // setting/unsetting TBCCL_ALGORITHM from inside the per-rank body
    // would race across those threads (concurrent, unsynchronized
    // getenv/setenv/unsetenv on the same name), and could even let two
    // ranks briefly observe different override values for the same
    // logical collective call, which is exactly the unsupported
    // mismatched-algorithm-across-ranks scenario the Phase 12 override
    // contract warns can deadlock. So each EnvOverride here is applied
    // once, on this single controlling thread, strictly before its
    // run_on_ranks() call spawns any rank threads — mirroring
    // algorithm_selector_test.cpp's forced-mode integration tests.
    void test_forced_public_api_algorithms()
    {
        constexpr std::size_t kSize = 3;

        {
            EnvOverride global("TBCCL_ALGORITHM", "reference");

            run_on_ranks(
                kForcedAlgoBase, kSize,
                [](tbccl::World &world, std::size_t rank)
                {
                    std::vector<std::uint8_t> send(
                        8, static_cast<std::uint8_t>(rank + 1));
                    std::vector<std::uint8_t> recv(8 * kSize, 0);
                    tbccl::all_gather(world, send.data(), recv.data(), 8);

                    for (std::size_t r = 0; r < kSize; ++r)
                    {
                        expect(
                            recv[r * 8] == static_cast<std::uint8_t>(r + 1),
                            "forced-reference all_gather mismatch");
                    }

                    expect(
                        RingExecutorAccess::stats(world)
                                .worker_start_count == 0,
                        "TBCCL_ALGORITHM=reference must never create a "
                        "ring executor");
                });
        }

        {
            EnvOverride global("TBCCL_ALGORITHM", "ring");

            run_on_ranks(
                static_cast<std::uint16_t>(kForcedAlgoBase + 3), kSize,
                [](tbccl::World &world, std::size_t rank)
                {
                    std::vector<std::uint8_t> send(
                        8, static_cast<std::uint8_t>(rank + 1));
                    std::vector<std::uint8_t> recv(8 * kSize, 0);
                    tbccl::all_gather(world, send.data(), recv.data(), 8);

                    for (std::size_t r = 0; r < kSize; ++r)
                    {
                        expect(
                            recv[r * 8] == static_cast<std::uint8_t>(r + 1),
                            "forced-ring all_gather mismatch");
                    }

                    expect(
                        RingExecutorAccess::stats(world)
                                .worker_start_count == 1,
                        "TBCCL_ALGORITHM=ring should have created the "
                        "executor");
                });
        }

        {
            // N=3, 8 bytes/rank is below every configured Auto
            // threshold for AllGather, so this must resolve Reference.
            EnvOverride global("TBCCL_ALGORITHM", "auto");

            run_on_ranks(
                static_cast<std::uint16_t>(kForcedAlgoBase + 6), kSize,
                [](tbccl::World &world, std::size_t rank)
                {
                    std::vector<std::uint8_t> send(
                        8, static_cast<std::uint8_t>(rank + 1));
                    std::vector<std::uint8_t> recv(8 * kSize, 0);
                    tbccl::all_gather(world, send.data(), recv.data(), 8);

                    for (std::size_t r = 0; r < kSize; ++r)
                    {
                        expect(
                            recv[r * 8] == static_cast<std::uint8_t>(r + 1),
                            "auto-mode all_gather mismatch");
                    }

                    expect(
                        RingExecutorAccess::stats(world)
                                .worker_start_count == 0,
                        "auto mode at a below-threshold size must "
                        "resolve reference, not create a ring executor");
                });
        }

        std::cout << "[PASS] test_forced_public_api_algorithms\n";
    }

    void test_repeated_all_gather(
        std::uint16_t base_port,
        std::size_t size,
        int iterations)
    {
        run_on_ranks(
            base_port, size,
            [iterations](tbccl::World &world, std::size_t rank)
            {
                constexpr std::size_t kBytesPerRank = 64;
                const std::size_t total = world.size() * kBytesPerRank;

                std::vector<std::uint8_t> send(
                    kBytesPerRank, static_cast<std::uint8_t>(rank + 1));
                std::vector<std::uint8_t> recv(total, 0);

                for (int i = 0; i < iterations; ++i)
                {
                    all_gather_ring(
                        world, send.data(), recv.data(), kBytesPerRank);

                    for (std::size_t r = 0; r < world.size(); ++r)
                    {
                        expect(
                            recv[r * kBytesPerRank] ==
                                static_cast<std::uint8_t>(r + 1),
                            "all_gather_ring output mismatch on iteration " +
                                std::to_string(i));
                    }
                }

                const RingExecutorStats stats =
                    RingExecutorAccess::stats(world);
                expect(
                    stats.worker_start_count == 1,
                    "one worker across all repeated all_gather calls");
                expect(
                    stats.submitted_jobs ==
                        static_cast<std::size_t>(iterations),
                    "expected " + std::to_string(iterations) +
                        " submitted jobs, got " +
                        std::to_string(stats.submitted_jobs));
                expect(
                    stats.completed_jobs ==
                        static_cast<std::size_t>(iterations),
                    "expected " + std::to_string(iterations) +
                        " completed jobs, got " +
                        std::to_string(stats.completed_jobs));
            },
            20000);

        std::cout << "[PASS] test_repeated_all_gather (size=" << size
                   << ", iterations=" << iterations << ")\n";
    }

    void test_repeated_reduce_scatter(
        std::uint16_t base_port,
        std::size_t size,
        int iterations)
    {
        run_on_ranks(
            base_port, size,
            [iterations](tbccl::World &world, std::size_t rank)
            {
                constexpr std::size_t kRecvCount = 4;
                const std::size_t total_count = world.size() * kRecvCount;

                const std::array<tbccl::ReduceOp, 4> ops = {
                    tbccl::ReduceOp::Sum, tbccl::ReduceOp::Max,
                    tbccl::ReduceOp::Min, tbccl::ReduceOp::Product};

                std::vector<std::int32_t> send(total_count);
                std::vector<std::int32_t> recv(kRecvCount, 0);

                for (int i = 0; i < iterations; ++i)
                {
                    const tbccl::ReduceOp op =
                        ops[static_cast<std::size_t>(i) % ops.size()];

                    for (std::size_t j = 0; j < total_count; ++j)
                    {
                        send[j] = static_cast<std::int32_t>(
                            (rank + 1) + (i % 3));
                    }

                    // Correctness of the reduction result itself is
                    // exhaustively covered by reduce_scatter_ring_test
                    // .cpp; this loop's job is to exercise many
                    // repeated calls (rotating input/op) and check
                    // worker reuse below.
                    reduce_scatter_ring(
                        world, send.data(), recv.data(), kRecvCount,
                        tbccl::DataType::Int32, op);
                }

                const RingExecutorStats stats =
                    RingExecutorAccess::stats(world);
                expect(
                    stats.worker_start_count == 1,
                    "one worker across all repeated reduce_scatter "
                    "calls");
                expect(
                    stats.submitted_jobs ==
                        static_cast<std::size_t>(iterations),
                    "submitted_jobs mismatch for repeated reduce_scatter");
                expect(
                    stats.completed_jobs ==
                        static_cast<std::size_t>(iterations),
                    "completed_jobs mismatch for repeated reduce_scatter");
            },
            20000);

        std::cout << "[PASS] test_repeated_reduce_scatter (size=" << size
                   << ", iterations=" << iterations << ")\n";
    }

    void test_repeated_all_reduce()
    {
        constexpr std::size_t kSize = 3;
        constexpr int kIterations = 1000;

        run_on_ranks(
            kRepeatedAllReduceBase, kSize,
            [](tbccl::World &world, std::size_t rank)
            {
                constexpr std::size_t kCount = 3; // divisible by kSize

                std::vector<std::int32_t> send(
                    kCount, static_cast<std::int32_t>(rank + 1));
                std::vector<std::int32_t> recv(kCount, 0);

                constexpr std::int32_t kExpected =
                    static_cast<std::int32_t>(kSize * (kSize + 1) / 2);

                for (int i = 0; i < kIterations; ++i)
                {
                    all_reduce_ring(
                        world, send.data(), recv.data(), kCount,
                        tbccl::DataType::Int32, tbccl::ReduceOp::Sum);

                    for (std::size_t j = 0; j < kCount; ++j)
                    {
                        expect(
                            recv[j] == kExpected,
                            "all_reduce_ring drifted on a repeated call, "
                            "iteration " +
                                std::to_string(i));
                    }
                }

                const RingExecutorStats stats =
                    RingExecutorAccess::stats(world);
                expect(
                    stats.worker_start_count == 1,
                    "one worker across 1000 repeated all_reduce calls");
                expect(
                    stats.submitted_jobs ==
                        static_cast<std::size_t>(kIterations) * 2,
                    "expected 2 submitted jobs per all_reduce_ring call "
                    "(reduce_scatter phase + all_gather phase), got " +
                        std::to_string(stats.submitted_jobs));
                expect(
                    stats.completed_jobs ==
                        static_cast<std::size_t>(kIterations) * 2,
                    "completed_jobs mismatch for repeated all_reduce");
            },
            30000);

        std::cout << "[PASS] test_repeated_all_reduce\n";
    }

    void test_world_destruction_and_recreation()
    {
        constexpr int kIterations = 100;

        for (int i = 0; i < kIterations; ++i)
        {
            const std::uint16_t base = kWorldDestructionBases
                [static_cast<std::size_t>(i) % kWorldDestructionBases.size()];

            run_on_ranks(
                base, 2,
                [](tbccl::World &world, std::size_t rank)
                {
                    std::vector<std::uint8_t> send(
                        16, static_cast<std::uint8_t>(rank + 1));
                    std::vector<std::uint8_t> recv(32, 0);
                    all_gather_ring(world, send.data(), recv.data(), 16);

                    expect(
                        RingExecutorAccess::stats(world)
                                .worker_start_count == 1,
                        "each fresh World should get exactly one worker "
                        "on its first ring call");
                },
                3000);
            // The World (and its RingExecutor, if any) is destroyed
            // here, inside run_rank(), before run_on_ranks() returns —
            // the next iteration's fresh World cannot possibly reuse a
            // stale executor from this one.
        }

        std::cout << "[PASS] test_world_destruction_and_recreation ("
                   << kIterations << " iterations)\n";
    }

    void test_two_independent_worlds()
    {
        RingExecutorStats stats_a;
        RingExecutorStats stats_b;

        run_on_ranks(
            kIndependentWorldABase, 2,
            [&stats_a](tbccl::World &world, std::size_t rank)
            {
                std::vector<std::uint8_t> send(
                    8, static_cast<std::uint8_t>(rank + 1));
                std::vector<std::uint8_t> recv(16, 0);
                all_gather_ring(world, send.data(), recv.data(), 8);
                all_gather_ring(world, send.data(), recv.data(), 8);
                all_gather_ring(world, send.data(), recv.data(), 8);

                if (rank == 0)
                {
                    stats_a = RingExecutorAccess::stats(world);
                }
            });

        run_on_ranks(
            kIndependentWorldBBase, 2,
            [&stats_b](tbccl::World &world, std::size_t rank)
            {
                std::vector<std::uint8_t> send(
                    8, static_cast<std::uint8_t>(rank + 1));
                std::vector<std::uint8_t> recv(16, 0);
                all_gather_ring(world, send.data(), recv.data(), 8);

                if (rank == 0)
                {
                    stats_b = RingExecutorAccess::stats(world);
                }
            });

        expect(stats_a.worker_start_count == 1, "world A: one worker");
        expect(stats_b.worker_start_count == 1, "world B: one worker");
        expect(
            stats_a.submitted_jobs == 3, "world A: three submitted jobs");
        expect(
            stats_b.submitted_jobs == 1,
            "world B: one submitted job, independent of world A's three");

        std::cout << "[PASS] test_two_independent_worlds\n";
    }

} // namespace

int main()
{
    try
    {
        test_first_job_executes();
        test_jobs_execute_in_order();
        test_exception_from_sender_propagates();
        test_exception_from_receiver_propagates();
        test_receiver_exception_preferred_when_both_fail();
        test_ring_session_step_progress();
        test_ring_session_failure_unblocks_waiter();
        test_operation_scoped_state_resets_between_jobs();
        test_idle_worker_no_busy_spin();
        test_destroying_idle_executor_exits_cleanly();
        test_multiple_executors_independent();

        test_world_worker_lazy_and_reused();
        test_zero_count_no_executor();
        test_single_rank_no_executor();
        test_mixed_collective_reuse();
        test_reference_to_ring_transitions();
        test_forced_public_api_algorithms();

        test_repeated_all_gather(kRepeatedAllGather3Base, 3, 1000);
        test_repeated_all_gather(kRepeatedAllGather5Base, 5, 100);
        test_repeated_all_gather(kRepeatedAllGather8Base, 8, 100);

        test_repeated_reduce_scatter(
            kRepeatedReduceScatter3Base, 3, 1000);
        test_repeated_reduce_scatter(kRepeatedReduceScatter5Base, 5, 100);
        test_repeated_reduce_scatter(kRepeatedReduceScatter8Base, 8, 100);

        test_repeated_all_reduce();

        test_world_destruction_and_recreation();
        test_two_independent_worlds();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
