#pragma once

// Internal (non-installed) persistent execution infrastructure for
// ring collective algorithms. Not part of the public tbccl:: surface
// — kept under src/ and exposed to ring collective implementations,
// World's own translation unit, and tests only via a private include
// path, not the public include/ tree.
//
// This is an execution-engine optimization, not a collective-
// algorithm optimization: it exists solely to eliminate the
// per-invocation std::thread create/join cost the pre-Phase-13 ring
// implementations each paid on every call, by giving every World a
// single, lazily-created, reusable sender worker thread instead.
// Chunk formulas, reduction arithmetic, buffer semantics, and public
// API behavior are unchanged.

#include <tbccl/world.hpp>

#include "ring_trace.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace tbccl::detail
{

// Coordinates one ring collective invocation's step-by-step
// sender/receiver progress and failure state. Scoped to exactly one
// RingExecutor::execute() call — every ring algorithm call constructs
// a fresh RingSession on the stack, so a previous, unrelated
// operation's completed-steps count or failure state can never
// spuriously satisfy — or block — a later one. This is the same
// mutex/condition_variable/completed_steps/failed coordination the
// pre-Phase-13 ring implementations each duplicated inline (as
// "RingSyncState"), now shared in one place.
class RingSession
{
public:
    // Called by the receiver side once it has fully written a
    // received step's data into its output slot (and, for
    // ReduceScatter, only after apply_reduction() has finished folding
    // it in — never before). Wakes any sender currently blocked in
    // wait_for_completed_steps().
    void complete_receive_step()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++completed_steps_;
        cv_.notify_all();
    }

    // Called by the sender side before it is allowed to forward the
    // data belonging to ring step `step` (step 0 needs no wait — it
    // forwards the caller's own original contribution, already valid
    // before the sender starts). Blocks until at least `step`
    // receive-steps have completed. Returns false — without the
    // caller having sent anything for this step — once
    // report_failure() has been called by either side; the sender
    // must stop immediately in that case rather than send.
    bool wait_for_completed_steps(std::size_t step)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(
            lock,
            [&]() { return completed_steps_ >= step || failed_; });

        return !failed_;
    }

    // Marks the session failed and wakes every waiter, so a sender
    // blocked on a receive step that will now never arrive (because
    // the other side just threw) does not hang forever. Safe to call
    // from either side, including after the other side already called
    // it.
    void report_failure()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_ = true;
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::size_t completed_steps_ = 0;
    bool failed_ = false;
};

// Test-only observability (never a public API): lets tests confirm
// that repeated ring collective calls on the same World reuse one
// worker thread rather than repeatedly spawning new ones. Prefer
// these internal counters over comparing OS thread IDs, which may be
// reused after a thread terminates.
struct RingExecutorStats
{
    std::size_t worker_start_count = 0;
    std::size_t submitted_jobs = 0;
    std::size_t completed_jobs = 0;
};

// One persistent sender worker thread, lazily created on a World's
// first ring collective call and reused by every subsequent ring
// collective on that same World — including both phases of ring
// AllReduce, which submit two sequential jobs to the same worker
// rather than restarting it in between. Not a general-purpose thread
// pool: it runs exactly one job at a time, in submission order, and
// exists specifically to back ring collectives.
//
// execute() runs `sender` on the persistent worker and `receiver` on
// the calling thread concurrently, and does not return until both
// have finished — from the caller's point of view this has the same
// blocking, synchronous behavior as the pre-Phase-13 "create a
// thread, join it" pattern, just without repeatedly paying
// thread-creation cost. Every execute() call is a fresh job: nothing
// about it is retained by the worker once it completes, so the worker
// holds no dangling reference to a finished operation's buffers.
class RingExecutor
{
public:
    RingExecutor();

    RingExecutor(const RingExecutor &) = delete;
    RingExecutor &operator=(const RingExecutor &) = delete;

    // Sets the stopping flag, wakes the worker, and joins it. Never
    // throws. Requires that no job is currently executing — guaranteed
    // by World's existing exclusive-use contract, under which a World
    // is never destroyed while a collective is still in flight on it,
    // so the worker is always already idle here.
    ~RingExecutor() noexcept;

    // Runs `sender` on the persistent worker thread and `receiver` on
    // the calling thread, concurrently; blocks until both have
    // returned. Neither callable's exceptions are allowed to escape
    // the worker thread: if `sender` throws, its exception is
    // captured internally and, once `receiver` has also finished,
    // rethrown to the caller — unless `receiver` also threw, in which
    // case `receiver`'s exception is rethrown instead (matching the
    // pre-existing ring algorithms' "receiver's exception preferred"
    // rule). The ring-step coordination (a RingSession) that lets
    // `sender` wait for `receiver`'s progress, and that unblocks a
    // waiting `sender` when `receiver` fails, is the caller's
    // responsibility, not this class's — RingExecutor only knows how
    // to run two callables concurrently and report their outcomes.
    //
    // `trace_operation_id` is diagnostic-only (see ring_trace.hpp): it
    // correlates this call's T0 (submit)/T1 (worker wake)/T2 (sender
    // begin)/T3 (sender end)/T4 (caller observes completion) trace
    // events, which are recorded here and in worker_loop(). Rank is
    // deliberately not tracked at this layer — RingExecutor has no
    // notion of World/rank — callers that want rank-correlated stage
    // events (e.g. send/recv/reduce timings) record those themselves,
    // using the same operation_id, around their sender/receiver
    // callables. Passing 0 (the default, and what
    // ring_trace_next_operation_id() itself returns when tracing is
    // compiled out) means "don't trace this call".
    template <typename SenderFn, typename ReceiverFn>
    void execute(
        SenderFn &&sender,
        ReceiverFn &&receiver,
        std::uint64_t trace_operation_id = 0)
    {
        ring_trace_record(
            trace_operation_id, 0, RingTraceRole::Sender,
            RingTraceEventType::SenderSubmit);

        std::exception_ptr sender_exception;

        submit(
            [&sender, &sender_exception, trace_operation_id]()
            {
                ring_trace_record(
                    trace_operation_id, 0, RingTraceRole::Sender,
                    RingTraceEventType::SenderBegin);

                try
                {
                    sender();
                }
                catch (...)
                {
                    sender_exception = std::current_exception();
                }

                ring_trace_record(
                    trace_operation_id, 0, RingTraceRole::Sender,
                    RingTraceEventType::SenderEnd);
            },
            trace_operation_id);

        std::exception_ptr receiver_exception;

        try
        {
            receiver();
        }
        catch (...)
        {
            receiver_exception = std::current_exception();
        }

        wait_for_job();

        ring_trace_record(
            trace_operation_id, 0, RingTraceRole::Receiver,
            RingTraceEventType::CallerObserveDone);

        if (receiver_exception)
        {
            std::rethrow_exception(receiver_exception);
        }

        if (sender_exception)
        {
            std::rethrow_exception(sender_exception);
        }
    }

    RingExecutorStats stats() const;

#if defined(TBCCL_ENABLE_RING_TRACE)
    // Diagnostic-only. The worker thread's T1/T2/T3 and sender-side
    // stage events live in ITS OWN thread-local trace buffer (see
    // ring_trace.hpp) — only the worker thread could drain that
    // directly, and it never calls back into caller code to do so. So
    // after each job, the worker copies its just-drained events into
    // worker_trace_ (under the same mutex_ it already takes once per
    // job for job_done_/stats_ bookkeeping — no new hot-path
    // contention). This method retrieves that copy from any thread.
    std::vector<RingTraceEvent> drain_worker_trace();
#endif

private:
    void submit(std::function<void()> job, std::uint64_t trace_operation_id);
    void wait_for_job();
    void worker_loop();

    // Declaration order matters here, independent of the constructor's
    // member-initializer-list order: class members are always
    // constructed in declaration order. worker_ is declared (and
    // therefore constructed) LAST, after every piece of state
    // worker_loop() touches (mutex_, cv_, stopping_, job_ready_,
    // job_done_, job_, stats_). std::thread's constructor launches its
    // thread function immediately, running concurrently with the rest
    // of the enclosing object's construction — so if worker_ were
    // constructed any earlier, the newly-started worker thread could
    // begin executing worker_loop() and touch mutex_/cv_/the flags
    // below while they were still mid-construction on the constructing
    // thread, which is undefined behavior (observed in practice as an
    // intermittent startup hang/race, not a reliably-reproducing one).
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    bool stopping_ = false;
    bool job_ready_ = false;
    bool job_done_ = true;
    std::function<void()> job_;
    std::uint64_t job_trace_operation_id_ = 0;

    RingExecutorStats stats_;

#if defined(TBCCL_ENABLE_RING_TRACE)
    std::vector<RingTraceEvent> worker_trace_;
#endif

    std::thread worker_;
};

// Grants ring collective implementations (and tests) access to a
// World's private ring_executor_ slot without exposing it as part of
// World's own public interface. Kept internal-only.
class RingExecutorAccess
{
public:
    // Returns `world`'s RingExecutor, constructing it on first call.
    static RingExecutor &get_or_create(World &world);

    // Test-only: returns a snapshot of `world`'s ring executor stats,
    // or a default-constructed (all-zero) RingExecutorStats if no
    // ring collective has run on this World yet — never triggers
    // lazy construction itself.
    static RingExecutorStats stats(const World &world);
};

} // namespace tbccl::detail
