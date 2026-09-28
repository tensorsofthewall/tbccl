#pragma once

// Internal (non-installed), optional diagnostic tracing for the ring
// execution path. Disabled by default (CMake option
// TBCCL_ENABLE_RING_TRACE, OFF) — every function here compiles to a
// trivial inline no-op when disabled, so production Release builds
// pay nothing for its existence. Never part of the public tbccl::
// surface.
//
// Recording is purely thread-local (no shared state touched on the
// hot path), so calling ring_trace_record() from the sender worker or
// the calling/receiver thread never introduces new mutex contention.
// Events accumulate in a fixed-capacity per-thread buffer and are only
// read out via an explicit drain call, which callers must make after
// the timed invocation has fully returned — never from inside a timed
// region or from the sender worker itself (see RingExecutor's
// drain_worker_trace(), which safely retrieves the worker thread's
// buffer by copying it out under the executor's existing
// once-per-job lock, not a new per-event one).

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tbccl::detail
{

enum class RingTraceEventType
{
    SenderSubmit,      // T0: calling thread submits the sender job
    WorkerWake,        // T1: worker wakes / accepts the job
    SenderBegin,       // T2: sender callable begins
    SenderEnd,         // T3: sender callable completes
    CallerObserveDone, // T4: calling thread observes job completion

    InputCopyBegin,
    InputCopyEnd,
    SendBegin,
    SendEnd,
    RecvBegin,
    RecvEnd,
    ReduceBegin,
    ReduceEnd,
    StepComplete,
    OutputCopyBegin,
    OutputCopyEnd,
};

enum class RingTraceRole
{
    Sender,
    Receiver,
};

struct RingTraceEvent
{
    std::uint64_t operation_id;
    std::size_t rank;
    RingTraceRole role;
    RingTraceEventType type;
    std::int64_t timestamp_ns; // steady_clock; only meaningful for
                                // deltas within one process/run, never
                                // compared across machines.
    std::size_t message_size;
};

const char *ring_trace_event_name(RingTraceEventType type);

#if defined(TBCCL_ENABLE_RING_TRACE)

// A fresh, process-wide-unique id used to correlate one ring
// collective invocation's sender-side and receiver-side events.
std::uint64_t ring_trace_next_operation_id();

void ring_trace_record(
    std::uint64_t operation_id,
    std::size_t rank,
    RingTraceRole role,
    RingTraceEventType type,
    std::size_t message_size = 0);

// Returns and clears every event recorded so far on the calling
// thread.
std::vector<RingTraceEvent> ring_trace_drain_this_thread();

#else

inline std::uint64_t ring_trace_next_operation_id()
{
    return 0;
}

inline void ring_trace_record(
    std::uint64_t,
    std::size_t,
    RingTraceRole,
    RingTraceEventType,
    std::size_t = 0)
{
}

inline std::vector<RingTraceEvent> ring_trace_drain_this_thread()
{
    return {};
}

#endif

} // namespace tbccl::detail
