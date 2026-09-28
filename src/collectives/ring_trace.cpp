#include "ring_trace.hpp"

#if defined(TBCCL_ENABLE_RING_TRACE)

#include <atomic>
#include <chrono>

namespace tbccl::detail
{
namespace
{

    // Fixed-capacity — recording never allocates once the buffer has
    // grown to this size; excess events are silently dropped rather
    // than degrading the timed path with reallocation. Generous
    // enough for any single ring invocation's handful of events.
    constexpr std::size_t kCapacity = 4096;

    thread_local std::vector<RingTraceEvent> g_thread_events = [] {
        std::vector<RingTraceEvent> events;
        events.reserve(kCapacity);
        return events;
    }();

    std::atomic<std::uint64_t> g_next_operation_id{1};

} // namespace

    std::uint64_t ring_trace_next_operation_id()
    {
        return g_next_operation_id.fetch_add(1, std::memory_order_relaxed);
    }

    void ring_trace_record(
        std::uint64_t operation_id,
        std::size_t rank,
        RingTraceRole role,
        RingTraceEventType type,
        std::size_t message_size)
    {
        if (g_thread_events.size() >= kCapacity)
        {
            return;
        }

        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(now)
                .count();

        g_thread_events.push_back(
            RingTraceEvent{
                operation_id, rank, role, type, ns, message_size});
    }

    std::vector<RingTraceEvent> ring_trace_drain_this_thread()
    {
        std::vector<RingTraceEvent> drained;
        drained.swap(g_thread_events);
        g_thread_events.reserve(kCapacity);
        return drained;
    }

} // namespace tbccl::detail

#endif // TBCCL_ENABLE_RING_TRACE

namespace tbccl::detail
{

    const char *ring_trace_event_name(RingTraceEventType type)
    {
        switch (type)
        {
        case RingTraceEventType::SenderSubmit:
            return "sender_submit";
        case RingTraceEventType::WorkerWake:
            return "worker_wake";
        case RingTraceEventType::SenderBegin:
            return "sender_begin";
        case RingTraceEventType::SenderEnd:
            return "sender_end";
        case RingTraceEventType::CallerObserveDone:
            return "caller_observe_done";
        case RingTraceEventType::InputCopyBegin:
            return "input_copy_begin";
        case RingTraceEventType::InputCopyEnd:
            return "input_copy_end";
        case RingTraceEventType::SendBegin:
            return "send_begin";
        case RingTraceEventType::SendEnd:
            return "send_end";
        case RingTraceEventType::RecvBegin:
            return "recv_begin";
        case RingTraceEventType::RecvEnd:
            return "recv_end";
        case RingTraceEventType::ReduceBegin:
            return "reduce_begin";
        case RingTraceEventType::ReduceEnd:
            return "reduce_end";
        case RingTraceEventType::StepComplete:
            return "step_complete";
        case RingTraceEventType::OutputCopyBegin:
            return "output_copy_begin";
        case RingTraceEventType::OutputCopyEnd:
            return "output_copy_end";
        }

        return "unknown";
    }

} // namespace tbccl::detail
