#include "reduce_scatter_internal.hpp"
#include "reduction_internal.hpp"
#include "ring_executor.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace tbccl::detail
{
namespace
{

    // Ring reduce-scatter over N-1 steps. Each rank's local `work`
    // array starts as N chunks of recv_count elements — a copy of its
    // own full N*recv_count contribution — and chunk i in `work` is
    // combined, over the course of the ring, with every other rank's
    // chunk i, until it holds the fully reduced segment i.
    //
    // At step s (0-indexed), a rank sends chunk `(rank+N-s-1)%N` to
    // `next=(rank+1)%N` and receives a chunk from `prev=(rank+N-1)%N`
    // into `recv_chunk = (rank+N-s-2)%N`, reducing it into
    // `work[recv_chunk]` before forwarding is possible. This is
    // shifted by one position relative to all_gather_ring's formula:
    // there, a received chunk is immediately forwardable as-is; here,
    // it must be folded into the local partial sum first. At s == 0
    // the sent chunk is the rank's own untouched contribution (no
    // reduction has happened yet for that index); at every later step
    // it is exactly the chunk the rank finished reducing during the
    // previous step. The last step's recv_chunk is always `rank`
    // itself ((rank+N-(N-2)-2)%N == rank), so after N-1 steps
    // work[rank] holds the fully reduced segment `rank` — see
    // reduce_scatter_internal.hpp's comment and the ring correctness
    // tests for a worked N=4 example.
    //
    // Concurrency: identical design to all_gather_ring — the World's
    // persistent RingExecutor runs every send on its one reusable
    // worker thread while the calling thread handles every
    // receive-and-reduce, so send and receive progress concurrently
    // (required to avoid deadlock once messages are large enough to
    // fill socket buffers). For N == 2, next == prev, so this send and
    // recv run concurrently on the same full-duplex connection, which
    // World's contract explicitly permits. The sender for step s >= 1
    // waits (via a fresh, invocation-scoped RingSession) until the
    // receiver has finished reducing into work[send_chunk(s)] during
    // its own previous iteration before forwarding it —
    // apply_reduction() must complete before the completion signal, or
    // the sender could read a partially-combined chunk.
    template <typename T>
    void reduce_scatter_ring_typed(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t recv_count,
        ReduceOp op)
    {
        const std::size_t rank = world.rank();
        const std::size_t size = world.size();
        const std::size_t total_count = size * recv_count;
        const std::size_t segment_bytes = recv_count * sizeof(T);

        if (size == 1)
        {
            std::memcpy(recv_buffer, send_buffer, segment_bytes);
            return;
        }

        const std::uint64_t trace_op = ring_trace_next_operation_id();

        ring_trace_record(
            trace_op, rank, RingTraceRole::Receiver,
            RingTraceEventType::InputCopyBegin, segment_bytes);

        std::vector<T> work(total_count);
        std::memcpy(
            work.data(), send_buffer, total_count * sizeof(T));

        ring_trace_record(
            trace_op, rank, RingTraceRole::Receiver,
            RingTraceEventType::InputCopyEnd, segment_bytes);

        std::vector<T> incoming(recv_count);

        const std::size_t next = (rank + 1) % size;
        const std::size_t prev = (rank + size - 1) % size;

        RingSession session;
        RingExecutor &executor = RingExecutorAccess::get_or_create(world);

        executor.execute(
            /* sender */
            [&]()
            {
                try
                {
                    const std::size_t send_chunk0 = (rank + size - 1) % size;

                    ring_trace_record(
                        trace_op, rank, RingTraceRole::Sender,
                        RingTraceEventType::SendBegin, segment_bytes);
                    world.send(
                        next, work.data() + send_chunk0 * recv_count,
                        segment_bytes);
                    ring_trace_record(
                        trace_op, rank, RingTraceRole::Sender,
                        RingTraceEventType::SendEnd, segment_bytes);

                    for (std::size_t step = 1; step < size - 1; ++step)
                    {
                        if (!session.wait_for_completed_steps(step))
                        {
                            return;
                        }

                        const std::size_t send_chunk =
                            (rank + size - step - 1) % size;

                        ring_trace_record(
                            trace_op, rank, RingTraceRole::Sender,
                            RingTraceEventType::SendBegin, segment_bytes);
                        world.send(
                            next, work.data() + send_chunk * recv_count,
                            segment_bytes);
                        ring_trace_record(
                            trace_op, rank, RingTraceRole::Sender,
                            RingTraceEventType::SendEnd, segment_bytes);
                    }
                }
                catch (...)
                {
                    session.report_failure();
                    throw;
                }
            },
            /* receiver */
            [&]()
            {
                try
                {
                    for (std::size_t step = 0; step < size - 1; ++step)
                    {
                        const std::size_t recv_chunk =
                            (rank + size - step - 2) % size;

                        ring_trace_record(
                            trace_op, rank, RingTraceRole::Receiver,
                            RingTraceEventType::RecvBegin, segment_bytes);
                        world.recv(prev, incoming.data(), segment_bytes);
                        ring_trace_record(
                            trace_op, rank, RingTraceRole::Receiver,
                            RingTraceEventType::RecvEnd, segment_bytes);

                        ring_trace_record(
                            trace_op, rank, RingTraceRole::Receiver,
                            RingTraceEventType::ReduceBegin, segment_bytes);
                        apply_reduction(
                            work.data() + recv_chunk * recv_count,
                            incoming.data(), recv_count, op);
                        ring_trace_record(
                            trace_op, rank, RingTraceRole::Receiver,
                            RingTraceEventType::ReduceEnd, segment_bytes);

                        session.complete_receive_step();
                        ring_trace_record(
                            trace_op, rank, RingTraceRole::Receiver,
                            RingTraceEventType::StepComplete, segment_bytes);
                    }
                }
                catch (...)
                {
                    session.report_failure();
                    throw;
                }
            },
            trace_op);

        ring_trace_record(
            trace_op, rank, RingTraceRole::Receiver,
            RingTraceEventType::OutputCopyBegin, segment_bytes);
        std::memcpy(
            recv_buffer, work.data() + rank * recv_count, segment_bytes);
        ring_trace_record(
            trace_op, rank, RingTraceRole::Receiver,
            RingTraceEventType::OutputCopyEnd, segment_bytes);
    }

} // namespace

    void reduce_scatter_ring(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t recv_count,
        DataType datatype,
        ReduceOp op)
    {
        validate_reduce_scatter_args(
            world, send_buffer, recv_buffer, recv_count, datatype, op);

        if (recv_count == 0)
        {
            return;
        }

        switch (datatype)
        {
        case DataType::Int32:
            reduce_scatter_ring_typed<std::int32_t>(
                world, send_buffer, recv_buffer, recv_count, op);
            break;

        case DataType::Int64:
            reduce_scatter_ring_typed<std::int64_t>(
                world, send_buffer, recv_buffer, recv_count, op);
            break;

        case DataType::Float32:
            reduce_scatter_ring_typed<float>(
                world, send_buffer, recv_buffer, recv_count, op);
            break;

        case DataType::Float64:
            reduce_scatter_ring_typed<double>(
                world, send_buffer, recv_buffer, recv_count, op);
            break;
        }
    }

} // namespace tbccl::detail
