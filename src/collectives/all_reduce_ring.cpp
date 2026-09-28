#include "all_gather_internal.hpp"
#include "all_reduce_internal.hpp"
#include "reduce_scatter_internal.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tbccl::detail
{
namespace
{

    // Ring AllReduce is exactly ring ReduceScatter (over the full
    // `count`-element input) followed by ring AllGather (of each
    // rank's now-reduced `count/N`-element segment) — no new
    // reduction arithmetic or ring transport engine, just composing
    // the two already-validated ring primitives. This composition also
    // means no dedicated changes were needed here for Phase 13's
    // persistent ring worker: both calls below resolve `world`'s
    // RingExecutor (see ring_executor.hpp) themselves, so the two
    // phases automatically submit two sequential jobs to the same
    // already-running worker rather than restarting it in between.
    // `local_segment` is
    // typed storage (not a byte buffer reinterpreted later), so it is
    // correctly aligned for T by construction, and it is entirely
    // separate from both `send_buffer` and `recv_buffer` — this is
    // exactly what makes send_buffer == recv_buffer safe: by the time
    // ring reduce_scatter's internal working copy of `send_buffer` is
    // made (before any communication), the caller's send_buffer
    // content is no longer needed, and ring all_gather only ever
    // writes into `recv_buffer` from `local_segment`, never reading
    // `send_buffer` again. No barrier is needed between the two
    // phases: reduce_scatter_ring() only returns once every rank
    // already owns its final reduced segment, so all_gather_ring() can
    // begin immediately.
    template <typename T>
    void all_reduce_ring_typed(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t count,
        DataType datatype,
        ReduceOp op)
    {
        const std::size_t size = world.size();
        const std::size_t recv_count = count / size;
        const std::size_t segment_bytes = recv_count * sizeof(T);

        std::vector<T> local_segment(recv_count);

        reduce_scatter_ring(
            world, send_buffer, local_segment.data(), recv_count, datatype,
            op);

        all_gather_ring(
            world, local_segment.data(), recv_buffer, segment_bytes);
    }

} // namespace

    void all_reduce_ring(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t count,
        DataType datatype,
        ReduceOp op)
    {
        validate_all_reduce_args(
            send_buffer, recv_buffer, count, datatype, op);

        if (count == 0)
        {
            return;
        }

        const std::size_t size = world.size();

        if (count % size != 0)
        {
            throw std::runtime_error(
                "all_reduce_ring: element count " + std::to_string(count) +
                " is not divisible by world size " + std::to_string(size));
        }

        switch (datatype)
        {
        case DataType::Int32:
            all_reduce_ring_typed<std::int32_t>(
                world, send_buffer, recv_buffer, count, datatype, op);
            break;

        case DataType::Int64:
            all_reduce_ring_typed<std::int64_t>(
                world, send_buffer, recv_buffer, count, datatype, op);
            break;

        case DataType::Float32:
            all_reduce_ring_typed<float>(
                world, send_buffer, recv_buffer, count, datatype, op);
            break;

        case DataType::Float64:
            all_reduce_ring_typed<double>(
                world, send_buffer, recv_buffer, count, datatype, op);
            break;
        }
    }

} // namespace tbccl::detail
