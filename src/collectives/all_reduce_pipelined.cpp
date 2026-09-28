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

    // Pipelined ring AllReduce is exactly pipelined ring ReduceScatter
    // (over the full `count`-element input) followed by pipelined ring
    // AllGather (of each rank's now-reduced `count/N`-element
    // segment), mirroring all_reduce_ring()'s composition exactly — no
    // new reduction arithmetic or ring transport engine, and no fusion
    // across the two phases (all-gather does not begin until
    // reduce-scatter has fully returned; chunk_bytes pipelines within
    // each primitive, not across their boundary). `local_segment` is
    // typed storage, entirely separate from both `send_buffer` and
    // `recv_buffer` — the same reasoning as all_reduce_ring.cpp makes
    // send_buffer == recv_buffer safe here too: by the time pipelined
    // reduce-scatter's internal working copy of `send_buffer` is made
    // (before any communication), the caller's send_buffer content is
    // no longer needed, and pipelined all-gather only ever writes into
    // `recv_buffer` from `local_segment`.
    template <typename T>
    void all_reduce_pipelined_typed(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t count,
        DataType datatype,
        ReduceOp op,
        std::size_t chunk_bytes)
    {
        const std::size_t size = world.size();
        const std::size_t recv_count = count / size;
        const std::size_t segment_bytes = recv_count * sizeof(T);

        std::vector<T> local_segment(recv_count);

        reduce_scatter_pipelined(
            world, send_buffer, local_segment.data(), recv_count, datatype,
            op, chunk_bytes);

        all_gather_pipelined(
            world, local_segment.data(), recv_buffer, segment_bytes,
            chunk_bytes);
    }

} // namespace

    void all_reduce_pipelined(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t count,
        DataType datatype,
        ReduceOp op,
        std::size_t chunk_bytes)
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
                "all_reduce_pipelined: element count " +
                std::to_string(count) + " is not divisible by world size " +
                std::to_string(size));
        }

        switch (datatype)
        {
        case DataType::Int32:
            all_reduce_pipelined_typed<std::int32_t>(
                world, send_buffer, recv_buffer, count, datatype, op,
                chunk_bytes);
            break;

        case DataType::Int64:
            all_reduce_pipelined_typed<std::int64_t>(
                world, send_buffer, recv_buffer, count, datatype, op,
                chunk_bytes);
            break;

        case DataType::Float32:
            all_reduce_pipelined_typed<float>(
                world, send_buffer, recv_buffer, count, datatype, op,
                chunk_bytes);
            break;

        case DataType::Float64:
            all_reduce_pipelined_typed<double>(
                world, send_buffer, recv_buffer, count, datatype, op,
                chunk_bytes);
            break;
        }
    }

} // namespace tbccl::detail
