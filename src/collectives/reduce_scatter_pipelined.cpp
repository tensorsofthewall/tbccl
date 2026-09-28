#include "reduce_scatter_internal.hpp"
#include "reduction_internal.hpp"
#include "ring_executor.hpp"
#include "ring_pipeline_session.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace tbccl::detail
{
namespace
{

    // Same N-1 step ring topology and send_chunk(s)/recv_chunk(s)
    // formulas as reduce_scatter_ring() (see that file's comment for
    // the full derivation, including why the formulas are shifted by
    // one position relative to all_gather_ring's) — the only
    // difference is that each step's recv_count-element segment is
    // itself split into up to chunk_count =
    // ceil(segment_bytes / chunk_bytes) byte chunks (a whole number of
    // elements each), received and reduced one chunk at a time rather
    // than as a single recv(segment_bytes) + apply_reduction(
    // recv_count) call.
    //
    // Wire order matches all_gather_pipelined's: step-major, chunk
    // index ascending, identical on sender and receiver. The critical
    // ordering invariant (see ring_pipeline_session.hpp) is unchanged
    // from the non-pipelined version, just at chunk instead of step
    // granularity: apply_reduction() for chunk c of step s must
    // complete — not just the recv() — before
    // session.mark_chunk_complete(s) runs, or a sender waiting on that
    // chunk could forward a partially-combined value.
    template <typename T>
    void reduce_scatter_pipelined_typed(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t recv_count,
        ReduceOp op,
        std::size_t chunk_bytes)
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

        std::vector<T> work(total_count);
        std::memcpy(work.data(), send_buffer, total_count * sizeof(T));

        const std::size_t chunk_count =
            compute_chunk_count(segment_bytes, chunk_bytes);

        // Sized for the largest (first) chunk — chunks are
        // non-increasing in size, so this safely bounds every chunk,
        // including the case where chunk_bytes exceeds segment_bytes
        // entirely (chunk_count == 1, this chunk covers the whole
        // segment).
        const std::size_t max_chunk_elements =
            chunk_length(segment_bytes, chunk_bytes, 0) / sizeof(T);
        std::vector<T> incoming(max_chunk_elements);

        const std::size_t next = (rank + 1) % size;
        const std::size_t prev = (rank + size - 1) % size;

        RingPipelineSession session(size - 1);
        RingExecutor &executor = RingExecutorAccess::get_or_create(world);

        executor.execute(
            /* sender */
            [&]()
            {
                try
                {
                    for (std::size_t step = 0; step < size - 1; ++step)
                    {
                        const std::size_t send_chunk_idx =
                            (rank + size - step - 1) % size;
                        T *segment = work.data() + send_chunk_idx * recv_count;

                        for (std::size_t c = 0; c < chunk_count; ++c)
                        {
                            if (step > 0)
                            {
                                if (!session.wait_for_chunks(
                                        step - 1, c + 1))
                                {
                                    return;
                                }
                            }

                            const std::size_t byte_offset = c * chunk_bytes;
                            const std::size_t length = chunk_length(
                                segment_bytes, chunk_bytes, c);
                            const std::size_t element_offset =
                                byte_offset / sizeof(T);

                            world.send(
                                next, segment + element_offset, length);
                        }
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
                        const std::size_t recv_chunk_idx =
                            (rank + size - step - 2) % size;
                        T *segment = work.data() + recv_chunk_idx * recv_count;

                        for (std::size_t c = 0; c < chunk_count; ++c)
                        {
                            const std::size_t byte_offset = c * chunk_bytes;
                            const std::size_t length = chunk_length(
                                segment_bytes, chunk_bytes, c);
                            const std::size_t element_offset =
                                byte_offset / sizeof(T);
                            const std::size_t elements = length / sizeof(T);

                            world.recv(prev, incoming.data(), length);

                            apply_reduction(
                                segment + element_offset, incoming.data(),
                                elements, op);

                            session.mark_chunk_complete(step);
                        }
                    }
                }
                catch (...)
                {
                    session.report_failure();
                    throw;
                }
            });

        std::memcpy(
            recv_buffer, work.data() + rank * recv_count, segment_bytes);
    }

} // namespace

    void reduce_scatter_pipelined(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t recv_count,
        DataType datatype,
        ReduceOp op,
        std::size_t chunk_bytes)
    {
        validate_reduce_scatter_args(
            world, send_buffer, recv_buffer, recv_count, datatype, op);

        if (recv_count == 0)
        {
            return;
        }

        validate_chunk_bytes(
            "reduce_scatter_pipelined", chunk_bytes,
            datatype_size(datatype));

        switch (datatype)
        {
        case DataType::Int32:
            reduce_scatter_pipelined_typed<std::int32_t>(
                world, send_buffer, recv_buffer, recv_count, op,
                chunk_bytes);
            break;

        case DataType::Int64:
            reduce_scatter_pipelined_typed<std::int64_t>(
                world, send_buffer, recv_buffer, recv_count, op,
                chunk_bytes);
            break;

        case DataType::Float32:
            reduce_scatter_pipelined_typed<float>(
                world, send_buffer, recv_buffer, recv_count, op,
                chunk_bytes);
            break;

        case DataType::Float64:
            reduce_scatter_pipelined_typed<double>(
                world, send_buffer, recv_buffer, recv_count, op,
                chunk_bytes);
            break;
        }
    }

} // namespace tbccl::detail
