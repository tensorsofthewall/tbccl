#include "all_gather_internal.hpp"
#include "ring_executor.hpp"
#include "ring_pipeline_session.hpp"

#include <cstdint>
#include <cstring>

namespace tbccl::detail
{

    // Same N-1 step ring topology and send_chunk(s)/recv_chunk(s)
    // formulas as all_gather_ring() (see that file's comment for the
    // full derivation) — the only difference is that each step's
    // bytes_per_rank-sized segment is itself split into up to
    // chunk_count = ceil(bytes_per_rank / chunk_bytes) pieces, sent and
    // received one at a time rather than as a single
    // send()/recv(bytes_per_rank) call. This lets a rank begin
    // forwarding a chunk to its next neighbor as soon as that chunk
    // individually arrives, instead of waiting for the neighbor's
    // entire segment.
    //
    // Wire order (step-major, matching the plan's chosen schedule):
    // for each step s, for each chunk c, send/receive chunk c of that
    // step's segment. Sender and receiver both iterate in this exact
    // order, so every send() has a matching recv() of the identical
    // length at the identical point in the peer's stream — chunk_bytes
    // must be identical on every rank for one invocation (undocumented
    // elsewhere in the transport, since World::send/recv are
    // exact-byte-count and carry no framing of their own).
    //
    // The sender for step s > 0, chunk c may forward that chunk only
    // once the receiver has completed chunk c of step s-1 — tracked by
    // a RingPipelineSession, which (unlike RingSession) tracks a
    // monotonic completed-chunk count per step rather than one flag
    // per step, since forwarding must be able to wait for just the
    // next chunk rather than the whole segment.
    void all_gather_pipelined(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t bytes_per_rank,
        std::size_t chunk_bytes)
    {
        validate_all_gather_args(
            world, send_buffer, recv_buffer, bytes_per_rank);

        if (bytes_per_rank == 0)
        {
            return;
        }

        validate_chunk_bytes("all_gather_pipelined", chunk_bytes, 1);

        const std::size_t size = world.size();

        if (size == 1)
        {
            std::memcpy(recv_buffer, send_buffer, bytes_per_rank);
            return;
        }

        const std::size_t rank = world.rank();
        const std::size_t next = (rank + 1) % size;
        const std::size_t prev = (rank + size - 1) % size;

        auto *recv_bytes = static_cast<std::uint8_t *>(recv_buffer);

        // Our own contribution is both part of the final output and
        // the first chunk we forward, so it must be in place before
        // the sender job starts.
        std::memcpy(
            recv_bytes + rank * bytes_per_rank, send_buffer, bytes_per_rank);

        const std::size_t chunk_count =
            compute_chunk_count(bytes_per_rank, chunk_bytes);

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
                            (rank + size - step) % size;
                        std::uint8_t *segment =
                            recv_bytes + send_chunk_idx * bytes_per_rank;

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

                            const std::size_t offset = c * chunk_bytes;
                            const std::size_t length = chunk_length(
                                bytes_per_rank, chunk_bytes, c);

                            world.send(next, segment + offset, length);
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
                            (rank + size - step - 1) % size;
                        std::uint8_t *segment =
                            recv_bytes + recv_chunk_idx * bytes_per_rank;

                        for (std::size_t c = 0; c < chunk_count; ++c)
                        {
                            const std::size_t offset = c * chunk_bytes;
                            const std::size_t length = chunk_length(
                                bytes_per_rank, chunk_bytes, c);

                            world.recv(prev, segment + offset, length);

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
    }

} // namespace tbccl::detail
