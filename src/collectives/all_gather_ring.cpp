#include "all_gather_internal.hpp"
#include "ring_executor.hpp"

#include <cstdint>
#include <cstring>

namespace tbccl::detail
{

    // Ring all-gather over N-1 steps. At step s (0-indexed), each rank
    // sends chunk `(rank + N - s) % N` to `next = (rank + 1) % N` and
    // receives chunk `(rank + N - s - 1) % N` from
    // `prev = (rank + N - 1) % N`. At s == 0 that send chunk is the
    // rank's own original contribution; at every later step it is
    // exactly the chunk the rank received during the previous step, so
    // after N-1 steps every rank has forwarded every other rank's
    // contribution exactly once and owns the complete gathered array.
    //
    // Blocking send()/recv() on every rank in a fixed order (e.g.
    // send-then-recv, or recv-then-send, everywhere) can deadlock once
    // a message is large enough to fill socket buffers, since nothing
    // is then draining any rank's send. The World's persistent
    // RingExecutor (see ring_executor.hpp) runs every send on its one
    // reusable worker thread while the calling thread handles every
    // receive, so send and receive progress concurrently for the
    // whole operation — for N == 2, next == prev, so this send and
    // recv run concurrently on the very same full-duplex connection,
    // which World's contract explicitly permits. The sender and
    // receiver synchronize through a fresh, invocation-scoped
    // RingSession: the sender for step s >= 1 waits until the receiver
    // has finished step s-1 (i.e. has written the chunk the sender is
    // about to forward) before sending it.
    void all_gather_ring(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t bytes_per_rank)
    {
        validate_all_gather_args(
            world, send_buffer, recv_buffer, bytes_per_rank);

        if (bytes_per_rank == 0)
        {
            return;
        }

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

        RingSession session;
        RingExecutor &executor = RingExecutorAccess::get_or_create(world);

        executor.execute(
            /* sender */
            [&]()
            {
                try
                {
                    world.send(
                        next,
                        recv_bytes + rank * bytes_per_rank,
                        bytes_per_rank);

                    for (std::size_t step = 1; step < size - 1; ++step)
                    {
                        if (!session.wait_for_completed_steps(step))
                        {
                            return;
                        }

                        const std::size_t send_chunk =
                            (rank + size - step) % size;

                        world.send(
                            next,
                            recv_bytes + send_chunk * bytes_per_rank,
                            bytes_per_rank);
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
                            (rank + size - step - 1) % size;

                        world.recv(
                            prev, recv_bytes + recv_chunk * bytes_per_rank,
                            bytes_per_rank);

                        session.complete_receive_step();
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
