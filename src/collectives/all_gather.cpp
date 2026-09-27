#include <tbccl/collectives.hpp>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tbccl
{

    // Two-phase centralized all-gather: gather every contribution to
    // rank 0 (sequential blocking recv per peer, same ordering
    // rationale as barrier()/broadcast() — every non-root rank has
    // already entered all_gather() and is blocked in send(0, ...) by
    // the time rank 0 gets to it), then reuse the already-tested
    // broadcast() to hand the fully-assembled buffer to everyone. Not
    // optimized: O(N) at the root for the gather, then broadcast's own
    // O(N).
    void all_gather(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t bytes_per_rank)
    {
        const std::size_t size = world.size();

        if (bytes_per_rank != 0 &&
            size > std::numeric_limits<std::size_t>::max() / bytes_per_rank)
        {
            throw std::runtime_error(
                "all_gather: total receive size overflows size_t");
        }

        const std::size_t total_bytes = size * bytes_per_rank;

        if (bytes_per_rank > 0)
        {
            if (send_buffer == nullptr)
            {
                throw std::runtime_error(
                    "all_gather: send buffer is null for non-zero byte count");
            }

            if (recv_buffer == nullptr)
            {
                throw std::runtime_error(
                    "all_gather: receive buffer is null for non-zero byte "
                    "count");
            }
        }

        if (bytes_per_rank == 0)
        {
            return;
        }

        if (size == 1)
        {
            std::memcpy(recv_buffer, send_buffer, bytes_per_rank);
            return;
        }

        const std::size_t rank = world.rank();

        auto *recv_bytes = static_cast<std::uint8_t *>(recv_buffer);

        if (rank == 0)
        {
            std::memcpy(recv_bytes, send_buffer, bytes_per_rank);

            for (std::size_t peer = 1; peer < size; ++peer)
            {
                world.recv(
                    peer,
                    recv_bytes + peer * bytes_per_rank,
                    bytes_per_rank);
            }
        }
        else
        {
            world.send(0, send_buffer, bytes_per_rank);
        }

        broadcast(world, recv_buffer, total_bytes, 0);
    }

} // namespace tbccl
