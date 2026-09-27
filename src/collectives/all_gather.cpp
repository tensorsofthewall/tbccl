#include <tbccl/collectives.hpp>

#include "algorithm_selector.hpp"
#include "all_gather_internal.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace tbccl
{

    void all_gather(
        World &world,
        const void *send_buffer,
        void *recv_buffer,
        std::size_t bytes_per_rank)
    {
        detail::validate_all_gather_args(
            world, send_buffer, recv_buffer, bytes_per_rank);

        const auto mode =
            detail::resolve_algorithm_mode(detail::CollectiveKind::AllGather);
        const auto decision = detail::select_all_gather_algorithm(
            world.size(), bytes_per_rank, mode);

        switch (decision.algorithm)
        {
        case detail::CollectiveAlgorithm::Reference:
            detail::all_gather_reference(
                world, send_buffer, recv_buffer, bytes_per_rank);
            break;

        case detail::CollectiveAlgorithm::Ring:
            detail::all_gather_ring(
                world, send_buffer, recv_buffer, bytes_per_rank);
            break;
        }
    }

} // namespace tbccl

namespace tbccl::detail
{

    void validate_all_gather_args(
        const World &world,
        const void *send_buffer,
        const void *recv_buffer,
        std::size_t bytes_per_rank)
    {
        const std::size_t size = world.size();

        if (bytes_per_rank != 0 &&
            size > std::numeric_limits<std::size_t>::max() / bytes_per_rank)
        {
            throw std::runtime_error(
                "all_gather: total receive size overflows size_t");
        }

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
    }

    // Two-phase centralized all-gather: gather every contribution to
    // rank 0 (sequential blocking recv per peer, same ordering
    // rationale as barrier()/broadcast() — every non-root rank has
    // already entered all_gather() and is blocked in send(0, ...) by
    // the time rank 0 gets to it), then reuse the already-tested
    // broadcast() to hand the fully-assembled buffer to everyone. Not
    // optimized: O(N) at the root for the gather, then broadcast's own
    // O(N). This is the correctness oracle other all_gather algorithm
    // variants (e.g. all_gather_ring()) are compared against — its
    // behavior must not change.
    void all_gather_reference(
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

        const std::size_t total_bytes = size * bytes_per_rank;
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

} // namespace tbccl::detail
