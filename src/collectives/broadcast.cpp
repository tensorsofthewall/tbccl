#include <tbccl/collectives.hpp>

#include <stdexcept>
#include <string>

namespace tbccl
{

    // Linear root-based broadcast: root sends the payload to every
    // other rank in turn with a sequential blocking send per peer. This
    // is O(N) at the root and intentionally not optimized (no tree,
    // no pipelining, no concurrent sends) — see collectives.hpp for the
    // ordering contract this relies on: every non-root rank has already
    // entered broadcast() and is blocked in recv(root, ...) by the time
    // root reaches it, since all ranks call collectives in matching
    // order with exclusive use of the World connections.
    void broadcast(
        World &world,
        void *buffer,
        std::size_t bytes,
        std::size_t root)
    {
        const std::size_t size = world.size();

        if (root >= size)
        {
            throw std::runtime_error(
                "broadcast: root rank " + std::to_string(root) +
                " is invalid for world size " + std::to_string(size));
        }

        if (bytes > 0 && buffer == nullptr)
        {
            throw std::runtime_error(
                "broadcast: buffer is null for non-zero byte count");
        }

        if (size <= 1 || bytes == 0)
        {
            return;
        }

        const std::size_t rank = world.rank();

        if (rank == root)
        {
            for (std::size_t peer = 0; peer < size; ++peer)
            {
                if (peer == root)
                {
                    continue;
                }

                world.send(peer, buffer, bytes);
            }
        }
        else
        {
            world.recv(root, buffer, bytes);
        }
    }

} // namespace tbccl
