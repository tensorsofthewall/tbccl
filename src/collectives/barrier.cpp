#include <tbccl/collectives.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace tbccl
{
namespace
{

    constexpr std::uint8_t kArrived = 0xA1;
    constexpr std::uint8_t kRelease = 0xB1;

} // namespace

    // Centralized (star) barrier: rank 0 is the coordinator.
    //
    // Phase 1 (gather): every non-root rank sends a 1-byte ARRIVED
    // token to rank 0. Rank 0 receives one from every peer, in rank
    // order — if a higher-ranked peer's token arrives first, it simply
    // sits in that peer's own TCP receive buffer until rank 0 gets
    // around to it; this is correct, just not maximally prompt.
    //
    // Phase 2 (release): only after every ARRIVED token has been
    // received does rank 0 send a RELEASE token to every peer. These
    // phases must not be interleaved per peer (receive-then-immediately
    // -release for that one peer) — that would let a rank leave the
    // barrier while others haven't arrived yet, which is not a barrier.
    //
    // Non-root ranks: send ARRIVED, then block waiting for RELEASE.
    void barrier(World &world)
    {
        const std::size_t size = world.size();

        if (size <= 1)
        {
            return;
        }

        const std::size_t rank = world.rank();

        if (rank == 0)
        {
            for (std::size_t peer = 1; peer < size; ++peer)
            {
                std::uint8_t token = 0;

                world.recv(peer, &token, sizeof(token));

                if (token != kArrived)
                {
                    throw std::runtime_error(
                        "barrier: rank 0 received invalid arrival token "
                        "from rank " +
                        std::to_string(peer));
                }
            }

            for (std::size_t peer = 1; peer < size; ++peer)
            {
                const std::uint8_t token = kRelease;

                world.send(peer, &token, sizeof(token));
            }
        }
        else
        {
            const std::uint8_t arrived = kArrived;

            world.send(0, &arrived, sizeof(arrived));

            std::uint8_t response = 0;

            world.recv(0, &response, sizeof(response));

            if (response != kRelease)
            {
                throw std::runtime_error(
                    "barrier: rank " + std::to_string(rank) +
                    " received invalid release token from rank 0");
            }
        }
    }

} // namespace tbccl
