#pragma once

#include <cstddef>

namespace tbccl
{

// A rank/peer communication substrate: rank() identities, size() peers,
// and exact-byte send/recv addressed by peer rank. This is not a
// message protocol — no framing beyond what Connection already
// guarantees is added at this layer. Higher-level operation framing
// belongs to future collectives, not here.
//
// Thread-safety: one send() and one recv() may be in flight on a given
// peer at the same time (TCP is full-duplex), matching a typical
// communication protocol's alternating or independent send/recv use.
// Concurrent send()/send() or recv()/recv() calls on the *same* peer
// are not guaranteed thread-safe.
class World
{
public:
    virtual ~World() = default;

    World(const World &) = delete;
    World &operator=(const World &) = delete;

    virtual std::size_t rank() const noexcept = 0;

    virtual std::size_t size() const noexcept = 0;

    // Sends exactly `bytes` bytes to `peer`, or throws. `peer` must be
    // a valid rank other than the local rank.
    virtual void send(
        std::size_t peer,
        const void *data,
        std::size_t bytes) = 0;

    // Receives exactly `bytes` bytes from `peer`, or throws. `peer`
    // must be a valid rank other than the local rank.
    virtual void recv(
        std::size_t peer,
        void *data,
        std::size_t bytes) = 0;

protected:
    World() = default;
};

} // namespace tbccl
