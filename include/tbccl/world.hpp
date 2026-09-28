#pragma once

#include <cstddef>
#include <memory>

namespace tbccl::detail
{
class RingExecutor;
class RingExecutorAccess;
} // namespace tbccl::detail

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
// are not guaranteed thread-safe in this phase.
class World
{
public:
    virtual ~World();

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
    World();

private:
    // Backs every ring collective algorithm (see
    // src/collectives/ring_executor.hpp), lazily constructed by
    // detail::RingExecutorAccess on the first ring collective call —
    // a World that only ever executes reference collectives never
    // allocates one. Declared on this abstract base rather than on a
    // specific transport subclass (e.g. TcpWorld) so every current
    // and future World implementation gets ring-collective execution
    // support for free, without ring algorithm code needing to know
    // the concrete transport type.
    //
    // Held as a pointer to an incomplete type, so this class's
    // constructor/destructor are declared here but defined out of
    // line (src/core/world.cpp) where RingExecutor is complete.
    // Binary-layout note: this adds private state to World, changing
    // its size/layout; any out-of-tree subclass of World must be
    // recompiled against this header (source-level API compatibility
    // is preserved — no public signature changed).
    //
    // Destruction-order note: for any subclass, the subclass's own
    // members (e.g. TcpWorld's transport connections) are destroyed
    // before this base class's members, including ring_executor_ —
    // derived-class teardown always precedes base-class teardown in
    // C++. This is safe only because it is never a supported use case
    // for a World to be destroyed while a collective is still in
    // flight on it (see the thread-safety note above); the ring
    // worker is therefore always idle by the time any World's
    // destructor runs, and RingExecutor's shutdown path touches only
    // its own thread/mutex/condition_variable state, never the World
    // or its transport, so it never touches already-destroyed
    // transport resources.
    std::unique_ptr<detail::RingExecutor> ring_executor_;

    friend class detail::RingExecutorAccess;
};

} // namespace tbccl
