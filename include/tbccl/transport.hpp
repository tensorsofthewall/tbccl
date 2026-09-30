#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>

namespace tbccl
{

// Phase 32 Part F: characteristics a Transport actually has, so higher
// layers (the async tensor-transfer substrate) can make staging/chunking
// decisions from *capabilities*, never from a transport's concrete type
// (see transport_capabilities() below and Part F item 28's explicit
// "if transport == TCP" anti-pattern). A future RDMA/native-Thunderbolt
// transport reports different values here; nothing above this struct
// should need to change.
struct TransportCapabilities
{
    // No silent drops/reordering/duplication at this layer.
    bool reliable = true;
    bool ordered = true;

    // Whether the transport can move data without an intermediate
    // host-visible staging copy (false for TCP -- every byte crosses a
    // socket send/recv buffer).
    bool supports_zero_copy = false;

    // Whether the transport can operate directly on caller-registered
    // memory regions (RDMA memory registration, GPUDirect-style APIs).
    bool supports_registered_memory = false;

    // Whether the transport can read/write device memory directly
    // without host staging (false for TCP; a future GPUDirect-capable
    // RDMA transport would report true).
    bool supports_direct_device_memory = false;

    // A hint, not a hard requirement -- chunk planning should prefer
    // offsets/sizes aligned to this where it costs nothing to do so.
    std::size_t preferred_chunk_alignment = 1;

    // Upper bound on how many transfers this transport can usefully
    // have simultaneously in flight; 1 for a single blocking
    // send()/recv() stream like TCP.
    std::size_t max_in_flight = 1;
};

class Connection
{
public:
    virtual ~Connection() = default;

    Connection(const Connection &) = delete;
    Connection &operator=(const Connection &) = delete;

    // Sends exactly `bytes` bytes, or throws.
    virtual void send(
        const void *data,
        std::size_t bytes) = 0;

    // Receives exactly `bytes` bytes, or throws.
    virtual void recv(
        void *data,
        std::size_t bytes) = 0;

    // Human-readable "address:port" of the remote peer, for logging.
    virtual std::string peer_name() const = 0;

protected:
    Connection() = default;
};

class Listener
{
public:
    virtual ~Listener() = default;

    Listener(const Listener &) = delete;
    Listener &operator=(const Listener &) = delete;

    // Blocks until a client connects; the listener remains usable for
    // further sequential accepts afterward.
    virtual std::unique_ptr<Connection> accept() = 0;

    // Waits up to `timeout` for a client to connect. Returns nullptr
    // if no connection arrives within `timeout`; throws on a genuine
    // socket error. Never blocks longer than `timeout`.
    virtual std::unique_ptr<Connection> accept_for(
        std::chrono::milliseconds timeout) = 0;

protected:
    Listener() = default;
};

// Phase 32 Part E: the transport-abstraction boundary a future
// TcpTransport/RdmaTransport/NativeTbTransport family lives behind, so
// the async tensor-transfer substrate (TransferRequest/TransferWork/
// TensorCommWorker/StagingPool/ChunkPlan, async_transfer.hpp) can move
// bytes without knowing which concrete transport is underneath. This is
// deliberately NOT just "Connection with a different name" (Part E item
// 22): it is the seam a data-plane implementation with genuinely
// different semantics (registered memory, zero-copy, direct device
// access) can occupy later, by implementing this same interface and
// reporting different capabilities() -- no caller above this class
// needs to change.
//
// Phase 32 ships exactly one implementation: TcpTransport, adapting the
// existing Connection this header already defines. No other transport
// is implemented this phase (see docs/transport_architecture.md).
class Transport
{
public:
    virtual ~Transport() = default;

    Transport(const Transport &) = delete;
    Transport &operator=(const Transport &) = delete;

    // Same exact-byte-count contract as Connection::send/recv. A
    // reliable/ordered transport (capabilities().reliable/ordered) may
    // implement this as a direct pass-through; a transport that isn't
    // both would need to add its own framing/retry beneath this same
    // signature -- callers above never see the difference.
    virtual void send(const void *data, std::size_t bytes) = 0;
    virtual void recv(void *data, std::size_t bytes) = 0;

    virtual TransportCapabilities capabilities() const noexcept = 0;

    virtual std::string peer_name() const = 0;

protected:
    Transport() = default;
};

// Adapts an existing Connection (TCP today; any future Connection
// implementation) to the Transport interface. Takes ownership of the
// Connection. This is intentionally a thin pass-through -- Phase 32
// does not reimplement socket I/O, framing, or retry logic that
// Connection/TcpConnection already provide correctly (Part E item 24).
class TcpTransport final : public Transport
{
public:
    explicit TcpTransport(std::unique_ptr<Connection> connection);

    void send(const void *data, std::size_t bytes) override;
    void recv(void *data, std::size_t bytes) override;
    TransportCapabilities capabilities() const noexcept override;
    std::string peer_name() const override;

private:
    std::unique_ptr<Connection> connection_;
};

} // namespace tbccl
