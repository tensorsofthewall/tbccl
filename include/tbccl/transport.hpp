#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace tbccl
{

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

protected:
    Listener() = default;
};

} // namespace tbccl
