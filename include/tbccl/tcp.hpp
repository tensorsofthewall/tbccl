#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <tbccl/transport.hpp>

namespace tbccl
{

struct TcpOptions
{
    bool tcp_nodelay = true;

    // Linux SO_BUSY_POLL duration in microseconds.
    // 0 disables busy polling.
    int busy_poll_us = 0;
};

std::unique_ptr<Connection> tcp_connect(
    const std::string &host,
    std::uint16_t port,
    const TcpOptions &options = {});

std::unique_ptr<Listener> tcp_listen(
    const std::string &bind_address,
    std::uint16_t port,
    const TcpOptions &options = {});

// Whether this platform build can actually apply SO_BUSY_POLL. Callers
// that want to report busy-poll status (e.g. "unsupported" vs. "off")
// need this, since tcp_connect/tcp_listen succeeding for busy_poll_us
// == 0 doesn't distinguish "disabled" from "not supported here" on its
// own, and the transport itself does no such presentation logic.
bool busy_poll_supported();

} // namespace tbccl
