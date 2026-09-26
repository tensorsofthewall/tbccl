#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <tbccl/tcp.hpp>
#include <tbccl/world.hpp>

namespace tbccl
{

struct PeerEndpoint
{
    std::string host;
    std::uint16_t port = 0;
};

struct TcpWorldOptions
{
    std::size_t rank = 0;

    std::vector<PeerEndpoint> peers;

    // Empty means use peers[rank].host.
    std::string bind_address;

    TcpOptions tcp;

    std::chrono::milliseconds bootstrap_timeout{
        10000};

    std::chrono::milliseconds retry_delay{
        100};
};

// Bootstraps a World over TCP: connects to every lower-ranked peer,
// accepts from every higher-ranked peer, and performs a handshake on
// each connection before returning. World size is options.peers.size();
// it is not independently configurable. Throws on invalid options,
// handshake mismatches, or if bootstrap doesn't complete within
// options.bootstrap_timeout.
std::unique_ptr<World>
create_tcp_world(
    const TcpWorldOptions &options);

} // namespace tbccl
