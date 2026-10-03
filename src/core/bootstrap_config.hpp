#pragma once

// Turns CommunicatorOptions (explicit rank directory, or the legacy N=2 `peers` list) into one validated, explicit
// description of the world. Pure: no sockets. Private to libtbccl.

#include <tbccl/communicator.hpp>

namespace tbccl::detail
{

struct ResolvedBootstrap
{
    std::size_t rank = 0;
    std::size_t world_size = 0;
    CommunicatorId communicator_id;
    RankDirectory directory;
    bool legacy_peers = false;
};

// Throws std::runtime_error ("invalid_argument: ..." / "unsupported: ...") on anything wrong with the options.
ResolvedBootstrap resolve_bootstrap(const CommunicatorOptions &options);

} // namespace tbccl::detail
