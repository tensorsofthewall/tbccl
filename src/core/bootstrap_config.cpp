// Phase 50: CommunicatorOptions -> explicit, validated world description (bootstrap_config.hpp).

#include "bootstrap_config.hpp"

#include <stdexcept>
#include <string>

namespace tbccl::detail
{

namespace
{

[[noreturn]] void invalid(const std::string &message) { throw std::runtime_error("invalid_argument: " + message); }

} // namespace

ResolvedBootstrap resolve_bootstrap(const CommunicatorOptions &options)
{
    ResolvedBootstrap out;
    out.rank = options.rank;
    out.communicator_id = options.communicator_id;

    const bool has_directory = !options.rank_directory.entries.empty();
    const bool has_peers = !options.peers.empty();
    if (has_directory && has_peers) invalid("set either CommunicatorOptions::peers (legacy, world_size <= 2) or rank_directory, not both");
    if (!has_directory && !has_peers) invalid("CommunicatorOptions needs a rank_directory (or the legacy peers list)");

    if (has_directory)
    {
        out.directory = options.rank_directory;
        out.world_size = options.world_size != 0 ? options.world_size : out.directory.world_size();
    }
    else
    {
        out.legacy_peers = true;
        if (options.peers.size() > 2)
        {
            throw std::runtime_error(
                "unsupported: the legacy CommunicatorOptions::peers list supports world_size 1 or 2 (got " + std::to_string(options.peers.size()) +
                "); describe an N-rank world with rank_directory");
        }
        out.world_size = options.world_size != 0 ? options.world_size : options.peers.size();
        if (out.world_size != options.peers.size())
            invalid("world_size " + std::to_string(out.world_size) + " does not match peers.size() " + std::to_string(options.peers.size()));
        for (std::size_t r = 0; r < options.peers.size(); ++r)
        {
            const auto &p = options.peers[r];
            if (static_cast<unsigned>(p.port) + 1000U > 65535U) invalid("legacy peer " + std::to_string(r) + " port " + std::to_string(p.port) + " + 1000 exceeds 65535");
            RankEndpoint e;
            e.rank = r;
            e.control = p;
            e.data = {p.host, static_cast<std::uint16_t>(p.port + 1000)};
            out.directory.entries.push_back(e);
        }
    }

    if (out.rank >= out.world_size)
        invalid("options.rank " + std::to_string(out.rank) + " out of range for world_size " + std::to_string(out.world_size));
    validate_rank_directory(out.directory, out.world_size, out.rank, /*allow_local_zero_ports=*/false);
    return out;
}

} // namespace tbccl::detail
