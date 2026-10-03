// Phase 50: CommunicatorOptions resolution (explicit rank directory vs the legacy N=2 `peers` list). Pure: no sockets.

#include "bootstrap_config.hpp"

#include "test_utils.hpp"

#include <iostream>
#include <string>

using tbccl_test::expect;
using namespace tbccl;

namespace
{

    std::string failure(const CommunicatorOptions &o)
    {
        try
        {
            detail::resolve_bootstrap(o);
        }
        catch (const std::runtime_error &e)
        {
            return e.what();
        }
        return "";
    }

    bool has(const std::string &text, const std::string &needle) { return text.find(needle) != std::string::npos; }

    RankDirectory directory(std::size_t n)
    {
        RankDirectory d;
        for (std::size_t r = 0; r < n; ++r) d.entries.push_back({r, {"127.0.0.1", static_cast<std::uint16_t>(41000 + 7 * r)}, {"127.0.0.1", static_cast<std::uint16_t>(43000 + 3 * r)}});
        return d;
    }

} // namespace

int main()
{
    // Legacy N=2 peers: control = peers[r], data = peers[r].port + 1000, nil id, rank 0 is the only listener.
    {
        CommunicatorOptions o;
        o.rank = 1;
        o.peers = {{"10.0.0.1", 29100}, {"10.0.0.2", 29101}};
        const auto r = detail::resolve_bootstrap(o);
        expect(r.legacy_peers && r.world_size == 2 && r.rank == 1 && r.communicator_id.is_nil(), "legacy resolves to a nil-id world of 2");
        expect(r.directory.entries[0].control.host == "10.0.0.1" && r.directory.entries[0].control.port == 29100, "legacy control endpoint of rank 0");
        expect(r.directory.entries[0].data.host == "10.0.0.1" && r.directory.entries[0].data.port == 30100, "legacy data endpoint of rank 0 is control + 1000");
        expect(r.directory.entries[1].data.port == 30101, "legacy data endpoint of rank 1 (never bound: the last rank does not listen)");
    }
    {
        CommunicatorOptions o;
        o.peers = {{"127.0.0.1", 29100}};
        const auto r = detail::resolve_bootstrap(o);
        expect(r.world_size == 1, "legacy world of one");
    }
    {
        CommunicatorOptions o;
        o.peers = {{"h", 1}, {"h", 2}, {"h", 3}};
        expect(has(failure(o), "unsupported") && has(failure(o), "rank_directory"), "legacy peers refuse N>2 and point at rank_directory: " + failure(o));
        o.peers = {{"h", 1}, {"h", 2}};
        o.world_size = 3;
        expect(has(failure(o), "does not match peers.size()"), "world_size must agree with peers");
        o.world_size = 0;
        o.peers = {{"127.0.0.1", 64600}, {"127.0.0.1", 64601}};
        expect(has(failure(o), "exceeds 65535"), "port + 1000 overflow is caught: " + failure(o));
    }

    // Explicit directory
    for (std::size_t n = 1; n <= 4; ++n)
    {
        CommunicatorOptions o;
        o.rank = n - 1;
        o.rank_directory = directory(n);
        o.communicator_id = CommunicatorId::generate();
        const auto r = detail::resolve_bootstrap(o);
        expect(!r.legacy_peers && r.world_size == n && r.communicator_id == o.communicator_id, "explicit directory of " + std::to_string(n));
        expect(r.directory.entries.back().data.port == 43000 + 3 * (n - 1), "no hidden port arithmetic: the data endpoint is used as given");
    }
    {
        CommunicatorOptions o;
        o.rank_directory = directory(3);
        o.world_size = 4;
        expect(has(failure(o), "3 entries for world_size 4"), "world_size disagreeing with the directory");
        o.world_size = 3;
        o.rank = 3;
        expect(has(failure(o), "out of range"), "rank out of range");
        o.rank = 0;
        o.peers = {{"h", 1}};
        expect(has(failure(o), "not both"), "peers and rank_directory are exclusive");
        o.peers.clear();
        o.rank_directory.entries.clear();
        expect(has(failure(o), "needs a rank_directory"), "no description at all");
    }
    std::cout << "bootstrap_config_test passed\n";
    return 0;
}
