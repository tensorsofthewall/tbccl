// Phase 50: CommunicatorId and RankDirectory validation (pure functions; no sockets).

#include <tbccl/rank_directory.hpp>

#include "test_utils.hpp"

#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

using tbccl_test::expect;

namespace
{

    tbccl::RankDirectory make(std::size_t n, std::uint16_t base = 40000)
    {
        tbccl::RankDirectory d;
        for (std::size_t r = 0; r < n; ++r)
        {
            tbccl::RankEndpoint e;
            e.rank = r;
            e.control = {"127.0.0.1", static_cast<std::uint16_t>(base + 2 * r)};
            e.data = {"127.0.0.1", static_cast<std::uint16_t>(base + 2 * r + 1)};
            d.entries.push_back(e);
        }
        return d;
    }

    std::string failure(const tbccl::RankDirectory &d, std::size_t world, std::size_t rank, bool zero_ok = false)
    {
        try
        {
            tbccl::validate_rank_directory(d, world, rank, zero_ok);
        }
        catch (const std::runtime_error &e)
        {
            return e.what();
        }
        return "";
    }

    bool contains(const std::string &text, const std::string &needle) { return text.find(needle) != std::string::npos; }

} // namespace

int main()
{
    // CommunicatorId
    const auto a = tbccl::CommunicatorId::generate();
    const auto b = tbccl::CommunicatorId::generate();
    expect(!a.is_nil() && !b.is_nil(), "generated ids are never nil");
    expect(a != b, "two generated ids differ");
    expect(tbccl::CommunicatorId::from_hex(a.to_hex()) == a, "hex round trip");
    expect(a.to_hex().size() == 32 && a.prefix() == a.to_hex().substr(0, 8), "hex and prefix shape");
    expect(tbccl::CommunicatorId().is_nil() && tbccl::CommunicatorId() == tbccl::CommunicatorId(), "default id is the nil id");
    expect(contains(([&] { try { tbccl::CommunicatorId::from_hex("abc"); } catch (const std::runtime_error &e) { return std::string(e.what()); } return std::string(); })(), "invalid_argument"), "short hex is rejected");
    expect(contains(([&] { try { tbccl::CommunicatorId::from_hex(std::string(31, 'a') + "z"); } catch (const std::runtime_error &e) { return std::string(e.what()); } return std::string(); })(), "non-hex"), "non-hex is rejected");
    std::set<std::string> distinct;
    for (int i = 0; i < 256; ++i) distinct.insert(tbccl::CommunicatorId::generate().to_hex());
    expect(distinct.size() == 256, "256 generated ids are distinct");

    // The connection rule
    expect(tbccl::rank_accepts_connections(0, 2) && !tbccl::rank_accepts_connections(1, 2), "rank 1 of 2 binds nothing");
    expect(tbccl::rank_accepts_connections(2, 4) && !tbccl::rank_accepts_connections(3, 4), "only the last rank never accepts");
    expect(!tbccl::rank_accepts_connections(0, 1), "world_size 1 accepts nothing");

    // Valid directories
    for (std::size_t n = 1; n <= 4; ++n)
    {
        for (std::size_t r = 0; r < n; ++r) expect(failure(make(n), n, r).empty(), "valid directory of " + std::to_string(n) + " ranks");
    }

    // Count / range / order
    expect(contains(failure(make(3), 4, 0), "3 entries for world_size 4"), "wrong entry count");
    expect(contains(failure(make(3), 3, 3), "out of range"), "local rank out of range");
    expect(contains(failure(make(2), 0, 0), "at least 1"), "world_size 0");
    expect(contains(failure(make(9), 9, 0), "full-mesh limit"), "world_size above the mesh limit");
    {
        auto d = make(3);
        d.entries[2].rank = 1;
        expect(contains(failure(d, 3, 0), "more than once"), "duplicate rank");
        d = make(3);
        d.entries[2].rank = 7;
        expect(contains(failure(d, 3, 0), "outside"), "rank >= world_size");
        d = make(3);
        std::swap(d.entries[0], d.entries[1]);
        expect(contains(failure(d, 3, 0), "ordered by rank"), "entries out of order");
    }

    // Endpoints
    {
        auto d = make(3);
        d.entries[1].control.port = 0;
        expect(contains(failure(d, 3, 0), "has no port"), "a remote accepting rank needs a port");
        expect(failure(d, 3, 1, true).empty(), "the local rank may leave ports zero when listeners are pre-bound");
        expect(!failure(d, 3, 1, false).empty(), "local zero ports rejected without pre-bound listeners");
        d = make(3);
        d.entries[0].control.host.clear();
        expect(contains(failure(d, 3, 1), "empty host"), "empty host");
        d = make(3);
        d.entries[0].data = d.entries[0].control;
        expect(contains(failure(d, 3, 0), "already used"), "control == data");
        d = make(3);
        d.entries[1].control = d.entries[0].control;
        expect(contains(failure(d, 3, 0), "already used"), "two ranks advertise one endpoint");
        d = make(3);
        d.entries[2].control = {};
        d.entries[2].data = {};
        expect(failure(d, 3, 0).empty(), "the last rank never listens, so its endpoints may be empty");
    }

    std::cout << "rank_directory_test passed\n";
    return 0;
}
