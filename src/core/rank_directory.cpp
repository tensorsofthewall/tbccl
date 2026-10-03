// Phase 50: CommunicatorId and RankDirectory validation (include/tbccl/rank_directory.hpp). Pure functions: no sockets.

#include <tbccl/rank_directory.hpp>

#include <random>
#include <set>
#include <stdexcept>
#include <utility>

namespace tbccl
{

namespace
{

[[noreturn]] void invalid(const std::string &message) { throw std::runtime_error("invalid_argument: " + message); }

int hex_value(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

std::string describe(const Endpoint &e) { return e.host + ":" + std::to_string(e.port); }

} // namespace

CommunicatorId CommunicatorId::generate()
{
    std::random_device rd;
    std::array<std::uint8_t, kBytes> bytes{};
    for (std::size_t i = 0; i < kBytes; i += 4)
    {
        const std::uint32_t word = rd();
        for (std::size_t j = 0; j < 4; ++j) bytes[i + j] = static_cast<std::uint8_t>(word >> (8 * j));
    }
    CommunicatorId id(bytes);
    if (id.is_nil()) id.bytes_[0] = 1; // never hand out the legacy nil id
    return id;
}

CommunicatorId CommunicatorId::from_hex(const std::string &hex)
{
    if (hex.size() != 2 * kBytes) invalid("communicator id must be " + std::to_string(2 * kBytes) + " hex digits");
    std::array<std::uint8_t, kBytes> bytes{};
    for (std::size_t i = 0; i < kBytes; ++i)
    {
        const int hi = hex_value(hex[2 * i]);
        const int lo = hex_value(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) invalid("communicator id contains a non-hex character");
        bytes[i] = static_cast<std::uint8_t>(hi * 16 + lo);
    }
    return CommunicatorId(bytes);
}

bool CommunicatorId::is_nil() const noexcept
{
    for (auto b : bytes_)
    {
        if (b != 0) return false;
    }
    return true;
}

std::string CommunicatorId::to_hex() const
{
    static const char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(2 * kBytes);
    for (auto b : bytes_)
    {
        out.push_back(digits[b >> 4]);
        out.push_back(digits[b & 15]);
    }
    return out;
}

std::string CommunicatorId::prefix() const { return to_hex().substr(0, 8); }

void validate_rank_directory(
    const RankDirectory &directory, std::size_t world_size, std::size_t local_rank, bool allow_local_zero_ports)
{
    if (world_size == 0) invalid("world_size must be at least 1");
    if (world_size > kMaxFullMeshWorldSize)
    {
        throw std::runtime_error(
            "unsupported: world_size " + std::to_string(world_size) + " exceeds the full-mesh limit of " +
            std::to_string(kMaxFullMeshWorldSize) + " (sparse topologies are not implemented)");
    }
    if (local_rank >= world_size) invalid("rank " + std::to_string(local_rank) + " is out of range for world_size " + std::to_string(world_size));
    if (directory.entries.size() != world_size)
    {
        invalid(
            "rank directory has " + std::to_string(directory.entries.size()) + " entries for world_size " + std::to_string(world_size));
    }

    std::set<std::size_t> seen;
    for (std::size_t i = 0; i < directory.entries.size(); ++i)
    {
        const auto &entry = directory.entries[i];
        if (!seen.insert(entry.rank).second) invalid("rank directory lists rank " + std::to_string(entry.rank) + " more than once");
        if (entry.rank >= world_size)
            invalid("rank directory entry " + std::to_string(i) + " has rank " + std::to_string(entry.rank) + " outside [0, " + std::to_string(world_size) + ")");
        if (entry.rank != i)
            invalid("rank directory entry " + std::to_string(i) + " has rank " + std::to_string(entry.rank) + "; entries must be ordered by rank");
    }

    std::set<std::pair<std::string, std::uint16_t>> endpoints;
    for (const auto &entry : directory.entries)
    {
        if (!rank_accepts_connections(entry.rank, world_size)) continue; // the last rank never listens
        const bool local_zero_ok = allow_local_zero_ports && entry.rank == local_rank;
        for (const Endpoint *e : {&entry.control, &entry.data})
        {
            const char *role = e == &entry.control ? "control" : "data";
            if (e->host.empty()) invalid("rank " + std::to_string(entry.rank) + " " + role + " endpoint has an empty host");
            if (e->port == 0 && !local_zero_ok)
                invalid("rank " + std::to_string(entry.rank) + " " + role + " endpoint " + describe(*e) + " has no port");
            if (e->port != 0 && !endpoints.insert({e->host, e->port}).second)
                invalid("rank " + std::to_string(entry.rank) + " " + role + " endpoint " + describe(*e) + " is already used by another endpoint");
        }
    }
}

} // namespace tbccl
