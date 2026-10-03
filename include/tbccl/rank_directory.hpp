#pragma once

// The framework-neutral identity and addressing types of an N-rank Communicator. libtbccl never discovers peers itself
// (no c10d::Store, Python, Redis, etcd or exo KVS here): an adapter or application gathers, for every rank, where it
// listens, and hands the result over as a RankDirectory together with one CommunicatorId shared by the ranks.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tbccl
{

// The version of the Communicator wire protocol (Hello, control frames, collective descriptors). It is independent of
// the package version and of kProtocolVersion in peer_capabilities.hpp (the capability-record layout). Version 1 was the
// implicit original N=2 bootstrap (TcpWorld hello + capability record); version 2 added the N-rank bootstrap; version 3 adds the algorithm id to the
// collective verdict and the forced-algorithm field to the descriptor. Older versions are not wire-compatible and are rejected by the handshake.
constexpr std::uint32_t kWireProtocolVersion = 3;

// The full-mesh runtime opens two sockets per rank pair. Validated at world_size 1..4; larger meshes are refused until a
// sparse topology exists (the N>2 collective-selection work).
constexpr std::size_t kMaxFullMeshWorldSize = 8;

// A 128-bit identity token that keeps independent communicators between the same hosts from cross-connecting. It is not a
// secret and has no cryptographic property. The all-zero value is the "nil" id used by the legacy N=2 `peers` bootstrap.
class CommunicatorId
{
public:
    static constexpr std::size_t kBytes = 16;

    CommunicatorId() = default;
    explicit CommunicatorId(const std::array<std::uint8_t, kBytes> &bytes) : bytes_(bytes) {}

    // A fresh random id (std::random_device). One rank generates it and the adapter distributes it.
    static CommunicatorId generate();

    // Parses exactly 32 hex digits; throws std::runtime_error ("invalid_argument: ...") otherwise.
    static CommunicatorId from_hex(const std::string &hex);

    const std::array<std::uint8_t, kBytes> &bytes() const noexcept { return bytes_; }
    bool is_nil() const noexcept;
    std::string to_hex() const;
    // First 8 hex digits: enough to tell communicators apart in a log line.
    std::string prefix() const;

    friend bool operator==(const CommunicatorId &a, const CommunicatorId &b) noexcept { return a.bytes_ == b.bytes_; }
    friend bool operator!=(const CommunicatorId &a, const CommunicatorId &b) noexcept { return !(a == b); }

private:
    std::array<std::uint8_t, kBytes> bytes_{};
};

struct Endpoint
{
    std::string host;
    std::uint16_t port = 0;
};

// Where one rank listens. `control` carries the handshake, capability exchange and Abort/Goodbye frames; `data` carries
// payloads and collective descriptors. They are independent: no arithmetic relationship between the ports is assumed.
struct RankEndpoint
{
    std::size_t rank = 0;
    Endpoint control;
    Endpoint data;
};

struct RankDirectory
{
    std::vector<RankEndpoint> entries;

    std::size_t world_size() const noexcept { return entries.size(); }
};

// Connection rule of the full mesh: the lower rank of every pair connects, the higher rank accepts. A rank therefore
// listens only if a higher rank exists, and rank world_size-1 never binds anything.
constexpr bool rank_accepts_connections(std::size_t rank, std::size_t world_size) noexcept { return rank + 1 < world_size; }

// Validates `directory` before any network work. Throws std::runtime_error with an "invalid_argument: ..." message when:
// the entry count differs from `world_size`; ranks are not exactly 0..world_size-1 in order (duplicate, missing or out of
// range); `local_rank` is out of range; `world_size` is 0 or exceeds kMaxFullMeshWorldSize; a rank that must accept
// connections has an empty host or a zero port (the local rank's own ports may be zero when listeners are pre-bound, see
// `allow_local_zero_ports`); a rank's control and data endpoints coincide; or two ranks advertise the same endpoint.
void validate_rank_directory(
    const RankDirectory &directory, std::size_t world_size, std::size_t local_rank, bool allow_local_zero_ports);

} // namespace tbccl
