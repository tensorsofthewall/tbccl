#pragma once

// Phase 50: the Communicator wire protocol, private to libtbccl. Every integer is explicitly serialized big-endian into a
// fixed-size buffer (no struct is ever sent), so nothing depends on sizeof, padding or host endianness.
//
//   Hello / HelloAck   128 bytes   first message on every control and data connection (the dialing side sends Hello,
//                                  the accepting side answers HelloAck, carrying either Ok or a rejection reason)
//   ControlFrame       272 bytes   Abort / Goodbye on a control connection after the handshake
//   CollectiveDescriptor, CollectiveVerdict: see collective_protocol.hpp
//
// The handshake exists to fail fast and clearly: wrong communicator id, duplicate rank, rank outside [0, world_size),
// world-size mismatch, wire protocol version mismatch and an unexpected connection role are all answered with a rejection
// and surface as "protocol_mismatch: ..." on BOTH sides. See docs/phase50_runtime_audit.md.

#include <tbccl/rank_directory.hpp>
#include <tbccl/transport.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tbccl::detail
{

enum class ConnectionRole : std::uint32_t
{
    Control = 1,
    Data = 2,            // point-to-point payloads (framed)
    CollectiveData = 3,  // Phase 73: collective payloads, on their own connection so the two domains never share a byte stream
};

const char *connection_role_name(ConnectionRole role) noexcept;

enum class HelloStatus : std::uint32_t
{
    Ok = 0,
    BadVersion = 1,
    WrongCommunicator = 2,
    DuplicateRank = 3,
    RankOutOfRange = 4,
    WorldSizeMismatch = 5,
    UnexpectedRole = 6,
    UnexpectedRank = 7,
};

struct Hello
{
    std::uint32_t wire_version = kWireProtocolVersion;
    CommunicatorId communicator_id;
    std::uint32_t rank = 0;
    std::uint32_t world_size = 0;
    ConnectionRole role = ConnectionRole::Control;
};

constexpr std::size_t kHelloWireSize = 128;

// What an accepting rank expects of an incoming connection.
struct AcceptExpectation
{
    CommunicatorId communicator_id;
    std::size_t local_rank = 0;
    std::size_t world_size = 0;
    ConnectionRole role = ConnectionRole::Control;
    // Ranks that already completed a handshake of this role on this listener.
    const std::vector<bool> *already_connected = nullptr;
    // Phase 73: the data listener accepts two roles; the role in the Hello (never the arrival order) decides which connection a socket is.
    bool has_alt_role = false;
    ConnectionRole alt_role = ConnectionRole::CollectiveData;
    const std::vector<bool> *already_connected_alt = nullptr;
};

// Dialing side. Sends Hello, reads HelloAck, verifies the echoed identity. Throws "protocol_mismatch: ..." if the peer
// rejected the Hello or answered with a different identity; propagates Connection errors (including "timeout:").
// `expected_peer_rank` is the rank this connection was dialed to.
void dial_handshake(Connection &connection, const Hello &mine, std::size_t expected_peer_rank);

// Accepting side. Reads a Hello, validates it against `expect`, answers with Ok or a rejection, and returns the dialer's
// Hello. On a rejection the reply has already been sent when this throws "protocol_mismatch: ...". A connection that does
// not speak this protocol at all (bad magic, short read, timeout) throws "protocol_mismatch: not a TBCCL peer ..." without a reply.
Hello accept_handshake(Connection &connection, const AcceptExpectation &expect);

// ---- control frames -------------------------------------------------------------------------------------------------

enum class ControlFrameType : std::uint32_t
{
    Abort = 1,
    Goodbye = 2,
    // Phase 51: collective descriptors (rank -> rank 0) and verdicts (rank 0 -> rank), carried on the control plane in the frame's 256-byte area so the
    // data plane can stay sparse. FIFO per peer, like everything on a control connection.
    CollectiveDescriptor = 3,
    CollectiveVerdict = 4,
};

constexpr std::size_t kControlReasonBytes = 256;
constexpr std::size_t kControlFrameWireSize = 16 + kControlReasonBytes;

struct ControlFrame
{
    ControlFrameType type = ControlFrameType::Goodbye;
    std::uint32_t origin_rank = 0;
    std::string reason; // Abort: truncated to kControlReasonBytes - 1 on the wire
    std::vector<std::uint8_t> payload; // CollectiveDescriptor / CollectiveVerdict: up to kControlReasonBytes bytes
};

void send_control_frame(Connection &connection, const ControlFrame &frame);
ControlFrame recv_control_frame(Connection &connection);

// ---- shared helpers -------------------------------------------------------------------------------------------------

void put_u32(std::uint8_t *out, std::uint32_t value) noexcept;
std::uint32_t get_u32(const std::uint8_t *in) noexcept;
void put_u64(std::uint8_t *out, std::uint64_t value) noexcept;
std::uint64_t get_u64(const std::uint8_t *in) noexcept;
void put_text(std::uint8_t *out, std::size_t capacity, const std::string &text) noexcept; // NUL-padded, truncated
std::string get_text(const std::uint8_t *in, std::size_t capacity);

} // namespace tbccl::detail
