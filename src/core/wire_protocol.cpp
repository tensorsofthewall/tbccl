// Handshake and control-frame encoding (wire_protocol.hpp). Private to libtbccl.

#include <tbccl/error.hpp>
#include "wire_protocol.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace tbccl::detail
{

namespace
{

constexpr std::uint32_t kHelloMagic = 0x54424332U; // "TBC2": deliberately different from the pre-N-rank-runtime TcpWorld "TBCL"
constexpr std::uint32_t kMessageHello = 1;
constexpr std::uint32_t kMessageHelloAck = 2;
constexpr std::size_t kHelloTextBytes = 64;

// Layout: magic(4) version(4) message(4) role(4) id(16) rank(4) world_size(4) status(4) text(64) = 108, padded to 128.
struct WireHello
{
    std::uint32_t message = 0;
    Hello hello;
    HelloStatus status = HelloStatus::Ok;
    std::string text;
};

[[noreturn]] void mismatch(const std::string &message) { throw Error(ErrorCode::ProtocolMismatch, "protocol_mismatch: " + message); }

void encode(const WireHello &w, std::uint8_t (&buffer)[kHelloWireSize])
{
    std::memset(buffer, 0, sizeof(buffer));
    put_u32(buffer + 0, kHelloMagic);
    put_u32(buffer + 4, w.hello.wire_version);
    put_u32(buffer + 8, w.message);
    put_u32(buffer + 12, static_cast<std::uint32_t>(w.hello.role));
    std::memcpy(buffer + 16, w.hello.communicator_id.bytes().data(), CommunicatorId::kBytes);
    put_u32(buffer + 32, w.hello.rank);
    put_u32(buffer + 36, w.hello.world_size);
    put_u32(buffer + 40, static_cast<std::uint32_t>(w.status));
    put_text(buffer + 44, kHelloTextBytes, w.text);
}

// Returns false (and sets `why`) if the buffer is not a TBCCL v2 hello at all.
bool decode(const std::uint8_t (&buffer)[kHelloWireSize], WireHello &w, std::string &why)
{
    if (get_u32(buffer + 0) != kHelloMagic)
    {
        why = "bad magic (peer is not a TBCCL rank of this wire protocol version: an older TBCCL, or a different service)";
        return false;
    }
    w.hello.wire_version = get_u32(buffer + 4);
    w.message = get_u32(buffer + 8);
    w.hello.role = static_cast<ConnectionRole>(get_u32(buffer + 12));
    std::array<std::uint8_t, CommunicatorId::kBytes> id{};
    std::memcpy(id.data(), buffer + 16, CommunicatorId::kBytes);
    w.hello.communicator_id = CommunicatorId(id);
    w.hello.rank = get_u32(buffer + 32);
    w.hello.world_size = get_u32(buffer + 36);
    w.status = static_cast<HelloStatus>(get_u32(buffer + 40));
    w.text = get_text(buffer + 44, kHelloTextBytes);
    return true;
}

void send_wire(Connection &c, const WireHello &w)
{
    std::uint8_t buffer[kHelloWireSize];
    encode(w, buffer);
    c.send(buffer, sizeof(buffer));
}

} // namespace

const char *connection_role_name(ConnectionRole role) noexcept
{
    switch (role)
    {
    case ConnectionRole::Control: return "control";
    case ConnectionRole::Data: return "data";
    case ConnectionRole::CollectiveData: return "collective-data";
    }
    return "unknown";
}

void put_u32(std::uint8_t *out, std::uint32_t value) noexcept
{
    out[0] = static_cast<std::uint8_t>(value >> 24);
    out[1] = static_cast<std::uint8_t>(value >> 16);
    out[2] = static_cast<std::uint8_t>(value >> 8);
    out[3] = static_cast<std::uint8_t>(value);
}

std::uint32_t get_u32(const std::uint8_t *in) noexcept
{
    return (static_cast<std::uint32_t>(in[0]) << 24) | (static_cast<std::uint32_t>(in[1]) << 16) |
           (static_cast<std::uint32_t>(in[2]) << 8) | static_cast<std::uint32_t>(in[3]);
}

void put_u64(std::uint8_t *out, std::uint64_t value) noexcept
{
    put_u32(out, static_cast<std::uint32_t>(value >> 32));
    put_u32(out + 4, static_cast<std::uint32_t>(value));
}

std::uint64_t get_u64(const std::uint8_t *in) noexcept { return (static_cast<std::uint64_t>(get_u32(in)) << 32) | get_u32(in + 4); }

void put_text(std::uint8_t *out, std::size_t capacity, const std::string &text) noexcept
{
    std::memset(out, 0, capacity);
    std::memcpy(out, text.data(), std::min(text.size(), capacity - 1));
}

std::string get_text(const std::uint8_t *in, std::size_t capacity)
{
    std::size_t n = 0;
    while (n < capacity && in[n] != 0) ++n;
    return std::string(reinterpret_cast<const char *>(in), n);
}

void dial_handshake(Connection &connection, const Hello &mine, std::size_t expected_peer_rank)
{
    WireHello out;
    out.message = kMessageHello;
    out.hello = mine;
    send_wire(connection, out);

    std::uint8_t buffer[kHelloWireSize];
    connection.recv(buffer, sizeof(buffer));
    WireHello ack;
    std::string why;
    if (!decode(buffer, ack, why)) mismatch("rank " + std::to_string(expected_peer_rank) + " answered the " + connection_role_name(mine.role) + " handshake with: " + why);
    if (ack.message != kMessageHelloAck) mismatch("rank " + std::to_string(expected_peer_rank) + " sent an unexpected message type " + std::to_string(ack.message));
    if (ack.status != HelloStatus::Ok)
        mismatch("rank " + std::to_string(expected_peer_rank) + " rejected the " + connection_role_name(mine.role) + " connection: " + ack.text);
    if (ack.hello.wire_version != mine.wire_version)
        mismatch("rank " + std::to_string(expected_peer_rank) + " speaks wire protocol " + std::to_string(ack.hello.wire_version) + ", this rank " + std::to_string(mine.wire_version));
    if (ack.hello.communicator_id != mine.communicator_id)
        mismatch("rank " + std::to_string(expected_peer_rank) + " belongs to communicator " + ack.hello.communicator_id.prefix() + ", this rank to " + mine.communicator_id.prefix());
    if (ack.hello.rank != expected_peer_rank)
        mismatch("dialed rank " + std::to_string(expected_peer_rank) + " but the peer identifies as rank " + std::to_string(ack.hello.rank));
    if (ack.hello.world_size != mine.world_size)
        mismatch("rank " + std::to_string(expected_peer_rank) + " has world_size " + std::to_string(ack.hello.world_size) + ", this rank " + std::to_string(mine.world_size));
}

Hello accept_handshake(Connection &connection, const AcceptExpectation &expect)
{
    std::uint8_t buffer[kHelloWireSize];
    try
    {
        connection.recv(buffer, sizeof(buffer));
    }
    catch (const std::exception &e)
    {
        mismatch(std::string("not a TBCCL peer: no complete hello arrived (") + e.what() + ")");
    }
    WireHello in;
    std::string why;
    if (!decode(buffer, in, why)) mismatch("not a TBCCL peer: " + why);
    if (in.message != kMessageHello) mismatch("not a TBCCL peer: first message type " + std::to_string(in.message) + " is not a hello");

    WireHello reply;
    reply.message = kMessageHelloAck;
    reply.hello.communicator_id = expect.communicator_id;
    reply.hello.rank = static_cast<std::uint32_t>(expect.local_rank);
    reply.hello.world_size = static_cast<std::uint32_t>(expect.world_size);
    reply.hello.role = expect.role;
    reply.status = HelloStatus::Ok;

    const auto &h = in.hello;
    const bool alt = expect.has_alt_role && h.role == expect.alt_role;
    if (alt) reply.hello.role = expect.alt_role;
    const std::string who = "rank " + std::to_string(h.rank);
    if (h.wire_version != kWireProtocolVersion)
    {
        reply.status = HelloStatus::BadVersion;
        reply.text = "wire protocol " + std::to_string(h.wire_version) + " != " + std::to_string(kWireProtocolVersion);
    }
    else if (h.communicator_id != expect.communicator_id)
    {
        reply.status = HelloStatus::WrongCommunicator;
        reply.text = "communicator id " + h.communicator_id.prefix() + " != " + expect.communicator_id.prefix();
    }
    else if (h.world_size != expect.world_size)
    {
        reply.status = HelloStatus::WorldSizeMismatch;
        reply.text = "world_size " + std::to_string(h.world_size) + " != " + std::to_string(expect.world_size);
    }
    else if (h.rank >= expect.world_size)
    {
        reply.status = HelloStatus::RankOutOfRange;
        reply.text = "rank " + std::to_string(h.rank) + " outside [0, " + std::to_string(expect.world_size) + ")";
    }
    else if (h.role != expect.role && !alt)
    {
        reply.status = HelloStatus::UnexpectedRole;
        reply.text = std::string("a ") + connection_role_name(h.role) + " connection arrived on the " + connection_role_name(expect.role) + " endpoint";
    }
    else if (h.rank <= expect.local_rank)
    {
        reply.status = HelloStatus::UnexpectedRank;
        reply.text = "rank " + std::to_string(h.rank) + " should not connect to rank " + std::to_string(expect.local_rank) + " (the lower rank dials)";
    }
    else if (const auto *have = alt ? expect.already_connected_alt : expect.already_connected; have != nullptr && h.rank < have->size() && (*have)[h.rank])
    {
        reply.status = HelloStatus::DuplicateRank;
        reply.text = "duplicate rank " + std::to_string(h.rank) + ": another process already holds this rank";
    }
    send_wire(connection, reply);
    if (reply.status != HelloStatus::Ok) mismatch("rejected " + who + " " + connection_role_name(h.role) + " connection: " + reply.text);
    return h;
}

void send_control_frame(Connection &connection, const ControlFrame &frame)
{
    std::uint8_t buffer[kControlFrameWireSize];
    std::memset(buffer, 0, sizeof(buffer));
    put_u32(buffer + 0, static_cast<std::uint32_t>(frame.type));
    put_u32(buffer + 4, frame.origin_rank);
    if (frame.type == ControlFrameType::CollectiveDescriptor || frame.type == ControlFrameType::CollectiveVerdict)
    {
        const std::size_t n = std::min(frame.payload.size(), kControlReasonBytes);
        put_u32(buffer + 8, static_cast<std::uint32_t>(n));
        std::memcpy(buffer + 16, frame.payload.data(), n);
    }
    else
    {
        put_u32(buffer + 8, static_cast<std::uint32_t>(std::min(frame.reason.size(), kControlReasonBytes - 1)));
        put_text(buffer + 16, kControlReasonBytes, frame.reason);
    }
    connection.send(buffer, sizeof(buffer));
}

ControlFrame recv_control_frame(Connection &connection)
{
    std::uint8_t buffer[kControlFrameWireSize];
    connection.recv(buffer, sizeof(buffer));
    ControlFrame frame;
    frame.type = static_cast<ControlFrameType>(get_u32(buffer + 0));
    frame.origin_rank = get_u32(buffer + 4);
    if (frame.type == ControlFrameType::CollectiveDescriptor || frame.type == ControlFrameType::CollectiveVerdict)
    {
        const std::size_t n = std::min<std::size_t>(get_u32(buffer + 8), kControlReasonBytes);
        frame.payload.assign(buffer + 16, buffer + 16 + n);
    }
    else
    {
        frame.reason = get_text(buffer + 16, kControlReasonBytes);
    }
    return frame;
}

} // namespace tbccl::detail
