#include <tbccl/tcp_world.hpp>

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace tbccl
{
namespace
{

    // -----------------------------------------------------------------------------
    // Bootstrap handshake wire protocol (internal only; not part of the
    // public API). Six fixed-width fields, explicitly serialized in
    // network byte order as a flat 24-byte buffer — no raw struct is
    // ever sent, so this doesn't depend on sizeof/padding/endianness.
    // -----------------------------------------------------------------------------

    constexpr std::uint32_t kMagic = 0x5442434CU; // "TBCL"
    constexpr std::uint32_t kProtocolVersion = 1;

    constexpr std::uint32_t kMessageHello = 1;
    constexpr std::uint32_t kMessageAck = 2;

    constexpr std::size_t kHandshakeWireSize = 6 * sizeof(std::uint32_t);

    struct Handshake
    {
        std::uint32_t magic = kMagic;
        std::uint32_t protocol_version = kProtocolVersion;
        std::uint32_t message_type = 0;
        std::uint32_t source_rank = 0;
        std::uint32_t destination_rank = 0;
        std::uint32_t world_size = 0;
    };

    void encode_u32(std::uint32_t value, std::uint8_t *out)
    {
        out[0] = static_cast<std::uint8_t>((value >> 24) & 0xFFU);
        out[1] = static_cast<std::uint8_t>((value >> 16) & 0xFFU);
        out[2] = static_cast<std::uint8_t>((value >> 8) & 0xFFU);
        out[3] = static_cast<std::uint8_t>(value & 0xFFU);
    }

    std::uint32_t decode_u32(const std::uint8_t *in)
    {
        return (static_cast<std::uint32_t>(in[0]) << 24) |
               (static_cast<std::uint32_t>(in[1]) << 16) |
               (static_cast<std::uint32_t>(in[2]) << 8) |
               static_cast<std::uint32_t>(in[3]);
    }

    void send_handshake(Connection &connection, const Handshake &handshake)
    {
        std::uint8_t buffer[kHandshakeWireSize];

        encode_u32(handshake.magic, buffer + 0);
        encode_u32(handshake.protocol_version, buffer + 4);
        encode_u32(handshake.message_type, buffer + 8);
        encode_u32(handshake.source_rank, buffer + 12);
        encode_u32(handshake.destination_rank, buffer + 16);
        encode_u32(handshake.world_size, buffer + 20);

        connection.send(buffer, sizeof(buffer));
    }

    Handshake recv_handshake(Connection &connection)
    {
        std::uint8_t buffer[kHandshakeWireSize];

        connection.recv(buffer, sizeof(buffer));

        Handshake handshake;

        handshake.magic = decode_u32(buffer + 0);
        handshake.protocol_version = decode_u32(buffer + 4);
        handshake.message_type = decode_u32(buffer + 8);
        handshake.source_rank = decode_u32(buffer + 12);
        handshake.destination_rank = decode_u32(buffer + 16);
        handshake.world_size = decode_u32(buffer + 20);

        return handshake;
    }

    void validate_common(
        const Handshake &handshake,
        std::uint32_t expected_type,
        std::size_t world_size)
    {
        if (handshake.magic != kMagic)
        {
            throw std::runtime_error(
                "handshake magic mismatch (got 0x" +
                std::to_string(handshake.magic) + ")");
        }

        if (handshake.protocol_version != kProtocolVersion)
        {
            throw std::runtime_error(
                "handshake protocol version mismatch: peer sent " +
                std::to_string(handshake.protocol_version) +
                ", expected " + std::to_string(kProtocolVersion));
        }

        if (handshake.message_type != expected_type)
        {
            throw std::runtime_error(
                "handshake message type mismatch: got " +
                std::to_string(handshake.message_type) +
                ", expected " + std::to_string(expected_type));
        }

        if (handshake.world_size != world_size)
        {
            throw std::runtime_error(
                "handshake from rank " +
                std::to_string(handshake.source_rank) +
                " declared world size " +
                std::to_string(handshake.world_size) +
                ", but local rank expects world size " +
                std::to_string(world_size));
        }
    }

    // Validates an incoming HELLO against the deterministic topology
    // rule: only higher-numbered ranks may connect in to us.
    void validate_hello(
        const Handshake &hello,
        std::size_t local_rank,
        std::size_t world_size)
    {
        validate_common(hello, kMessageHello, world_size);

        if (hello.destination_rank != local_rank)
        {
            throw std::runtime_error(
                "handshake destination rank mismatch: expected " +
                std::to_string(local_rank) + ", got " +
                std::to_string(hello.destination_rank));
        }

        if (hello.source_rank >= world_size)
        {
            throw std::runtime_error(
                "invalid peer rank " + std::to_string(hello.source_rank) +
                " for world size " + std::to_string(world_size));
        }

        if (hello.source_rank == local_rank)
        {
            throw std::runtime_error(
                "received handshake claiming local rank " +
                std::to_string(local_rank) + " as its source rank");
        }

        if (hello.source_rank <= local_rank)
        {
            throw std::runtime_error(
                "unexpected connection from lower rank " +
                std::to_string(hello.source_rank) + " (local rank " +
                std::to_string(local_rank) +
                " only accepts incoming connections from higher ranks)");
        }
    }

    void validate_ack(
        const Handshake &ack,
        std::size_t local_rank,
        std::size_t peer_rank,
        std::size_t world_size)
    {
        validate_common(ack, kMessageAck, world_size);

        if (ack.source_rank != peer_rank)
        {
            throw std::runtime_error(
                "ACK source rank mismatch: expected " +
                std::to_string(peer_rank) + ", got " +
                std::to_string(ack.source_rank));
        }

        if (ack.destination_rank != local_rank)
        {
            throw std::runtime_error(
                "ACK destination rank mismatch: expected " +
                std::to_string(local_rank) + ", got " +
                std::to_string(ack.destination_rank));
        }
    }

    // -----------------------------------------------------------------------------
    // Options validation
    // -----------------------------------------------------------------------------

    void validate_options(const TcpWorldOptions &options)
    {
        if (options.peers.empty())
        {
            throw std::runtime_error(
                "TcpWorldOptions.peers must not be empty");
        }

        if (options.rank >= options.peers.size())
        {
            throw std::runtime_error(
                "rank " + std::to_string(options.rank) +
                " is out of range for world size " +
                std::to_string(options.peers.size()));
        }

        for (std::size_t i = 0; i < options.peers.size(); ++i)
        {
            const PeerEndpoint &peer = options.peers[i];

            if (peer.host.empty())
            {
                throw std::runtime_error(
                    "peer " + std::to_string(i) + " has an empty host");
            }

            if (peer.port == 0)
            {
                throw std::runtime_error(
                    "peer " + std::to_string(i) + " has port 0");
            }
        }

        for (std::size_t i = 0; i < options.peers.size(); ++i)
        {
            for (std::size_t j = i + 1; j < options.peers.size(); ++j)
            {
                if (options.peers[i].host == options.peers[j].host &&
                    options.peers[i].port == options.peers[j].port)
                {
                    throw std::runtime_error(
                        "duplicate peer endpoint " + options.peers[i].host +
                        ":" + std::to_string(options.peers[i].port) +
                        " (ranks " + std::to_string(i) + " and " +
                        std::to_string(j) + ")");
                }
            }
        }

        if (options.bootstrap_timeout.count() <= 0)
        {
            throw std::runtime_error("bootstrap_timeout must be > 0");
        }

        if (options.retry_delay.count() <= 0)
        {
            throw std::runtime_error("retry_delay must be > 0");
        }
    }

    // -----------------------------------------------------------------------------
    // Outgoing connect with retry/backoff
    // -----------------------------------------------------------------------------

    std::unique_ptr<Connection> connect_with_retry(
        std::size_t local_rank,
        std::size_t peer_rank,
        const PeerEndpoint &peer,
        const TcpOptions &tcp_options,
        std::chrono::milliseconds timeout,
        std::chrono::milliseconds retry_delay)
    {
        const auto deadline = std::chrono::steady_clock::now() + timeout;

        std::string last_error = "no attempt made";

        while (true)
        {
            try
            {
                return tcp_connect(peer.host, peer.port, tcp_options);
            }
            catch (const std::exception &error)
            {
                last_error = error.what();
            }

            if (std::chrono::steady_clock::now() >= deadline)
            {
                throw std::runtime_error(
                    "rank " + std::to_string(local_rank) +
                    " timed out connecting to rank " +
                    std::to_string(peer_rank) + " at " + peer.host + ":" +
                    std::to_string(peer.port) + " after " +
                    std::to_string(timeout.count()) + " ms: " + last_error);
            }

            std::this_thread::sleep_for(retry_delay);
        }
    }

    // -----------------------------------------------------------------------------
    // TcpWorld
    // -----------------------------------------------------------------------------

    class TcpWorld final : public World
    {
    public:
        TcpWorld(
            std::size_t rank,
            std::vector<std::unique_ptr<Connection>> connections)
            : rank_(rank), connections_(std::move(connections)) {}

        std::size_t rank() const noexcept override
        {
            return rank_;
        }

        std::size_t size() const noexcept override
        {
            return connections_.size();
        }

        void send(
            std::size_t peer,
            const void *data,
            std::size_t bytes) override
        {
            validate_peer(peer);
            connections_[peer]->send(data, bytes);
        }

        void recv(
            std::size_t peer,
            void *data,
            std::size_t bytes) override
        {
            validate_peer(peer);
            connections_[peer]->recv(data, bytes);
        }

    private:
        void validate_peer(std::size_t peer) const
        {
            if (peer >= connections_.size())
            {
                throw std::runtime_error(
                    "invalid peer rank " + std::to_string(peer) +
                    " for world size " +
                    std::to_string(connections_.size()));
            }

            if (peer == rank_)
            {
                throw std::runtime_error(
                    "cannot send/recv to local rank " +
                    std::to_string(rank_));
            }
        }

        std::size_t rank_;
        std::vector<std::unique_ptr<Connection>> connections_;
    };

} // namespace

std::unique_ptr<World> create_tcp_world(const TcpWorldOptions &options)
{
    validate_options(options);

    const std::size_t world_size = options.peers.size();
    const std::size_t rank = options.rank;
    const PeerEndpoint &self = options.peers[rank];

    const std::string bind_address =
        options.bind_address.empty() ? self.host : options.bind_address;

    auto listener = tcp_listen(bind_address, self.port, options.tcp);

    std::vector<std::unique_ptr<Connection>> connections(world_size);

    // Outgoing: connect to every lower-ranked peer. Lower rank always
    // accepts, higher rank always connects, so exactly one TCP
    // connection exists per unordered rank pair.
    for (std::size_t peer_rank = 0; peer_rank < rank; ++peer_rank)
    {
        auto connection =
            connect_with_retry(
                rank,
                peer_rank,
                options.peers[peer_rank],
                options.tcp,
                options.bootstrap_timeout,
                options.retry_delay);

        Handshake hello;
        hello.message_type = kMessageHello;
        hello.source_rank = static_cast<std::uint32_t>(rank);
        hello.destination_rank = static_cast<std::uint32_t>(peer_rank);
        hello.world_size = static_cast<std::uint32_t>(world_size);

        send_handshake(*connection, hello);

        const Handshake ack = recv_handshake(*connection);

        validate_ack(ack, rank, peer_rank, world_size);

        connections[peer_rank] = std::move(connection);
    }

    // Incoming: accept from every higher-ranked peer, bounded by a
    // single shared deadline for this whole phase.
    const std::size_t incoming_count = world_size - 1 - rank;

    if (incoming_count > 0)
    {
        const auto deadline =
            std::chrono::steady_clock::now() + options.bootstrap_timeout;

        std::size_t accepted = 0;

        while (accepted < incoming_count)
        {
            const auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now());

            if (remaining.count() <= 0)
            {
                throw std::runtime_error(
                    "rank " + std::to_string(rank) + " timed out after " +
                    std::to_string(options.bootstrap_timeout.count()) +
                    " ms waiting for " +
                    std::to_string(incoming_count - accepted) +
                    " more incoming connection(s)");
            }

            auto connection = listener->accept_for(remaining);

            if (!connection)
            {
                // Loop back around; the deadline check above will
                // throw once the budget is actually exhausted.
                continue;
            }

            const Handshake hello = recv_handshake(*connection);

            validate_hello(hello, rank, world_size);

            if (connections[hello.source_rank] != nullptr)
            {
                throw std::runtime_error(
                    "duplicate connection from rank " +
                    std::to_string(hello.source_rank));
            }

            Handshake ack;
            ack.message_type = kMessageAck;
            ack.source_rank = static_cast<std::uint32_t>(rank);
            ack.destination_rank = hello.source_rank;
            ack.world_size = static_cast<std::uint32_t>(world_size);

            send_handshake(*connection, ack);

            connections[hello.source_rank] = std::move(connection);
            ++accepted;
        }
    }

    // The bootstrap listener has served its purpose; it goes out of
    // scope here and closes automatically (RAII).

    return std::make_unique<TcpWorld>(rank, std::move(connections));
}

} // namespace tbccl
