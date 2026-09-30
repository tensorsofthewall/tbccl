#include <tbccl/peer_capabilities.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{

    constexpr std::uint16_t kBasePort = 28620;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    // local_capabilities() always reports at least TCP + Host, matching
    // this build (no CUDA/Metal macros visible to the core library).
    void test_local_capabilities_baseline()
    {
        const auto caps = tbccl::local_capabilities();

        expect(caps.protocol_version == tbccl::kProtocolVersion,
               "local_capabilities() should report the current protocol version");

        bool has_tcp = false;
        for (auto t : caps.transports)
        {
            if (t == tbccl::TransportKind::Tcp) has_tcp = true;
        }
        expect(has_tcp, "local_capabilities() must always advertise TCP");

        bool has_host = false;
        for (auto m : caps.memory_backends)
        {
            if (m == tbccl::MemoryBackendKind::Host) has_host = true;
        }
        expect(has_host, "local_capabilities() must always advertise Host memory");

        std::cout << "[PASS] test_local_capabilities_baseline\n";
    }

    // exchange_capabilities() over a real loopback TCP connection
    // round-trips a non-trivial PeerCapabilities exactly.
    void test_exchange_capabilities_roundtrip()
    {
        const std::uint16_t port = kBasePort;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        tbccl::PeerCapabilities server_caps;
        server_caps.os = tbccl::OsKind::Linux;
        server_caps.transports = {tbccl::TransportKind::Tcp, tbccl::TransportKind::Rdma};
        server_caps.memory_backends = {tbccl::MemoryBackendKind::Host,
                                        tbccl::MemoryBackendKind::CudaPinned};
        server_caps.async_capabilities = {tbccl::AsyncCapability::CudaEvents};
        server_caps.max_chunk = 4 * 1024 * 1024;
        server_caps.preferred_alignment = 64;

        tbccl::PeerCapabilities client_caps;
        client_caps.os = tbccl::OsKind::MacOS;
        client_caps.transports = {tbccl::TransportKind::Tcp};
        client_caps.memory_backends = {tbccl::MemoryBackendKind::Host,
                                        tbccl::MemoryBackendKind::MetalShared};
        client_caps.async_capabilities = {tbccl::AsyncCapability::MetalEvents};
        client_caps.max_chunk = 2 * 1024 * 1024;
        client_caps.preferred_alignment = 16;

        tbccl::PeerCapabilities received_by_server;

        std::thread server(
            [&]()
            {
                auto connection = listener->accept();
                received_by_server =
                    tbccl::exchange_capabilities(*connection, server_caps);
            });

        auto client = tbccl::tcp_connect("127.0.0.1", port, {});
        const tbccl::PeerCapabilities received_by_client =
            tbccl::exchange_capabilities(*client, client_caps);

        server.join();

        // Client's send-then-recv paired with server's recv-then-send
        // ordering: client sees the server's caps.
        expect(received_by_client.os == tbccl::OsKind::Linux,
               "client should receive server's OS");
        expect(received_by_client.transports.size() == 2,
               "client should receive server's transport list intact");
        expect(received_by_client.memory_backends.size() == 2,
               "client should receive server's memory backend list intact");
        expect(received_by_client.max_chunk == 4 * 1024 * 1024,
               "client should receive server's max_chunk");
        expect(received_by_client.preferred_alignment == 64,
               "client should receive server's preferred_alignment");

        expect(received_by_server.os == tbccl::OsKind::MacOS,
               "server should receive client's OS");
        expect(received_by_server.async_capabilities.size() == 1 &&
                   received_by_server.async_capabilities[0] ==
                       tbccl::AsyncCapability::MetalEvents,
               "server should receive client's async capabilities intact");

        std::cout << "[PASS] test_exchange_capabilities_roundtrip\n";
    }

    void test_negotiate_success_common_tcp()
    {
        tbccl::PeerCapabilities a;
        a.transports = {tbccl::TransportKind::Tcp};
        a.memory_backends = {tbccl::MemoryBackendKind::Host,
                              tbccl::MemoryBackendKind::CudaPinned};

        tbccl::PeerCapabilities b;
        b.transports = {tbccl::TransportKind::Tcp, tbccl::TransportKind::Rdma};
        b.memory_backends = {tbccl::MemoryBackendKind::Host,
                              tbccl::MemoryBackendKind::MetalShared};

        const auto result = tbccl::negotiate(a, b);

        expect(result.ok, "negotiate() should succeed when both sides support TCP");
        expect(result.transport == tbccl::TransportKind::Tcp,
               "negotiate() should select TCP");
        expect(result.common_memory_backends.size() == 1 &&
                   result.common_memory_backends[0] ==
                       tbccl::MemoryBackendKind::Host,
               "negotiate() should intersect memory backends to just Host");

        std::cout << "[PASS] test_negotiate_success_common_tcp\n";
    }

    // Unsupported transport: peer only claims RDMA, which this build
    // does not implement -- negotiation must fail cleanly, not silently
    // pick something unimplemented (Part D item 20 / Part Z item 100).
    void test_negotiate_unsupported_transport()
    {
        tbccl::PeerCapabilities a;
        a.transports = {tbccl::TransportKind::Tcp};

        tbccl::PeerCapabilities b;
        b.transports = {tbccl::TransportKind::Rdma};

        const auto result = tbccl::negotiate(a, b);

        expect(!result.ok, "negotiate() must fail with no common implemented transport");
        expect(!result.failure_reason.empty(),
               "negotiate() failure must carry a reason");

        std::cout << "[PASS] test_negotiate_unsupported_transport\n";
    }

    void test_negotiate_incompatible_protocol_version()
    {
        tbccl::PeerCapabilities a;
        a.protocol_version = 1;
        a.transports = {tbccl::TransportKind::Tcp};

        tbccl::PeerCapabilities b;
        b.protocol_version = 2;
        b.transports = {tbccl::TransportKind::Tcp};

        const auto result = tbccl::negotiate(a, b);

        expect(!result.ok, "negotiate() must fail on protocol version mismatch");
        expect(result.failure_reason.find("version") != std::string::npos,
               "negotiate() failure reason should mention version");

        std::cout << "[PASS] test_negotiate_incompatible_protocol_version\n";
    }

    // Unknown capability value (e.g. a future enum member from a newer
    // peer this build doesn't recognize) must not crash negotiate() --
    // it should simply never match anything this build implements.
    void test_negotiate_unknown_capability_does_not_crash()
    {
        tbccl::PeerCapabilities a;
        a.transports = {tbccl::TransportKind::Tcp};

        tbccl::PeerCapabilities b;
        b.transports = {static_cast<tbccl::TransportKind>(99)};

        const auto result = tbccl::negotiate(a, b);

        expect(!result.ok, "negotiate() must fail gracefully on an unrecognized-only transport");

        std::cout << "[PASS] test_negotiate_unknown_capability_does_not_crash\n";
    }

    // Backend mismatch: no common memory backend is legal at the
    // negotiate() level (it only decides transport) -- ok stays true,
    // common_memory_backends is simply empty.
    void test_negotiate_backend_mismatch_still_ok_for_transport()
    {
        tbccl::PeerCapabilities a;
        a.transports = {tbccl::TransportKind::Tcp};
        a.memory_backends = {tbccl::MemoryBackendKind::CudaPinned};

        tbccl::PeerCapabilities b;
        b.transports = {tbccl::TransportKind::Tcp};
        b.memory_backends = {tbccl::MemoryBackendKind::MetalShared};

        const auto result = tbccl::negotiate(a, b);

        expect(result.ok, "negotiate() should still succeed on transport alone");
        expect(result.common_memory_backends.empty(),
               "no common memory backend should yield an empty intersection, not a failure");

        std::cout << "[PASS] test_negotiate_backend_mismatch_still_ok_for_transport\n";
    }

    // TcpTransport is a thin, correct pass-through over a real
    // Connection.
    void test_tcp_transport_passthrough()
    {
        const std::uint16_t port = kBasePort + 1;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::thread server(
            [&]()
            {
                tbccl::TcpTransport transport(listener->accept());

                std::uint8_t buffer[4] = {};
                transport.recv(buffer, sizeof(buffer));
                transport.send(buffer, sizeof(buffer));
            });

        tbccl::TcpTransport client(tbccl::tcp_connect("127.0.0.1", port, {}));

        const std::uint8_t sent[4] = {1, 2, 3, 4};
        client.send(sent, sizeof(sent));

        std::uint8_t echoed[4] = {};
        client.recv(echoed, sizeof(echoed));

        server.join();

        for (int i = 0; i < 4; ++i)
        {
            expect(echoed[i] == sent[i], "TcpTransport echo mismatch");
        }

        const auto caps = client.capabilities();
        expect(caps.reliable && caps.ordered,
               "TcpTransport must report reliable/ordered");
        expect(!caps.supports_zero_copy && !caps.supports_direct_device_memory,
               "TcpTransport must not claim zero-copy/direct-device capabilities it lacks");

        std::cout << "[PASS] test_tcp_transport_passthrough\n";
    }

} // namespace

int main()
{
    try
    {
        test_local_capabilities_baseline();
        test_exchange_capabilities_roundtrip();
        test_negotiate_success_common_tcp();
        test_negotiate_unsupported_transport();
        test_negotiate_incompatible_protocol_version();
        test_negotiate_unknown_capability_does_not_crash();
        test_negotiate_backend_mismatch_still_ok_for_transport();
        test_tcp_transport_passthrough();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
