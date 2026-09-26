#include <tbccl/tcp.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

    constexpr std::uint16_t kBasePort = 28515;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    // Test A: a client sends a known small payload, the server verifies
    // and echoes it, and the client verifies the echo.
    void test_small_payload()
    {
        const std::uint16_t port = kBasePort;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::vector<std::uint8_t> sent(64);

        for (std::size_t i = 0; i < sent.size(); ++i)
        {
            sent[i] = static_cast<std::uint8_t>(i);
        }

        std::thread server(
            [&]()
            {
                auto connection = listener->accept();

                std::vector<std::uint8_t> buffer(sent.size());
                connection->recv(buffer.data(), buffer.size());

                expect(buffer == sent, "server received unexpected small payload");

                connection->send(buffer.data(), buffer.size());
            });

        auto client = tbccl::tcp_connect("127.0.0.1", port, {});

        client->send(sent.data(), sent.size());

        std::vector<std::uint8_t> echoed(sent.size());
        client->recv(echoed.data(), echoed.size());

        server.join();

        expect(echoed == sent, "client received unexpected small payload echo");

        std::cout << "[PASS] test_small_payload\n";
    }

    // Test B: a >= 1 MiB payload with a deterministic pattern round-trips
    // byte-for-byte.
    void test_large_payload()
    {
        const std::uint16_t port = kBasePort + 1;
        constexpr std::size_t kSize = 1ULL * 1024 * 1024;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::vector<std::uint8_t> sent(kSize);

        for (std::size_t i = 0; i < sent.size(); ++i)
        {
            sent[i] = static_cast<std::uint8_t>((i * 2654435761u) & 0xFFu);
        }

        std::thread server(
            [&]()
            {
                auto connection = listener->accept();

                std::vector<std::uint8_t> buffer(sent.size());
                connection->recv(buffer.data(), buffer.size());

                expect(buffer == sent, "server received unexpected large payload");

                connection->send(buffer.data(), buffer.size());
            });

        auto client = tbccl::tcp_connect("127.0.0.1", port, {});

        client->send(sent.data(), sent.size());

        std::vector<std::uint8_t> echoed(sent.size());
        client->recv(echoed.data(), echoed.size());

        server.join();

        expect(echoed == sent, "client received unexpected large payload echo");

        std::cout << "[PASS] test_large_payload\n";
    }

    // Test C: the listener survives two sequential client connections,
    // matching the benchmark's persistent, one-client-at-a-time server.
    void test_persistent_listener()
    {
        const std::uint16_t port = kBasePort + 2;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::thread server(
            [&]()
            {
                for (std::uint8_t round = 0; round < 2; ++round)
                {
                    auto connection = listener->accept();

                    std::uint8_t value = 0;
                    connection->recv(&value, sizeof(value));

                    expect(
                        value == round,
                        "persistent listener received unexpected round marker");

                    connection->send(&value, sizeof(value));
                }
            });

        for (std::uint8_t round = 0; round < 2; ++round)
        {
            auto client = tbccl::tcp_connect("127.0.0.1", port, {});

            client->send(&round, sizeof(round));

            std::uint8_t echoed = 0;
            client->recv(&echoed, sizeof(echoed));

            expect(
                echoed == round,
                "persistent listener echoed unexpected round marker");
        }

        server.join();

        std::cout << "[PASS] test_persistent_listener\n";
    }

    // Test D: recv() on a connection whose peer has already closed
    // throws, rather than hanging or returning a short/garbage read.
    void test_peer_disconnect()
    {
        const std::uint16_t port = kBasePort + 3;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::thread server(
            [&]()
            {
                // Accept and immediately drop the connection.
                auto connection = listener->accept();
                (void)connection;
            });

        auto client = tbccl::tcp_connect("127.0.0.1", port, {});

        server.join();

        bool threw = false;

        try
        {
            std::uint8_t value = 0;
            client->recv(&value, sizeof(value));
        }
        catch (const std::exception &)
        {
            threw = true;
        }

        expect(threw, "recv() after peer disconnect should throw");

        std::cout << "[PASS] test_peer_disconnect\n";
    }

    // accept_for() must return nullptr, promptly, when nothing connects.
    void test_accept_for_timeout()
    {
        const std::uint16_t port = kBasePort + 4;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        const auto start = std::chrono::steady_clock::now();

        auto connection = listener->accept_for(std::chrono::milliseconds(200));

        const auto elapsed = std::chrono::steady_clock::now() - start;

        expect(connection == nullptr, "accept_for() should time out to nullptr");

        expect(
            elapsed < std::chrono::milliseconds(2000),
            "accept_for() timeout took far longer than requested");

        std::cout << "[PASS] test_accept_for_timeout\n";
    }

    // accept_for() must still return a working connection when a client
    // connects before the deadline.
    void test_accept_for_success()
    {
        const std::uint16_t port = kBasePort + 5;

        auto listener = tbccl::tcp_listen("127.0.0.1", port, {});

        std::thread server(
            [&]()
            {
                auto connection =
                    listener->accept_for(std::chrono::milliseconds(5000));

                expect(
                    connection != nullptr,
                    "accept_for() should have accepted a real connection");

                std::uint8_t value = 0;
                connection->recv(&value, sizeof(value));
                connection->send(&value, sizeof(value));
            });

        // Give the server a moment to reach accept_for() before connecting,
        // so this also exercises the "connection arrives mid-wait" path.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        auto client = tbccl::tcp_connect("127.0.0.1", port, {});

        std::uint8_t sent = 0x7A;
        client->send(&sent, sizeof(sent));

        std::uint8_t echoed = 0;
        client->recv(&echoed, sizeof(echoed));

        server.join();

        expect(echoed == sent, "accept_for() connection echo mismatch");

        std::cout << "[PASS] test_accept_for_success\n";
    }

} // namespace

int main()
{
    try
    {
        test_small_payload();
        test_large_payload();
        test_persistent_listener();
        test_peer_disconnect();
        test_accept_for_timeout();
        test_accept_for_success();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
