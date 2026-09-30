// Phase 33 diagnostic-only: measures raw Transport::send/recv in a
// loop with the identical ack-based timing scope as
// tbccl_async_transfer_bench, but WITHOUT TensorCommWorker at all --
// isolates whether the queue/thread-handoff mechanism itself costs
// anything beyond what a direct Transport call would cost on its own.
// Not part of the async substrate's public surface; a standalone
// measurement tool only.

#include <tbccl/peer_capabilities.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    struct PeerEndpoint { std::string host; std::uint16_t port = 0; };

    PeerEndpoint parse_peer(const std::string &text)
    {
        const auto colon = text.rfind(':');
        return {text.substr(0, colon), static_cast<std::uint16_t>(std::stoul(text.substr(colon + 1)))};
    }
}

int main(int argc, char **argv)
{
    try
    {
        std::size_t rank = 0, bytes = 64 * 1024 * 1024, warmup = 5, iterations = 20, source_rank = 0;
        std::vector<PeerEndpoint> peers;
        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            auto next = [&]() { return std::string(argv[++i]); };
            if (arg == "--rank") rank = std::stoul(next());
            else if (arg == "--peers") {
                std::string v = next(); auto comma = v.find(',');
                peers = {parse_peer(v.substr(0, comma)), parse_peer(v.substr(comma + 1))};
            }
            else if (arg == "--bytes") bytes = std::stoull(next());
            else if (arg == "--warmup") warmup = std::stoull(next());
            else if (arg == "--iterations") iterations = std::stoull(next());
            else if (arg == "--source-rank") source_rank = std::stoul(next());
        }
        const bool is_sender = (rank == source_rank);

        std::unique_ptr<tbccl::Connection> connection;
        if (rank == 0)
        {
            auto listener = tbccl::tcp_listen(peers[0].host, peers[0].port, {});
            connection = listener->accept();
        }
        else
        {
            connection = tbccl::tcp_connect(peers[0].host, peers[0].port, {});
        }
        const auto local_caps = tbccl::local_capabilities();
        tbccl::exchange_capabilities(*connection, local_caps);
        tbccl::TcpTransport transport(std::move(connection));

        std::vector<std::uint8_t> buffer(bytes, is_sender ? 0x42 : 0);

        std::vector<double> completion_us;
        const std::size_t total = warmup + iterations;
        for (std::size_t round = 0; round < total; ++round)
        {
            const auto start = std::chrono::steady_clock::now();
            if (is_sender)
            {
                transport.send(buffer.data(), buffer.size());
            }
            else
            {
                transport.recv(buffer.data(), buffer.size());
            }
            std::uint8_t ack = 0;
            if (is_sender) transport.recv(&ack, sizeof(ack));
            else { ack = 1; transport.send(&ack, sizeof(ack)); }
            const auto end = std::chrono::steady_clock::now();
            if (round >= warmup)
                completion_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }

        if (is_sender)
        {
            std::sort(completion_us.begin(), completion_us.end());
            double median = completion_us[completion_us.size() / 2];
            double gib_s = (bytes / (1024.0*1024.0*1024.0)) / (median / 1e6);
            std::cout << "raw_transport median_us=" << median << " gib_s=" << gib_s << "\n";
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
