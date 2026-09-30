// Phase 32 Part AA-AH: host<->host large-tensor benchmark for the async
// tensor-transfer substrate (TensorCommWorker/StagingPool/ChunkPlan).
// Two processes (one per physical machine), one designated sender (the
// --source-rank), one receiver. Connection setup: rank 0 listens and
// accepts, rank 1 connects (same convention as create_tcp_world's
// "connect to every lower-ranked peer" rule for the 2-peer case),
// followed by a real PeerCapabilities exchange/negotiation (Part Y) --
// this is deliberately NOT skipped even though only one transport
// exists, to exercise the real control-plane path this phase built.
//
// Every run measures ONE (bytes, chunk_bytes, pipeline_depth) point:
// warmup iterations (untimed), then measured iterations (timed),
// reusing one persistent connection/worker/staging pool throughout
// (Part AX: never benchmark initialization as steady-state).

#include <tbccl/async_transfer.hpp>
#include <tbccl/peer_capabilities.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include "tensor/host_async_backend.hpp"
#include "tensor/tensor_backend.hpp"
#include "tensor/tensor_backend_async_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    struct PeerEndpoint
    {
        std::string host;
        std::uint16_t port = 0;
    };

    PeerEndpoint parse_peer(const std::string &text)
    {
        const auto colon = text.rfind(':');
        if (colon == std::string::npos)
        {
            throw std::runtime_error("invalid peer endpoint (expected HOST:PORT): " + text);
        }
        PeerEndpoint peer;
        peer.host = text.substr(0, colon);
        peer.port = static_cast<std::uint16_t>(std::stoul(text.substr(colon + 1)));
        return peer;
    }

    std::vector<PeerEndpoint> parse_peers(const std::string &input)
    {
        std::vector<PeerEndpoint> peers;
        std::size_t start = 0;
        while (start <= input.size())
        {
            const auto comma = input.find(',', start);
            const auto end = (comma == std::string::npos) ? input.size() : comma;
            peers.push_back(parse_peer(input.substr(start, end - start)));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        return peers;
    }

    struct Options
    {
        std::size_t rank = 0;
        std::vector<PeerEndpoint> peers;
        std::size_t source_rank = 0;
        std::size_t bytes = 1 * 1024 * 1024;
        std::size_t chunk_bytes = 0; // 0 = whole payload, one chunk
        std::size_t pipeline_depth = 2;
        std::size_t warmup = 5;
        std::size_t iterations = 20;
        std::string output;
        std::string label; // free-form tag echoed into output, for sweep bookkeeping
        std::string backend = "host"; // host, cuda-pageable, cuda-pinned, metal-shared, metal-private-staged
    };

    std::string require_value(const std::string &flag, int &i, int argc, char **argv)
    {
        if (i + 1 >= argc)
        {
            throw std::runtime_error("missing value for " + flag);
        }
        return argv[++i];
    }

    Options parse_args(int argc, char **argv)
    {
        Options options;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--rank") options.rank = std::stoul(require_value(arg, i, argc, argv));
            else if (arg == "--peers") options.peers = parse_peers(require_value(arg, i, argc, argv));
            else if (arg == "--source-rank") options.source_rank = std::stoul(require_value(arg, i, argc, argv));
            else if (arg == "--bytes") options.bytes = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--chunk-bytes") options.chunk_bytes = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--depth") options.pipeline_depth = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--warmup") options.warmup = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--iterations") options.iterations = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--output") options.output = require_value(arg, i, argc, argv);
            else if (arg == "--label") options.label = require_value(arg, i, argc, argv);
            else if (arg == "--backend") options.backend = require_value(arg, i, argc, argv);
            else throw std::runtime_error("unknown argument: " + arg);
        }
        if (options.peers.size() != 2)
        {
            throw std::runtime_error("--peers must list exactly 2 endpoints");
        }
        return options;
    }

    std::uint8_t pattern_byte(std::size_t i, std::uint32_t seed)
    {
        const std::uint64_t index = static_cast<std::uint64_t>(i);
        const std::uint64_t value =
            index * 131u + (index >> 8) * 17u + static_cast<std::uint64_t>(seed);
        return static_cast<std::uint8_t>(value & 0xffu);
    }

    double percentile(std::vector<double> values, double fraction)
    {
        std::sort(values.begin(), values.end());
        const std::size_t index = static_cast<std::size_t>(
            std::max<double>(0, std::ceil(fraction * static_cast<double>(values.size())) - 1));
        return values[std::min(index, values.size() - 1)];
    }

    std::unique_ptr<tbccl::Connection> establish_connection(const Options &options)
    {
        if (options.rank == 0)
        {
            // Bind to this rank's own specific address (peers[0].host),
            // matching create_tcp_world's proven convention -- not
            // "0.0.0.0". A wildcard bind was observed to leave accept()
            // permanently blocked on macOS even after the peer's
            // connect() completed and the kernel showed the socket
            // ESTABLISHED, while binding to the specific TB4 interface
            // address does not exhibit this.
            auto listener = tbccl::tcp_listen(
                options.peers[0].host, options.peers[0].port, {});
            return listener->accept();
        }
        return tbccl::tcp_connect(
            options.peers[0].host, options.peers[0].port, {});
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_args(argc, argv);
        const bool is_sender = (options.rank == options.source_rank);

        auto connection = establish_connection(options);

        // Real control-plane exchange (Part Y): negotiate before doing
        // anything data-plane, even though this phase only has one
        // transport to select.
        const auto local_caps = tbccl::local_capabilities();
        const auto remote_caps = tbccl::exchange_capabilities(*connection, local_caps);
        const auto negotiation = tbccl::negotiate(local_caps, remote_caps);
        if (!negotiation.ok)
        {
            throw std::runtime_error("capability negotiation failed: " + negotiation.failure_reason);
        }

        tbccl::TcpTransport transport(std::move(connection));

        const bool use_device_backend = (options.backend != "host");

        // Host path: a plain buffer + HostAsyncBackend (Part Q).
        std::vector<std::uint8_t> buffer;
        std::unique_ptr<tbccl_bench::tensor::HostAsyncBackend> host_backend;

        // Device path: a real TensorBackend (CUDA/Metal) wrapped by
        // TensorBackendAsyncAdapter (Commit 4) -- exercises the actual
        // device staging cost, not a host memcpy stand-in.
        std::unique_ptr<tbccl_bench::tensor::TensorBackend> device_backend;
        std::unique_ptr<tbccl_bench::tensor::TensorBackendAsyncAdapter> device_adapter;

        tbccl::AsyncMemoryBackend *backend = nullptr;

        if (use_device_backend)
        {
            const auto kind = tbccl_bench::tensor::parse_backend_kind(options.backend);
            if (!tbccl_bench::tensor::backend_kind_available(kind))
            {
                throw std::runtime_error("backend not available in this build: " + options.backend);
            }
            device_backend = tbccl_bench::tensor::make_backend(kind);
            device_backend->allocate(options.bytes);
            device_adapter = std::make_unique<tbccl_bench::tensor::TensorBackendAsyncAdapter>(*device_backend);
            backend = device_adapter.get();
        }
        else
        {
            buffer.resize(options.bytes);
            if (is_sender)
            {
                for (std::size_t i = 0; i < options.bytes; ++i)
                {
                    buffer[i] = pattern_byte(i, 0xA5A5A5A5u);
                }
            }
            host_backend = std::make_unique<tbccl_bench::tensor::HostAsyncBackend>(buffer.data(), buffer.size());
            backend = host_backend.get();
        }

        tbccl::TensorCommWorker worker(options.pipeline_depth, /*queue_depth=*/2);

        std::vector<double> enqueue_us;
        std::vector<double> completion_us;
        bool verify_ok = true;

        const auto chunk_plan = tbccl::plan_chunks(options.bytes, options.chunk_bytes, 1);
        const std::size_t chunk_count = chunk_plan.empty() ? 1 : chunk_plan.size();

        const std::size_t total_rounds = options.warmup + options.iterations;
        for (std::size_t round = 0; round < total_rounds; ++round)
        {
            if (use_device_backend)
            {
                if (is_sender)
                {
                    device_backend->initialize_source(0xA5A5A5A5u);
                    device_backend->prepare_source();
                }
                device_adapter->begin_transfer(chunk_count);
            }
            else if (!is_sender)
            {
                std::fill(buffer.begin(), buffer.end(), 0);
            }

            tbccl::TransferRequest request;
            request.transfer_id = static_cast<std::uint64_t>(round);
            request.direction = is_sender ? tbccl::TransferDirection::Send
                                           : tbccl::TransferDirection::Recv;
            request.backend = backend;
            request.transport = &transport;
            request.total_bytes = options.bytes;
            request.chunk_hint = options.chunk_bytes;

            const auto enqueue_start = std::chrono::steady_clock::now();
            auto work = worker.enqueue(request);
            const auto enqueue_end = std::chrono::steady_clock::now();

            work.wait();
            const auto completion_end = std::chrono::steady_clock::now();

            if (work.has_error())
            {
                throw std::runtime_error("transfer failed: " + work.error());
            }

            if (round >= options.warmup)
            {
                enqueue_us.push_back(
                    std::chrono::duration<double, std::micro>(enqueue_end - enqueue_start).count());
                completion_us.push_back(
                    std::chrono::duration<double, std::micro>(completion_end - enqueue_start).count());
            }

            if (!is_sender)
            {
                if (use_device_backend)
                {
                    if (!device_backend->verify_destination(0xA5A5A5A5u))
                    {
                        verify_ok = false;
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < options.bytes; ++i)
                    {
                        if (buffer[i] != pattern_byte(i, 0xA5A5A5A5u))
                        {
                            verify_ok = false;
                            break;
                        }
                    }
                }
            }
        }

        std::ostringstream json;
        json << "{\n"
             << "  \"label\": \"" << options.label << "\",\n"
             << "  \"rank\": " << options.rank << ",\n"
             << "  \"role\": \"" << (is_sender ? "sender" : "receiver") << "\",\n"
             << "  \"bytes\": " << options.bytes << ",\n"
             << "  \"chunk_bytes\": " << options.chunk_bytes << ",\n"
             << "  \"pipeline_depth\": " << options.pipeline_depth << ",\n"
             << "  \"warmup\": " << options.warmup << ",\n"
             << "  \"iterations\": " << options.iterations << ",\n"
             << "  \"verify_ok\": " << (verify_ok ? "true" : "false") << ",\n"
             << "  \"negotiated_transport\": \"" << tbccl::transport_kind_name(negotiation.transport) << "\",\n";

        if (is_sender)
        {
            const double median = percentile(completion_us, 0.5);
            const double p95 = percentile(completion_us, 0.95);
            const double p99 = percentile(completion_us, 0.99);
            double max_us = 0;
            double sum_us = 0;
            for (double v : completion_us) { max_us = std::max(max_us, v); sum_us += v; }
            const double mean_us = completion_us.empty() ? 0 : sum_us / static_cast<double>(completion_us.size());
            const double gib_per_s = (mean_us > 0)
                ? (static_cast<double>(options.bytes) / (1024.0 * 1024.0 * 1024.0)) / (mean_us / 1e6)
                : 0.0;

            const double enqueue_median = percentile(enqueue_us, 0.5);

            json << "  \"completion_median_us\": " << median << ",\n"
                 << "  \"completion_p95_us\": " << p95 << ",\n"
                 << "  \"completion_p99_us\": " << p99 << ",\n"
                 << "  \"completion_max_us\": " << max_us << ",\n"
                 << "  \"completion_mean_us\": " << mean_us << ",\n"
                 << "  \"gib_per_s\": " << gib_per_s << ",\n"
                 << "  \"enqueue_median_us\": " << enqueue_median << ",\n"
                 << "  \"completion_samples_us\": [";
            for (std::size_t i = 0; i < completion_us.size(); ++i)
            {
                if (i) json << ", ";
                json << completion_us[i];
            }
            json << "]\n";
        }
        else
        {
            json << "  \"note\": \"receiver: timing not authoritative, see sender output\"\n";
        }
        json << "}\n";

        if (!options.output.empty())
        {
            std::ofstream out(options.output);
            out << json.str();
        }
        std::cout << json.str();

        return verify_ok ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
