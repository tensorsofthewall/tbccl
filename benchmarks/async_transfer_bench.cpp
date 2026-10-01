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
#include "tensor/metal_shared_direct_async_backend.hpp"
#include "tensor/tensor_backend.hpp"
#include "tensor/tensor_backend_async_adapter.hpp"

#if defined(TBCCL_ENABLE_CUDA)
#include "tensor/cuda_chunked_async_backend.hpp"
#endif

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
        std::string metal_async_path = "adapter"; // adapter, direct -- only meaningful for --backend metal-shared
        // Phase 34 diagnostic: the receiver's per-round std::fill +
        // byte-pattern verify loop runs BEFORE the ack is sent, so it
        // is included in the sender's own measured completion time
        // (the ack is the sender's proof of destination-visible
        // completion -- Part D). Disabling verification isolates
        // whether this receiver-side CPU work, not Transport itself,
        // is inflating the measured regression. Default true to match
        // every prior phase's correctness-checked measurements.
        bool verify = true;
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
            else if (arg == "--metal-async-path") options.metal_async_path = require_value(arg, i, argc, argv);
            else if (arg == "--verify") {
                const std::string v = require_value(arg, i, argc, argv);
                if (v == "on") options.verify = true;
                else if (v == "off") options.verify = false;
                else throw std::runtime_error("--verify must be 'on' or 'off'");
            }
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

        const bool use_cuda_chunked_backend = (options.backend == "cuda-chunked");
        const bool use_device_backend = (options.backend != "host") && !use_cuda_chunked_backend;

        // Host path: a plain buffer + HostAsyncBackend (Part Q).
        std::vector<std::uint8_t> buffer;
        std::unique_ptr<tbccl_bench::tensor::HostAsyncBackend> host_backend;

        // Device path: a real TensorBackend (CUDA/Metal) wrapped by
        // TensorBackendAsyncAdapter (Commit 4) -- exercises the actual
        // device staging cost, not a host memcpy stand-in.
        std::unique_ptr<tbccl_bench::tensor::TensorBackend> device_backend;
        std::unique_ptr<tbccl_bench::tensor::TensorBackendAsyncAdapter> device_adapter;
        std::unique_ptr<tbccl_bench::tensor::MetalSharedDirectAsyncBackend> device_direct;
        const bool use_metal_direct =
            (options.backend == "metal-shared") && (options.metal_async_path == "direct");

#if defined(TBCCL_ENABLE_CUDA)
        // Phase 35: the true per-chunk async CUDA D2H/H2D staging
        // backend (docs/phase35_device_pipeline_design.md), selected
        // via --backend cuda-chunked. Bypasses TensorBackend/
        // TensorBackendAsyncAdapter's whole-buffer-only staging
        // entirely.
        std::unique_ptr<tbccl_bench::tensor::CudaChunkedAsyncBackend> cuda_chunked_backend;
#endif

        tbccl::AsyncMemoryBackend *backend = nullptr;

        if (use_cuda_chunked_backend)
        {
#if defined(TBCCL_ENABLE_CUDA)
            cuda_chunked_backend = std::make_unique<tbccl_bench::tensor::CudaChunkedAsyncBackend>();
            const std::size_t max_chunk_bytes =
                (options.chunk_bytes == 0) ? options.bytes : options.chunk_bytes;
            cuda_chunked_backend->allocate(options.bytes, max_chunk_bytes);
            backend = cuda_chunked_backend.get();
#else
            throw std::runtime_error("--backend cuda-chunked requires a CUDA-enabled build");
#endif
        }
        else if (use_device_backend)
        {
            if (options.metal_async_path != "adapter" && options.metal_async_path != "direct")
                throw std::runtime_error("unknown --metal-async-path: " + options.metal_async_path);

            const auto kind = tbccl_bench::tensor::parse_backend_kind(options.backend);
            if (!tbccl_bench::tensor::backend_kind_available(kind))
            {
                throw std::runtime_error("backend not available in this build: " + options.backend);
            }
            device_backend = tbccl_bench::tensor::make_backend(kind);
            device_backend->allocate(options.bytes);
            if (use_metal_direct)
            {
                device_direct = std::make_unique<tbccl_bench::tensor::MetalSharedDirectAsyncBackend>(*device_backend);
                backend = device_direct.get();
            }
            else
            {
                device_adapter = std::make_unique<tbccl_bench::tensor::TensorBackendAsyncAdapter>(*device_backend);
                backend = device_adapter.get();
            }
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

        // Phase 34 finding: a per-round std::fill()+byte-verify on the
        // receiver, done INSIDE this timed loop, was found to inflate
        // every measured round's completion time by tens of
        // milliseconds -- the ack for round N is only sent after the
        // receiver's own work.wait() returns, but the receiver doesn't
        // re-post its next recv() until it finishes that round's
        // verify, so large sends back up against TCP flow control
        // waiting on a receiver that's busy verifying instead of
        // reading. tbccl_tensor_transfer_bench never had this problem:
        // it verifies exactly ONCE, in an explicitly untimed round
        // after the whole measured loop (see its own comment: "One
        // untimed, full-byte-verified round -- never folded into the
        // timing above"). This benchmark now matches that convention.
        auto run_one_transfer = [&](std::uint64_t transfer_id)
        {
            if (use_device_backend)
            {
                if (is_sender)
                {
                    device_backend->initialize_source(0xA5A5A5A5u);
                    device_backend->prepare_source();
                }
                if (use_metal_direct) device_direct->begin_transfer(chunk_count);
                else device_adapter->begin_transfer(chunk_count);
            }
#if defined(TBCCL_ENABLE_CUDA)
            else if (use_cuda_chunked_backend && is_sender)
            {
                cuda_chunked_backend->initialize_source(0xA5A5A5A5u);
            }
#endif

            tbccl::TransferRequest request;
            request.transfer_id = transfer_id;
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

            if (work.has_error())
            {
                throw std::runtime_error("transfer failed: " + work.error());
            }

            // Part AE: match the synchronous benchmark's timing scope
            // exactly -- completion_confirmed_us there is the SENDER's
            // own local time from source-ready to receiving a 1-byte
            // application-level ACK the receiver only sends after its
            // OWN host_recv_data()+stage_host_to_device() have
            // returned. Without this, TransferWork::wait() for a Send
            // only proves the local kernel accepted the bytes into its
            // send buffer -- not that the peer's destination commit
            // ever happened. This ack round-trip is a benchmark-level
            // measurement concern, not a TensorCommWorker/TransferWork
            // semantic change (the library's own completion contract
            // is unchanged); it exists solely so this tool's numbers
            // are comparable to tbccl_tensor_transfer_bench's.
            std::uint8_t ack = 0;
            if (is_sender)
            {
                transport.recv(&ack, sizeof(ack));
            }
            else
            {
                ack = 1;
                transport.send(&ack, sizeof(ack));
            }
            const auto completion_end = std::chrono::steady_clock::now();
            return std::make_pair(enqueue_end - enqueue_start, completion_end - enqueue_start);
        };

        const std::size_t total_rounds = options.warmup + options.iterations;
        for (std::size_t round = 0; round < total_rounds; ++round)
        {
            const auto [enqueue_dur, completion_dur] = run_one_transfer(static_cast<std::uint64_t>(round));
            if (round >= options.warmup)
            {
                enqueue_us.push_back(std::chrono::duration<double, std::micro>(enqueue_dur).count());
                completion_us.push_back(std::chrono::duration<double, std::micro>(completion_dur).count());
            }
        }

        if (options.verify)
        {
            if (!use_device_backend && !use_cuda_chunked_backend && !is_sender)
            {
                std::fill(buffer.begin(), buffer.end(), 0);
            }
            run_one_transfer(static_cast<std::uint64_t>(total_rounds));
            if (!is_sender)
            {
                if (use_device_backend)
                {
                    verify_ok = device_backend->verify_destination(0xA5A5A5A5u);
                }
#if defined(TBCCL_ENABLE_CUDA)
                else if (use_cuda_chunked_backend)
                {
                    verify_ok = cuda_chunked_backend->verify_destination(0xA5A5A5A5u);
                }
#endif
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
             << "  \"backend\": \"" << options.backend << "\",\n"
             << "  \"metal_async_path\": \"" << options.metal_async_path << "\",\n"
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
