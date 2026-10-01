// Phase 38 Part Y: the heterogeneous N=2 SUM AllReduce benchmark.
// Linux CUDA rank <-> Mac Metal-shared rank (or host<->host for a
// portable reference run), comparing:
//   --algo sync  : the EXISTING, unmodified tbccl::reduce()+broadcast()
//                  (Part AC/AB's "existing synchronous collective
//                  baseline"), called in-place directly over a
//                  TensorBackend's host-visible staging pointer.
//   --algo async : the new n2_all_reduce_tensor() (hetero_allreduce.hpp),
//                  using the real async tensor-transfer substrate.
// Both operate on the SAME backend abstraction (tensor/tensor_backend.hpp
// for host/metal-shared; CudaChunkedAsyncBackend for the CUDA async path,
// a plain TensorBackend CudaPinned instance for the CUDA sync path, since
// tbccl::reduce()/broadcast() need a World, which CudaChunkedAsyncBackend
// was never designed to support), so the only thing that differs between
// --algo sync and --algo async is the communication substrate itself.

#include <tbccl/collectives.hpp>
#include <tbccl/hetero_allreduce.hpp>
#include <tbccl/peer_capabilities.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/tcp_world.hpp>
#include <tbccl/transport.hpp>

#include "tensor/host_dual_buffer_backend.hpp"
#include "tensor/host_reduce_backend.hpp"
#include "tensor/metal_shared_direct_async_backend.hpp"
#include "tensor/tensor_backend.hpp"
#include "tensor/tensor_backend_async_adapter.hpp"
#include "tensor/host_async_backend.hpp"

#if defined(TBCCL_ENABLE_CUDA)
#include "tensor/cuda_chunked_async_backend.hpp"
#include "tensor/cuda_reduce_backend.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using Clock = std::chrono::steady_clock;

    double now_us(Clock::time_point start)
    {
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    }

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
        std::vector<PeerEndpoint> peers; // base port pair; world uses it as-is, transport uses port+1
        std::size_t root = 0;
        std::string backend = "host"; // host, cuda, metal-shared
        std::size_t count = 1024 * 1024; // float32 elements
        std::size_t chunk_bytes = 262144; // Phase 35 anchor, Part X
        std::size_t warmup = 3;
        std::size_t iterations = 10;
        std::string algo = "both"; // sync, async, both
        std::string metal_async_path = "adapter"; // adapter, direct
        std::string label;
        std::string output;
    };

    std::string require_value(const std::string &flag, int &i, int argc, char **argv)
    {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
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
            else if (arg == "--root") options.root = std::stoul(require_value(arg, i, argc, argv));
            else if (arg == "--backend") options.backend = require_value(arg, i, argc, argv);
            else if (arg == "--count") options.count = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--chunk-bytes") options.chunk_bytes = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--warmup") options.warmup = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--iterations") options.iterations = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--algo") options.algo = require_value(arg, i, argc, argv);
            else if (arg == "--metal-async-path") options.metal_async_path = require_value(arg, i, argc, argv);
            else if (arg == "--label") options.label = require_value(arg, i, argc, argv);
            else if (arg == "--output") options.output = require_value(arg, i, argc, argv);
            else throw std::runtime_error("unknown argument: " + arg);
        }
        if (options.peers.size() != 2) throw std::runtime_error("--peers must list exactly 2 endpoints");
        if (options.root > 1) throw std::runtime_error("--root must be 0 or 1 (N=2 only)");
        return options;
    }

    double percentile(std::vector<double> values, double fraction)
    {
        if (values.empty()) return 0.0;
        std::sort(values.begin(), values.end());
        const std::size_t index = static_cast<std::size_t>(
            std::max<double>(0, std::ceil(fraction * static_cast<double>(values.size())) - 1));
        return values[std::min(index, values.size() - 1)];
    }

    float value_for(std::size_t i, std::uint32_t seed, std::uint32_t modulus)
    {
        return static_cast<float>((i + seed) % modulus);
    }

    // Per-rank deterministic seed/modulus (Part F): small integer-valued
    // floats, sums stay exactly representable, so verification uses exact
    // equality.
    constexpr std::uint32_t kSeedRank0 = 11, kModRank0 = 127;
    constexpr std::uint32_t kSeedRank1 = 97, kModRank1 = 113;

    struct Stats
    {
        double median_us = 0, p95_us = 0, p99_us = 0, mean_us = 0, max_us = 0;
    };

    Stats compute_stats(const std::vector<double> &samples)
    {
        Stats s;
        if (samples.empty()) return s;
        double sum = 0, mx = 0;
        for (double v : samples) { sum += v; mx = std::max(mx, v); }
        s.mean_us = sum / static_cast<double>(samples.size());
        s.max_us = mx;
        s.median_us = percentile(samples, 0.5);
        s.p95_us = percentile(samples, 0.95);
        s.p99_us = percentile(samples, 0.99);
        return s;
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_args(argc, argv);
        const std::size_t rank = options.rank;
        const std::size_t count = options.count;
        const std::size_t bytes = count * sizeof(float);
        const bool is_root = (rank == options.root);
        const std::uint32_t local_seed = (rank == 0) ? kSeedRank0 : kSeedRank1;
        const std::uint32_t local_mod = (rank == 0) ? kModRank0 : kModRank1;

        std::vector<float> expected(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            expected[i] = value_for(i, kSeedRank0, kModRank0) + value_for(i, kSeedRank1, kModRank1);
        }

        const bool run_sync = (options.algo == "sync" || options.algo == "both");
        const bool run_async = (options.algo == "async" || options.algo == "both");

        std::ostringstream json;
        json << "{\n";
        json << "  \"label\": \"" << options.label << "\",\n";
        json << "  \"rank\": " << rank << ",\n";
        json << "  \"root\": " << options.root << ",\n";
        json << "  \"backend\": \"" << options.backend << "\",\n";
        json << "  \"metal_async_path\": \"" << options.metal_async_path << "\",\n";
        json << "  \"count\": " << count << ",\n";
        json << "  \"bytes\": " << bytes << ",\n";
        json << "  \"chunk_bytes\": " << options.chunk_bytes << ",\n";
        json << "  \"warmup\": " << options.warmup << ",\n";
        json << "  \"iterations\": " << options.iterations << ",\n";

        // A World connection is always established, even for --algo async
        // alone: besides serving the sync reference phase, it doubles as a
        // readiness barrier for the async phase's separate TcpTransport
        // connection below (Part CI -- do not regress to sleep-only
        // coordination; barrier() is an existing, unmodified collective).
        tbccl::TcpWorldOptions world_opts;
        world_opts.rank = rank;
        for (const auto &p : options.peers) world_opts.peers.push_back({p.host, p.port});
        auto world = tbccl::create_tcp_world(world_opts);

        // -----------------------------------------------------------
        // SYNC reference: unmodified tbccl::reduce()+broadcast() over a
        // TensorBackend's host-visible staging pointer (Part AC/AB).
        // -----------------------------------------------------------
        if (run_sync)
        {
            std::unique_ptr<tbccl_bench::tensor::TensorBackend> backend;
            if (options.backend == "host")
            {
                backend = tbccl_bench::tensor::make_backend(tbccl_bench::tensor::BackendKind::Host);
            }
            else if (options.backend == "cuda")
            {
                backend = tbccl_bench::tensor::make_backend(tbccl_bench::tensor::BackendKind::CudaPinned);
            }
            else if (options.backend == "metal-shared")
            {
                backend = tbccl_bench::tensor::make_backend(tbccl_bench::tensor::BackendKind::MetalShared);
            }
            else
            {
                throw std::runtime_error("unknown --backend: " + options.backend);
            }
            backend->allocate(bytes);

            std::vector<double> completion_us;
            bool verify_ok = true;

            const std::size_t total_rounds = options.warmup + options.iterations;
            for (std::size_t round = 0; round < total_rounds; ++round)
            {
                // Fill local input directly into source staging (host-
                // visible for every backend kind), outside the timer.
                auto *src = static_cast<float *>(
                    const_cast<void *>(backend->source_staging_data()));
                // source_staging_data() is declared const; the backend
                // guarantees it IS the real backing store for host/
                // cuda-pinned/metal-shared (never a read-only copy), so
                // writing the deterministic test input here (before any
                // timed region, matching every prior phase's benchmark-
                // methodology convention) is safe and is how every
                // existing device backend's own initialize_source()
                // would have filled it anyway.
                for (std::size_t i = 0; i < count; ++i) src[i] = value_for(i, local_seed, local_mod);

                const auto t0 = Clock::now();
                tbccl::reduce(*world, src, src, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum, options.root);
                tbccl::broadcast(*world, src, bytes, options.root);
                backend->stage_host_to_device();
                const double elapsed = now_us(t0);

                if (round >= options.warmup)
                {
                    completion_us.push_back(elapsed);
                }
            }

            // One untimed, fully-verified round (Part Z).
            {
                auto *src = static_cast<float *>(const_cast<void *>(backend->source_staging_data()));
                for (std::size_t i = 0; i < count; ++i) src[i] = value_for(i, local_seed, local_mod);
                tbccl::reduce(*world, src, src, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum, options.root);
                tbccl::broadcast(*world, src, bytes, options.root);
                backend->stage_host_to_device();
                for (std::size_t i = 0; i < count; ++i)
                {
                    if (src[i] != expected[i]) { verify_ok = false; break; }
                }
            }

            const auto stats = compute_stats(completion_us);
            json << "  \"sync_median_us\": " << stats.median_us << ",\n";
            json << "  \"sync_p95_us\": " << stats.p95_us << ",\n";
            json << "  \"sync_p99_us\": " << stats.p99_us << ",\n";
            json << "  \"sync_mean_us\": " << stats.mean_us << ",\n";
            json << "  \"sync_verify_ok\": " << (verify_ok ? "true" : "false") << ",\n";
        }

        // -----------------------------------------------------------
        // ASYNC: n2_all_reduce_tensor() over the real async substrate.
        // -----------------------------------------------------------
        if (run_async)
        {
            auto async_listener_port = static_cast<std::uint16_t>(options.peers[0].port + 1000);
            std::unique_ptr<tbccl::Connection> connection;
            std::unique_ptr<tbccl::Listener> listener;
            if (rank == 0)
            {
                listener = tbccl::tcp_listen(options.peers[0].host, async_listener_port, {});
            }
            // Both ranks already share a working World (above); use it as
            // a readiness barrier so rank 1's connect() cannot race ahead
            // of rank 0's listen() -- a fixed sleep() cannot guarantee
            // this ordering (Part CI).
            tbccl::barrier(*world);
            if (rank == 0)
            {
                connection = listener->accept();
            }
            else
            {
                connection = tbccl::tcp_connect(options.peers[0].host, async_listener_port, {});
            }

            const auto local_caps = tbccl::local_capabilities();
            const auto remote_caps = tbccl::exchange_capabilities(*connection, local_caps);
            const auto negotiation = tbccl::negotiate(local_caps, remote_caps);
            if (!negotiation.ok) throw std::runtime_error("capability negotiation failed: " + negotiation.failure_reason);

            tbccl::TcpTransport transport(std::move(connection));
            tbccl::TensorCommWorker worker(/*pipeline_depth=*/2, /*queue_depth=*/2);

            const auto chunk_plan = tbccl::plan_chunks(bytes, options.chunk_bytes, 1);
            const std::size_t chunk_count = chunk_plan.empty() ? 1 : chunk_plan.size();

            std::vector<double> completion_us;
            bool verify_ok = true;

            // Backend setup, kind-specific (Part W).
            tbccl::AsyncMemoryBackend *recv_backend = nullptr;
            tbccl::AsyncMemoryBackend *send_backend = nullptr;
            tbccl::LocalReduceBackend *reduce_backend = nullptr;

            // Host path.
            std::unique_ptr<tbccl_bench::tensor::HostDualBufferBackend> host_backend;
            std::unique_ptr<tbccl_bench::tensor::HostReduceBackend> host_reduce;

            // Metal-shared path.
            std::unique_ptr<tbccl_bench::tensor::TensorBackend> metal_backend;
            std::unique_ptr<tbccl_bench::tensor::TensorBackendAsyncAdapter> metal_adapter;
            std::unique_ptr<tbccl_bench::tensor::MetalSharedDirectAsyncBackend> metal_direct;
            std::unique_ptr<tbccl_bench::tensor::HostAsyncBackend> metal_send_from_result;
            std::unique_ptr<tbccl_bench::tensor::HostReduceBackend> metal_reduce;
            const bool metal_async_direct = (options.metal_async_path == "direct");

#if defined(TBCCL_ENABLE_CUDA)
            std::unique_ptr<tbccl_bench::tensor::CudaChunkedAsyncBackend> cuda_backend;
            std::unique_ptr<tbccl_bench::tensor::CudaReduceBackend> cuda_reduce;
#endif

            if (options.backend == "host")
            {
                host_backend = std::make_unique<tbccl_bench::tensor::HostDualBufferBackend>(bytes);
                recv_backend = host_backend.get();
                send_backend = host_backend.get();
                if (is_root)
                {
                    host_reduce = std::make_unique<tbccl_bench::tensor::HostReduceBackend>(
                        host_backend->source_data(), host_backend->destination_data());
                    reduce_backend = host_reduce.get();
                }
            }
            else if (options.backend == "cuda")
            {
#if defined(TBCCL_ENABLE_CUDA)
                cuda_backend = std::make_unique<tbccl_bench::tensor::CudaChunkedAsyncBackend>();
                cuda_backend->allocate(bytes, options.chunk_bytes == 0 ? bytes : options.chunk_bytes);
                cuda_backend->initialize_source(local_seed == kSeedRank0 ? 0xC0DEu : 0xBEEFu);
                recv_backend = cuda_backend.get();
                send_backend = cuda_backend.get();
                if (is_root)
                {
                    cuda_reduce = std::make_unique<tbccl_bench::tensor::CudaReduceBackend>(*cuda_backend, nullptr);
                    reduce_backend = cuda_reduce.get();
                }
#else
                throw std::runtime_error("--backend cuda requires a CUDA-enabled build");
#endif
            }
            else if (options.backend == "metal-shared")
            {
                if (options.metal_async_path != "adapter" && options.metal_async_path != "direct")
                    throw std::runtime_error("unknown --metal-async-path: " + options.metal_async_path);

                metal_backend = tbccl_bench::tensor::make_backend(tbccl_bench::tensor::BackendKind::MetalShared);
                metal_backend->allocate(bytes);

                tbccl::AsyncMemoryBackend *metal_async = nullptr;
                if (metal_async_direct)
                {
                    metal_direct = std::make_unique<tbccl_bench::tensor::MetalSharedDirectAsyncBackend>(*metal_backend);
                    metal_async = metal_direct.get();
                }
                else
                {
                    metal_adapter = std::make_unique<tbccl_bench::tensor::TensorBackendAsyncAdapter>(*metal_backend);
                    metal_async = metal_adapter.get();
                }
                recv_backend = metal_async;
                if (is_root)
                {
                    // Root's send-back leg must read the post-reduce
                    // result from destination_staging_data() (mutable),
                    // not source_staging_data() (const) -- see
                    // docs/phase38_collective_design.md Part 4/5.
                    metal_send_from_result = std::make_unique<tbccl_bench::tensor::HostAsyncBackend>(
                        metal_backend->destination_staging_data(), bytes);
                    send_backend = metal_send_from_result.get();
                    metal_reduce = std::make_unique<tbccl_bench::tensor::HostReduceBackend>(
                        metal_backend->destination_staging_data(), metal_backend->source_staging_data());
                    reduce_backend = metal_reduce.get();
                }
                else
                {
                    send_backend = metal_async;
                }
            }
            else
            {
                throw std::runtime_error("unknown --backend: " + options.backend);
            }

            const std::size_t total_rounds = options.warmup + options.iterations;
            for (std::size_t round = 0; round < total_rounds; ++round)
            {
                // Prepare this rank's local input, outside the timer.
                if (options.backend == "host")
                {
                    auto *src = static_cast<float *>(host_backend->source_data());
                    for (std::size_t i = 0; i < count; ++i) src[i] = value_for(i, local_seed, local_mod);
                }
                else if (options.backend == "metal-shared")
                {
                    metal_backend->initialize_source(local_seed);
                    metal_backend->prepare_source();
                    metal_backend->stage_device_to_host();
                    auto *src = static_cast<float *>(
                        const_cast<void *>(metal_backend->source_staging_data()));
                    for (std::size_t i = 0; i < count; ++i) src[i] = value_for(i, local_seed, local_mod);
                }
#if defined(TBCCL_ENABLE_CUDA)
                else if (options.backend == "cuda")
                {
                    std::vector<float> local_values(count);
                    for (std::size_t i = 0; i < count; ++i) local_values[i] = value_for(i, local_seed, local_mod);
                    tbccl_bench::tensor::cuda_copy_host_to_device(
                        local_values.data(), cuda_backend->source_device_ptr(), bytes);
                }
#endif

                if (options.backend == "metal-shared")
                {
                    if (metal_async_direct) metal_direct->begin_transfer(chunk_count);
                    else metal_adapter->begin_transfer(chunk_count);
                    if (is_root) { /* send-leg uses metal_send_from_result, no begin_transfer needed */ }
                }

                const auto t0 = Clock::now();
                tbccl::n2_all_reduce_tensor(
                    transport, worker, *recv_backend, *send_backend, reduce_backend,
                    rank, options.root, bytes, options.chunk_bytes, count, tbccl::DataType::Float32);
                const double elapsed = now_us(t0);

                if (round >= options.warmup)
                {
                    completion_us.push_back(elapsed);
                }
            }

            // One untimed, fully-verified round.
            {
                if (options.backend == "host")
                {
                    auto *src = static_cast<float *>(host_backend->source_data());
                    for (std::size_t i = 0; i < count; ++i) src[i] = value_for(i, local_seed, local_mod);
                }
                else if (options.backend == "metal-shared")
                {
                    metal_backend->initialize_source(local_seed);
                    metal_backend->prepare_source();
                    metal_backend->stage_device_to_host();
                    auto *src = static_cast<float *>(
                        const_cast<void *>(metal_backend->source_staging_data()));
                    for (std::size_t i = 0; i < count; ++i) src[i] = value_for(i, local_seed, local_mod);
                    if (metal_async_direct) metal_direct->begin_transfer(chunk_count);
                    else metal_adapter->begin_transfer(chunk_count);
                }
#if defined(TBCCL_ENABLE_CUDA)
                else if (options.backend == "cuda")
                {
                    std::vector<float> local_values(count);
                    for (std::size_t i = 0; i < count; ++i) local_values[i] = value_for(i, local_seed, local_mod);
                    tbccl_bench::tensor::cuda_copy_host_to_device(
                        local_values.data(), cuda_backend->source_device_ptr(), bytes);
                }
#endif

                tbccl::n2_all_reduce_tensor(
                    transport, worker, *recv_backend, *send_backend, reduce_backend,
                    rank, options.root, bytes, options.chunk_bytes, count, tbccl::DataType::Float32);

                std::vector<float> result(count);
                if (options.backend == "host")
                {
                    const void *ptr = is_root ? host_backend->source_data() : host_backend->destination_data();
                    std::memcpy(result.data(), ptr, bytes);
                }
                else if (options.backend == "metal-shared")
                {
                    const void *ptr = metal_backend->destination_staging_data();
                    std::memcpy(result.data(), ptr, bytes);
                }
#if defined(TBCCL_ENABLE_CUDA)
                else if (options.backend == "cuda")
                {
                    const void *ptr = is_root ? cuda_backend->source_device_ptr() : cuda_backend->destination_device_ptr();
                    tbccl_bench::tensor::cuda_copy_device_to_host(ptr, result.data(), bytes);
                }
#endif
                verify_ok = true;
                for (std::size_t i = 0; i < count; ++i)
                {
                    if (result[i] != expected[i]) { verify_ok = false; break; }
                }
            }

            const auto stats = compute_stats(completion_us);
            json << "  \"async_median_us\": " << stats.median_us << ",\n";
            json << "  \"async_p95_us\": " << stats.p95_us << ",\n";
            json << "  \"async_p99_us\": " << stats.p99_us << ",\n";
            json << "  \"async_mean_us\": " << stats.mean_us << ",\n";
            json << "  \"async_verify_ok\": " << (verify_ok ? "true" : "false") << ",\n";
        }

        json << "  \"done\": true\n";
        json << "}\n";

        if (!options.output.empty())
        {
            std::ofstream out(options.output);
            out << json.str();
        }
        std::cout << json.str();
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
