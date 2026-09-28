// Phase 17 heterogeneous tensor-transfer benchmark. This first slice
// covers Part E (the host-only baseline) plus Part F/G's local
// staging-only measurement, generalized over the TensorBackend
// abstraction (benchmarks/tensor/tensor_backend.hpp) so it needs no
// changes once CUDA/Metal backends are registered in a later commit
// -- --local-backend cuda-pinned etc. will simply work once
// TBCCL_ENABLE_CUDA/TBCCL_ENABLE_METAL are on.
//
// Modes implemented here:
//
//   staging-only   Single process, no networking. Measures device-to-
//                  host and host-to-device staging cost in isolation,
//                  by copying directly between two backend instances'
//                  host-visible staging areas (bypassing World
//                  entirely) -- see Part F/G/H.
//
//   network-only   Exactly 2 ranks. --source-rank streams `--sizes`
//                  payloads to its peer back-to-back; reports the
//                  sender's per-send() latency distribution and the
//                  aggregate wall-clock throughput. Always uses the
//                  host backend (Part E's network-only baseline is
//                  explicitly host-only).
//
//   latency-floor  Exactly 2 ranks. A request/response (ping-pong)
//                  exchange at small payload sizes. Reports RTT
//                  percentiles and a clearly-labeled half-RTT
//                  approximation (never claimed as a directly
//                  measured one-way latency -- see Part 17). Always
//                  uses the host backend.
//
// end-to-end and ack-calibration (Part I/J, the cross-backend
// completion-confirmed protocol) are added once CUDA/Metal backends
// exist to make them meaningful.
//
// Only the source_rank (network-only/latency-floor) or the sole
// process (staging-only) writes CSV to stdout; every rank writes
// diagnostics to stderr. This matches benchmarks/collective_bench.cpp
// and scripts/run_collective_bench.py's rank0-stdout-is-CSV
// convention.

#include "tensor/tensor_backend.hpp"

#include <tbccl/tcp_world.hpp>
#include <tbccl/world.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    using tbccl_bench::tensor::AllocationStats;
    using tbccl_bench::tensor::BackendKind;
    using tbccl_bench::tensor::make_backend;
    using tbccl_bench::tensor::parse_backend_kind;
    using tbccl_bench::tensor::TensorBackend;

    // -----------------------------------------------------------------------------
    // CLI
    // -----------------------------------------------------------------------------

    struct Options
    {
        std::size_t rank = 0;
        std::vector<tbccl::PeerEndpoint> peers;
        std::string bind_address;
        int busy_poll_us = 0;

        std::string mode; // required
        BackendKind local_backend = BackendKind::Host;
        std::size_t source_rank = 0;

        std::vector<std::size_t> sizes;
        int warmup = 20;
        int iterations = 200;
    };

    void print_usage(const char *program)
    {
        std::cerr
            << "Usage:\n"
            << "  " << program
            << " --mode staging-only|network-only|latency-floor "
            << "[--rank R --peers HOST:PORT,...] "
            << "[--bind ADDRESS] [--busy-poll MICROSECONDS] "
            << "[--local-backend host|cuda-pageable|cuda-pinned|"
               "metal-shared|metal-private-staged] "
            << "[--source-rank N] [--sizes BYTES[,BYTES...]] "
            << "[--warmup N] [--iterations N]\n\n"
            << "staging-only needs no --rank/--peers (single process, no "
               "networking). network-only and latency-floor require "
               "exactly 2 --peers and always use the host backend "
               "(--local-backend is rejected if not 'host' for these two "
               "modes) -- see Part E of the Phase 17 plan.\n\n"
            << "Example:\n"
            << "  " << program
            << " --mode latency-floor --rank 0 --peers "
            << "127.0.0.1:31000,127.0.0.1:31001 "
            << "--sizes 2,8,16,32,64,128,256,1024,4096,16384,65536\n";
    }

    tbccl::PeerEndpoint parse_peer(const std::string &text)
    {
        const std::size_t colon = text.rfind(':');

        if (colon == std::string::npos || colon == 0 ||
            colon == text.size() - 1)
        {
            throw std::runtime_error("invalid peer endpoint: " + text);
        }

        tbccl::PeerEndpoint peer;
        peer.host = text.substr(0, colon);

        const std::string port_text = text.substr(colon + 1);

        unsigned long port_value = 0;

        try
        {
            std::size_t consumed = 0;
            port_value = std::stoul(port_text, &consumed);

            if (consumed != port_text.size())
            {
                throw std::runtime_error("trailing characters");
            }
        }
        catch (const std::exception &)
        {
            throw std::runtime_error("invalid port in peer endpoint: " + text);
        }

        if (port_value == 0 || port_value > 65535)
        {
            throw std::runtime_error("invalid port in peer endpoint: " + text);
        }

        peer.port = static_cast<std::uint16_t>(port_value);

        return peer;
    }

    std::vector<tbccl::PeerEndpoint> parse_peers(const std::string &input)
    {
        std::vector<tbccl::PeerEndpoint> peers;

        std::size_t start = 0;

        while (start < input.size())
        {
            const std::size_t comma = input.find(',', start);
            const std::size_t end =
                comma == std::string::npos ? input.size() : comma;

            peers.push_back(parse_peer(input.substr(start, end - start)));

            if (comma == std::string::npos)
            {
                break;
            }

            start = comma + 1;
        }

        return peers;
    }

    unsigned long parse_unsigned(
        const std::string &name,
        const std::string &value_text)
    {
        unsigned long value = 0;

        try
        {
            std::size_t consumed = 0;
            value = std::stoul(value_text, &consumed);

            if (consumed != value_text.size())
            {
                throw std::runtime_error("trailing characters");
            }
        }
        catch (const std::exception &)
        {
            throw std::runtime_error("invalid " + name + " value: " + value_text);
        }

        return value;
    }

    std::vector<std::size_t> parse_sizes(const std::string &input)
    {
        std::vector<std::size_t> sizes;

        std::size_t start = 0;

        while (start < input.size())
        {
            const std::size_t comma = input.find(',', start);
            const std::size_t end =
                comma == std::string::npos ? input.size() : comma;

            sizes.push_back(
                static_cast<std::size_t>(
                    parse_unsigned("--sizes", input.substr(start, end - start))));

            if (comma == std::string::npos)
            {
                break;
            }

            start = comma + 1;
        }

        return sizes;
    }

    std::vector<std::size_t> default_sizes_for_mode(const std::string &mode)
    {
        if (mode == "latency-floor")
        {
            return {2, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384, 65536};
        }

        // network-only and staging-only.
        return {
            64, 256, 1024, 4096, 16384, 65536, 262144,
            1048576, 4194304, 16777216};
    }

    Options parse_options(int argc, char **argv)
    {
        Options options;

        bool have_rank = false;
        bool have_peers = false;
        bool have_sizes = false;

        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];

            auto require_value = [&](const std::string &name)
            {
                if (i + 1 >= argc)
                {
                    throw std::runtime_error("missing value for " + name);
                }

                return std::string(argv[++i]);
            };

            if (argument == "--rank")
            {
                options.rank =
                    static_cast<std::size_t>(
                        parse_unsigned(argument, require_value(argument)));
                have_rank = true;
            }
            else if (argument == "--peers")
            {
                options.peers = parse_peers(require_value(argument));
                have_peers = true;
            }
            else if (argument == "--bind")
            {
                options.bind_address = require_value(argument);
            }
            else if (argument == "--busy-poll")
            {
                const unsigned long value =
                    parse_unsigned(argument, require_value(argument));

                if (value > 1000000)
                {
                    throw std::runtime_error("--busy-poll must be <= 1000000");
                }

                options.busy_poll_us = static_cast<int>(value);
            }
            else if (argument == "--mode")
            {
                options.mode = require_value(argument);
            }
            else if (argument == "--local-backend")
            {
                options.local_backend = parse_backend_kind(require_value(argument));
            }
            else if (argument == "--source-rank")
            {
                options.source_rank =
                    static_cast<std::size_t>(
                        parse_unsigned(argument, require_value(argument)));
            }
            else if (argument == "--sizes")
            {
                options.sizes = parse_sizes(require_value(argument));
                have_sizes = true;
            }
            else if (argument == "--warmup")
            {
                options.warmup =
                    static_cast<int>(
                        parse_unsigned(argument, require_value(argument)));
            }
            else if (argument == "--iterations")
            {
                options.iterations =
                    static_cast<int>(
                        parse_unsigned(argument, require_value(argument)));
            }
            else if (argument == "--help" || argument == "-h")
            {
                print_usage(argv[0]);
                std::exit(0);
            }
            else
            {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }

        if (options.mode != "staging-only" &&
            options.mode != "network-only" &&
            options.mode != "latency-floor")
        {
            throw std::runtime_error(
                "--mode must be staging-only, network-only, or "
                "latency-floor (this build)");
        }

        if (options.mode == "staging-only")
        {
            if (have_rank || have_peers)
            {
                throw std::runtime_error(
                    "--rank/--peers are not used by --mode staging-only");
            }
        }
        else
        {
            if (!have_rank)
            {
                throw std::runtime_error("--rank is required for this mode");
            }
            if (!have_peers || options.peers.size() != 2)
            {
                throw std::runtime_error(
                    "--peers must list exactly 2 endpoints for this mode");
            }
            if (options.source_rank >= options.peers.size())
            {
                throw std::runtime_error("--source-rank is out of range");
            }
            if (options.local_backend != BackendKind::Host)
            {
                throw std::runtime_error(
                    "--local-backend must be 'host' for --mode network-only "
                    "or latency-floor (Part E's host-only baseline)");
            }
        }

        if (!have_sizes)
        {
            options.sizes = default_sizes_for_mode(options.mode);
        }

        if (options.sizes.empty())
        {
            throw std::runtime_error("--sizes must not be empty");
        }

        if (options.warmup < 0)
        {
            throw std::runtime_error("--warmup must be >= 0");
        }

        if (options.iterations <= 0)
        {
            throw std::runtime_error("--iterations must be > 0");
        }

        return options;
    }

    // -----------------------------------------------------------------------------
    // Timing / statistics
    // -----------------------------------------------------------------------------

    struct LatencyStats
    {
        double min_us = 0.0;
        double median_us = 0.0;
        double p95_us = 0.0;
        double p99_us = 0.0;
        double max_us = 0.0;
    };

    // `samples` is sorted in place.
    LatencyStats compute_stats(std::vector<double> &samples)
    {
        LatencyStats stats;

        if (samples.empty())
        {
            return stats;
        }

        std::sort(samples.begin(), samples.end());

        auto percentile = [&](double fraction)
        {
            const std::size_t index =
                std::min(
                    samples.size() - 1,
                    static_cast<std::size_t>(fraction * static_cast<double>(samples.size())));
            return samples[index];
        };

        stats.min_us = samples.front();
        stats.median_us = percentile(0.50);
        stats.p95_us = percentile(0.95);
        stats.p99_us = percentile(0.99);
        stats.max_us = samples.back();

        return stats;
    }

    double microseconds_between(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point end)
    {
        return std::chrono::duration<double, std::micro>(end - start).count();
    }

    // -----------------------------------------------------------------------------
    // CSV
    // -----------------------------------------------------------------------------

    void print_csv_header()
    {
        std::cout
            << "mode,stage,source_backend,destination_backend,payload_bytes,"
               "iterations,warmup,"
               "median_us,p95_us,p99_us,min_us,max_us,half_rtt_us,"
               "effective_GBps,effective_Gbps,"
               "allocation_type,staging_capacity_bytes,allocation_count,"
               "reuse_count\n";
    }

    void print_csv_row(
        const std::string &mode,
        const std::string &stage,
        const std::string &source_backend,
        const std::string &destination_backend,
        std::size_t payload_bytes,
        int iterations,
        int warmup,
        const LatencyStats &stats,
        double half_rtt_us, // -1 for "not applicable"
        double effective_GBps,
        const AllocationStats &alloc_stats,
        const std::string &allocation_type)
    {
        std::cout
            << mode << ','
            << stage << ','
            << source_backend << ','
            << destination_backend << ','
            << payload_bytes << ','
            << iterations << ','
            << warmup << ','
            << stats.median_us << ','
            << stats.p95_us << ','
            << stats.p99_us << ','
            << stats.min_us << ','
            << stats.max_us << ',';

        if (half_rtt_us < 0.0)
        {
            std::cout << "NA,";
        }
        else
        {
            std::cout << half_rtt_us << ',';
        }

        std::cout
            << effective_GBps << ','
            << (effective_GBps * 8.0) << ','
            << allocation_type << ','
            << alloc_stats.capacity_bytes << ','
            << alloc_stats.allocation_count << ','
            << alloc_stats.reuse_count << '\n';
    }

    // -----------------------------------------------------------------------------
    // staging-only
    // -----------------------------------------------------------------------------

    void run_staging_only(const Options &options)
    {
        const std::string backend_name =
            tbccl_bench::tensor::backend_kind_name(options.local_backend);

        print_csv_header();

        for (std::size_t size : options.sizes)
        {
            auto source = make_backend(options.local_backend);
            auto destination = make_backend(options.local_backend);

            source->allocate(size);
            destination->allocate(size);

            std::vector<double> d2h_samples;
            std::vector<double> h2d_samples;
            d2h_samples.reserve(static_cast<std::size_t>(options.iterations));
            h2d_samples.reserve(static_cast<std::size_t>(options.iterations));

            const int total = options.warmup + options.iterations;

            for (int i = 0; i < total; ++i)
            {
                const std::uint32_t seed = static_cast<std::uint32_t>(0x1000 + i);

                source->initialize_source(seed);
                source->prepare_source();

                const auto d2h_start = std::chrono::steady_clock::now();
                source->stage_device_to_host();
                const auto d2h_end = std::chrono::steady_clock::now();

                if (size > 0)
                {
                    std::memcpy(
                        destination->destination_staging_data(),
                        source->source_staging_data(),
                        size);
                }

                const auto h2d_start = std::chrono::steady_clock::now();
                destination->stage_host_to_device();
                const auto h2d_end = std::chrono::steady_clock::now();

                if (i >= options.warmup)
                {
                    d2h_samples.push_back(microseconds_between(d2h_start, d2h_end));
                    h2d_samples.push_back(microseconds_between(h2d_start, h2d_end));
                }

                if (i == total - 1)
                {
                    if (!source->verify_source(seed))
                    {
                        throw std::runtime_error(
                            "staging-only: source failed verification for size " +
                            std::to_string(size));
                    }
                    if (!destination->verify_destination(seed))
                    {
                        throw std::runtime_error(
                            "staging-only: destination failed verification for "
                            "size " +
                            std::to_string(size));
                    }
                }
            }

            const LatencyStats d2h_stats = compute_stats(d2h_samples);
            const LatencyStats h2d_stats = compute_stats(h2d_samples);

            const double d2h_gbps =
                d2h_stats.median_us > 0.0
                    ? (static_cast<double>(size) / 1e9) / (d2h_stats.median_us / 1e6)
                    : 0.0;
            const double h2d_gbps =
                h2d_stats.median_us > 0.0
                    ? (static_cast<double>(size) / 1e9) / (h2d_stats.median_us / 1e6)
                    : 0.0;

            print_csv_row(
                "staging-only", "d2h", backend_name, "NA", size,
                options.iterations, options.warmup, d2h_stats, -1.0, d2h_gbps,
                source->stats(), backend_name);

            print_csv_row(
                "staging-only", "h2d", "NA", backend_name, size,
                options.iterations, options.warmup, h2d_stats, -1.0, h2d_gbps,
                destination->stats(), backend_name);

            std::cerr
                << "[staging-only] size=" << size
                << " d2h_median_us=" << d2h_stats.median_us
                << " h2d_median_us=" << h2d_stats.median_us << '\n';
        }
    }

    // -----------------------------------------------------------------------------
    // network-only
    // -----------------------------------------------------------------------------

    void run_network_only(const Options &options, tbccl::World &world)
    {
        const bool is_source = (options.rank == options.source_rank);
        const std::size_t peer = 1 - options.rank;

        if (is_source)
        {
            print_csv_header();
        }

        for (std::size_t size : options.sizes)
        {
            auto backend = make_backend(BackendKind::Host);
            backend->allocate(size);

            const int total = options.warmup + options.iterations;

            if (is_source)
            {
                backend->initialize_source(0xC0FFEEu);
                backend->prepare_source();
                backend->stage_device_to_host();

                std::vector<double> send_samples;
                send_samples.reserve(static_cast<std::size_t>(options.iterations));

                auto wall_start = std::chrono::steady_clock::now();

                for (int i = 0; i < total; ++i)
                {
                    if (i == options.warmup)
                    {
                        wall_start = std::chrono::steady_clock::now();
                    }

                    const auto start = std::chrono::steady_clock::now();
                    backend->host_send_data(world, peer);
                    const auto end = std::chrono::steady_clock::now();

                    if (i >= options.warmup)
                    {
                        send_samples.push_back(microseconds_between(start, end));
                    }
                }

                const auto wall_end = std::chrono::steady_clock::now();
                const double wall_seconds =
                    std::chrono::duration<double>(wall_end - wall_start).count();

                const LatencyStats stats = compute_stats(send_samples);

                const double effective_GBps =
                    wall_seconds > 0.0
                        ? (static_cast<double>(size) *
                           static_cast<double>(options.iterations) / 1e9) /
                              wall_seconds
                        : 0.0;

                print_csv_row(
                    "network-only", "send", "host", "host", size,
                    options.iterations, options.warmup, stats, -1.0,
                    effective_GBps, backend->stats(), "host");

                std::cerr
                    << "[network-only] size=" << size
                    << " send_median_us=" << stats.median_us
                    << " effective_GBps=" << effective_GBps << '\n';
            }
            else
            {
                for (int i = 0; i < total; ++i)
                {
                    backend->host_recv_data(world, peer);
                }
            }
        }
    }

    // -----------------------------------------------------------------------------
    // latency-floor
    // -----------------------------------------------------------------------------

    void run_latency_floor(const Options &options, tbccl::World &world)
    {
        const bool is_source = (options.rank == options.source_rank);
        const std::size_t peer = 1 - options.rank;

        if (is_source)
        {
            print_csv_header();
        }

        for (std::size_t size : options.sizes)
        {
            auto backend = make_backend(BackendKind::Host);
            backend->allocate(size);

            const int total = options.warmup + options.iterations;

            if (is_source)
            {
                backend->initialize_source(0xA5A5A5A5u);
                backend->prepare_source();
                backend->stage_device_to_host();

                std::vector<double> rtt_samples;
                rtt_samples.reserve(static_cast<std::size_t>(options.iterations));

                for (int i = 0; i < total; ++i)
                {
                    const auto start = std::chrono::steady_clock::now();
                    backend->host_send_data(world, peer);
                    backend->host_recv_data(world, peer);
                    const auto end = std::chrono::steady_clock::now();

                    if (i >= options.warmup)
                    {
                        rtt_samples.push_back(microseconds_between(start, end));
                    }
                }

                const LatencyStats stats = compute_stats(rtt_samples);
                const double half_rtt_us = stats.median_us / 2.0;

                // Round-trip moves `size` bytes in each direction.
                const double effective_GBps =
                    stats.median_us > 0.0
                        ? (2.0 * static_cast<double>(size) / 1e9) /
                              (stats.median_us / 1e6)
                        : 0.0;

                print_csv_row(
                    "latency-floor", "rtt", "host", "host", size,
                    options.iterations, options.warmup, stats, half_rtt_us,
                    effective_GBps, backend->stats(), "host");

                std::cerr
                    << "[latency-floor] size=" << size
                    << " rtt_median_us=" << stats.median_us
                    << " half_rtt_us(approx)=" << half_rtt_us << '\n';
            }
            else
            {
                // A true echo (send back exactly the bytes just
                // received) needs one buffer used for both directions
                // -- host_send_data()/host_recv_data() are one-way
                // (source_staging_data() vs destination_staging_data()
                // are distinct buffers), so this talks to `world`
                // directly on the destination staging buffer instead.
                void *buffer = backend->destination_staging_data();

                for (int i = 0; i < total; ++i)
                {
                    world.recv(peer, buffer, size);
                    world.send(peer, buffer, size);
                }
            }
        }
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_options(argc, argv);

        if (!tbccl_bench::tensor::backend_kind_available(options.local_backend))
        {
            throw std::runtime_error(
                "tensor backend '" +
                tbccl_bench::tensor::backend_kind_name(options.local_backend) +
                "' was not compiled into this build");
        }

        if (options.mode == "staging-only")
        {
            run_staging_only(options);
            return 0;
        }

        tbccl::TcpWorldOptions world_options;
        world_options.rank = options.rank;
        world_options.peers = options.peers;
        world_options.bind_address = options.bind_address;
        world_options.tcp.busy_poll_us = options.busy_poll_us;

        auto world = tbccl::create_tcp_world(world_options);

        if (options.mode == "network-only")
        {
            run_network_only(options, *world);
        }
        else
        {
            run_latency_floor(options, *world);
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
