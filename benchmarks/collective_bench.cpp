// Generic collective benchmark harness. Currently supports:
//   --collective all-gather --algorithm reference|ring
//
// The CLI/CSV shape is deliberately generic so later phases can add
// more collectives/algorithms without replacing this tool — see
// select_all_gather_algorithm() and run_all_gather_benchmark() as the
// pattern to follow for e.g. reduce-scatter or all-reduce.
//
// Bandwidth formulas (see collectives.hpp for AllGather's semantics):
//   N = world size, B = bytes_per_rank contributed by each rank,
//   T = measured collective duration (max across ranks, per iteration
//   median).
//   algbw = (N * B) / T             — total gathered bytes per rank
//                                      over time.
//   busbw = algbw * (N - 1) / N
//         = ((N - 1) * B) / T       — useful peer-delivered traffic
//                                      per rank for an ideal AllGather.
// Bandwidth is reported in decimal GB/s (1 GB = 1e9 bytes), not GiB/s.

#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "all_gather_internal.hpp"

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

    struct Options
    {
        std::size_t rank = 0;
        std::vector<tbccl::PeerEndpoint> peers;
        std::string bind_address;
        int busy_poll_us = 0;
        std::string collective = "all-gather";
        std::string algorithm = "reference";
        std::vector<std::size_t> sizes;
        int iterations = 100;
        int warmup = 20;
    };

    void print_usage(const char *program)
    {
        std::cerr
            << "Usage:\n"
            << "  " << program
            << " --rank R --peers HOST:PORT,HOST:PORT,... "
            << "[--bind ADDRESS] [--busy-poll MICROSECONDS] "
            << "[--collective all-gather] [--algorithm reference|ring] "
            << "[--sizes BYTES[,BYTES...]] [--iterations N] "
            << "[--warmup N]\n\n"
            << "Example:\n"
            << "  " << program
            << " --rank 0 --peers "
            << "127.0.0.1:29500,127.0.0.1:29501,127.0.0.1:29502 "
            << "--collective all-gather --algorithm ring "
            << "--sizes 64,256,1024,4096,16384,65536,262144,1048576 "
            << "--iterations 100 --warmup 20\n";
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
                    parse_unsigned("sizes", input.substr(start, end - start))));

            if (comma == std::string::npos)
            {
                break;
            }

            start = comma + 1;
        }

        return sizes;
    }

    std::vector<std::size_t> default_sizes()
    {
        return {
            64,
            256,
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            4194304,
            16777216,
        };
    }

    Options parse_options(int argc, char **argv)
    {
        Options options;

        bool have_rank = false;
        bool have_peers = false;

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
            else if (argument == "--collective")
            {
                options.collective = require_value(argument);
            }
            else if (argument == "--algorithm")
            {
                options.algorithm = require_value(argument);
            }
            else if (argument == "--sizes")
            {
                options.sizes = parse_sizes(require_value(argument));
            }
            else if (argument == "--iterations")
            {
                const std::string value_text = require_value(argument);

                long value = 0;
                std::size_t consumed = 0;

                try
                {
                    value = std::stol(value_text, &consumed);
                }
                catch (const std::exception &)
                {
                    throw std::runtime_error(
                        "invalid --iterations value: " + value_text);
                }

                if (consumed != value_text.size() || value <= 0)
                {
                    throw std::runtime_error(
                        "invalid --iterations value: " + value_text +
                        " (must be a positive integer)");
                }

                options.iterations = static_cast<int>(value);
            }
            else if (argument == "--warmup")
            {
                const std::string value_text = require_value(argument);

                long value = 0;
                std::size_t consumed = 0;

                try
                {
                    value = std::stol(value_text, &consumed);
                }
                catch (const std::exception &)
                {
                    throw std::runtime_error(
                        "invalid --warmup value: " + value_text);
                }

                if (consumed != value_text.size() || value < 0)
                {
                    throw std::runtime_error(
                        "invalid --warmup value: " + value_text +
                        " (must be a non-negative integer)");
                }

                options.warmup = static_cast<int>(value);
            }
            else if (argument == "--busy-poll")
            {
                const std::string value_text = require_value(argument);

                long value = 0;
                std::size_t consumed = 0;

                try
                {
                    value = std::stol(value_text, &consumed);
                }
                catch (const std::exception &)
                {
                    throw std::runtime_error(
                        "invalid --busy-poll value: " + value_text);
                }

                if (consumed != value_text.size() || value < 0 ||
                    value > 1000000)
                {
                    throw std::runtime_error(
                        "invalid --busy-poll value: " + value_text +
                        " (must be an integer between 0 and 1000000)");
                }

                options.busy_poll_us = static_cast<int>(value);
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

        if (!have_rank)
        {
            throw std::runtime_error("--rank is required");
        }

        if (!have_peers)
        {
            throw std::runtime_error("--peers is required");
        }

        if (options.sizes.empty())
        {
            options.sizes = default_sizes();
        }

        return options;
    }

    std::vector<std::uint8_t> deterministic_buffer(
        std::size_t size,
        std::uint32_t seed)
    {
        std::vector<std::uint8_t> buffer(size);

        for (std::size_t i = 0; i < size; ++i)
        {
            buffer[i] = static_cast<std::uint8_t>(
                (static_cast<std::uint32_t>(i) * 2654435761u + seed) &
                0xFFu);
        }

        return buffer;
    }

    using AllGatherFn = void (*)(
        tbccl::World &, const void *, void *, std::size_t);

    AllGatherFn select_all_gather_algorithm(const std::string &algorithm)
    {
        if (algorithm == "reference")
        {
            return tbccl::detail::all_gather_reference;
        }

        if (algorithm == "ring")
        {
            return tbccl::detail::all_gather_ring;
        }

        throw std::runtime_error("unknown --algorithm: " + algorithm);
    }

    struct LatencyStats
    {
        std::size_t iterations = 0;
        double median_us = 0.0;
        double p95_us = 0.0;
        double min_us = 0.0;
        double max_us = 0.0;
    };

    double percentile(const std::vector<double> &sorted_values, double p)
    {
        if (sorted_values.empty())
        {
            return 0.0;
        }

        const double index =
            p * (static_cast<double>(sorted_values.size()) - 1.0);
        const auto lo = static_cast<std::size_t>(index);
        const std::size_t hi =
            std::min(lo + 1, sorted_values.size() - 1);
        const double frac = index - static_cast<double>(lo);

        return sorted_values[lo] +
               frac * (sorted_values[hi] - sorted_values[lo]);
    }

    LatencyStats summarize(std::vector<double> iteration_us)
    {
        std::sort(iteration_us.begin(), iteration_us.end());

        LatencyStats stats;
        stats.iterations = iteration_us.size();
        stats.median_us = percentile(iteration_us, 0.5);
        stats.p95_us = percentile(iteration_us, 0.95);
        stats.min_us = iteration_us.empty() ? 0.0 : iteration_us.front();
        stats.max_us = iteration_us.empty() ? 0.0 : iteration_us.back();

        return stats;
    }

    // Runs one (algorithm, bytes_per_rank) AllGather benchmark: an
    // untimed correctness gate, `warmup` untimed rounds, `iterations`
    // timed rounds (barrier -> collective, timed only around the
    // collective call itself), and a final untimed correctness
    // recheck. The timing exchange used to find each iteration's
    // max-across-ranks duration is plain World::send/recv, never the
    // collective being benchmarked. Only rank 0's LatencyStats are
    // meaningful; other ranks return a default-constructed result.
    LatencyStats run_all_gather_benchmark(
        tbccl::World &world,
        AllGatherFn collective,
        std::size_t bytes_per_rank,
        int iterations,
        int warmup)
    {
        const std::size_t rank = world.rank();
        const std::size_t size = world.size();
        const std::size_t total_bytes = size * bytes_per_rank;

        const auto send =
            deterministic_buffer(
                bytes_per_rank, static_cast<std::uint32_t>(rank) + 0x9000u);
        std::vector<std::uint8_t> recv(total_bytes, 0);

        auto check_correctness = [&](const char *stage)
        {
            for (std::size_t r = 0; r < size; ++r)
            {
                const auto expected =
                    deterministic_buffer(
                        bytes_per_rank,
                        static_cast<std::uint32_t>(r) + 0x9000u);

                if (std::memcmp(
                        recv.data() + r * bytes_per_rank, expected.data(),
                        bytes_per_rank) != 0)
                {
                    throw std::runtime_error(
                        std::string("correctness gate failed (") + stage +
                        ") at bytes_per_rank=" +
                        std::to_string(bytes_per_rank) + ", slot " +
                        std::to_string(r));
                }
            }
        };

        // Correctness gate, untimed.
        collective(world, send.data(), recv.data(), bytes_per_rank);
        check_correctness("pre-benchmark");

        // Warmup, untimed.
        for (int i = 0; i < warmup; ++i)
        {
            tbccl::barrier(world);
            collective(world, send.data(), recv.data(), bytes_per_rank);
        }

        std::vector<double> iteration_us;
        iteration_us.reserve(static_cast<std::size_t>(iterations));

        for (int i = 0; i < iterations; ++i)
        {
            tbccl::barrier(world);

            const auto start = std::chrono::steady_clock::now();
            collective(world, send.data(), recv.data(), bytes_per_rank);
            const auto stop = std::chrono::steady_clock::now();

            const double local_us =
                std::chrono::duration<double, std::micro>(stop - start)
                    .count();

            if (rank == 0)
            {
                double max_us = local_us;

                for (std::size_t peer = 1; peer < size; ++peer)
                {
                    double peer_us = 0.0;
                    world.recv(peer, &peer_us, sizeof(peer_us));
                    max_us = std::max(max_us, peer_us);
                }

                iteration_us.push_back(max_us);
            }
            else
            {
                world.send(0, &local_us, sizeof(local_us));
            }
        }

        // Final correctness recheck, untimed.
        collective(world, send.data(), recv.data(), bytes_per_rank);
        check_correctness("post-benchmark");

        if (rank != 0)
        {
            return LatencyStats{};
        }

        return summarize(std::move(iteration_us));
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_options(argc, argv);

        if (options.collective != "all-gather")
        {
            throw std::runtime_error(
                "unsupported --collective: " + options.collective +
                " (only all-gather is supported so far)");
        }

        const AllGatherFn collective =
            select_all_gather_algorithm(options.algorithm);

        tbccl::TcpWorldOptions world_options;
        world_options.rank = options.rank;
        world_options.peers = options.peers;
        world_options.bind_address = options.bind_address;
        world_options.tcp.busy_poll_us = options.busy_poll_us;

        auto world = tbccl::create_tcp_world(world_options);
        const std::size_t size = world->size();

        std::cerr
            << "rank=" << world->rank() << " world_size=" << size
            << " collective=" << options.collective
            << " algorithm=" << options.algorithm << '\n';

        if (world->rank() == 0)
        {
            std::cout
                << "collective,algorithm,world_size,bytes_per_rank,"
                   "output_bytes_per_rank,iterations,warmup,median_us,"
                   "p95_us,min_us,max_us,algbw_GBps,busbw_GBps,"
                   "busy_poll_us\n";
        }

        for (std::size_t bytes_per_rank : options.sizes)
        {
            const auto stats =
                run_all_gather_benchmark(
                    *world, collective, bytes_per_rank, options.iterations,
                    options.warmup);

            std::cerr
                << "rank " << world->rank()
                << ": completed bytes_per_rank=" << bytes_per_rank << '\n';

            if (world->rank() == 0)
            {
                const double median_seconds = stats.median_us * 1e-6;
                const double gathered_bytes =
                    static_cast<double>(size) *
                    static_cast<double>(bytes_per_rank);

                const double algbw_GBps =
                    median_seconds > 0.0
                        ? (gathered_bytes / median_seconds) / 1e9
                        : 0.0;
                const double busbw_GBps =
                    algbw_GBps * static_cast<double>(size - 1) /
                    static_cast<double>(size);

                std::cout
                    << options.collective << ',' << options.algorithm << ','
                    << size << ',' << bytes_per_rank << ','
                    << (size * bytes_per_rank) << ',' << stats.iterations
                    << ',' << options.warmup << ',' << stats.median_us
                    << ',' << stats.p95_us << ',' << stats.min_us << ','
                    << stats.max_us << ',' << algbw_GBps << ','
                    << busbw_GBps << ',' << options.busy_poll_us << '\n';
            }
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
