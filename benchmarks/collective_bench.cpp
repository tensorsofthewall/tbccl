// Generic collective benchmark harness. Currently supports:
//   --collective all-gather     --algorithm reference|ring|ring-pipelined|auto
//   --collective reduce-scatter --algorithm reference|ring|ring-pipelined|auto
//   --collective all-reduce     --algorithm reference|ring|ring-pipelined|auto
//                                --datatype int32|int64|float32|float64
//                                --op sum|product|min|max
//
// --algorithm auto resolves the algorithm via the same pure selector
// policy the public collective API uses (see algorithm_selector.hpp),
// independently for every --sizes value — never via a
// TBCCL_ALGORITHM/TBCCL_*_ALGORITHM environment override, so a
// benchmark run's results are reproducible regardless of the user's
// shell environment. Each CSV row records both requested_algorithm
// ("reference"/"ring"/"ring-pipelined"/"auto") and the resolved
// algorithm actually benchmarked, plus selection_reason for auto rows.
//
// --algorithm ring-pipelined selects the experimental chunked/
// pipelined ring implementations (see all_gather_internal.hpp /
// reduce_scatter_internal.hpp / all_reduce_internal.hpp) and requires
// --chunk-bytes; it is never a possible Auto resolution — Auto only
// ever resolves to "reference" or "ring", exactly as before. --chunk-
// bytes is ignored (has no effect) for reference/ring/auto.
//
// The CLI/CSV shape is deliberately generic so later work can add
// more collectives/algorithms without replacing this tool — see
// run_all_gather_benchmark()/run_reduce_scatter_benchmark() as the
// pattern to follow for e.g. all-reduce.
//
// Bandwidth formulas, using a single "segment_bytes" (B) meaning that
// differs slightly by collective (see collectives.hpp for exact
// semantics): for AllGather, B is the contribution bytes each rank
// sends; for ReduceScatter, B is the output segment bytes each rank
// receives. N = world size, T = measured collective duration (max
// across ranks, per-iteration median).
//   algbw = (N * B) / T             — total bytes moved per rank over
//                                      time (gathered for AllGather,
//                                      logically-input for
//                                      ReduceScatter).
//   busbw = algbw * (N - 1) / N
//         = ((N - 1) * B) / T       — useful peer-delivered traffic
//                                      per rank for an ideal ring
//                                      implementation of either
//                                      collective.
// Bandwidth is reported in decimal GB/s (1 GB = 1e9 bytes), not GiB/s.

#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "algorithm_selector.hpp"
#include "all_gather_internal.hpp"
#include "all_reduce_internal.hpp"
#include "reduce_scatter_internal.hpp"
#include "ring_pipeline_session.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
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
        tbccl::DataType datatype = tbccl::DataType::Float32;
        tbccl::ReduceOp op = tbccl::ReduceOp::Sum;
        std::vector<std::size_t> sizes;
        int iterations = 100;
        int warmup = 20;
        // 0 means "not set"; required (> 0) when algorithm ==
        // "ring-pipelined", ignored otherwise.
        std::size_t chunk_bytes = 0;
    };

    void print_usage(const char *program)
    {
        std::cerr
            << "Usage:\n"
            << "  " << program
            << " --rank R --peers HOST:PORT,HOST:PORT,... "
            << "[--bind ADDRESS] [--busy-poll MICROSECONDS] "
            << "[--collective all-gather|reduce-scatter|all-reduce] "
            << "[--algorithm reference|ring|ring-pipelined|auto] "
            << "[--chunk-bytes N] "
            << "[--datatype int32|int64|float32|float64] "
            << "[--op sum|product|min|max] "
            << "[--sizes BYTES[,BYTES...]] [--iterations N] "
            << "[--warmup N]\n\n"
            << "For all-gather, each --sizes value is contribution "
               "bytes per rank. For reduce-scatter, each --sizes value "
               "is output segment bytes per rank. For all-reduce, each "
               "--sizes value is total tensor bytes per rank (element "
               "count must divide the world size for the ring and "
               "ring-pipelined algorithms). All must be a multiple of "
               "the datatype size where a datatype applies; "
               "--datatype/--op select the reduction for "
               "reduce-scatter/all-reduce.\n\n"
            << "--algorithm ring-pipelined is an experimental chunked/"
               "pipelined ring variant, never a possible --algorithm "
               "auto resolution; it requires --chunk-bytes (the "
               "requested chunk size in bytes; must be a multiple of "
               "the datatype size for reduce-scatter/all-reduce). "
               "--chunk-bytes has no effect for reference/ring/auto.\n\n"
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

    tbccl::DataType parse_datatype(const std::string &text)
    {
        if (text == "int32") return tbccl::DataType::Int32;
        if (text == "int64") return tbccl::DataType::Int64;
        if (text == "float32") return tbccl::DataType::Float32;
        if (text == "float64") return tbccl::DataType::Float64;

        throw std::runtime_error("invalid --datatype value: " + text);
    }

    tbccl::ReduceOp parse_op(const std::string &text)
    {
        if (text == "sum") return tbccl::ReduceOp::Sum;
        if (text == "product") return tbccl::ReduceOp::Product;
        if (text == "min") return tbccl::ReduceOp::Min;
        if (text == "max") return tbccl::ReduceOp::Max;

        throw std::runtime_error("invalid --op value: " + text);
    }

    const char *datatype_name(tbccl::DataType datatype)
    {
        switch (datatype)
        {
        case tbccl::DataType::Int32: return "int32";
        case tbccl::DataType::Int64: return "int64";
        case tbccl::DataType::Float32: return "float32";
        case tbccl::DataType::Float64: return "float64";
        }

        return "unknown";
    }

    const char *op_name(tbccl::ReduceOp op)
    {
        switch (op)
        {
        case tbccl::ReduceOp::Sum: return "sum";
        case tbccl::ReduceOp::Product: return "product";
        case tbccl::ReduceOp::Min: return "min";
        case tbccl::ReduceOp::Max: return "max";
        }

        return "unknown";
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
            else if (argument == "--datatype")
            {
                options.datatype = parse_datatype(require_value(argument));
            }
            else if (argument == "--op")
            {
                options.op = parse_op(require_value(argument));
            }
            else if (argument == "--sizes")
            {
                options.sizes = parse_sizes(require_value(argument));
            }
            else if (argument == "--chunk-bytes")
            {
                options.chunk_bytes =
                    static_cast<std::size_t>(
                        parse_unsigned(argument, require_value(argument)));
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

    // Deterministic values kept small ([1, 7]) so Product cannot
    // explode numerically.
    template <typename T>
    T generate_value(
        std::size_t rank, std::size_t global_index, std::uint32_t seed)
    {
        const long value =
            (static_cast<long>(rank) * 131 +
             static_cast<long>(global_index) * 7 +
             static_cast<long>(seed)) %
                7 +
            1;

        return static_cast<T>(value);
    }

    template <typename T>
    void combine(T &acc, T value, tbccl::ReduceOp op)
    {
        switch (op)
        {
        case tbccl::ReduceOp::Sum:
            acc = acc + value;
            break;
        case tbccl::ReduceOp::Product:
            acc = acc * value;
            break;
        case tbccl::ReduceOp::Min:
            acc = acc < value ? acc : value;
            break;
        case tbccl::ReduceOp::Max:
            acc = acc > value ? acc : value;
            break;
        }
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

    // One shared CSV row shape for every collective this tool
    // supports. `datatype`/`op` are empty for collectives (like
    // AllGather) that have no reduction. `requested_algorithm` is
    // whatever --algorithm asked for ("reference"/"ring"/"auto");
    // `algorithm` is what was actually benchmarked for this row
    // (always == requested_algorithm unless requested_algorithm ==
    // "auto", in which case it is the selector's resolved choice for
    // this row's specific size). `selection_reason` is empty for
    // explicit reference/ring requests (no selector was consulted) and
    // the selector's SelectionReason name for auto rows.
    struct BenchRow
    {
        std::string collective;
        std::string requested_algorithm;
        std::string algorithm;
        std::string selection_reason;
        std::string datatype;
        std::string op;
        std::size_t world_size = 0;
        std::size_t segment_bytes = 0;
        std::size_t input_bytes_per_rank = 0;
        std::size_t output_bytes_per_rank = 0;
        LatencyStats stats;
        int warmup = 0;
        int busy_poll_us = 0;
        // busbw = algbw * busbw_multiplier * (N-1)/N. 1.0 for AllGather
        // and ReduceScatter (each moves ~1 collective's worth of
        // traffic); 2.0 for AllReduce, which is normalized as
        // ReduceScatter + AllGather network volume per rank.
        double busbw_multiplier = 1.0;
        // 0 for reference/ring/auto rows (not applicable).
        // ring-pipelined rows carry the requested chunk size and the
        // resulting chunk count per segment (ceil(segment_bytes /
        // chunk_bytes)), so a chunk-size sweep's CSV is
        // self-describing without needing to cross-reference the
        // command line that produced it.
        std::size_t chunk_bytes = 0;
        std::size_t chunks_per_segment = 0;
    };

    void print_csv_header()
    {
        std::cout
            << "collective,requested_algorithm,algorithm,selection_reason,"
               "datatype,op,world_size,segment_bytes,input_bytes_per_rank,"
               "output_bytes_per_rank,chunk_bytes,chunks_per_segment,"
               "iterations,warmup,median_us,p95_us,min_us,max_us,"
               "algbw_GBps,busbw_GBps,busy_poll_us\n";
    }

    void print_csv_row(const BenchRow &row)
    {
        const double median_seconds = row.stats.median_us * 1e-6;
        const double moved_bytes =
            static_cast<double>(row.world_size) *
            static_cast<double>(row.segment_bytes);

        const double algbw_GBps =
            median_seconds > 0.0 ? (moved_bytes / median_seconds) / 1e9
                                  : 0.0;
        const double busbw_GBps =
            algbw_GBps * row.busbw_multiplier *
            static_cast<double>(row.world_size - 1) /
            static_cast<double>(row.world_size);

        std::cout
            << row.collective << ',' << row.requested_algorithm << ','
            << row.algorithm << ',' << row.selection_reason << ','
            << row.datatype << ',' << row.op << ',' << row.world_size << ','
            << row.segment_bytes << ',' << row.input_bytes_per_rank << ','
            << row.output_bytes_per_rank << ',' << row.chunk_bytes << ','
            << row.chunks_per_segment << ',' << row.stats.iterations << ','
            << row.warmup << ',' << row.stats.median_us << ','
            << row.stats.p95_us << ',' << row.stats.min_us << ','
            << row.stats.max_us << ',' << algbw_GBps << ',' << busbw_GBps
            << ',' << row.busy_poll_us << '\n';
    }

    using AllGatherFn = std::function<void(
        tbccl::World &, const void *, void *, std::size_t)>;

    struct AllGatherResolution
    {
        AllGatherFn function;
        std::string resolved_name;
        std::string reason_name;
    };

    // Resolves --algorithm to a concrete callable for one --sizes
    // value. For "auto", this calls the same pure selector policy the
    // public all_gather() uses (never a TBCCL_ALGORITHM environment
    // override), so the resolution is reproducible and independent of
    // the caller's shell environment; different --sizes values in the
    // same run may resolve differently. "ring-pipelined" is never a
    // possible Auto resolution — it is only reachable by requesting it
    // explicitly, with the experimental algorithm distinctly labeled
    // in the returned name (never conflated with "ring").
    AllGatherResolution resolve_all_gather_function(
        const std::string &requested,
        std::size_t world_size,
        std::size_t bytes_per_rank,
        std::size_t chunk_bytes)
    {
        if (requested == "reference")
        {
            return {tbccl::detail::all_gather_reference, "reference", ""};
        }

        if (requested == "ring")
        {
            return {tbccl::detail::all_gather_ring, "ring", ""};
        }

        if (requested == "ring-pipelined")
        {
            return {
                [chunk_bytes](
                    tbccl::World &world, const void *send_buffer,
                    void *recv_buffer, std::size_t bytes)
                {
                    tbccl::detail::all_gather_pipelined(
                        world, send_buffer, recv_buffer, bytes,
                        chunk_bytes);
                },
                "ring-pipelined", ""};
        }

        if (requested == "auto")
        {
            const auto decision = tbccl::detail::select_all_gather_algorithm(
                world_size, bytes_per_rank,
                tbccl::detail::AlgorithmMode::Auto);

            const AllGatherFn function =
                decision.algorithm ==
                        tbccl::detail::CollectiveAlgorithm::Ring
                    ? tbccl::detail::all_gather_ring
                    : tbccl::detail::all_gather_reference;

            return {function,
                    tbccl::detail::algorithm_name(decision.algorithm),
                    tbccl::detail::selection_reason_name(decision.reason)};
        }

        throw std::runtime_error("unknown --algorithm: " + requested);
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

    using ReduceScatterFn = std::function<void(
        tbccl::World &, const void *, void *, std::size_t, tbccl::DataType,
        tbccl::ReduceOp)>;

    struct ReduceScatterResolution
    {
        ReduceScatterFn function;
        std::string resolved_name;
        std::string reason_name;
    };

    ReduceScatterResolution resolve_reduce_scatter_function(
        const std::string &requested,
        std::size_t world_size,
        std::size_t segment_bytes,
        std::size_t chunk_bytes)
    {
        if (requested == "reference")
        {
            return {tbccl::detail::reduce_scatter_reference, "reference",
                    ""};
        }

        if (requested == "ring")
        {
            return {tbccl::detail::reduce_scatter_ring, "ring", ""};
        }

        if (requested == "ring-pipelined")
        {
            return {
                [chunk_bytes](
                    tbccl::World &world, const void *send_buffer,
                    void *recv_buffer, std::size_t recv_count,
                    tbccl::DataType datatype, tbccl::ReduceOp op)
                {
                    tbccl::detail::reduce_scatter_pipelined(
                        world, send_buffer, recv_buffer, recv_count,
                        datatype, op, chunk_bytes);
                },
                "ring-pipelined", ""};
        }

        if (requested == "auto")
        {
            const auto decision =
                tbccl::detail::select_reduce_scatter_algorithm(
                    world_size, segment_bytes,
                    tbccl::detail::AlgorithmMode::Auto);

            const ReduceScatterFn function =
                decision.algorithm ==
                        tbccl::detail::CollectiveAlgorithm::Ring
                    ? tbccl::detail::reduce_scatter_ring
                    : tbccl::detail::reduce_scatter_reference;

            return {function,
                    tbccl::detail::algorithm_name(decision.algorithm),
                    tbccl::detail::selection_reason_name(decision.reason)};
        }

        throw std::runtime_error("unknown --algorithm: " + requested);
    }

    // Same structure as run_all_gather_benchmark: untimed correctness
    // gate, untimed warmup, timed iterations (barrier + collective
    // only), untimed final recheck. `recv_count` elements are
    // generated/verified per rank via generate_value()/combine(),
    // independently of whichever algorithm is under test.
    template <typename T>
    LatencyStats run_reduce_scatter_benchmark(
        tbccl::World &world,
        ReduceScatterFn collective,
        std::size_t recv_count,
        tbccl::DataType datatype,
        tbccl::ReduceOp op,
        int iterations,
        int warmup)
    {
        const std::size_t rank = world.rank();
        const std::size_t size = world.size();
        const std::size_t total_count = size * recv_count;
        constexpr std::uint32_t kSeed = 0x9000u;

        std::vector<T> send(total_count);

        for (std::size_t i = 0; i < total_count; ++i)
        {
            send[i] = generate_value<T>(rank, i, kSeed);
        }

        std::vector<T> expected(recv_count);

        for (std::size_t i = 0; i < recv_count; ++i)
        {
            const std::size_t global_index = rank * recv_count + i;
            T acc = generate_value<T>(0, global_index, kSeed);

            for (std::size_t p = 1; p < size; ++p)
            {
                combine(acc, generate_value<T>(p, global_index, kSeed), op);
            }

            expected[i] = acc;
        }

        std::vector<T> recv(recv_count, T{});

        auto check_correctness = [&](const char *stage)
        {
            for (std::size_t i = 0; i < recv_count; ++i)
            {
                if (recv[i] != expected[i])
                {
                    throw std::runtime_error(
                        std::string("correctness gate failed (") + stage +
                        ") at recv_count=" + std::to_string(recv_count) +
                        ", element " + std::to_string(i));
                }
            }
        };

        collective(
            world, send.data(), recv.data(), recv_count, datatype, op);
        check_correctness("pre-benchmark");

        for (int i = 0; i < warmup; ++i)
        {
            tbccl::barrier(world);
            collective(
                world, send.data(), recv.data(), recv_count, datatype, op);
        }

        std::vector<double> iteration_us;
        iteration_us.reserve(static_cast<std::size_t>(iterations));

        for (int i = 0; i < iterations; ++i)
        {
            tbccl::barrier(world);

            const auto start = std::chrono::steady_clock::now();
            collective(
                world, send.data(), recv.data(), recv_count, datatype, op);
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

        collective(
            world, send.data(), recv.data(), recv_count, datatype, op);
        check_correctness("post-benchmark");

        if (rank != 0)
        {
            return LatencyStats{};
        }

        return summarize(std::move(iteration_us));
    }

    LatencyStats run_reduce_scatter_benchmark_dispatch(
        tbccl::World &world,
        ReduceScatterFn collective,
        std::size_t recv_count,
        tbccl::DataType datatype,
        tbccl::ReduceOp op,
        int iterations,
        int warmup)
    {
        switch (datatype)
        {
        case tbccl::DataType::Int32:
            return run_reduce_scatter_benchmark<std::int32_t>(
                world, collective, recv_count, datatype, op, iterations,
                warmup);
        case tbccl::DataType::Int64:
            return run_reduce_scatter_benchmark<std::int64_t>(
                world, collective, recv_count, datatype, op, iterations,
                warmup);
        case tbccl::DataType::Float32:
            return run_reduce_scatter_benchmark<float>(
                world, collective, recv_count, datatype, op, iterations,
                warmup);
        case tbccl::DataType::Float64:
            return run_reduce_scatter_benchmark<double>(
                world, collective, recv_count, datatype, op, iterations,
                warmup);
        }

        throw std::runtime_error("unreachable: unknown DataType");
    }

    using AllReduceFn = std::function<void(
        tbccl::World &, const void *, void *, std::size_t, tbccl::DataType,
        tbccl::ReduceOp)>;

    struct AllReduceResolution
    {
        AllReduceFn function;
        std::string resolved_name;
        std::string reason_name;
    };

    AllReduceResolution resolve_all_reduce_function(
        const std::string &requested,
        std::size_t world_size,
        std::size_t tensor_bytes,
        std::size_t element_count,
        std::size_t chunk_bytes)
    {
        if (requested == "reference")
        {
            return {tbccl::detail::all_reduce_reference, "reference", ""};
        }

        if (requested == "ring")
        {
            return {tbccl::detail::all_reduce_ring, "ring", ""};
        }

        if (requested == "ring-pipelined")
        {
            return {
                [chunk_bytes](
                    tbccl::World &world, const void *send_buffer,
                    void *recv_buffer, std::size_t count,
                    tbccl::DataType datatype, tbccl::ReduceOp op)
                {
                    tbccl::detail::all_reduce_pipelined(
                        world, send_buffer, recv_buffer, count, datatype,
                        op, chunk_bytes);
                },
                "ring-pipelined", ""};
        }

        if (requested == "auto")
        {
            const auto decision = tbccl::detail::select_all_reduce_algorithm(
                world_size, tensor_bytes, element_count,
                tbccl::detail::AlgorithmMode::Auto);

            const AllReduceFn function =
                decision.algorithm ==
                        tbccl::detail::CollectiveAlgorithm::Ring
                    ? tbccl::detail::all_reduce_ring
                    : tbccl::detail::all_reduce_reference;

            return {function,
                    tbccl::detail::algorithm_name(decision.algorithm),
                    tbccl::detail::selection_reason_name(decision.reason)};
        }

        throw std::runtime_error("unknown --algorithm: " + requested);
    }

    // Same structure as run_reduce_scatter_benchmark: untimed
    // correctness gate, untimed warmup, timed iterations (barrier +
    // collective only), untimed final recheck. Every rank's full
    // `count`-element output is independently verified, not just
    // compared against another algorithm.
    template <typename T>
    LatencyStats run_all_reduce_benchmark(
        tbccl::World &world,
        AllReduceFn collective,
        std::size_t count,
        tbccl::DataType datatype,
        tbccl::ReduceOp op,
        int iterations,
        int warmup)
    {
        const std::size_t rank = world.rank();
        const std::size_t size = world.size();
        constexpr std::uint32_t kSeed = 0x9000u;

        std::vector<T> send(count);

        for (std::size_t i = 0; i < count; ++i)
        {
            send[i] = generate_value<T>(rank, i, kSeed);
        }

        std::vector<T> expected(count);

        for (std::size_t i = 0; i < count; ++i)
        {
            T acc = generate_value<T>(0, i, kSeed);

            for (std::size_t p = 1; p < size; ++p)
            {
                combine(acc, generate_value<T>(p, i, kSeed), op);
            }

            expected[i] = acc;
        }

        std::vector<T> recv(count, T{});

        auto check_correctness = [&](const char *stage)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                if (recv[i] != expected[i])
                {
                    throw std::runtime_error(
                        std::string("correctness gate failed (") + stage +
                        ") at count=" + std::to_string(count) +
                        ", element " + std::to_string(i));
                }
            }
        };

        collective(world, send.data(), recv.data(), count, datatype, op);
        check_correctness("pre-benchmark");

        for (int i = 0; i < warmup; ++i)
        {
            tbccl::barrier(world);
            collective(world, send.data(), recv.data(), count, datatype, op);
        }

        std::vector<double> iteration_us;
        iteration_us.reserve(static_cast<std::size_t>(iterations));

        for (int i = 0; i < iterations; ++i)
        {
            tbccl::barrier(world);

            const auto start = std::chrono::steady_clock::now();
            collective(world, send.data(), recv.data(), count, datatype, op);
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

        collective(world, send.data(), recv.data(), count, datatype, op);
        check_correctness("post-benchmark");

        if (rank != 0)
        {
            return LatencyStats{};
        }

        return summarize(std::move(iteration_us));
    }

    LatencyStats run_all_reduce_benchmark_dispatch(
        tbccl::World &world,
        AllReduceFn collective,
        std::size_t count,
        tbccl::DataType datatype,
        tbccl::ReduceOp op,
        int iterations,
        int warmup)
    {
        switch (datatype)
        {
        case tbccl::DataType::Int32:
            return run_all_reduce_benchmark<std::int32_t>(
                world, collective, count, datatype, op, iterations, warmup);
        case tbccl::DataType::Int64:
            return run_all_reduce_benchmark<std::int64_t>(
                world, collective, count, datatype, op, iterations, warmup);
        case tbccl::DataType::Float32:
            return run_all_reduce_benchmark<float>(
                world, collective, count, datatype, op, iterations, warmup);
        case tbccl::DataType::Float64:
            return run_all_reduce_benchmark<double>(
                world, collective, count, datatype, op, iterations, warmup);
        }

        throw std::runtime_error("unreachable: unknown DataType");
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_options(argc, argv);

        if (options.collective != "all-gather" &&
            options.collective != "reduce-scatter" &&
            options.collective != "all-reduce")
        {
            throw std::runtime_error(
                "unsupported --collective: " + options.collective +
                " (only all-gather, reduce-scatter, and all-reduce are "
                "supported so far)");
        }

        if (options.algorithm != "reference" && options.algorithm != "ring" &&
            options.algorithm != "ring-pipelined" &&
            options.algorithm != "auto")
        {
            throw std::runtime_error(
                "unsupported --algorithm: " + options.algorithm +
                " (only reference, ring, ring-pipelined, and auto are "
                "supported)");
        }

        if (options.algorithm == "ring-pipelined" &&
            options.chunk_bytes == 0)
        {
            throw std::runtime_error(
                "--chunk-bytes is required (and must be > 0) for "
                "--algorithm ring-pipelined");
        }

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
            << " algorithm=" << options.algorithm;

        if (options.collective == "reduce-scatter" ||
            options.collective == "all-reduce")
        {
            std::cerr
                << " datatype=" << datatype_name(options.datatype)
                << " op=" << op_name(options.op);
        }

        std::cerr << '\n';

        if (world->rank() == 0)
        {
            print_csv_header();
        }

        if (options.collective == "all-gather")
        {
            for (std::size_t bytes_per_rank : options.sizes)
            {
                const auto resolution = resolve_all_gather_function(
                    options.algorithm, size, bytes_per_rank,
                    options.chunk_bytes);

                const auto stats = run_all_gather_benchmark(
                    *world, resolution.function, bytes_per_rank,
                    options.iterations, options.warmup);

                std::cerr
                    << "rank " << world->rank()
                    << ": completed bytes_per_rank=" << bytes_per_rank
                    << " algorithm=" << resolution.resolved_name << '\n';

                if (world->rank() == 0)
                {
                    BenchRow row;
                    row.collective = options.collective;
                    row.requested_algorithm = options.algorithm;
                    row.algorithm = resolution.resolved_name;
                    row.selection_reason = resolution.reason_name;
                    row.world_size = size;
                    row.segment_bytes = bytes_per_rank;
                    row.input_bytes_per_rank = bytes_per_rank;
                    row.output_bytes_per_rank = size * bytes_per_rank;
                    row.stats = stats;
                    row.warmup = options.warmup;
                    row.busy_poll_us = options.busy_poll_us;

                    if (resolution.resolved_name == "ring-pipelined")
                    {
                        row.chunk_bytes = options.chunk_bytes;
                        row.chunks_per_segment =
                            tbccl::detail::compute_chunk_count(
                                bytes_per_rank, options.chunk_bytes);
                    }

                    print_csv_row(row);
                }
            }
        }
        else if (options.collective == "reduce-scatter")
        {
            const std::size_t element_size =
                tbccl::datatype_size(options.datatype);

            for (std::size_t segment_bytes : options.sizes)
            {
                if (segment_bytes % element_size != 0)
                {
                    throw std::runtime_error(
                        "--sizes value " + std::to_string(segment_bytes) +
                        " is not a multiple of the " +
                        datatype_name(options.datatype) + " element size (" +
                        std::to_string(element_size) + " bytes)");
                }

                const std::size_t recv_count = segment_bytes / element_size;

                const auto resolution = resolve_reduce_scatter_function(
                    options.algorithm, size, segment_bytes,
                    options.chunk_bytes);

                const auto stats = run_reduce_scatter_benchmark_dispatch(
                    *world, resolution.function, recv_count,
                    options.datatype, options.op, options.iterations,
                    options.warmup);

                std::cerr
                    << "rank " << world->rank()
                    << ": completed segment_bytes=" << segment_bytes
                    << " algorithm=" << resolution.resolved_name << '\n';

                if (world->rank() == 0)
                {
                    BenchRow row;
                    row.collective = options.collective;
                    row.requested_algorithm = options.algorithm;
                    row.algorithm = resolution.resolved_name;
                    row.selection_reason = resolution.reason_name;
                    row.datatype = datatype_name(options.datatype);
                    row.op = op_name(options.op);
                    row.world_size = size;
                    row.segment_bytes = segment_bytes;
                    row.input_bytes_per_rank = size * segment_bytes;
                    row.output_bytes_per_rank = segment_bytes;
                    row.stats = stats;
                    row.warmup = options.warmup;
                    row.busy_poll_us = options.busy_poll_us;

                    if (resolution.resolved_name == "ring-pipelined")
                    {
                        row.chunk_bytes = options.chunk_bytes;
                        row.chunks_per_segment =
                            tbccl::detail::compute_chunk_count(
                                segment_bytes, options.chunk_bytes);
                    }

                    print_csv_row(row);
                }
            }
        }
        else
        {
            // all-reduce: each --sizes value B is the total tensor
            // bytes per rank (input and output are the same size).
            // The benchmark requires B % datatype_size == 0 and the
            // resulting element count % world_size == 0 for both
            // algorithms here, so segment_bytes (B/N) is always exact
            // and directly comparable between reference and ring rows
            // — ring itself would reject a non-divisible count anyway.
            const std::size_t element_size =
                tbccl::datatype_size(options.datatype);

            for (std::size_t tensor_bytes : options.sizes)
            {
                if (tensor_bytes % element_size != 0)
                {
                    throw std::runtime_error(
                        "--sizes value " + std::to_string(tensor_bytes) +
                        " is not a multiple of the " +
                        datatype_name(options.datatype) + " element size (" +
                        std::to_string(element_size) + " bytes)");
                }

                const std::size_t count = tensor_bytes / element_size;

                if (count % size != 0)
                {
                    throw std::runtime_error(
                        "--sizes value " + std::to_string(tensor_bytes) +
                        " (" + std::to_string(count) +
                        " elements) is not divisible by world size " +
                        std::to_string(size));
                }

                const auto resolution = resolve_all_reduce_function(
                    options.algorithm, size, tensor_bytes, count,
                    options.chunk_bytes);

                const auto stats = run_all_reduce_benchmark_dispatch(
                    *world, resolution.function, count, options.datatype,
                    options.op, options.iterations, options.warmup);

                std::cerr
                    << "rank " << world->rank()
                    << ": completed tensor_bytes=" << tensor_bytes
                    << " algorithm=" << resolution.resolved_name << '\n';

                if (world->rank() == 0)
                {
                    BenchRow row;
                    row.collective = options.collective;
                    row.requested_algorithm = options.algorithm;
                    row.algorithm = resolution.resolved_name;
                    row.selection_reason = resolution.reason_name;
                    row.datatype = datatype_name(options.datatype);
                    row.op = op_name(options.op);
                    row.world_size = size;
                    row.segment_bytes = tensor_bytes / size;
                    row.input_bytes_per_rank = tensor_bytes;
                    row.output_bytes_per_rank = tensor_bytes;
                    row.stats = stats;
                    row.warmup = options.warmup;
                    row.busy_poll_us = options.busy_poll_us;
                    row.busbw_multiplier = 2.0;

                    if (resolution.resolved_name == "ring-pipelined")
                    {
                        row.chunk_bytes = options.chunk_bytes;
                        row.chunks_per_segment =
                            tbccl::detail::compute_chunk_count(
                                row.segment_bytes, options.chunk_bytes);
                    }

                    print_csv_row(row);
                }
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
