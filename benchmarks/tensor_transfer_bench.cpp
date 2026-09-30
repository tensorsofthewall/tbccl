// the CUDA tensor-benchmark work heterogeneous tensor-transfer benchmark, built entirely
// over the TensorBackend abstraction (benchmarks/tensor/
// tensor_backend.hpp) -- --local-backend cuda-pinned/metal-shared/etc.
// work automatically depending on which of TBCCL_ENABLE_CUDA/
// TBCCL_ENABLE_METAL this build has on; no mode below is aware of any
// concrete backend.
//
// Modes:
//
//   staging-only     Single process, no networking. Measures device-
//                    to-host and host-to-device staging cost in
//                    isolation, by copying directly between two
//                    backend instances' host-visible staging areas
//                    (bypassing World entirely) -- see.
//
//   network-only     Exactly 2 ranks. --source-rank streams `--sizes`
//                    payloads to its peer back-to-back; reports the
//                    sender's per-send() latency distribution and the
//                    aggregate wall-clock throughput. Always uses the
//                    host backend (the network-only baseline is
//                    explicitly host-only).
//
//   latency-floor    Exactly 2 ranks. A request/response (ping-pong)
//                    exchange at small payload sizes. Reports RTT
//                    percentiles and a clearly-labeled half-RTT
//                    approximation (never claimed as a directly
//                    measured one-way latency -- see Part 17). Always
//                    uses the host backend.
//
//   end-to-end       Exactly 2 ranks, each using its own
//                    --local-backend (they may legitimately differ --
//                    that is the heterogeneous transfer this mode
//                    measures). Implements the control protocol
//                    (a fixed-width handshake validating both
//                    invocations agree before any timed transfer) and
//                    the destination-completion ACK: the
//                    completion-confirmed latency spans from the
//                    source tensor being ready through the
//                    destination's accelerator-completion ACK being
//                    observed, explicitly including the ACK's own
//                    return trip (Part 35) -- never presented as a
//                    one-way measurement.
//
//   ack-calibration  The same end-to-end protocol, with the backend
//                    forced to host regardless of --local-backend
//                    (Part 37) and an extra 0-byte payload measured
//                    first. Comparing its numbers against a real
//                    end-to-end run at the same sizes isolates
//                    protocol/ACK overhead from actual accelerator
//                    staging cost.
//
// Only the source_rank (every networked mode) or the sole process
// (staging-only) writes CSV to stdout; every rank writes diagnostics
// to stderr. This matches benchmarks/collective_bench.cpp and
// scripts/run_collective_bench.py's rank0-stdout-is-CSV convention.

#include "tensor/tensor_backend.hpp"
#include "tensor/transfer_diagnostics.hpp"
#include "tensor/transfer_trace.hpp"

#include <tbccl/tcp_world.hpp>
#include <tbccl/world.hpp>

#include <arpa/inet.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
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
        bool diagnostics = false;
        bool trace = false;
        std::string run_id = "unspecified";
        std::string physical_machine = "unspecified";
        int source_gap_us = 0;

        std::string mode; // required
        BackendKind local_backend = BackendKind::Host;
        std::size_t source_rank = 0;

        // "ready" (default, matches the CUDA tensor-benchmark work's
        // only behavior exactly -- see Part 18's timing audit) or
        // "produce". Only end-to-end and ack-calibration modes consult
        // this; ignored otherwise.
        std::string timing_scope = "ready";

        std::vector<std::size_t> sizes;
        int warmup = 20;
        int iterations = 200;
    };

    void print_usage(const char *program)
    {
        std::cerr
            << "Usage:\n"
            << "  " << program
            << " --mode staging-only|network-only|latency-floor|"
               "end-to-end|ack-calibration "
            << "[--rank R --peers HOST:PORT,...] "
            << "[--bind ADDRESS] [--busy-poll MICROSECONDS] "
            << "[--local-backend host|cuda-pageable|cuda-pinned|"
               "metal-shared|metal-private-staged] "
            << "[--source-rank N] [--sizes BYTES[,BYTES...]] "
            << "[--warmup N] [--iterations N] "
            << "[--timing-scope ready|produce] [--diagnostics] [--trace] "
               "[--run-id ID] [--physical-machine NAME] [--source-gap-us N]\n\n"
            << "staging-only needs no --rank/--peers (single process, no "
               "networking). Every other mode requires exactly 2 --peers. "
               "network-only and latency-floor always use the host "
               "backend (--local-backend is rejected if not 'host') -- "
               "see the benchmark methodology. end-to-end uses "
               "--local-backend as each rank's OWN backend (they may "
               "legitimately differ, e.g. cuda-pinned on one rank and "
               "metal-shared on the other -- that IS the heterogeneous "
               "transfer this mode measures); each rank must be invoked "
               "with its own correct --local-backend. ack-calibration "
               "always forces the host backend regardless of "
               "--local-backend (Part 37's host-only control baseline) "
               "and implicitly measures an extra 0-byte payload first.\n\n"
            << "--timing-scope applies to end-to-end/ack-calibration only "
               "(see the benchmark methodology). 'ready' (default, matches the earlier "
               "measurements exactly): source generation and "
               "producer synchronization happen BEFORE the primary timer "
               "-- completion_confirmed_us then measures the cost of "
               "communicating an already-ready tensor. 'produce': the "
               "timer starts BEFORE source generation, so "
               "completion_confirmed_us also includes producer cost. "
               "These are never the same number and must not be "
               "compared as if they were.\n\n"
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
            else if (argument == "--diagnostics")
            {
                options.diagnostics = true;
            }
            else if (argument == "--trace")
            {
                options.trace = true;
            }
            else if (argument == "--run-id")
            {
                options.run_id = require_value(argument);
            }
            else if (argument == "--physical-machine")
            {
                options.physical_machine = require_value(argument);
            }
            else if (argument == "--source-gap-us")
            {
                const unsigned long value =
                    parse_unsigned(argument, require_value(argument));
                if (value > 10000000)
                    throw std::runtime_error("--source-gap-us must be <= 10000000");
                options.source_gap_us = static_cast<int>(value);
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
            else if (argument == "--timing-scope")
            {
                options.timing_scope = require_value(argument);
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
            options.mode != "latency-floor" &&
            options.mode != "end-to-end" &&
            options.mode != "ack-calibration")
        {
            throw std::runtime_error(
                "--mode must be staging-only, network-only, latency-floor, "
                "end-to-end, or ack-calibration");
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
            if ((options.mode == "network-only" || options.mode == "latency-floor") &&
                options.local_backend != BackendKind::Host)
            {
                throw std::runtime_error(
                    "--local-backend must be 'host' for --mode network-only "
                    "or latency-floor (the host-only baseline)");
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

        if (options.timing_scope != "ready" && options.timing_scope != "produce")
        {
            throw std::runtime_error("--timing-scope must be ready or produce");
        }

        if (options.diagnostics && options.mode != "end-to-end" &&
            options.mode != "ack-calibration")
            throw std::runtime_error("--diagnostics requires end-to-end or ack-calibration mode");

        if ((options.trace || options.source_gap_us != 0) &&
            options.mode != "end-to-end" && options.mode != "ack-calibration")
            throw std::runtime_error(
                "--trace/--source-gap-us require end-to-end or ack-calibration mode");

        if (options.trace &&
            (options.run_id == "unspecified" || options.physical_machine == "unspecified"))
            throw std::runtime_error(
                "--trace requires --run-id and --physical-machine");

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
    //
    // One unified schema across every mode (Part 40's column list,
    // extended with a couple of fields ("stage", "half_rtt_us")
    // already useful from the first three modes -- Part 40 says "at
    // minimum", not "exactly"). A field a given mode/row doesn't
    // measure is printed as the literal text "NA" (kNA sentinel below
    // for the optional doubles), never a bogus zero (Part 40's
    // explicit instruction).
    // -----------------------------------------------------------------------------

    constexpr double kNA = -1.0;

    struct CsvRow
    {
        std::string mode;
        std::string stage = "NA";
        long long source_rank = -1; // < 0 => NA
        std::string source_backend = "NA";
        std::string destination_backend = "NA";
        std::size_t payload_bytes = 0;
        int iterations = 0;
        int warmup = 0;
        LatencyStats stats; // the row's primary latency distribution
        double half_rtt_us = kNA;
        double source_staging_us = kNA;
        double source_sync_us = kNA;
        double sender_network_us = kNA;
        double receiver_network_us = kNA;
        double destination_staging_us = kNA;
        double destination_sync_us = kNA;
        double completion_confirmed_us = kNA;
        double effective_GBps = 0.0;
        std::string allocation_type = "NA";
        AllocationStats alloc_stats;

        // The CUDA synchronization-audit work, which of the two
        // disjoint timing boundaries
        // `stats`/`completion_confirmed_us` describes -- "ready"
        // (producer excluded, the original only
        // behavior) or "produce" (producer included). "NA" for
        // modes/rows where the distinction doesn't apply
        // (staging-only/network-only/ latency-floor).
        // producer_enqueue_us is the CPU time to *submit* the
        // producer's GPU work, kept separate from source_sync_us (the
        // time spent *waiting* for it, via prepare_source()) these
        // answered very differently in the CUDA synchronization-audit
        // CUDA audit (enqueue ~4us, sync ~1.3-1.9ms).
        std::string timing_scope = "NA";
        double producer_enqueue_us = kNA;
    };

    void print_csv_header()
    {
        std::cout
            << "mode,stage,source_rank,source_backend,destination_backend,"
               "payload_bytes,iterations,warmup,"
               "median_us,p95_us,p99_us,min_us,max_us,half_rtt_us,"
               "source_staging_us,source_sync_us,sender_network_us,"
               "receiver_network_us,destination_staging_us,"
               "destination_sync_us,completion_confirmed_us,"
               "effective_GBps,effective_Gbps,"
               "allocation_type,staging_capacity_bytes,allocation_count,"
               "reuse_count,timing_scope,producer_enqueue_us\n";
    }

    void print_optional(double value, char separator = ',')
    {
        if (value < 0.0)
        {
            std::cout << "NA" << separator;
        }
        else
        {
            std::cout << value << separator;
        }
    }

    void print_csv_row(const CsvRow &row)
    {
        std::cout << row.mode << ',' << row.stage << ',';

        if (row.source_rank < 0)
        {
            std::cout << "NA,";
        }
        else
        {
            std::cout << row.source_rank << ',';
        }

        std::cout
            << row.source_backend << ','
            << row.destination_backend << ','
            << row.payload_bytes << ','
            << row.iterations << ','
            << row.warmup << ','
            << row.stats.median_us << ','
            << row.stats.p95_us << ','
            << row.stats.p99_us << ','
            << row.stats.min_us << ','
            << row.stats.max_us << ',';

        print_optional(row.half_rtt_us);
        print_optional(row.source_staging_us);
        print_optional(row.source_sync_us);
        print_optional(row.sender_network_us);
        print_optional(row.receiver_network_us);
        print_optional(row.destination_staging_us);
        print_optional(row.destination_sync_us);
        print_optional(row.completion_confirmed_us);

        std::cout
            << row.effective_GBps << ','
            << (row.effective_GBps * 8.0) << ','
            << row.allocation_type << ','
            << row.alloc_stats.capacity_bytes << ','
            << row.alloc_stats.allocation_count << ','
            << row.alloc_stats.reuse_count << ','
            << row.timing_scope << ',';

        print_optional(row.producer_enqueue_us, '\n');
    }

    // -----------------------------------------------------------------------------
    // Control protocol: before any timed end-to-end/ack-
    // calibration transfer, both ranks exchange and validate a small
    // fixed-width control message so a configuration mismatch
    // (different --sizes/--iterations/--warmup between the two
    // invocations) fails clearly up front rather than hanging or
    // corrupting the data stream. Deterministic send/recv ordering
    // (lower rank sends first) avoids deadlock independently of which
    // rank is the transfer's source. Fixed-width types, explicit byte
    // order (network byte order via htonl/hton64, matching
    // benchmarks/tcp_pingpong.cpp's own small helper -- duplicated
    // rather than shared, per this repo's self-contained-benchmark-
    // tool convention).
    // -----------------------------------------------------------------------------

    std::uint64_t hton64(std::uint64_t value)
    {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return (static_cast<std::uint64_t>(
                    htonl(static_cast<std::uint32_t>(value & 0xffffffffULL)))
                << 32) |
               htonl(static_cast<std::uint32_t>(value >> 32));
#else
        return value;
#endif
    }

    std::uint64_t ntoh64(std::uint64_t value)
    {
        return hton64(value);
    }

    constexpr std::uint32_t kControlProtocolVersion = 2;
    constexpr std::uint32_t kTransferModeEndToEnd = 1;
    constexpr std::uint32_t kTransferModeAckCalibration = 2;

    struct ControlMessage
    {
        std::uint32_t protocol_version = kControlProtocolVersion;
        std::uint32_t rank = 0;
        std::uint32_t local_backend = 0; // BackendKind
        std::uint32_t transfer_mode = 0;
        std::uint64_t payload_bytes = 0;
        std::uint32_t iterations = 0;
        std::uint32_t warmup = 0;
        std::uint32_t verify_enabled = 1;
        std::uint32_t source_gap_us = 0;
    };

    struct WireControlMessage
    {
        std::uint32_t protocol_version;
        std::uint32_t rank;
        std::uint32_t local_backend;
        std::uint32_t transfer_mode;
        std::uint64_t payload_bytes;
        std::uint32_t iterations;
        std::uint32_t warmup;
        std::uint32_t verify_enabled;
        std::uint32_t source_gap_us;
    };

    WireControlMessage to_wire(const ControlMessage &message)
    {
        WireControlMessage wire;
        wire.protocol_version = htonl(message.protocol_version);
        wire.rank = htonl(message.rank);
        wire.local_backend = htonl(message.local_backend);
        wire.transfer_mode = htonl(message.transfer_mode);
        wire.payload_bytes = hton64(message.payload_bytes);
        wire.iterations = htonl(message.iterations);
        wire.warmup = htonl(message.warmup);
        wire.verify_enabled = htonl(message.verify_enabled);
        wire.source_gap_us = htonl(message.source_gap_us);
        return wire;
    }

    ControlMessage from_wire(const WireControlMessage &wire)
    {
        ControlMessage message;
        message.protocol_version = ntohl(wire.protocol_version);
        message.rank = ntohl(wire.rank);
        message.local_backend = ntohl(wire.local_backend);
        message.transfer_mode = ntohl(wire.transfer_mode);
        message.payload_bytes = ntoh64(wire.payload_bytes);
        message.iterations = ntohl(wire.iterations);
        message.warmup = ntohl(wire.warmup);
        message.verify_enabled = ntohl(wire.verify_enabled);
        message.source_gap_us = ntohl(wire.source_gap_us);
        return message;
    }

    // Lower rank sends first, higher rank receives first -- a fixed,
    // deterministic rule independent of which rank is the transfer's
    // logical source, so it can never deadlock.
    ControlMessage exchange_control(
        tbccl::World &world,
        std::size_t self_rank,
        std::size_t peer,
        const ControlMessage &mine)
    {
        const WireControlMessage wire_mine = to_wire(mine);
        WireControlMessage wire_peer{};

        if (self_rank < peer)
        {
            world.send(peer, &wire_mine, sizeof(wire_mine));
            world.recv(peer, &wire_peer, sizeof(wire_peer));
        }
        else
        {
            world.recv(peer, &wire_peer, sizeof(wire_peer));
            world.send(peer, &wire_mine, sizeof(wire_mine));
        }

        return from_wire(wire_peer);
    }

    void validate_control_agreement(
        const ControlMessage &mine,
        const ControlMessage &peer_message)
    {
        // rank and local_backend are expected to differ (that is the
        // whole point of a heterogeneous transfer) and are excluded
        // from this check.
        auto require_equal = [&](
                                  const char *field,
                                  std::uint64_t mine_value,
                                  std::uint64_t peer_value)
        {
            if (mine_value != peer_value)
            {
                throw std::runtime_error(
                    std::string("control protocol mismatch: ") + field +
                    " differs between ranks (check --mode/--sizes/"
                    "--iterations/--warmup are identical on both "
                    "invocations)");
            }
        };

        require_equal(
            "protocol_version", mine.protocol_version, peer_message.protocol_version);
        require_equal("transfer_mode", mine.transfer_mode, peer_message.transfer_mode);
        require_equal("payload_bytes", mine.payload_bytes, peer_message.payload_bytes);
        require_equal("iterations", mine.iterations, peer_message.iterations);
        require_equal("warmup", mine.warmup, peer_message.warmup);
        require_equal(
            "verify_enabled", mine.verify_enabled, peer_message.verify_enabled);
        require_equal("source_gap_us", mine.source_gap_us, peer_message.source_gap_us);
    }

    // -----------------------------------------------------------------------------
    // A one-time (per size, after the timed loop -- never per-
    // iteration, so it never perturbs the ACK-based completion timing)
    // summary the destination sends back to the source, carrying the
    // receiver-side per-stage numbers Part 40's CSV schema asks for
    // that only the destination rank can measure directly (Part 38:
    // "each rank may record its own stage durations... collected
    // outside the primary timed interval"). Microseconds are sent as
    // fixed-point (x1000) 64-bit integers rather than raw doubles, to
    // keep the wire format unambiguous regardless of either
    // endpoint's float representation.
    // -----------------------------------------------------------------------------

    struct StageSummary
    {
        double receiver_network_us = kNA;
        double destination_staging_us = kNA;
        double destination_sync_us = kNA;
    };

    struct WireStageSummary
    {
        std::uint64_t receiver_network_us_x1000;
        std::uint64_t destination_staging_us_x1000;
        std::uint64_t destination_sync_us_x1000;
    };

    constexpr std::uint64_t kWireNA = 0xFFFFFFFFFFFFFFFFULL;

    std::uint64_t scale_us(double microseconds)
    {
        return microseconds < 0.0
                   ? kWireNA
                   : static_cast<std::uint64_t>(microseconds * 1000.0);
    }

    double unscale_us(std::uint64_t scaled)
    {
        return scaled == kWireNA ? kNA : static_cast<double>(scaled) / 1000.0;
    }

    WireStageSummary to_wire(const StageSummary &summary)
    {
        WireStageSummary wire;
        wire.receiver_network_us_x1000 = hton64(scale_us(summary.receiver_network_us));
        wire.destination_staging_us_x1000 =
            hton64(scale_us(summary.destination_staging_us));
        wire.destination_sync_us_x1000 = hton64(scale_us(summary.destination_sync_us));
        return wire;
    }

    StageSummary from_wire(const WireStageSummary &wire)
    {
        StageSummary summary;
        summary.receiver_network_us = unscale_us(ntoh64(wire.receiver_network_us_x1000));
        summary.destination_staging_us =
            unscale_us(ntoh64(wire.destination_staging_us_x1000));
        summary.destination_sync_us = unscale_us(ntoh64(wire.destination_sync_us_x1000));
        return summary;
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

            CsvRow d2h_row;
            d2h_row.mode = "staging-only";
            d2h_row.stage = "d2h";
            d2h_row.source_backend = backend_name;
            d2h_row.payload_bytes = size;
            d2h_row.iterations = options.iterations;
            d2h_row.warmup = options.warmup;
            d2h_row.stats = d2h_stats;
            d2h_row.effective_GBps = d2h_gbps;
            d2h_row.allocation_type = backend_name;
            d2h_row.alloc_stats = source->stats();
            print_csv_row(d2h_row);

            CsvRow h2d_row;
            h2d_row.mode = "staging-only";
            h2d_row.stage = "h2d";
            h2d_row.destination_backend = backend_name;
            h2d_row.payload_bytes = size;
            h2d_row.iterations = options.iterations;
            h2d_row.warmup = options.warmup;
            h2d_row.stats = h2d_stats;
            h2d_row.effective_GBps = h2d_gbps;
            h2d_row.allocation_type = backend_name;
            h2d_row.alloc_stats = destination->stats();
            print_csv_row(h2d_row);

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

                CsvRow row;
                row.mode = "network-only";
                row.stage = "send";
                row.source_rank = static_cast<long long>(options.source_rank);
                row.source_backend = "host";
                row.destination_backend = "host";
                row.payload_bytes = size;
                row.iterations = options.iterations;
                row.warmup = options.warmup;
                row.stats = stats;
                row.sender_network_us = stats.median_us;
                row.effective_GBps = effective_GBps;
                row.allocation_type = "host";
                row.alloc_stats = backend->stats();
                print_csv_row(row);

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

                CsvRow row;
                row.mode = "latency-floor";
                row.stage = "rtt";
                row.source_rank = static_cast<long long>(options.source_rank);
                row.source_backend = "host";
                row.destination_backend = "host";
                row.payload_bytes = size;
                row.iterations = options.iterations;
                row.warmup = options.warmup;
                row.stats = stats;
                row.half_rtt_us = half_rtt_us;
                row.effective_GBps = effective_GBps;
                row.allocation_type = "host";
                row.alloc_stats = backend->stats();
                print_csv_row(row);

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

    // -----------------------------------------------------------------------------
    // end-to-end / ack-calibration
    //
    // Shared implementation: ack-calibration is exactly the
    // end-to-end protocol with the backend forced to Host regardless
    // of --local-backend (Part 37's "host-only control benchmark"),
    // and an implicit 0-byte "empty/control request" case prepended
    // -- comparing its host-payload+ACK numbers against a real
    // end-to-end run's numbers at the same sizes is what isolates
    // protocol/ACK overhead from actual accelerator staging cost.
    //
    // Per size: exchange_control() validates both invocations agree;
    // then `total` iterations of source (init -> sync -> stage ->
    // send -> wait for a 1-byte ACK) / destination (recv -> stage ->
    // send a 1-byte ACK) -- the completion-confirmed latency (Part 35)
    // is measured on the source from "source tensor ready" (right
    // after prepare_source()) through the ACK being observed, which
    // is exactly source_staging + sender_network + destination-side
    // work + the ACK's own return trip, never presented as a one-way
    // measurement. After the timed loop, the destination sends back a
    // one-time StageSummary, and both sides do one untimed
    // full-byte-verification round (a fresh seed, not part of any
    // per-iteration timing) before moving to the next size.
    // -----------------------------------------------------------------------------

    void run_end_to_end_protocol(
        const Options &options,
        tbccl::World &world,
        const std::string &mode_name,
        std::uint32_t wire_transfer_mode,
        BackendKind effective_backend)
    {
        const bool is_source = (options.rank == options.source_rank);
        const std::size_t peer = 1 - options.rank;

        std::vector<std::size_t> sizes = options.sizes;

        if (wire_transfer_mode == kTransferModeAckCalibration &&
            (sizes.empty() || sizes.front() != 0))
        {
            sizes.insert(sizes.begin(), 0);
        }

        if (is_source)
        {
            print_csv_header();
        }

        const std::string local_backend_name =
            tbccl_bench::tensor::backend_kind_name(effective_backend);

        for (std::size_t size : sizes)
        {
            auto backend = make_backend(effective_backend);
            backend->allocate(size);

            ControlMessage mine;
            mine.rank = static_cast<std::uint32_t>(options.rank);
            mine.local_backend = static_cast<std::uint32_t>(effective_backend);
            mine.transfer_mode = wire_transfer_mode;
            mine.payload_bytes = size;
            mine.iterations = static_cast<std::uint32_t>(options.iterations);
            mine.warmup = static_cast<std::uint32_t>(options.warmup);
            mine.verify_enabled = 1;
            mine.source_gap_us = static_cast<std::uint32_t>(options.source_gap_us);

            const ControlMessage peer_message =
                exchange_control(world, options.rank, peer, mine);
            validate_control_agreement(mine, peer_message);

            const std::string peer_backend_name =
                tbccl_bench::tensor::backend_kind_name(
                    static_cast<BackendKind>(peer_message.local_backend));

            const int total = options.warmup + options.iterations;
            const std::size_t reserve_n =
                static_cast<std::size_t>(options.iterations);

            tbccl_bench::diagnostics::CpuWindow cpu(options.diagnostics);
            const std::string direction =
                is_source
                    ? local_backend_name + "->" + peer_backend_name
                    : peer_backend_name + "->" + local_backend_name;
            tbccl_bench::trace::Batch trace(
                options.trace,
                reserve_n,
                options.run_id,
                options.physical_machine,
                options.rank,
                options.source_rank,
                direction,
                size,
                options.busy_poll_us,
                options.timing_scope,
                options.source_gap_us);

            if (is_source)
            {
                const bool produce_scope = (options.timing_scope == "produce");

                std::vector<double> completion_samples;
                std::vector<double> staging_samples;
                std::vector<double> sync_samples;
                std::vector<double> enqueue_samples;
                std::vector<double> send_samples;
                completion_samples.reserve(reserve_n);
                staging_samples.reserve(reserve_n);
                sync_samples.reserve(reserve_n);
                enqueue_samples.reserve(reserve_n);
                send_samples.reserve(reserve_n);
                std::int64_t previous_iteration_end_ns = -1;

                for (int i = 0; i < total; ++i)
                {
                    if (i > 0 && options.source_gap_us > 0)
                        std::this_thread::sleep_for(
                            std::chrono::microseconds(options.source_gap_us));
                    tbccl_bench::trace::Entry trace_entry;
                    const std::int64_t iteration_begin_ns = trace.now();
                    if (i >= options.warmup)
                    {
                        trace_entry.iteration_index = i - options.warmup;
                        trace_entry.iteration_begin_ns = iteration_begin_ns;
                        if (previous_iteration_end_ns >= 0)
                            trace_entry.observed_source_gap_ns =
                                iteration_begin_ns - previous_iteration_end_ns;
                    }
                    if (i == options.warmup) cpu.begin();
                    const std::uint32_t seed = static_cast<std::uint32_t>(0x3000 + i);

                    // the earlier behavior, this is the ONLY difference
                    // between the two timing scopes -- where
                    // `interval_start` (the primary timer's start) is
                    // captured relative to the producer's work.
                    //
                    // "ready": generate + synchronize the source BEFORE
                    // starting the timer (the original only behavior,
                    // and the default here) -- completion_confirmed_us
                    // then measures the cost of communicating an
                    // ALREADY-ready tensor, deliberately excluding
                    // producer cost.
                    //
                    // "produce": the timer starts first, so producer
                    // enqueue+sync work is INSIDE completion_confirmed_us.
                    //
                    // Every other stage (staging/send/ACK-wait) is
                    // measured identically in both scopes -- only the
                    // interval's start point and what precedes it
                    // differ (Part 12: "share... only the timing
                    // boundaries... differ").
                    std::chrono::steady_clock::time_point interval_start;
                    std::chrono::steady_clock::time_point enqueue_start;
                    std::chrono::steady_clock::time_point enqueue_end;
                    std::chrono::steady_clock::time_point sync_end;

                    if (produce_scope)
                    {
                        interval_start = std::chrono::steady_clock::now();
                        enqueue_start = interval_start;
                        backend->initialize_source(seed);
                        enqueue_end = std::chrono::steady_clock::now();
                        backend->prepare_source();
                        sync_end = std::chrono::steady_clock::now();
                    }
                    else
                    {
                        enqueue_start = std::chrono::steady_clock::now();
                        backend->initialize_source(seed);
                        enqueue_end = std::chrono::steady_clock::now();
                        backend->prepare_source();
                        sync_end = std::chrono::steady_clock::now();
                        // "source tensor ready" -- the primary timer
                        // starts here, deliberately after producer
                        // work (Part 13).
                        interval_start = sync_end;
                    }

                    if (i >= options.warmup)
                    {
                        trace_entry.source_ready_ns = trace.now();
                        trace_entry.staging_begin_ns = trace.now();
                    }
                    backend->stage_device_to_host();
                    const auto staging_end = std::chrono::steady_clock::now();

                    if (i >= options.warmup)
                    {
                        trace_entry.staging_end_ns = trace.now();
                        trace_entry.send_begin_ns = trace.now();
                    }
                    backend->host_send_data(world, peer);
                    const auto send_end = std::chrono::steady_clock::now();

                    if (i >= options.warmup)
                    {
                        trace_entry.send_end_ns = trace.now();
                        trace_entry.ack_wait_begin_ns = trace.now();
                    }
                    std::uint8_t ack = 0;
                    world.recv(peer, &ack, sizeof(ack));
                    const auto ack_observed = std::chrono::steady_clock::now();
                    const std::int64_t iteration_end_ns = trace.now();
                    previous_iteration_end_ns = iteration_end_ns;

                    // Part 22's per-iteration invariant: in "produce"
                    // scope, the timer starts at/before the producer's
                    // own enqueue, so this iteration's completion
                    // duration must be >= its own producer-completion
                    // duration. Checked per iteration on raw values,
                    // never against aggregated/cross-iteration medians
                    // (Part 22's explicit warning). steady_clock is
                    // monotonic, so this can only fail from an actual
                    // logic error in the timestamps above.
                    if (produce_scope)
                    {
                        const double producer_completion_this_iter =
                            microseconds_between(enqueue_start, sync_end);
                        const double completion_this_iter =
                            microseconds_between(interval_start, ack_observed);

                        if (completion_this_iter < producer_completion_this_iter)
                        {
                            throw std::runtime_error(
                                "timing invariant violated: produce-scope "
                                "completion (" +
                                std::to_string(completion_this_iter) +
                                "us) < producer completion (" +
                                std::to_string(producer_completion_this_iter) +
                                "us) at iteration " + std::to_string(i));
                        }
                    }

                    if (i >= options.warmup)
                    {
                        enqueue_samples.push_back(microseconds_between(enqueue_start, enqueue_end));
                        sync_samples.push_back(microseconds_between(enqueue_end, sync_end));
                        staging_samples.push_back(microseconds_between(sync_end, staging_end));
                        send_samples.push_back(microseconds_between(staging_end, send_end));
                        completion_samples.push_back(microseconds_between(interval_start, ack_observed));
                        trace_entry.ack_received_ns = iteration_end_ns;
                        trace_entry.iteration_end_ns = iteration_end_ns;
                        trace.add(trace_entry);
                    }
                }

                cpu.end();
                trace.flush();
                cpu.report(options.rank, size);
                tbccl_bench::diagnostics::samples(options.diagnostics, options.rank, size,
                                                "completion_confirmed", completion_samples);

                const LatencyStats completion_stats = compute_stats(completion_samples);
                const LatencyStats staging_stats = compute_stats(staging_samples);
                const LatencyStats sync_stats = compute_stats(sync_samples);
                const LatencyStats enqueue_stats = compute_stats(enqueue_samples);
                const LatencyStats send_stats = compute_stats(send_samples);

                WireStageSummary wire_summary{};
                world.recv(peer, &wire_summary, sizeof(wire_summary));
                const StageSummary summary = from_wire(wire_summary);

                // One untimed, full-byte-verified round -- never
                // folded into the timing above.
                const std::uint32_t verify_seed = 0xDEADBEEFu;
                backend->initialize_source(verify_seed);
                backend->prepare_source();
                backend->stage_device_to_host();
                backend->host_send_data(world, peer);

                std::uint8_t verify_ok = 0;
                world.recv(peer, &verify_ok, sizeof(verify_ok));

                if (verify_ok == 0)
                {
                    throw std::runtime_error(
                        mode_name + ": destination failed verification for size " +
                        std::to_string(size));
                }

                const double effective_GBps =
                    completion_stats.median_us > 0.0
                        ? (static_cast<double>(size) / 1e9) /
                              (completion_stats.median_us / 1e6)
                        : 0.0;

                CsvRow row;
                row.mode = mode_name;
                row.source_rank = static_cast<long long>(options.source_rank);
                row.source_backend = local_backend_name;
                row.destination_backend = peer_backend_name;
                row.payload_bytes = size;
                row.iterations = options.iterations;
                row.warmup = options.warmup;
                row.stats = completion_stats;
                row.source_staging_us = staging_stats.median_us;
                row.source_sync_us = sync_stats.median_us;
                row.sender_network_us = send_stats.median_us;
                row.receiver_network_us = summary.receiver_network_us;
                row.destination_staging_us = summary.destination_staging_us;
                row.destination_sync_us = summary.destination_sync_us;
                row.completion_confirmed_us = completion_stats.median_us;
                row.effective_GBps = effective_GBps;
                row.allocation_type = local_backend_name;
                row.alloc_stats = backend->stats();
                row.timing_scope = options.timing_scope;
                row.producer_enqueue_us = enqueue_stats.median_us;
                print_csv_row(row);

                std::cerr
                    << '[' << mode_name << '/' << options.timing_scope
                    << "] size=" << size
                    << " completion_confirmed_median_us=" << completion_stats.median_us
                    << " effective_GBps=" << effective_GBps << '\n';
            }
            else
            {
                std::vector<double> recv_samples;
                std::vector<double> staging_samples;
                recv_samples.reserve(reserve_n);
                staging_samples.reserve(reserve_n);

                for (int i = 0; i < total; ++i)
                {
                    if (i == options.warmup) cpu.begin();
                    tbccl_bench::trace::Entry trace_entry;
                    if (i >= options.warmup)
                    {
                        trace_entry.iteration_index = i - options.warmup;
                        trace_entry.iteration_begin_ns = trace.now();
                        trace_entry.recv_begin_ns = trace.now();
                    }
                    const auto recv_start = std::chrono::steady_clock::now();
                    backend->host_recv_data(world, peer);
                    const auto recv_end = std::chrono::steady_clock::now();

                    if (i >= options.warmup)
                    {
                        trace_entry.recv_end_ns = trace.now();
                        trace_entry.destination_staging_begin_ns = trace.now();
                    }
                    backend->stage_host_to_device();
                    const auto stage_end = std::chrono::steady_clock::now();

                    if (i >= options.warmup)
                    {
                        trace_entry.destination_staging_end_ns = trace.now();
                        trace_entry.destination_sync_end_ns =
                            trace_entry.destination_staging_end_ns;
                        trace_entry.ack_send_begin_ns = trace.now();
                    }
                    const std::uint8_t ack = 1;
                    world.send(peer, &ack, sizeof(ack));

                    if (i >= options.warmup)
                    {
                        recv_samples.push_back(microseconds_between(recv_start, recv_end));
                        staging_samples.push_back(microseconds_between(recv_end, stage_end));
                        trace_entry.ack_send_end_ns = trace.now();
                        trace_entry.iteration_end_ns = trace_entry.ack_send_end_ns;
                        trace.add(trace_entry);
                    }
                }

                cpu.end();
                trace.flush();
                cpu.report(options.rank, size);

                StageSummary summary;
                summary.receiver_network_us = compute_stats(recv_samples).median_us;
                summary.destination_staging_us = compute_stats(staging_samples).median_us;
                // This interface has no device-side synchronization
                // stage distinct from stage_host_to_device() itself
                // (that call already blocks until the destination
                // accelerator operation is complete -- see Part 8),
                // so there is nothing separate to report here.
                summary.destination_sync_us = kNA;

                const WireStageSummary wire_summary = to_wire(summary);
                world.send(peer, &wire_summary, sizeof(wire_summary));

                const std::uint32_t verify_seed = 0xDEADBEEFu;
                backend->host_recv_data(world, peer);
                backend->stage_host_to_device();

                const std::uint8_t verify_ok =
                    backend->verify_destination(verify_seed) ? 1 : 0;
                world.send(peer, &verify_ok, sizeof(verify_ok));
            }
        }
    }

    void run_end_to_end(const Options &options, tbccl::World &world)
    {
        run_end_to_end_protocol(
            options, world, "end-to-end", kTransferModeEndToEnd,
            options.local_backend);
    }

    void run_ack_calibration(const Options &options, tbccl::World &world)
    {
        run_end_to_end_protocol(
            options, world, "ack-calibration", kTransferModeAckCalibration,
            BackendKind::Host);
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

        if (options.busy_poll_us > 0 && !tbccl::busy_poll_supported())
            std::cerr << "busy-poll unsupported on this platform; requested_us="
                      << options.busy_poll_us << " is not enabled\n";

        auto world = tbccl::create_tcp_world(world_options);
        if (options.diagnostics) tbccl_bench::diagnostics::sockets(options.busy_poll_us);

        if (options.mode == "network-only")
        {
            run_network_only(options, *world);
        }
        else if (options.mode == "latency-floor")
        {
            run_latency_floor(options, *world);
        }
        else if (options.mode == "end-to-end")
        {
            run_end_to_end(options, *world);
        }
        else
        {
            run_ack_calibration(options, *world);
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
