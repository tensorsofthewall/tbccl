// Compute/communication overlap microbenchmark.
// Producer side does real, deterministic work on each bucket (not
// sleep), then asynchronously transfers it while
// continuing to compute the next bucket. Compares against a serial
// control (compute everything, then communicate everything) using the
// SAME buckets/buffers, over the SAME real TCP/TB4 connection.
//
// Scope note: the "compute" here is a
// deterministic CPU-bound workload (repeated FNV-1a-style hashing over
// the buffer), not a CUDA/Metal kernel -- building and validating a new
// GPU compute kernel was out of scope for the time this benchmark could
// receive this phase. This still tests the real architectural claim
// (Part AM item 139-142: does asynchronous transfer hide otherwise-
// serial work), just with host-side compute standing in for
// device-side compute. It does NOT claim to demonstrate GPU
// compute/network overlap specifically -- only CPU compute/network
// overlap, which is the more conservative (harder to hide, since CPU
// compute and the async worker's own CPU-side staging both compete for
// CPU time on the same cores) of the two claims.

#include <tbccl/async_transfer.hpp>
#include <tbccl/peer_capabilities.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include "tensor/host_async_backend.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
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
        PeerEndpoint peer;
        peer.host = text.substr(0, colon);
        peer.port = static_cast<std::uint16_t>(std::stoul(text.substr(colon + 1)));
        return peer;
    }

    struct Options
    {
        std::size_t rank = 0;
        std::vector<PeerEndpoint> peers;
        std::size_t source_rank = 0;
        std::size_t bucket_bytes = 4 * 1024 * 1024;
        std::size_t bucket_count = 4;
        std::size_t compute_rounds = 40; // hashing passes over the buffer, tuned so compute is comparable to comm time
        std::string label;
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
            else if (arg == "--peers")
            {
                const std::string value = require_value(arg, i, argc, argv);
                const auto comma = value.find(',');
                options.peers = {parse_peer(value.substr(0, comma)), parse_peer(value.substr(comma + 1))};
            }
            else if (arg == "--source-rank") options.source_rank = std::stoul(require_value(arg, i, argc, argv));
            else if (arg == "--bucket-bytes") options.bucket_bytes = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--bucket-count") options.bucket_count = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--compute-rounds") options.compute_rounds = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--label") options.label = require_value(arg, i, argc, argv);
            else throw std::runtime_error("unknown argument: " + arg);
        }
        return options;
    }

    // Deterministic, genuinely CPU-bound (not sleep-based) workload:
    // repeated FNV-1a-style passes over the buffer, each pass depending
    // on the previous pass's output so the compiler cannot elide it.
    void compute_bucket(std::vector<std::uint8_t> &buffer, std::size_t rounds)
    {
        std::uint64_t hash = 1469598103934665603ull;
        for (std::size_t round = 0; round < rounds; ++round)
        {
            for (std::size_t i = 0; i < buffer.size(); ++i)
            {
                hash ^= buffer[i];
                hash *= 1099511628211ull;
                buffer[i] = static_cast<std::uint8_t>(hash);
            }
        }
    }

    std::unique_ptr<tbccl::Connection> establish_connection(const Options &options)
    {
        if (options.rank == 0)
        {
            auto listener = tbccl::tcp_listen(options.peers[0].host, options.peers[0].port, {});
            return listener->accept();
        }
        return tbccl::tcp_connect(options.peers[0].host, options.peers[0].port, {});
    }

    double elapsed_us(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point end)
    {
        return std::chrono::duration<double, std::micro>(end - start).count();
    }

    // Runs `bucket_count` transfers (all Send or all Recv, matching
    // `is_sender`) over `transport`/`worker`, waiting for all of them.
    // Returns the wall time for the whole batch.
    double run_comm_batch(
        tbccl::TensorCommWorker &worker,
        tbccl::Transport &transport,
        bool is_sender,
        const std::vector<std::unique_ptr<tbccl_bench::tensor::HostAsyncBackend>> &backends,
        std::size_t bucket_bytes,
        std::uint64_t transfer_id_base)
    {
        const auto start = std::chrono::steady_clock::now();
        std::vector<tbccl::TransferWork> works;
        for (std::size_t i = 0; i < backends.size(); ++i)
        {
            tbccl::TransferRequest request;
            request.transfer_id = transfer_id_base + i;
            request.direction = is_sender ? tbccl::TransferDirection::Send : tbccl::TransferDirection::Recv;
            request.backend = backends[i].get();
            request.transport = &transport;
            request.total_bytes = bucket_bytes;
            works.push_back(worker.enqueue(request));
        }
        for (auto &w : works)
        {
            w.wait();
            if (w.has_error())
            {
                throw std::runtime_error("comm batch transfer failed: " + w.error());
            }
        }
        const auto end = std::chrono::steady_clock::now();
        return elapsed_us(start, end);
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_args(argc, argv);
        const bool is_sender = (options.rank == options.source_rank);

        auto connection = establish_connection(options);
        const auto local_caps = tbccl::local_capabilities();
        const auto remote_caps = tbccl::exchange_capabilities(*connection, local_caps);
        const auto negotiation = tbccl::negotiate(local_caps, remote_caps);
        if (!negotiation.ok)
        {
            throw std::runtime_error("negotiation failed: " + negotiation.failure_reason);
        }
        tbccl::TcpTransport transport(std::move(connection));

        std::vector<std::vector<std::uint8_t>> buckets(options.bucket_count);
        std::vector<std::unique_ptr<tbccl_bench::tensor::HostAsyncBackend>> backends;
        for (auto &bucket : buckets)
        {
            bucket.assign(options.bucket_bytes, 0x11);
            backends.push_back(std::make_unique<tbccl_bench::tensor::HostAsyncBackend>(
                bucket.data(), bucket.size()));
        }

        // ---- T_compute_only: compute every bucket, no communication.
        auto t0 = std::chrono::steady_clock::now();
        if (is_sender)
        {
            for (auto &bucket : buckets) compute_bucket(bucket, options.compute_rounds);
        }
        auto t1 = std::chrono::steady_clock::now();
        const double compute_only_us = elapsed_us(t0, t1);

        // ---- T_comm_only: communicate every bucket, no interleaved
        // compute (a fresh worker, single connection reused).
        double comm_only_us = 0;
        {
            tbccl::TensorCommWorker worker(1, options.bucket_count + 1);
            comm_only_us = run_comm_batch(
                worker, transport, is_sender, backends, options.bucket_bytes, 2000);
        }

        // ---- T_serial: compute bucket 0..N-1 in full, THEN communicate
        // bucket 0..N-1 in full (no overlap by construction).
        double serial_us = 0;
        {
            const auto start = std::chrono::steady_clock::now();
            if (is_sender)
            {
                for (auto &bucket : buckets) compute_bucket(bucket, options.compute_rounds);
            }
            tbccl::TensorCommWorker worker(1, options.bucket_count + 1);
            run_comm_batch(worker, transport, is_sender, backends, options.bucket_bytes, 3000);
            const auto end = std::chrono::steady_clock::now();
            serial_us = elapsed_us(start, end);
        }

        // ---- T_async_overlap: compute bucket i, enqueue transfer i,
        // move on to compute bucket i+1 immediately -- transfer i
        // proceeds on TensorCommWorker's own threads while THIS thread
        // computes bucket i+1.
        double async_overlap_us = 0;
        {
            const auto start = std::chrono::steady_clock::now();
            tbccl::TensorCommWorker worker(1, options.bucket_count + 1);
            std::vector<tbccl::TransferWork> works;
            for (std::size_t i = 0; i < options.bucket_count; ++i)
            {
                if (is_sender)
                {
                    compute_bucket(buckets[i], options.compute_rounds);
                }
                tbccl::TransferRequest request;
                request.transfer_id = 4000 + i;
                request.direction = is_sender ? tbccl::TransferDirection::Send : tbccl::TransferDirection::Recv;
                request.backend = backends[i].get();
                request.transport = &transport;
                request.total_bytes = options.bucket_bytes;
                works.push_back(worker.enqueue(request));
            }
            for (auto &w : works)
            {
                w.wait();
                if (w.has_error()) throw std::runtime_error("async-overlap transfer failed: " + w.error());
            }
            const auto end = std::chrono::steady_clock::now();
            async_overlap_us = elapsed_us(start, end);
        }

        const double hidden_us = serial_us - async_overlap_us;
        const double hidden_fraction = (comm_only_us > 0) ? (hidden_us / comm_only_us) : 0.0;

        std::ostringstream json;
        json << "{\n"
             << "  \"label\": \"" << options.label << "\",\n"
             << "  \"rank\": " << options.rank << ",\n"
             << "  \"role\": \"" << (is_sender ? "sender" : "receiver") << "\",\n"
             << "  \"bucket_bytes\": " << options.bucket_bytes << ",\n"
             << "  \"bucket_count\": " << options.bucket_count << ",\n"
             << "  \"compute_rounds\": " << options.compute_rounds << ",\n"
             << "  \"compute_only_us\": " << compute_only_us << ",\n"
             << "  \"comm_only_us\": " << comm_only_us << ",\n"
             << "  \"serial_us\": " << serial_us << ",\n"
             << "  \"async_overlap_us\": " << async_overlap_us << ",\n"
             << "  \"hidden_us\": " << hidden_us << ",\n"
             << "  \"hidden_fraction\": " << hidden_fraction << ",\n"
             << "  \"speedup\": " << (async_overlap_us > 0 ? serial_us / async_overlap_us : 0.0) << "\n"
             << "}\n";
        std::cout << json.str();

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
