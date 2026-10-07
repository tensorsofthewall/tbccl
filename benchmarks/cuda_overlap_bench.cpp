// Real CUDA compute / TBCCL communication overlap benchmark.
// Simulates a DDP-like gradient-bucket production pattern (NOT actual
// DDP) -- N independent CUDA buckets, each genuinely computed
// by a real GPU kernel on the SENDER (always Linux/CUDA), then
// communicated with TBCCL's existing per-chunk async CUDA staging
// (CudaChunkedAsyncBackend, the CUDA/Metal device-pipeline work) to a portable RECEIVER backend
// (host or Metal shared, matching async_transfer_bench.cpp's existing
// cross-platform pattern -- the receiver does no compute
// and must build/run on a non-CUDA machine).
//
// This is a plain .cpp (not .cu): all CUDA-specific code is guarded
// behind TBCCL_ENABLE_CUDA, using the CUDA-type-free headers
// (cuda_bucket_compute.hpp, cuda_chunked_async_backend.hpp -- both use
// void* for device pointers/streams) so this file itself never
// includes cuda_runtime.h or CUDA syntax. The sender role REQUIRES a
// CUDA-enabled build (real CUDA kernel execution only, never
// a CPU/sleep substitute for the authoritative result); the receiver
// role requires only tensor_backend.hpp's portable backend set.
//
// Central design decision: this benchmark deliberately uses
// ONLY the existing CudaChunkedAsyncBackend's cudaStreamSynchronize()-
// per-chunk design, with ZERO new cudaEvent_t machinery. Correctness
// of "communication never reads a bucket before its compute finishes"
// is established the simple way: the host thread launches bucket i's
// compute kernel on a dedicated compute stream, then explicitly
// synchronizes that stream (confirming bucket i's compute is complete
// and globally visible) BEFORE enqueueing bucket i's TransferRequest.
// Because that synchronize call returns before this loop moves on to
// launch bucket i+1's kernel, and because worker.enqueue() does not
// block waiting for the transfer to complete, bucket i's communication
// (handled by TensorCommWorker's own threads) and bucket i+1's GPU
// compute (launched immediately after) proceed concurrently -- the
// overlap this phase exists to measure, achieved with no new
// synchronization primitives.
//
// Schedules: "serial" computes every bucket, synchronizes
// once, THEN enqueues every transfer, THEN waits all -- the
// authoritative non-overlapped baseline. "overlap" interleaves compute
// and enqueue per bucket as described above. "compute-only" and
// "comm-only" isolate each half for the speedup/hidden-fraction
// calculation.

#include <tbccl/async_transfer.hpp>
#include <tbccl/peer_capabilities.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include "tensor/tensor_backend.hpp"
#include "tensor/tensor_backend_async_adapter.hpp"

// cuda_bucket_compute.hpp's expected_bucket_byte() is a portable
// inline function (plain integer arithmetic, no CUDA dependency) --
// included unconditionally so a non-CUDA receiver build (e.g. the
// Mac's Metal-shared destination) can still verify CUDA-computed
// bytes it received. Only launch_bucket_compute()/
// launch_bucket_fill_input() (actual kernel launches) require
// TBCCL_ENABLE_CUDA and are never called outside that guard below.
#include "tensor/cuda_bucket_compute.hpp"

#if defined(TBCCL_ENABLE_CUDA)
#include "tensor/cuda_chunked_async_backend.hpp"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    struct PeerEndpoint { std::string host; std::uint16_t port = 0; };

    PeerEndpoint parse_peer(const std::string &text)
    {
        const auto colon = text.rfind(':');
        if (colon == std::string::npos)
            throw std::runtime_error("invalid peer endpoint (expected HOST:PORT): " + text);
        return {text.substr(0, colon), static_cast<std::uint16_t>(std::stoul(text.substr(colon + 1)))};
    }

    struct Options
    {
        std::size_t rank = 0;
        std::vector<PeerEndpoint> peers;
        std::size_t source_rank = 0;
        std::size_t bucket_bytes = 4 * 1024 * 1024;
        std::size_t bucket_count = 4;
        int compute_rounds = 0;
        std::size_t chunk_bytes = 256 * 1024;
        std::size_t depth = 1;
        std::size_t warmup = 3;
        std::size_t iterations = 15;
        std::string schedule = "overlap"; // serial | overlap | compute-only | comm-only
        std::string receiver_backend = "host"; // host | metal-shared
        std::string label;
        std::string output;
        std::string timeline_output; // if set, dump the LAST measured iteration's per-bucket timeline as CSV
        bool calibrate = false;
        std::size_t calibrate_target_us = 1000;
    };

    std::string require_value(const std::string &flag, int &i, int argc, char **argv)
    {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
        return argv[++i];
    }

    Options parse_args(int argc, char **argv)
    {
        Options o;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--rank") o.rank = std::stoul(require_value(arg, i, argc, argv));
            else if (arg == "--peers")
            {
                std::string v = require_value(arg, i, argc, argv);
                auto comma = v.find(',');
                o.peers = {parse_peer(v.substr(0, comma)), parse_peer(v.substr(comma + 1))};
            }
            else if (arg == "--source-rank") o.source_rank = std::stoul(require_value(arg, i, argc, argv));
            else if (arg == "--bucket-bytes") o.bucket_bytes = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--bucket-count") o.bucket_count = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--compute-rounds") o.compute_rounds = std::stoi(require_value(arg, i, argc, argv));
            else if (arg == "--chunk-bytes") o.chunk_bytes = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--depth") o.depth = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--warmup") o.warmup = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--iterations") o.iterations = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--schedule") o.schedule = require_value(arg, i, argc, argv);
            else if (arg == "--receiver-backend") o.receiver_backend = require_value(arg, i, argc, argv);
            else if (arg == "--label") o.label = require_value(arg, i, argc, argv);
            else if (arg == "--output") o.output = require_value(arg, i, argc, argv);
            else if (arg == "--timeline-output") o.timeline_output = require_value(arg, i, argc, argv);
            else if (arg == "--calibrate") o.calibrate = true;
            else if (arg == "--calibrate-target-us") o.calibrate_target_us = std::stoull(require_value(arg, i, argc, argv));
            else throw std::runtime_error("unknown argument: " + arg);
        }
        return o;
    }

    // Deterministic per-(round, bucket) seed. For comm-only mode
    // (do_compute == false), content is never recomputed per round --
    // callers pass content_round=0 consistently for both the one-time
    // pre-fill and the final verify step.
    std::uint32_t bucket_seed(std::size_t content_round, std::size_t bucket)
    {
        return 0xA5A5A5A5u + static_cast<std::uint32_t>(content_round) * 1000u +
               static_cast<std::uint32_t>(bucket);
    }

    double percentile(std::vector<double> values, double fraction)
    {
        std::sort(values.begin(), values.end());
        const std::size_t index = static_cast<std::size_t>(
            std::max<double>(0, std::ceil(fraction * static_cast<double>(values.size())) - 1));
        return values[std::min(index, values.size() - 1)];
    }

    // One bucket's compute+communicate state for the sender side.
    // TransferWork's default constructor is private (only
    // TensorCommWorker constructs a live one), so `work` must be an
    // optional, not a bare member, for this struct to stay default-
    // constructible.
    struct SenderBucket
    {
        std::optional<tbccl::TransferWork> work;
        bool work_valid = false;
    };

    struct TimelineEvent
    {
        std::size_t bucket = 0;
        double compute_start_us = 0;
        double compute_end_us = 0;
        double enqueue_start_us = 0;
        double enqueue_end_us = 0;
    };

#if defined(TBCCL_ENABLE_CUDA)
    using tbccl_bench::tensor::CudaChunkedAsyncBackend;

    // --calibrate mode: no networking at all. Finds, by simple scaling
    // + linear interpolation, a round count whose kernel time is close
    // to --calibrate-target-us for --bucket-bytes, then prints it.
    // Calibration happens once, outside any measured loop; this
    // mode's whole purpose is to be run once ahead of time to pick
    // --compute-rounds for the real experiment. CUDA-only (the sender
    // side is always CUDA; calibration only makes sense there).
    void run_calibration(const Options &options)
    {
        auto backend = std::make_unique<CudaChunkedAsyncBackend>();
        backend->allocate(options.bucket_bytes, options.bucket_bytes);

        // A dedicated throwaway stream for timing only -- not the
        // backend's own copy stream.
        tbccl_bench::tensor::launch_bucket_fill_input(backend->source_device_ptr(), options.bucket_bytes, 1u, nullptr);

        const double target_us = static_cast<double>(options.calibrate_target_us);
        auto measure = [&](int rounds) {
            const auto start = std::chrono::steady_clock::now();
            tbccl_bench::tensor::launch_bucket_compute(
                backend->source_device_ptr(), options.bucket_bytes, 1u, rounds, nullptr);
            // No explicit stream handle here (nullptr => default
            // stream), so a subsequent synchronous CUDA op on the same
            // default stream naturally waits -- use the backend's own
            // verify (a synchronous readback) purely as a wait point.
            backend->verify_source(1u); // return value unused: just forces completion
            const auto end = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::micro>(end - start).count();
        };

        measure(10); // warm up (context/JIT overhead)
        int rounds = 100;
        double us = measure(rounds);
        if (us > 0) rounds = std::max(1, static_cast<int>(rounds * (target_us / us)));
        us = measure(rounds);
        if (us > 0 && std::abs(us - target_us) / target_us > 0.1)
        {
            rounds = std::max(1, static_cast<int>(rounds * (target_us / us)));
            us = measure(rounds);
        }

        std::cout << "{\n"
                  << "  \"bucket_bytes\": " << options.bucket_bytes << ",\n"
                  << "  \"calibrate_target_us\": " << options.calibrate_target_us << ",\n"
                  << "  \"compute_rounds\": " << rounds << ",\n"
                  << "  \"achieved_us\": " << us << "\n"
                  << "}\n";
    }
#endif

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_args(argc, argv);

#if defined(TBCCL_ENABLE_CUDA)
        if (options.calibrate)
        {
            run_calibration(options);
            return 0;
        }
#else
        if (options.calibrate)
        {
            throw std::runtime_error("--calibrate requires a CUDA-enabled build");
        }
#endif

        if (options.peers.size() != 2)
            throw std::runtime_error("--peers must list exactly 2 endpoints");

        const bool is_sender = (options.rank == options.source_rank);
        if (is_sender)
        {
#if !defined(TBCCL_ENABLE_CUDA)
            throw std::runtime_error(
                "the sender role requires a CUDA-enabled build (Part AS: real CUDA "
                "compute is the authoritative result, never a CPU substitute)");
#endif
        }

        std::unique_ptr<tbccl::Connection> connection;
        if (options.rank == 0)
        {
            auto listener = tbccl::tcp_listen(options.peers[0].host, options.peers[0].port, {});
            std::fprintf(stderr, "READY\n");
            std::fflush(stderr);
            connection = listener->accept();
        }
        else
        {
            connection = tbccl::tcp_connect(options.peers[0].host, options.peers[0].port, {});
        }

        const auto local_caps = tbccl::local_capabilities();
        tbccl::exchange_capabilities(*connection, local_caps);
        tbccl::TcpTransport transport(std::move(connection));

        const bool do_compute = (options.schedule != "comm-only");
        const bool do_comm = (options.schedule != "compute-only");

        // Generic AsyncMemoryBackend handles for whichever concrete
        // type this role actually uses -- TransferRequest/
        // TensorCommWorker never know the difference.
        std::vector<tbccl::AsyncMemoryBackend *> backend_ptrs(options.bucket_count, nullptr);

#if defined(TBCCL_ENABLE_CUDA)
        std::vector<std::unique_ptr<CudaChunkedAsyncBackend>> cuda_backends;
#endif

        // Receiver-side (or any non-CUDA role): portable TensorBackend
        // (host or metal-shared) wrapped by the existing async
        // tensor-transfer adapter -- identical to
        // async_transfer_bench.cpp's pattern.
        std::vector<std::unique_ptr<tbccl_bench::tensor::TensorBackend>> device_backends;
        std::vector<std::unique_ptr<tbccl_bench::tensor::TensorBackendAsyncAdapter>> device_adapters;

        // --- Allocate backends for this rank's role ---
#if defined(TBCCL_ENABLE_CUDA)
        void *compute_stream = nullptr;
        void *copy_stream = nullptr;
        if (is_sender)
        {
            if (do_compute)
            {
                cudaStream_t raw_compute_stream;
                cudaStreamCreate(&raw_compute_stream);
                compute_stream = raw_compute_stream;
            }
            if (do_comm)
            {
                cudaStream_t raw_copy_stream;
                cudaStreamCreate(&raw_copy_stream);
                copy_stream = raw_copy_stream;
            }
            for (std::size_t i = 0; i < options.bucket_count; ++i)
            {
                auto backend = std::make_unique<CudaChunkedAsyncBackend>(do_comm ? copy_stream : nullptr);
                const std::size_t max_chunk = (options.chunk_bytes == 0) ? options.bucket_bytes : options.chunk_bytes;
                backend->allocate(options.bucket_bytes, max_chunk);
                backend_ptrs[i] = backend.get();
                cuda_backends.push_back(std::move(backend));
            }
        }
#endif
        if (!is_sender)
        {
            const auto kind = tbccl_bench::tensor::parse_backend_kind(options.receiver_backend);
            if (!tbccl_bench::tensor::backend_kind_available(kind))
                throw std::runtime_error("receiver backend not available in this build: " + options.receiver_backend);
            for (std::size_t i = 0; i < options.bucket_count; ++i)
            {
                auto backend = tbccl_bench::tensor::make_backend(kind);
                backend->allocate(options.bucket_bytes);
                auto adapter = std::make_unique<tbccl_bench::tensor::TensorBackendAsyncAdapter>(*backend);
                backend_ptrs[i] = adapter.get();
                device_backends.push_back(std::move(backend));
                device_adapters.push_back(std::move(adapter));
            }
        }

        // Queue_depth sized to the whole batch so enqueue() never
        // blocks waiting for TensorCommWorker to drain mid-batch --
        // that would serialize the very overlap this benchmark
        // measures. Measured and reported via enqueue_us.
        std::unique_ptr<tbccl::TensorCommWorker> worker;
        if (do_comm)
        {
            worker = std::make_unique<tbccl::TensorCommWorker>(
                options.depth, /*queue_depth=*/options.bucket_count + 1);
        }

        // TensorBackendAsyncAdapter::begin_transfer() resets its
        // "already staged this round"/"chunks committed this round"
        // bookkeeping and must be called once before EVERY transfer
        // that reuses it (the async tensor-transfer work convention)
        // -- the chunk count is fixed for a given (bucket_bytes,
        // chunk_bytes), so compute it once here and call
        // begin_transfer() per round below.
        const auto chunk_plan = tbccl::plan_chunks(options.bucket_bytes, options.chunk_bytes, 1);
        const std::size_t chunk_count = chunk_plan.empty() ? 1 : chunk_plan.size();

#if defined(TBCCL_ENABLE_CUDA)
        // Comm-only mode uses precomputed buckets -- filled ONCE here,
        // outside the timed loop, with content_round=0 (no compute
        // happens per round in this mode, so content never changes
        // between rounds; the final verify step below uses the same
        // content_round=0 to match).
        if (is_sender && !do_compute)
        {
            for (std::size_t i = 0; i < options.bucket_count; ++i)
            {
                tbccl_bench::tensor::launch_bucket_fill_input(
                    cuda_backends[i]->source_device_ptr(), options.bucket_bytes, bucket_seed(0, i), copy_stream);
            }
            cudaStreamSynchronize(static_cast<cudaStream_t>(copy_stream));
        }
#endif

        std::vector<double> total_us_samples;
        std::vector<double> enqueue_us_samples;
        bool verify_ok = true;
        std::vector<TimelineEvent> last_timeline;

        const std::size_t total_rounds = options.warmup + options.iterations;
        for (std::size_t round = 0; round < total_rounds; ++round)
        {
            std::vector<TimelineEvent> timeline(options.bucket_count);
            const auto iter_start = std::chrono::steady_clock::now();
            // Timeline timestamps are relative to iter_start (not
            // absolute epoch time): an absolute now_us() value is
            // ~5*10^10 at microsecond resolution, and default ofstream
            // formatting (6 significant digits) silently truncates
            // sub-millisecond differences between events at that
            // magnitude -- relative timestamps stay small and
            // human-readable, and are exactly what overlap analysis
            // needs anyway.
            [[maybe_unused]] auto rel_us = [&]() {
                return std::chrono::duration<double, std::micro>(
                           std::chrono::steady_clock::now() - iter_start)
                    .count();
            };
            for (std::size_t i = 0; i < options.bucket_count; ++i) timeline[i].bucket = i;

            if (is_sender)
            {
                std::vector<SenderBucket> state(options.bucket_count);

#if defined(TBCCL_ENABLE_CUDA)
                auto launch_and_sync = [&](std::size_t i) {
                    const std::uint32_t seed = bucket_seed(round, i);
                    timeline[i].compute_start_us = rel_us();
                    if (do_compute)
                    {
                        tbccl_bench::tensor::launch_bucket_fill_input(
                            cuda_backends[i]->source_device_ptr(), options.bucket_bytes, seed, compute_stream);
                        tbccl_bench::tensor::launch_bucket_compute(
                            cuda_backends[i]->source_device_ptr(), options.bucket_bytes, seed,
                            options.compute_rounds, compute_stream);
                        cudaStreamSynchronize(static_cast<cudaStream_t>(compute_stream));
                    }
                    timeline[i].compute_end_us = rel_us();
                };

                auto enqueue_bucket = [&](std::size_t i) {
                    tbccl::TransferRequest request;
                    request.transfer_id = static_cast<std::uint64_t>(i);
                    request.direction = tbccl::TransferDirection::Send;
                    request.backend = backend_ptrs[i];
                    request.transport = &transport;
                    request.total_bytes = options.bucket_bytes;
                    request.chunk_hint = options.chunk_bytes;
                    timeline[i].enqueue_start_us = rel_us();
                    state[i].work = worker->enqueue(request);
                    timeline[i].enqueue_end_us = rel_us();
                    state[i].work_valid = true;
                };

                if (options.schedule == "serial")
                {
                    for (std::size_t i = 0; i < options.bucket_count; ++i) launch_and_sync(i);
                    if (do_comm)
                        for (std::size_t i = 0; i < options.bucket_count; ++i) enqueue_bucket(i);
                }
                else if (options.schedule == "overlap" || options.schedule == "compute-only")
                {
                    for (std::size_t i = 0; i < options.bucket_count; ++i)
                    {
                        launch_and_sync(i);
                        if (do_comm) enqueue_bucket(i);
                    }
                }
                else if (options.schedule == "comm-only")
                {
                    for (std::size_t i = 0; i < options.bucket_count; ++i) enqueue_bucket(i);
                }
                else
                {
                    throw std::runtime_error("unknown --schedule: " + options.schedule);
                }
#endif

                if (do_comm)
                {
                    for (auto &s : state)
                    {
                        if (s.work_valid)
                        {
                            s.work->wait();
                            if (s.work->has_error())
                                throw std::runtime_error("bucket transfer failed: " + s.work->error());
                        }
                    }
                    // One batch-level ack, not per-bucket --
                    // establishes destination-confirmed completion for
                    // the whole iteration, matching the established
                    // sync-vs-async timing-scope convention at the
                    // batch granularity instead of per-transfer.
                    std::uint8_t ack = 0;
                    transport.recv(&ack, sizeof(ack));
                }
            }
            else // receiver
            {
                if (do_comm)
                {
                    for (auto &adapter : device_adapters) adapter->begin_transfer(chunk_count);
                    std::vector<tbccl::TransferWork> works;
                    works.reserve(options.bucket_count);
                    for (std::size_t i = 0; i < options.bucket_count; ++i)
                    {
                        tbccl::TransferRequest request;
                        request.transfer_id = static_cast<std::uint64_t>(i);
                        request.direction = tbccl::TransferDirection::Recv;
                        request.backend = backend_ptrs[i];
                        request.transport = &transport;
                        request.total_bytes = options.bucket_bytes;
                        request.chunk_hint = options.chunk_bytes;
                        works.push_back(worker->enqueue(request));
                    }
                    for (auto &w : works)
                    {
                        w.wait();
                        if (w.has_error()) throw std::runtime_error("bucket recv failed: " + w.error());
                    }
                    std::uint8_t ack = 1;
                    transport.send(&ack, sizeof(ack));
                }
            }

            const auto iter_end = std::chrono::steady_clock::now();
            if (round >= options.warmup)
            {
                total_us_samples.push_back(std::chrono::duration<double, std::micro>(iter_end - iter_start).count());
                if (is_sender)
                {
                    for (auto &t : timeline)
                        enqueue_us_samples.push_back(t.enqueue_end_us - t.enqueue_start_us);
                }
            }
            if (round == total_rounds - 1) last_timeline = timeline;
        }

        // the benchmark methodology
        // (docs/development/benchmark-methodology.md): verification
        // happens ONCE, untimed, after the whole measured loop --
        // never inside it. content_round matches whichever content is
        // actually in the sender's device buffers right now: for
        // compute-driving schedules, that's the last measured round;
        // for comm-only, content was fixed once at content_round=0.
        const std::size_t content_round = do_compute ? (total_rounds - 1) : 0;
#if defined(TBCCL_ENABLE_CUDA)
        if (is_sender && do_compute)
        {
            for (std::size_t i = 0; i < options.bucket_count; ++i)
            {
                const std::uint32_t seed = bucket_seed(content_round, i);
                tbccl_bench::tensor::launch_bucket_fill_input(
                    cuda_backends[i]->source_device_ptr(), options.bucket_bytes, seed, compute_stream);
                tbccl_bench::tensor::launch_bucket_compute(
                    cuda_backends[i]->source_device_ptr(), options.bucket_bytes, seed, options.compute_rounds,
                    compute_stream);
            }
            cudaStreamSynchronize(static_cast<cudaStream_t>(compute_stream));
        }
#endif
        if (do_comm)
        {
            if (is_sender)
            {
                for (std::size_t i = 0; i < options.bucket_count; ++i)
                {
                    tbccl::TransferRequest request;
                    request.transfer_id = 1000000 + static_cast<std::uint64_t>(i);
                    request.direction = tbccl::TransferDirection::Send;
                    request.backend = backend_ptrs[i];
                    request.transport = &transport;
                    request.total_bytes = options.bucket_bytes;
                    request.chunk_hint = options.chunk_bytes;
                    auto w = worker->enqueue(request);
                    w.wait();
                }
                std::uint8_t ack = 0;
                transport.recv(&ack, sizeof(ack));
            }
            else
            {
                for (auto &adapter : device_adapters) adapter->begin_transfer(chunk_count);
                std::vector<tbccl::TransferWork> works;
                works.reserve(options.bucket_count);
                for (std::size_t i = 0; i < options.bucket_count; ++i)
                {
                    tbccl::TransferRequest request;
                    request.transfer_id = 1000000 + static_cast<std::uint64_t>(i);
                    request.direction = tbccl::TransferDirection::Recv;
                    request.backend = backend_ptrs[i];
                    request.transport = &transport;
                    request.total_bytes = options.bucket_bytes;
                    request.chunk_hint = options.chunk_bytes;
                    works.push_back(worker->enqueue(request));
                }
                for (auto &w : works) w.wait();
                // Portable verify: TensorBackend::destination_staging_data()
                // is a host-visible pointer on every backend kind
                // (host/metal-shared/cuda-pinned/...), and
                // expected_bucket_byte()/pattern_byte() are both plain
                // portable inline functions -- this comparison needs no
                // CUDA/Metal-specific code at all, even when this build
                // itself has no CUDA (e.g. the Mac receiver verifying
                // bytes a CUDA peer computed and sent).
                for (std::size_t i = 0; i < options.bucket_count; ++i)
                {
                    const std::uint32_t seed = bucket_seed(content_round, i);
                    const auto *host_copy =
                        static_cast<const std::uint8_t *>(device_backends[i]->destination_staging_data());
                    for (std::size_t b = 0; b < options.bucket_bytes; ++b)
                    {
                        const std::uint8_t expected = do_compute
                            ? tbccl_bench::tensor::expected_bucket_byte(b, seed, options.compute_rounds)
                            : tbccl_bench::tensor::pattern_byte(b, seed);
                        if (host_copy[b] != expected) { verify_ok = false; break; }
                    }
                }
                std::uint8_t ack = 1;
                transport.send(&ack, sizeof(ack));
            }
        }

        if (!options.timeline_output.empty() && is_sender)
        {
            std::ofstream f(options.timeline_output);
            f << "bucket,compute_start_us,compute_end_us,enqueue_start_us,enqueue_end_us\n";
            for (const auto &t : last_timeline)
            {
                f << t.bucket << "," << t.compute_start_us << "," << t.compute_end_us << ","
                  << t.enqueue_start_us << "," << t.enqueue_end_us << "\n";
            }
        }

        std::ostringstream json;
        json << "{\n"
             << "  \"label\": \"" << options.label << "\",\n"
             << "  \"rank\": " << options.rank << ",\n"
             << "  \"role\": \"" << (is_sender ? "sender" : "receiver") << "\",\n"
             << "  \"schedule\": \"" << options.schedule << "\",\n"
             << "  \"bucket_bytes\": " << options.bucket_bytes << ",\n"
             << "  \"bucket_count\": " << options.bucket_count << ",\n"
             << "  \"compute_rounds\": " << options.compute_rounds << ",\n"
             << "  \"chunk_bytes\": " << options.chunk_bytes << ",\n"
             << "  \"depth\": " << options.depth << ",\n"
             << "  \"warmup\": " << options.warmup << ",\n"
             << "  \"iterations\": " << options.iterations << ",\n"
             << "  \"verify_ok\": " << (verify_ok ? "true" : "false") << ",\n";

        if (is_sender)
        {
            const double median = percentile(total_us_samples, 0.5);
            const double p95 = percentile(total_us_samples, 0.95);
            const double enqueue_median = enqueue_us_samples.empty() ? 0.0 : percentile(enqueue_us_samples, 0.5);
            const double enqueue_max = enqueue_us_samples.empty()
                ? 0.0
                : *std::max_element(enqueue_us_samples.begin(), enqueue_us_samples.end());
            json << "  \"total_median_us\": " << median << ",\n"
                 << "  \"total_p95_us\": " << p95 << ",\n"
                 << "  \"enqueue_median_us\": " << enqueue_median << ",\n"
                 << "  \"enqueue_max_us\": " << enqueue_max << "\n";
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

#if defined(TBCCL_ENABLE_CUDA)
        if (compute_stream) cudaStreamDestroy(static_cast<cudaStream_t>(compute_stream));
        if (copy_stream) cudaStreamDestroy(static_cast<cudaStream_t>(copy_stream));
#endif

        return verify_ok ? 0 : 1;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
