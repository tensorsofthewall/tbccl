// The central experiment -- real CUDA compute(bucket N+1)
// overlapping a real heterogeneous N=2 SUM AllReduce(bucket N), over the
// real Linux<->Mac TB4 link. Reuses, unchanged: the CUDA compute-overlap work's
// cuda_bucket_compute kernel, the CudaChunkedAsyncBackend, and
// n2_all_reduce_tensor() -- the only new scheduling component
// is BucketAllReduceWorker (bucket_allreduce_worker.hpp), which guarantees
// exactly one AllReduce protocol is ever in flight on the wire while
// letting the CUDA rank's compute run ahead (Part 3/4 of the plan).
//
// AllReduce datatype: Int32, not Float32. The CUDA compute kernel
// (cuda_bucket_compute) produces an arbitrary deterministic byte pattern
// (a hash-mix avalanche transform), which this benchmark reinterprets as
// a buffer of int32 words for the AllReduce SUM -- integer wraparound is
// well-defined two's-complement arithmetic (host_reduce_backend.hpp/
// cuda_reduce_backend.cu's wrapping-add fix, the bucketed all-reduce overlap work), so exact-equality
// verification holds regardless of how the compute kernel's
// output happens to be distributed, unlike reinterpreting arbitrary bytes
// as Float32 (NaN/Inf hazard). This is a deliberate adaptation documented
//.md, not an accidental scope creep.

#include <tbccl/collectives.hpp>
#include <tbccl/hetero_allreduce.hpp>
#include <tbccl/peer_capabilities.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/tcp_world.hpp>
#include <tbccl/transport.hpp>

#include "bucket_allreduce_worker.hpp"
#include "tensor/host_async_backend.hpp"
#include "tensor/host_reduce_backend.hpp"
#include "tensor/metal_shared_direct_async_backend.hpp"
#include "tensor/tensor_backend.hpp"
#include "tensor/tensor_backend_async_adapter.hpp"

// cuda_bucket_compute.hpp's expected_bucket_byte() is portable, CUDA-type-
// free C++ (the compute-overlap design: "critical for the
// receiver role, which commonly runs on a non-CUDA machine but still needs
// to verify CUDA-computed bytes it received") -- included unconditionally
// so the Mac (non-CUDA) rank can compute the correct expected value for
// its own verification. Only
// launch_bucket_compute()/launch_bucket_fill_input() (the actual kernel
// launches, implemented in the .cu) require TBCCL_ENABLE_CUDA, and those
// are never called outside that guard below -- matching
// cuda_overlap_bench.cpp's identical precedent.
#include "tensor/cuda_bucket_compute.hpp"

#if defined(TBCCL_ENABLE_CUDA)
#include "tensor/cuda_chunked_async_backend.hpp"
#include "tensor/cuda_reduce_backend.hpp"
#include <cuda_runtime.h>
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

    double elapsed_us(Clock::time_point start, Clock::time_point end)
    {
        return std::chrono::duration<double, std::micro>(end - start).count();
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
            throw std::runtime_error("invalid peer endpoint (expected HOST:PORT): " + text);
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
        std::size_t root = 0;
        std::size_t bucket_bytes = 4 * 1024 * 1024;
        std::size_t bucket_count = 4;
        int compute_rounds = 0;
        std::size_t chunk_bytes = 262144;
        std::size_t warmup = 3;
        std::size_t iterations = 10;
        std::string schedule = "overlap"; // compute-only, collective-only, serial, overlap
        std::string metal_async_path = "adapter"; // adapter, direct
        std::string label;
        std::string output;
        std::string timeline_output;
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
            else if (arg == "--peers") o.peers = parse_peers(require_value(arg, i, argc, argv));
            else if (arg == "--root") o.root = std::stoul(require_value(arg, i, argc, argv));
            else if (arg == "--bucket-bytes") o.bucket_bytes = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--bucket-count") o.bucket_count = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--compute-rounds") o.compute_rounds = std::stoi(require_value(arg, i, argc, argv));
            else if (arg == "--chunk-bytes") o.chunk_bytes = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--warmup") o.warmup = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--iterations") o.iterations = std::stoull(require_value(arg, i, argc, argv));
            else if (arg == "--schedule") o.schedule = require_value(arg, i, argc, argv);
            else if (arg == "--metal-async-path") o.metal_async_path = require_value(arg, i, argc, argv);
            else if (arg == "--label") o.label = require_value(arg, i, argc, argv);
            else if (arg == "--output") o.output = require_value(arg, i, argc, argv);
            else if (arg == "--timeline-output") o.timeline_output = require_value(arg, i, argc, argv);
            else if (arg == "--calibrate") o.calibrate = true;
            else if (arg == "--calibrate-target-us") o.calibrate_target_us = std::stoull(require_value(arg, i, argc, argv));
            else throw std::runtime_error("unknown argument: " + arg);
        }
        return o;
    }

    double percentile(std::vector<double> values, double fraction)
    {
        if (values.empty()) return 0.0;
        std::sort(values.begin(), values.end());
        const std::size_t index = static_cast<std::size_t>(
            std::max<double>(0, std::ceil(fraction * static_cast<double>(values.size())) - 1));
        return values[std::min(index, values.size() - 1)];
    }

    struct Stats
    {
        double median_us = 0, p95_us = 0, mean_us = 0, max_us = 0;
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
        return s;
    }

    // Mac-side per-byte local pattern (no compute -- Metal rank
    // participates only in AllReduce). Same pattern_byte() formula used
    // throughout the codebase.
    std::uint8_t mac_pattern_byte(std::size_t i, std::uint32_t seed)
    {
        const std::uint64_t index = static_cast<std::uint64_t>(i);
        const std::uint64_t value = index * 131u + (index >> 8) * 17u + static_cast<std::uint64_t>(seed);
        return static_cast<std::uint8_t>(value & 0xffu);
    }

    std::uint32_t wrapping_add_u8x4(std::uint32_t a, std::uint32_t b)
    {
        return a + b; // unsigned: well-defined modular wraparound
    }

    // Expected post-AllReduce int32 word at element `elem` for a bucket
    // computed with (cuda_seed, rounds) on the CUDA side and (mac_seed)
    // on the Mac side -- reinterprets 4 consecutive expected_bucket_byte()/
    // mac_pattern_byte() outputs as one little-endian-packed uint32 word,
    // matching how the benchmark's own buffers are reinterpreted via
    // reinterpret_cast<std::int32_t*> elsewhere in this file (byte order
    // is whatever the local machine uses on both sides of this function,
    // so it is self-consistent regardless of actual endianness).
    std::int32_t expected_sum_word(
        std::size_t elem, std::uint32_t cuda_seed, int rounds, std::uint32_t mac_seed)
    {
        std::uint32_t cuda_word = 0;
        std::uint32_t mac_word = 0;
        for (int b = 0; b < 4; ++b)
        {
            // expected_bucket_byte() is portable (no CUDA types/build
            // dependency) -- both ranks need the real value here, since
            // the NON-CUDA rank must verify the full reduced result it
            // receives, which includes the CUDA side's real contribution.
            const std::uint8_t cb = tbccl_bench::tensor::expected_bucket_byte(elem * 4 + b, cuda_seed, rounds);
            const std::uint8_t mb = mac_pattern_byte(elem * 4 + b, mac_seed);
            cuda_word |= static_cast<std::uint32_t>(cb) << (8 * b);
            mac_word |= static_cast<std::uint32_t>(mb) << (8 * b);
        }
        return static_cast<std::int32_t>(wrapping_add_u8x4(cuda_word, mac_word));
    }

    struct TimelineEvent
    {
        std::size_t bucket = 0;
        double compute_start_us = 0, compute_end_us = 0;
        double submit_us = 0, complete_us = 0;
    };

#if defined(TBCCL_ENABLE_CUDA)
    void run_calibration(const Options &options)
    {
        auto backend = std::make_unique<tbccl_bench::tensor::CudaChunkedAsyncBackend>();
        backend->allocate(options.bucket_bytes, options.bucket_bytes);
        tbccl_bench::tensor::launch_bucket_fill_input(
            backend->source_device_ptr(), options.bucket_bytes, 1u, nullptr);

        const double target_us = static_cast<double>(options.calibrate_target_us);
        auto measure = [&](int rounds)
        {
            const auto start = Clock::now();
            tbccl_bench::tensor::launch_bucket_compute(
                backend->source_device_ptr(), options.bucket_bytes, 1u, rounds, nullptr);
            backend->verify_source(1u);
            return elapsed_us(start, Clock::now());
        };

        measure(10);
        int rounds = 100;
        double us = measure(rounds);
        if (us > 0) rounds = std::max(1, static_cast<int>(rounds * (target_us / us)));
        us = measure(rounds);
        if (us > 0 && std::abs(us - target_us) / target_us > 0.1)
        {
            rounds = std::max(1, static_cast<int>(rounds * (target_us / us)));
            us = measure(rounds);
        }

        std::cout << "{\n  \"bucket_bytes\": " << options.bucket_bytes
                  << ",\n  \"calibrate_target_us\": " << options.calibrate_target_us
                  << ",\n  \"compute_rounds\": " << rounds
                  << ",\n  \"achieved_us\": " << us << "\n}\n";
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
            throw std::runtime_error("--calibrate requires a CUDA-enabled build");
#endif

        if (options.peers.size() != 2) throw std::runtime_error("--peers must list exactly 2 endpoints");
        if (options.root > 1) throw std::runtime_error("--root must be 0 or 1 (N=2 only)");

        const std::size_t rank = options.rank;
        const std::size_t count = options.bucket_bytes / sizeof(std::int32_t);
        const bool is_root = (rank == options.root);
        const bool is_cuda_rank = (rank == 0); // convention: rank 0 = Linux/CUDA, rank 1 = Mac/Metal
        [[maybe_unused]] const bool do_compute = (options.schedule != "collective-only");
        const bool do_collective = (options.schedule != "compute-only");

        // World: readiness barrier only (no sleep-only
        // coordination), matching tbccl_hetero_allreduce_bench.cpp's
        // established pattern.
        tbccl::TcpWorldOptions world_opts;
        world_opts.rank = rank;
        for (const auto &p : options.peers) world_opts.peers.push_back({p.host, p.port});
        auto world = tbccl::create_tcp_world(world_opts);

        std::unique_ptr<tbccl::Connection> connection;
        std::unique_ptr<tbccl::Listener> listener;
        const auto data_port = static_cast<std::uint16_t>(options.peers[0].port + 1000);
        if (rank == 0) listener = tbccl::tcp_listen(options.peers[0].host, data_port, {});
        tbccl::barrier(*world);
        if (rank == 0) connection = listener->accept();
        else connection = tbccl::tcp_connect(options.peers[0].host, data_port, {});

        const auto local_caps = tbccl::local_capabilities();
        const auto remote_caps = tbccl::exchange_capabilities(*connection, local_caps);
        const auto negotiation = tbccl::negotiate(local_caps, remote_caps);
        if (!negotiation.ok) throw std::runtime_error("capability negotiation failed: " + negotiation.failure_reason);

        tbccl::TcpTransport transport(std::move(connection));
        tbccl::TensorCommWorker comm_worker(/*pipeline_depth=*/2, /*queue_depth=*/options.bucket_count + 1);
        tbccl_bench::BucketAllReduceWorker allreduce_worker(/*queue_depth=*/options.bucket_count + 1);

        const auto chunk_plan = tbccl::plan_chunks(options.bucket_bytes, options.chunk_bytes, 1);
        const std::size_t chunk_count = chunk_plan.empty() ? 1 : chunk_plan.size();

        // ---- Backend setup: one persistent set of buffers per bucket,
        // allocated once before the measured loop. ----
        std::vector<tbccl::AsyncMemoryBackend *> recv_backends(options.bucket_count);
        std::vector<tbccl::AsyncMemoryBackend *> send_backends(options.bucket_count);
        std::vector<tbccl::LocalReduceBackend *> reduce_backends(options.bucket_count, nullptr);

#if defined(TBCCL_ENABLE_CUDA)
        std::vector<std::unique_ptr<tbccl_bench::tensor::CudaChunkedAsyncBackend>> cuda_backends;
        std::vector<std::unique_ptr<tbccl_bench::tensor::CudaReduceBackend>> cuda_reduces;
        void *compute_stream = nullptr;
        void *copy_stream = nullptr;
#endif
        if (options.metal_async_path != "adapter" && options.metal_async_path != "direct")
            throw std::runtime_error("unknown --metal-async-path: " + options.metal_async_path);
        const bool metal_direct = (options.metal_async_path == "direct");

        std::vector<std::unique_ptr<tbccl_bench::tensor::TensorBackend>> metal_backends;
        std::vector<std::unique_ptr<tbccl_bench::tensor::TensorBackendAsyncAdapter>> metal_adapters;
        std::vector<std::unique_ptr<tbccl_bench::tensor::MetalSharedDirectAsyncBackend>> metal_directs;
        std::vector<std::unique_ptr<tbccl_bench::tensor::HostAsyncBackend>> metal_send_wrappers;
        std::vector<std::unique_ptr<tbccl_bench::tensor::HostReduceBackend>> metal_reduces;

        if (is_cuda_rank)
        {
#if defined(TBCCL_ENABLE_CUDA)
            if (cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&compute_stream)) != cudaSuccess)
                throw std::runtime_error("cudaStreamCreate(compute_stream) failed");
            if (cudaStreamCreate(reinterpret_cast<cudaStream_t *>(&copy_stream)) != cudaSuccess)
                throw std::runtime_error("cudaStreamCreate(copy_stream) failed");

            for (std::size_t b = 0; b < options.bucket_count; ++b)
            {
                auto backend = std::make_unique<tbccl_bench::tensor::CudaChunkedAsyncBackend>(copy_stream);
                backend->allocate(options.bucket_bytes, options.chunk_bytes == 0 ? options.bucket_bytes : options.chunk_bytes);
                recv_backends[b] = backend.get();
                send_backends[b] = backend.get();
                if (is_root)
                {
                    cuda_reduces.push_back(
                        std::make_unique<tbccl_bench::tensor::CudaReduceBackend>(*backend, compute_stream));
                    reduce_backends[b] = cuda_reduces.back().get();
                }
                cuda_backends.push_back(std::move(backend));
            }
#else
            throw std::runtime_error("CUDA rank requires a CUDA-enabled build");
#endif
        }
        else
        {
            for (std::size_t b = 0; b < options.bucket_count; ++b)
            {
                auto backend = tbccl_bench::tensor::make_backend(tbccl_bench::tensor::BackendKind::MetalShared);
                backend->allocate(options.bucket_bytes);

                tbccl::AsyncMemoryBackend *metal_async = nullptr;
                if (metal_direct)
                {
                    auto direct = std::make_unique<tbccl_bench::tensor::MetalSharedDirectAsyncBackend>(*backend);
                    metal_async = direct.get();
                    metal_directs.push_back(std::move(direct));
                }
                else
                {
                    auto adapter = std::make_unique<tbccl_bench::tensor::TensorBackendAsyncAdapter>(*backend);
                    metal_async = adapter.get();
                    metal_adapters.push_back(std::move(adapter));
                }
                recv_backends[b] = metal_async;
                if (is_root)
                {
                    metal_send_wrappers.push_back(std::make_unique<tbccl_bench::tensor::HostAsyncBackend>(
                        const_cast<void *>(backend->destination_staging_data()), options.bucket_bytes));
                    send_backends[b] = metal_send_wrappers.back().get();
                    metal_reduces.push_back(std::make_unique<tbccl_bench::tensor::HostReduceBackend>(
                        const_cast<void *>(backend->destination_staging_data()), backend->source_staging_data()));
                    reduce_backends[b] = metal_reduces.back().get();
                }
                else
                {
                    send_backends[b] = metal_async;
                }
                metal_backends.push_back(std::move(backend));
            }
        }

        std::vector<double> compute_only_us, collective_only_us, serial_us, overlap_us;
        std::vector<double> submit_us_samples;
        bool verify_ok = true;
        std::vector<TimelineEvent> last_timeline;
        const bool capture_timeline = !options.timeline_output.empty();

        const std::size_t total_rounds = options.warmup + options.iterations;
        for (std::size_t round = 0; round < total_rounds; ++round)
        {
            const auto iter_start = Clock::now();
            std::vector<TimelineEvent> timeline(options.bucket_count);
            for (std::size_t b = 0; b < options.bucket_count; ++b) timeline[b].bucket = b;

            auto rel_us = [&](Clock::time_point t) { return elapsed_us(iter_start, t); };

            auto compute_bucket = [&](std::size_t b)
            {
                timeline[b].compute_start_us = rel_us(Clock::now());
#if defined(TBCCL_ENABLE_CUDA)
                if (is_cuda_rank && do_compute)
                {
                    const std::uint32_t seed = static_cast<std::uint32_t>(1000 * round + b);
                    tbccl_bench::tensor::launch_bucket_fill_input(
                        cuda_backends[b]->source_device_ptr(), options.bucket_bytes, seed, compute_stream);
                    tbccl_bench::tensor::launch_bucket_compute(
                        cuda_backends[b]->source_device_ptr(), options.bucket_bytes, seed,
                        options.compute_rounds, compute_stream);
                    cudaStreamSynchronize(static_cast<cudaStream_t>(compute_stream));
                }
#endif
                timeline[b].compute_end_us = rel_us(Clock::now());
            };

            // Mac rank's own deterministic local input (prepared before
            // any timed submission, matching the "input prep outside
            // the collective timer" -- here it's outside the compute
            // timer too, since Mac never computes).
            if (!is_cuda_rank)
            {
                for (std::size_t b = 0; b < options.bucket_count; ++b)
                {
                    const std::uint32_t seed = static_cast<std::uint32_t>(1000 * round + b);
                    auto &backend = metal_backends[b];
                    auto *src = static_cast<std::uint8_t *>(
                        const_cast<void *>(backend->source_staging_data()));
                    for (std::size_t i = 0; i < options.bucket_bytes; ++i) src[i] = mac_pattern_byte(i, seed);
                    if (metal_direct) metal_directs[b]->begin_transfer(chunk_count);
                    else metal_adapters[b]->begin_transfer(chunk_count);
                    if (is_root)
                    {
                        // Root's send-back wrapper must also reset per the
                        // async tensor-transfer work convention even
                        // though HostAsyncBackend itself has no
                        // begin_transfer bookkeeping -- N/A here,
                        // intentionally omitted (no-op by design).
                    }
                }
            }

            std::vector<tbccl_bench::BucketAllReduceWork> works;
            works.reserve(options.bucket_count);

            auto submit_bucket = [&](std::size_t b)
            {
                tbccl_bench::BucketAllReduceJob job;
                job.transport = &transport;
                job.worker = &comm_worker;
                job.recv_backend = recv_backends[b];
                job.send_backend = send_backends[b];
                job.reduce_backend = reduce_backends[b];
                job.rank = rank;
                job.root = options.root;
                job.total_bytes = options.bucket_bytes;
                job.chunk_hint = options.chunk_bytes;
                job.count = count;
                job.datatype = tbccl::DataType::Int32;
                job.bucket_index = b;

                const auto submit_start = Clock::now();
                works.push_back(allreduce_worker.enqueue(job));
                const auto submit_end = Clock::now();
                timeline[b].submit_us = rel_us(submit_end);
                if (round >= options.warmup)
                {
                    submit_us_samples.push_back(elapsed_us(submit_start, submit_end));
                }
            };

            const auto t0 = Clock::now();

            if (options.schedule == "compute-only")
            {
                for (std::size_t b = 0; b < options.bucket_count; ++b) compute_bucket(b);
            }
            else if (options.schedule == "collective-only")
            {
                for (std::size_t b = 0; b < options.bucket_count; ++b) submit_bucket(b);
            }
            else if (options.schedule == "serial")
            {
                for (std::size_t b = 0; b < options.bucket_count; ++b) compute_bucket(b);
                for (std::size_t b = 0; b < options.bucket_count; ++b)
                {
                    submit_bucket(b);
                    works.back().wait();
                    if (works.back().has_error())
                        throw std::runtime_error("bucket " + std::to_string(b) + " failed: " + works.back().error());
                    timeline[b].complete_us = rel_us(works.back().completed_at());
                }
            }
            else if (options.schedule == "overlap")
            {
                for (std::size_t b = 0; b < options.bucket_count; ++b)
                {
                    compute_bucket(b);
                    submit_bucket(b);
                }
            }
            else
            {
                throw std::runtime_error("unknown --schedule: " + options.schedule);
            }

            if (do_collective && options.schedule != "serial")
            {
                for (std::size_t b = 0; b < options.bucket_count; ++b)
                {
                    works[b].wait();
                    if (works[b].has_error())
                        throw std::runtime_error("bucket " + std::to_string(b) + " failed: " + works[b].error());
                    timeline[b].complete_us = rel_us(works[b].completed_at());
                }
            }

            const double elapsed = elapsed_us(t0, Clock::now());

            if (round >= options.warmup)
            {
                if (options.schedule == "compute-only") compute_only_us.push_back(elapsed);
                else if (options.schedule == "collective-only") collective_only_us.push_back(elapsed);
                else if (options.schedule == "serial") serial_us.push_back(elapsed);
                else if (options.schedule == "overlap") overlap_us.push_back(elapsed);

                if (capture_timeline) last_timeline = timeline;
            }
        }

        // One untimed, fully-verified round -- always runs
        // compute+collective regardless of --schedule, so correctness is
        // checked for the real end-to-end path.
        {
            std::vector<tbccl_bench::BucketAllReduceWork> works;
            works.reserve(options.bucket_count);
            const std::uint32_t verify_round_seed_base = 999000;

            if (!is_cuda_rank)
            {
                for (std::size_t b = 0; b < options.bucket_count; ++b)
                {
                    const std::uint32_t seed = verify_round_seed_base + static_cast<std::uint32_t>(b);
                    auto &backend = metal_backends[b];
                    auto *src = static_cast<std::uint8_t *>(const_cast<void *>(backend->source_staging_data()));
                    for (std::size_t i = 0; i < options.bucket_bytes; ++i) src[i] = mac_pattern_byte(i, seed);
                    if (metal_direct) metal_directs[b]->begin_transfer(chunk_count);
                    else metal_adapters[b]->begin_transfer(chunk_count);
                }
            }

            for (std::size_t b = 0; b < options.bucket_count; ++b)
            {
#if defined(TBCCL_ENABLE_CUDA)
                if (is_cuda_rank)
                {
                    const std::uint32_t seed = verify_round_seed_base + static_cast<std::uint32_t>(b);
                    tbccl_bench::tensor::launch_bucket_fill_input(
                        cuda_backends[b]->source_device_ptr(), options.bucket_bytes, seed, compute_stream);
                    tbccl_bench::tensor::launch_bucket_compute(
                        cuda_backends[b]->source_device_ptr(), options.bucket_bytes, seed,
                        options.compute_rounds, compute_stream);
                    cudaStreamSynchronize(static_cast<cudaStream_t>(compute_stream));
                }
#endif
                tbccl_bench::BucketAllReduceJob job;
                job.transport = &transport;
                job.worker = &comm_worker;
                job.recv_backend = recv_backends[b];
                job.send_backend = send_backends[b];
                job.reduce_backend = reduce_backends[b];
                job.rank = rank;
                job.root = options.root;
                job.total_bytes = options.bucket_bytes;
                job.chunk_hint = options.chunk_bytes;
                job.count = count;
                job.datatype = tbccl::DataType::Int32;
                job.bucket_index = b;
                works.push_back(allreduce_worker.enqueue(job));
            }

            verify_ok = true;
            for (std::size_t b = 0; b < options.bucket_count; ++b)
            {
                works[b].wait();
                if (works[b].has_error()) { verify_ok = false; continue; }

                std::vector<std::int32_t> result(count);
#if defined(TBCCL_ENABLE_CUDA)
                if (is_cuda_rank)
                {
                    const void *ptr = is_root ? cuda_backends[b]->source_device_ptr()
                                               : cuda_backends[b]->destination_device_ptr();
                    tbccl_bench::tensor::cuda_copy_device_to_host(ptr, result.data(), options.bucket_bytes);
                }
#endif
                if (!is_cuda_rank)
                {
                    const void *ptr = metal_backends[b]->destination_staging_data();
                    std::memcpy(result.data(), ptr, options.bucket_bytes);
                }

                const std::uint32_t cuda_seed = verify_round_seed_base + static_cast<std::uint32_t>(b);
                const std::uint32_t mac_seed = verify_round_seed_base + static_cast<std::uint32_t>(b);
                for (std::size_t i = 0; i < count; ++i)
                {
                    const std::int32_t expected = expected_sum_word(i, cuda_seed, options.compute_rounds, mac_seed);
                    if (result[i] != expected) { verify_ok = false; break; }
                }
            }
        }

        // ---- Output ----
        std::ostringstream json;
        json << "{\n";
        json << "  \"label\": \"" << options.label << "\",\n";
        json << "  \"rank\": " << rank << ",\n";
        json << "  \"root\": " << options.root << ",\n";
        json << "  \"schedule\": \"" << options.schedule << "\",\n";
        json << "  \"metal_async_path\": \"" << options.metal_async_path << "\",\n";
        json << "  \"bucket_bytes\": " << options.bucket_bytes << ",\n";
        json << "  \"bucket_count\": " << options.bucket_count << ",\n";
        json << "  \"compute_rounds\": " << options.compute_rounds << ",\n";
        json << "  \"chunk_bytes\": " << options.chunk_bytes << ",\n";
        json << "  \"warmup\": " << options.warmup << ",\n";
        json << "  \"iterations\": " << options.iterations << ",\n";

        auto emit_stats = [&](const char *name, const std::vector<double> &samples)
        {
            const auto s = compute_stats(samples);
            json << "  \"" << name << "_median_us\": " << s.median_us << ",\n";
            json << "  \"" << name << "_p95_us\": " << s.p95_us << ",\n";
            json << "  \"" << name << "_mean_us\": " << s.mean_us << ",\n";
        };
        if (!compute_only_us.empty()) emit_stats("compute_only", compute_only_us);
        if (!collective_only_us.empty()) emit_stats("collective_only", collective_only_us);
        if (!serial_us.empty()) emit_stats("serial", serial_us);
        if (!overlap_us.empty()) emit_stats("overlap", overlap_us);
        if (!submit_us_samples.empty())
        {
            const auto s = compute_stats(submit_us_samples);
            json << "  \"submit_median_us\": " << s.median_us << ",\n";
            json << "  \"submit_max_us\": " << s.max_us << ",\n";
        }
        json << "  \"verify_ok\": " << (verify_ok ? "true" : "false") << "\n";
        json << "}\n";

        if (!options.output.empty())
        {
            std::ofstream out(options.output);
            out << json.str();
        }
        std::cout << json.str();

        if (capture_timeline && !last_timeline.empty())
        {
            std::ofstream tl(options.timeline_output);
            tl << "bucket,compute_start_us,compute_end_us,submit_us,complete_us\n";
            for (const auto &e : last_timeline)
            {
                tl << e.bucket << "," << e.compute_start_us << "," << e.compute_end_us << ","
                   << e.submit_us << "," << e.complete_us << "\n";
            }
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
