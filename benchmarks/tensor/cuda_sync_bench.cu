// the CUDA synchronization-audit work standalone CUDA diagnostic: isolates kernel-launch,
// device-execution, and stream-synchronization costs, and
// investigates the CUDA tensor-benchmark cuda-pinned H2D latency spikes.
// No TBCCL networking or TensorBackend involvement -- this is a
// direct CUDA-only microbenchmark, intentionally decoupled from
// benchmarks/tensor/cuda_backend.cu so its timing can't be confused
// with (or accidentally perturbed by) TensorBackend's own machinery.
// fill_pattern_kernel is an intentional duplicate of cuda_backend.cu's
// kernel of the same name, for the same reason tensor_backend.hpp's
// pattern_byte() is duplicated there in the first place.
//
// Every experiment's setup (context init, cudaMalloc, cudaHostAlloc,
// stream/event creation, warmup) happens outside its timed loop
// (Part 31). Nothing here changes TBCCL's transport, collectives, or
// the tensor backend abstraction; it exists purely to answer Part 18's
// hypotheses about where CUDA time actually goes.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

    void check_cuda(cudaError_t status, const char *what)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("CUDA error in ") + what + ": " +
                cudaGetErrorString(status));
        }
    }

} // namespace

#define TBCCL_CUDA_CHECK(expr) check_cuda((expr), #expr)

namespace
{

    __global__ void fill_pattern_kernel(
        std::uint8_t *data,
        std::size_t count,
        std::uint32_t seed)
    {
        const std::size_t i =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i >= count)
        {
            return;
        }

        const std::uint64_t index = static_cast<std::uint64_t>(i);
        const std::uint64_t value =
            index * 131u + (index >> 8) * 17u + static_cast<std::uint64_t>(seed);

        data[i] = static_cast<std::uint8_t>(value & 0xffu);
    }

    void launch_fill_pattern(
        std::uint8_t *device_data,
        std::size_t count,
        std::uint32_t seed,
        cudaStream_t stream)
    {
        // A 0-block grid launch is an invalid CUDA launch
        // configuration (cudaErrorInvalidConfiguration), not simply a
        // launch that does no work -- so a genuinely empty payload
        // must skip the launch entirely, matching how
        // benchmarks/tensor/cuda_backend.cu guards its own
        // capacity_==0 case. This was previously untested: no the
        // CUDA synchronization-audit work experiment's hardcoded size
        // list ever included 0, so the bug was latent until the CUDA
        // diagnostics work's --sizes override made a 0-byte run
        // possible.
        if (count == 0)
        {
            return;
        }

        constexpr int kThreadsPerBlock = 256;
        const int blocks =
            static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);

        fill_pattern_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            device_data, count, seed);

        TBCCL_CUDA_CHECK(cudaGetLastError());
    }

    // -----------------------------------------------------------------------------
    // Statistics / CSV
    // -----------------------------------------------------------------------------

    struct Stats
    {
        double min_us = 0.0;
        double median_us = 0.0;
        double p95_us = 0.0;
        double p99_us = 0.0;
        double max_us = 0.0;
        std::size_t gt_500us = 0;
        std::size_t gt_1000us = 0;
        std::size_t count = 0;
    };

    Stats compute_stats(std::vector<double> &samples)
    {
        Stats stats;
        stats.count = samples.size();

        if (samples.empty())
        {
            return stats;
        }

        for (double v : samples)
        {
            if (v > 500.0)
            {
                ++stats.gt_500us;
            }
            if (v > 1000.0)
            {
                ++stats.gt_1000us;
            }
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

    double us_between(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point end)
    {
        return std::chrono::duration<double, std::micro>(end - start).count();
    }

    // The calling thread's own CPU consumption (Linux
    // CLOCK_THREAD_CPUTIME_ID), distinct from wall-clock time --
    // distinguishes a thread that consumes CPU while polling from one
    // that spends wall time blocked/descheduled waiting for a
    // completion notification.
    double thread_cpu_time_us()
    {
        struct timespec ts
        {
        };

        if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0)
        {
            return -1.0;
        }

        return static_cast<double>(ts.tv_sec) * 1e6 +
               static_cast<double>(ts.tv_nsec) / 1e3;
    }

    void print_csv_header()
    {
        std::cout
            << "experiment,kind,size_bytes,run_index,idle_ms,sample_count,"
               "min_us,median_us,p95_us,p99_us,max_us,gt_500us_count,"
               "gt_1000us_count,sync_mode,thread_cpu_median_us,"
               "poll_iterations_median,fallback_count\n";
    }

    // The CUDA synchronization-audit work's three experiments (kernel
    // sweep, idle-interval, H2D comparison) call this with no
    // sync-mode-specific data -- they all use blocking
    // cudaStreamSynchronize()/cudaMemcpy(Async)+sync throughout (Part
    // B item 8: their methodology is intentionally left unchanged
    // from CUDA synchronization-audit while reproducing the frozen
    // baseline), so their rows report sync_mode="stream" and "NA" for
    // the CUDA diagnostics-only columns that don't apply to them.
    void print_stats_row(
        const std::string &experiment,
        const std::string &kind,
        std::size_t size_bytes,
        int run_index,
        int idle_ms,
        const Stats &stats,
        const std::string &sync_mode = "stream",
        double thread_cpu_median_us = -1.0,
        double poll_iterations_median = -1.0,
        long fallback_count = -1)
    {
        std::cout
            << experiment << ',' << kind << ',' << size_bytes << ',' << run_index
            << ',' << idle_ms << ',' << stats.count << ',' << stats.min_us << ','
            << stats.median_us << ',' << stats.p95_us << ',' << stats.p99_us << ','
            << stats.max_us << ',' << stats.gt_500us << ',' << stats.gt_1000us
            << ',' << sync_mode << ',';

        if (thread_cpu_median_us < 0.0)
        {
            std::cout << "NA,";
        }
        else
        {
            std::cout << thread_cpu_median_us << ',';
        }

        if (poll_iterations_median < 0.0)
        {
            std::cout << "NA,";
        }
        else
        {
            std::cout << poll_iterations_median << ',';
        }

        if (fallback_count < 0)
        {
            std::cout << "NA\n";
        }
        else
        {
            std::cout << fallback_count << '\n';
        }

        std::cerr
            << '[' << experiment << '/' << kind << '/' << sync_mode
            << "] size=" << size_bytes << " run=" << run_index
            << " idle_ms=" << idle_ms << " median_us=" << stats.median_us
            << " p99_us=" << stats.p99_us << " max_us=" << stats.max_us
            << " n>500us=" << stats.gt_500us << " n>1000us=" << stats.gt_1000us
            << '\n';
    }

    // -----------------------------------------------------------------------------
    // Synchronization-strategy comparison.
    //
    // Four ways to wait for the same GPU work, all given equivalent
    // completion semantics (Part 12): a single stop event is recorded
    // on the stream immediately after the tested operation, and
    // "complete" always means "that event has fired" -- for stream-
    // based modes this is implicit (the stream has drained), for
    // event-based modes it is checked directly. This is fair for a
    // single-operation-per-iteration workload (the only kind this
    // tool ever issues): there is no other preceding stream work an
    // event could ambiguously stand in for.
    // -----------------------------------------------------------------------------

    enum class SyncMode
    {
        Stream,
        Event,
        StreamPoll,
        EventPoll
    };

    std::string sync_mode_name(SyncMode mode)
    {
        switch (mode)
        {
        case SyncMode::Stream:
            return "stream";
        case SyncMode::Event:
            return "event";
        case SyncMode::StreamPoll:
            return "stream-poll";
        case SyncMode::EventPoll:
            return "event-poll";
        }

        throw std::runtime_error("sync_mode_name: unhandled SyncMode");
    }

    SyncMode parse_sync_mode(const std::string &name)
    {
        if (name == "stream")
        {
            return SyncMode::Stream;
        }
        if (name == "event")
        {
            return SyncMode::Event;
        }
        if (name == "stream-poll")
        {
            return SyncMode::StreamPoll;
        }
        if (name == "event-poll")
        {
            return SyncMode::EventPoll;
        }

        throw std::runtime_error("unknown --sync-mode: " + name);
    }

    struct WaitResult
    {
        bool fallback_used = false;
        long poll_iterations = 0;
    };

    // Waits for `stop_event` (already recorded on `stream` by the
    // caller) to complete, using the requested strategy. Polling
    // modes are bounded by `poll_bound` iterations (Part 14: never an
    // infinite busy-spin); if the bound is reached without observing
    // completion, falls back to the equivalent blocking call and
    // records that the fallback fired (Part 14/15).
    WaitResult wait_for_completion(
        SyncMode mode,
        cudaStream_t stream,
        cudaEvent_t stop_event,
        long poll_bound)
    {
        WaitResult result;

        switch (mode)
        {
        case SyncMode::Stream:
            TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));
            return result;

        case SyncMode::Event:
            TBCCL_CUDA_CHECK(cudaEventSynchronize(stop_event));
            return result;

        case SyncMode::StreamPoll:
            for (; result.poll_iterations < poll_bound; ++result.poll_iterations)
            {
                const cudaError_t status = cudaStreamQuery(stream);

                if (status == cudaSuccess)
                {
                    return result;
                }

                if (status != cudaErrorNotReady)
                {
                    TBCCL_CUDA_CHECK(status);
                }
            }

            result.fallback_used = true;
            TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));
            return result;

        case SyncMode::EventPoll:
            for (; result.poll_iterations < poll_bound; ++result.poll_iterations)
            {
                const cudaError_t status = cudaEventQuery(stop_event);

                if (status == cudaSuccess)
                {
                    return result;
                }

                if (status != cudaErrorNotReady)
                {
                    TBCCL_CUDA_CHECK(status);
                }
            }

            result.fallback_used = true;
            TBCCL_CUDA_CHECK(cudaEventSynchronize(stop_event));
            return result;
        }

        throw std::runtime_error("wait_for_completion: unhandled SyncMode");
    }

    // Runs one kernel-launch workload under every one of the four
    // synchronization strategies, at each requested size, alternating
    // mode order across runs (item 16) so no mode is systematically
    // favored/disfavored by warm-up drift or thermal/scheduling state
    // that changes over the course of the whole sweep.
    void run_sync_mode_comparison(
        const std::vector<std::size_t> &sizes,
        int warmup,
        int iterations,
        int runs,
        long poll_bound,
        const std::vector<SyncMode> &modes)
    {
        cudaStream_t stream;
        TBCCL_CUDA_CHECK(cudaStreamCreate(&stream));

        cudaEvent_t start_event;
        cudaEvent_t stop_event;
        TBCCL_CUDA_CHECK(cudaEventCreate(&start_event));
        TBCCL_CUDA_CHECK(cudaEventCreate(&stop_event));

        for (std::size_t size : sizes)
        {
            std::uint8_t *device_buffer = nullptr;
            TBCCL_CUDA_CHECK(cudaMalloc(&device_buffer, size));

            // Warm up every mode once before any measured run, so a
            // cold-start cost never lands specifically on whichever
            // mode happens to run first.
            for (SyncMode mode : modes)
            {
                for (int i = 0; i < warmup; ++i)
                {
                    launch_fill_pattern(
                        device_buffer, size, static_cast<std::uint32_t>(i), stream);
                    TBCCL_CUDA_CHECK(cudaEventRecord(stop_event, stream));
                    wait_for_completion(mode, stream, stop_event, poll_bound);
                }
            }

            for (int run = 1; run <= runs; ++run)
            {
                // Alternates mode order across runs (item 16): run 1
                // is `modes` as given, run 2 reverses it, and so on.
                std::vector<SyncMode> order = modes;

                if (run % 2 == 0)
                {
                    std::reverse(order.begin(), order.end());
                }

                for (SyncMode mode : order)
                {
                    std::vector<double> enqueue_samples;
                    std::vector<double> sync_samples;
                    std::vector<double> cpu_total_samples;
                    std::vector<double> event_samples;
                    std::vector<double> thread_cpu_samples;
                    std::vector<double> poll_iteration_samples;
                    long fallback_count = 0;

                    enqueue_samples.reserve(static_cast<std::size_t>(iterations));
                    sync_samples.reserve(static_cast<std::size_t>(iterations));
                    cpu_total_samples.reserve(static_cast<std::size_t>(iterations));
                    event_samples.reserve(static_cast<std::size_t>(iterations));
                    thread_cpu_samples.reserve(static_cast<std::size_t>(iterations));
                    poll_iteration_samples.reserve(static_cast<std::size_t>(iterations));

                    for (int i = 0; i < iterations; ++i)
                    {
                        const std::uint32_t seed = static_cast<std::uint32_t>(0x6000 + i);

                        const double thread_cpu_before = thread_cpu_time_us();
                        const auto t0 = std::chrono::steady_clock::now();

                        TBCCL_CUDA_CHECK(cudaEventRecord(start_event, stream));
                        launch_fill_pattern(device_buffer, size, seed, stream);
                        TBCCL_CUDA_CHECK(cudaEventRecord(stop_event, stream));

                        const auto t1 = std::chrono::steady_clock::now();

                        const WaitResult wait_result =
                            wait_for_completion(mode, stream, stop_event, poll_bound);

                        const auto t2 = std::chrono::steady_clock::now();
                        const double thread_cpu_after = thread_cpu_time_us();

                        float event_ms = 0.0f;
                        TBCCL_CUDA_CHECK(
                            cudaEventElapsedTime(&event_ms, start_event, stop_event));

                        enqueue_samples.push_back(us_between(t0, t1));
                        sync_samples.push_back(us_between(t1, t2));
                        cpu_total_samples.push_back(us_between(t0, t2));
                        event_samples.push_back(static_cast<double>(event_ms) * 1000.0);
                        thread_cpu_samples.push_back(thread_cpu_after - thread_cpu_before);
                        poll_iteration_samples.push_back(
                            static_cast<double>(wait_result.poll_iterations));

                        if (wait_result.fallback_used)
                        {
                            ++fallback_count;
                        }
                    }

                    // Part 49: the selected synchronization strategy
                    // must not report completion before the GPU output
                    // is actually ready -- verified here by reading
                    // back the device buffer (never inferred solely
                    // from a cudaSuccess return code) against the last
                    // iteration's known seed. wait_for_completion()
                    // already guaranteed completion for that iteration,
                    // so this read is untimed and does not perturb the
                    // measurements above.
                    if (iterations > 0)
                    {
                        const std::uint32_t last_seed =
                            static_cast<std::uint32_t>(0x6000 + (iterations - 1));

                        std::vector<std::uint8_t> readback(size);
                        TBCCL_CUDA_CHECK(cudaMemcpy(
                            readback.data(), device_buffer, size,
                            cudaMemcpyDeviceToHost));

                        for (std::size_t i = 0; i < size; ++i)
                        {
                            const std::uint64_t index = static_cast<std::uint64_t>(i);
                            const std::uint64_t value =
                                index * 131u + (index >> 8) * 17u +
                                static_cast<std::uint64_t>(last_seed);
                            const std::uint8_t expected =
                                static_cast<std::uint8_t>(value & 0xffu);

                            if (readback[i] != expected)
                            {
                                throw std::runtime_error(
                                    "sync_mode correctness check failed: mode=" +
                                    sync_mode_name(mode) +
                                    " size=" + std::to_string(size) +
                                    " byte=" + std::to_string(i));
                            }
                        }
                    }

                    const std::string mode_name = sync_mode_name(mode);
                    const Stats cpu_total_stats = compute_stats(cpu_total_samples);
                    const Stats thread_cpu_stats = compute_stats(thread_cpu_samples);
                    const Stats poll_stats = compute_stats(poll_iteration_samples);

                    print_stats_row(
                        "sync_mode", "enqueue", size, run, 0,
                        compute_stats(enqueue_samples), mode_name);
                    print_stats_row(
                        "sync_mode", "wait", size, run, 0,
                        compute_stats(sync_samples), mode_name);
                    print_stats_row(
                        "sync_mode", "cpu_total", size, run, 0, cpu_total_stats,
                        mode_name, thread_cpu_stats.median_us, poll_stats.median_us,
                        fallback_count);
                    print_stats_row(
                        "sync_mode", "event_duration", size, run, 0,
                        compute_stats(event_samples), mode_name);
                }
            }

            cudaFree(device_buffer);
        }

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaStreamDestroy(stream);
    }

    // -----------------------------------------------------------------------------
    // Kernel launch / device-execution / stream-sync sweep.
    //
    // Every iteration measures FOUR distinct quantities for the same
    // kernel launch, never conflated (Part 19):
    //   enqueue_us     -- CPU time to submit the launch + event records
    //                     (should be small; async launch is expected
    //                     to return quickly regardless of payload size)
    //   sync_us        -- CPU time blocked in cudaStreamSynchronize()
    //                     after the enqueue returns
    //   cpu_total_us   -- enqueue_us + sync_us, i.e. the CPU-observed
    //                     enqueue-to-completion duration (this is what
    //                     TensorBackend::prepare_source() effectively
    //                     measures end-to-end as source_sync_us, since
    //                     that field's own timer starts before
    //                     prepare_source() is called and stops after
    //                     -- see the Final Report's timing audit)
    //   event_us       -- CUDA-event device-reported execution
    //                     duration (start/stop events bracketing only
    //                     the kernel launch itself, not the sync call)
    // -----------------------------------------------------------------------------

    void run_kernel_sweep(
        const std::vector<std::size_t> &sizes,
        int warmup,
        int iterations,
        int runs)
    {
        cudaStream_t stream;
        TBCCL_CUDA_CHECK(cudaStreamCreate(&stream));

        cudaEvent_t start_event;
        cudaEvent_t stop_event;
        TBCCL_CUDA_CHECK(cudaEventCreate(&start_event));
        TBCCL_CUDA_CHECK(cudaEventCreate(&stop_event));

        for (std::size_t size : sizes)
        {
            std::uint8_t *device_buffer = nullptr;
            TBCCL_CUDA_CHECK(cudaMalloc(&device_buffer, size));

            // Warmup: excludes first-touch/JIT costs from steady state
            // (Part 31).
            for (int i = 0; i < warmup; ++i)
            {
                launch_fill_pattern(
                    device_buffer, size, static_cast<std::uint32_t>(i), stream);
                TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            for (int run = 1; run <= runs; ++run)
            {
                std::vector<double> enqueue_samples;
                std::vector<double> sync_samples;
                std::vector<double> cpu_total_samples;
                std::vector<double> event_samples;

                enqueue_samples.reserve(static_cast<std::size_t>(iterations));
                sync_samples.reserve(static_cast<std::size_t>(iterations));
                cpu_total_samples.reserve(static_cast<std::size_t>(iterations));
                event_samples.reserve(static_cast<std::size_t>(iterations));

                for (int i = 0; i < iterations; ++i)
                {
                    const std::uint32_t seed = static_cast<std::uint32_t>(0x4000 + i);

                    const auto t0 = std::chrono::steady_clock::now();

                    TBCCL_CUDA_CHECK(cudaEventRecord(start_event, stream));
                    launch_fill_pattern(device_buffer, size, seed, stream);
                    TBCCL_CUDA_CHECK(cudaEventRecord(stop_event, stream));

                    const auto t1 = std::chrono::steady_clock::now();

                    TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));

                    const auto t2 = std::chrono::steady_clock::now();

                    float event_ms = 0.0f;
                    TBCCL_CUDA_CHECK(
                        cudaEventElapsedTime(&event_ms, start_event, stop_event));

                    enqueue_samples.push_back(us_between(t0, t1));
                    sync_samples.push_back(us_between(t1, t2));
                    cpu_total_samples.push_back(us_between(t0, t2));
                    event_samples.push_back(static_cast<double>(event_ms) * 1000.0);
                }

                print_stats_row(
                    "kernel", "enqueue", size, run, 0,
                    compute_stats(enqueue_samples));
                print_stats_row(
                    "kernel", "stream_sync", size, run, 0,
                    compute_stats(sync_samples));
                print_stats_row(
                    "kernel", "cpu_total", size, run, 0,
                    compute_stats(cpu_total_samples));
                print_stats_row(
                    "kernel", "event_duration", size, run, 0,
                    compute_stats(event_samples));
            }

            cudaFree(device_buffer);
        }

        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
        cudaStreamDestroy(stream);
    }

    // -----------------------------------------------------------------------------
    // Controlled idle-interval experiment. Tests the
    // "GPU power-state transition" hypothesis directly by measuring
    // CPU-observed enqueue-to-completion latency after a deliberate
    // idle gap of varying length, at one representative size.
    // -----------------------------------------------------------------------------

    void run_idle_interval_experiment(std::size_t size, int samples_per_level)
    {
        cudaStream_t stream;
        TBCCL_CUDA_CHECK(cudaStreamCreate(&stream));

        std::uint8_t *device_buffer = nullptr;
        TBCCL_CUDA_CHECK(cudaMalloc(&device_buffer, size));

        // Warmup.
        for (int i = 0; i < 20; ++i)
        {
            launch_fill_pattern(
                device_buffer, size, static_cast<std::uint32_t>(i), stream);
            TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        const int idle_levels_ms[] = {0, 10, 100, 500};

        for (int idle_ms : idle_levels_ms)
        {
            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(samples_per_level));

            for (int i = 0; i < samples_per_level; ++i)
            {
                if (idle_ms > 0)
                {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(idle_ms));
                }

                const std::uint32_t seed = static_cast<std::uint32_t>(0x5000 + i);

                const auto t0 = std::chrono::steady_clock::now();
                launch_fill_pattern(device_buffer, size, seed, stream);
                TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto t1 = std::chrono::steady_clock::now();

                samples.push_back(us_between(t0, t1));
            }

            print_stats_row(
                "idle_interval", "cpu_total", size, 1, idle_ms,
                compute_stats(samples));
        }

        cudaFree(device_buffer);
        cudaStreamDestroy(stream);
    }

    // -----------------------------------------------------------------------------
    // Cuda-pageable vs cuda-pinned H2D comparison. Alternates
    // pageable/pinned across runs (item 35) rather than running one
    // kind entirely before the other, and retains every raw sample
    // (via compute_stats' gt_500us/gt_1000us counters and max_us) so a
    // sporadic spike is never averaged away (item 34).
    // -----------------------------------------------------------------------------

    void run_h2d_comparison(
        const std::vector<std::size_t> &sizes,
        int warmup,
        int iterations,
        int runs)
    {
        for (std::size_t size : sizes)
        {
            std::uint8_t *device_src = nullptr;
            TBCCL_CUDA_CHECK(cudaMalloc(&device_src, size));

            cudaStream_t stream;
            TBCCL_CUDA_CHECK(cudaStreamCreate(&stream));
            launch_fill_pattern(device_src, size, 0xABCDu, stream);
            TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));

            std::vector<std::uint8_t> pageable_host(size);

            void *pinned_host = nullptr;
            TBCCL_CUDA_CHECK(cudaHostAlloc(&pinned_host, size, cudaHostAllocDefault));

            // Real (non-zero) deterministic source content, so the
            // correctness check below the run loop is meaningful -- a
            // copy of all-zero bytes could silently "pass" even if
            // the copy never actually ran.
            for (std::size_t i = 0; i < size; ++i)
            {
                const std::uint64_t index = static_cast<std::uint64_t>(i);
                const std::uint64_t value = index * 131u + (index >> 8) * 17u + 0x1234u;
                const std::uint8_t byte = static_cast<std::uint8_t>(value & 0xffu);
                pageable_host[i] = byte;
                static_cast<std::uint8_t *>(pinned_host)[i] = byte;
            }

            std::uint8_t *device_dst_pageable = nullptr;
            std::uint8_t *device_dst_pinned = nullptr;
            TBCCL_CUDA_CHECK(cudaMalloc(&device_dst_pageable, size));
            TBCCL_CUDA_CHECK(cudaMalloc(&device_dst_pinned, size));

            // Warmup both paths.
            for (int i = 0; i < warmup; ++i)
            {
                TBCCL_CUDA_CHECK(cudaMemcpy(
                    device_dst_pageable, pageable_host.data(), size,
                    cudaMemcpyHostToDevice));
                TBCCL_CUDA_CHECK(cudaMemcpyAsync(
                    device_dst_pinned, pinned_host, size, cudaMemcpyHostToDevice,
                    stream));
                TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            // Alternate pageable/pinned across runs (item 35).
            for (int run = 1; run <= runs; ++run)
            {
                std::vector<double> pageable_samples;
                std::vector<double> pinned_enqueue_samples;
                std::vector<double> pinned_sync_samples;
                std::vector<double> pinned_total_samples;
                pageable_samples.reserve(static_cast<std::size_t>(iterations));
                pinned_enqueue_samples.reserve(static_cast<std::size_t>(iterations));
                pinned_sync_samples.reserve(static_cast<std::size_t>(iterations));
                pinned_total_samples.reserve(static_cast<std::size_t>(iterations));

                const bool pageable_first = (run % 2) == 1;

                auto measure_pageable = [&]()
                {
                    for (int i = 0; i < iterations; ++i)
                    {
                        const auto t0 = std::chrono::steady_clock::now();
                        TBCCL_CUDA_CHECK(cudaMemcpy(
                            device_dst_pageable, pageable_host.data(), size,
                            cudaMemcpyHostToDevice));
                        const auto t1 = std::chrono::steady_clock::now();
                        pageable_samples.push_back(us_between(t0, t1));
                    }
                };

                auto measure_pinned = [&]()
                {
                    for (int i = 0; i < iterations; ++i)
                    {
                        const auto t0 = std::chrono::steady_clock::now();
                        TBCCL_CUDA_CHECK(cudaMemcpyAsync(
                            device_dst_pinned, pinned_host, size,
                            cudaMemcpyHostToDevice, stream));
                        const auto t1 = std::chrono::steady_clock::now();
                        TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));
                        const auto t2 = std::chrono::steady_clock::now();

                        pinned_enqueue_samples.push_back(us_between(t0, t1));
                        pinned_sync_samples.push_back(us_between(t1, t2));
                        pinned_total_samples.push_back(us_between(t0, t2));
                    }
                };

                if (pageable_first)
                {
                    measure_pageable();
                    measure_pinned();
                }
                else
                {
                    measure_pinned();
                    measure_pageable();
                }

                print_stats_row(
                    "h2d", "pageable_total", size, run, 0,
                    compute_stats(pageable_samples));
                print_stats_row(
                    "h2d", "pinned_enqueue", size, run, 0,
                    compute_stats(pinned_enqueue_samples));
                print_stats_row(
                    "h2d", "pinned_stream_sync", size, run, 0,
                    compute_stats(pinned_sync_samples));
                print_stats_row(
                    "h2d", "pinned_total", size, run, 0,
                    compute_stats(pinned_total_samples));
            }

            // Verify correctness once per size, after (never during)
            // the timed loop.
            {
                std::vector<std::uint8_t> readback_pageable(size);
                std::vector<std::uint8_t> readback_pinned(size);

                TBCCL_CUDA_CHECK(cudaMemcpy(
                    readback_pageable.data(), device_dst_pageable, size,
                    cudaMemcpyDeviceToHost));
                TBCCL_CUDA_CHECK(cudaMemcpy(
                    readback_pinned.data(), device_dst_pinned, size,
                    cudaMemcpyDeviceToHost));

                bool pageable_ok = true;
                bool pinned_ok = true;

                for (std::size_t i = 0; i < size; ++i)
                {
                    const std::uint64_t index = static_cast<std::uint64_t>(i);
                    const std::uint64_t value =
                        index * 131u + (index >> 8) * 17u + 0x1234u;
                    const std::uint8_t expected = static_cast<std::uint8_t>(value & 0xffu);

                    if (readback_pageable[i] != expected)
                    {
                        pageable_ok = false;
                    }
                    if (readback_pinned[i] != expected)
                    {
                        pinned_ok = false;
                    }
                }

                if (!pageable_ok || !pinned_ok)
                {
                    throw std::runtime_error(
                        "h2d correctness check failed at size " +
                        std::to_string(size) +
                        " (pageable_ok=" + std::to_string(pageable_ok) +
                        " pinned_ok=" + std::to_string(pinned_ok) + ")");
                }
            }

            cudaFreeHost(pinned_host);
            cudaFree(device_dst_pageable);
            cudaFree(device_dst_pinned);
            cudaFree(device_src);
            cudaStreamDestroy(stream);
        }
    }

    // -----------------------------------------------------------------------------
    // D2H control. Same structure as run_h2d_comparison
    // (alternating pageable/pinned order across runs), transfer
    // direction reversed, at a smaller matching size subset -- exists
    // to answer one question: is the below-threshold latency spike
    // specific to H2D, or does the same pattern appear for D2H too?
    // -----------------------------------------------------------------------------

    void run_d2h_comparison(
        const std::vector<std::size_t> &sizes,
        int warmup,
        int iterations,
        int runs)
    {
        for (std::size_t size : sizes)
        {
            std::uint8_t *device_src = nullptr;
            TBCCL_CUDA_CHECK(cudaMalloc(&device_src, size));

            cudaStream_t stream;
            TBCCL_CUDA_CHECK(cudaStreamCreate(&stream));
            launch_fill_pattern(device_src, size, 0xABCDu, stream);
            TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));

            std::vector<std::uint8_t> pageable_host(size);

            void *pinned_host = nullptr;
            TBCCL_CUDA_CHECK(cudaHostAlloc(&pinned_host, size, cudaHostAllocDefault));

            // Warmup both paths.
            for (int i = 0; i < warmup; ++i)
            {
                TBCCL_CUDA_CHECK(cudaMemcpy(
                    pageable_host.data(), device_src, size, cudaMemcpyDeviceToHost));
                TBCCL_CUDA_CHECK(cudaMemcpyAsync(
                    pinned_host, device_src, size, cudaMemcpyDeviceToHost, stream));
                TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));
            }

            for (int run = 1; run <= runs; ++run)
            {
                std::vector<double> pageable_samples;
                std::vector<double> pinned_enqueue_samples;
                std::vector<double> pinned_sync_samples;
                std::vector<double> pinned_total_samples;
                pageable_samples.reserve(static_cast<std::size_t>(iterations));
                pinned_enqueue_samples.reserve(static_cast<std::size_t>(iterations));
                pinned_sync_samples.reserve(static_cast<std::size_t>(iterations));
                pinned_total_samples.reserve(static_cast<std::size_t>(iterations));

                const bool pageable_first = (run % 2) == 1;

                auto measure_pageable = [&]()
                {
                    for (int i = 0; i < iterations; ++i)
                    {
                        const auto t0 = std::chrono::steady_clock::now();
                        TBCCL_CUDA_CHECK(cudaMemcpy(
                            pageable_host.data(), device_src, size,
                            cudaMemcpyDeviceToHost));
                        const auto t1 = std::chrono::steady_clock::now();
                        pageable_samples.push_back(us_between(t0, t1));
                    }
                };

                auto measure_pinned = [&]()
                {
                    for (int i = 0; i < iterations; ++i)
                    {
                        const auto t0 = std::chrono::steady_clock::now();
                        TBCCL_CUDA_CHECK(cudaMemcpyAsync(
                            pinned_host, device_src, size, cudaMemcpyDeviceToHost,
                            stream));
                        const auto t1 = std::chrono::steady_clock::now();
                        TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream));
                        const auto t2 = std::chrono::steady_clock::now();

                        pinned_enqueue_samples.push_back(us_between(t0, t1));
                        pinned_sync_samples.push_back(us_between(t1, t2));
                        pinned_total_samples.push_back(us_between(t0, t2));
                    }
                };

                if (pageable_first)
                {
                    measure_pageable();
                    measure_pinned();
                }
                else
                {
                    measure_pinned();
                    measure_pageable();
                }

                print_stats_row(
                    "d2h", "pageable_total", size, run, 0,
                    compute_stats(pageable_samples));
                print_stats_row(
                    "d2h", "pinned_enqueue", size, run, 0,
                    compute_stats(pinned_enqueue_samples));
                print_stats_row(
                    "d2h", "pinned_stream_sync", size, run, 0,
                    compute_stats(pinned_sync_samples));
                print_stats_row(
                    "d2h", "pinned_total", size, run, 0,
                    compute_stats(pinned_total_samples));
            }

            // Verify correctness once per size, after (never during)
            // the timed loop -- both destinations must still hold the
            // source's actual pattern.
            bool pageable_ok = true;
            bool pinned_ok = true;

            for (std::size_t i = 0; i < size; ++i)
            {
                const std::uint64_t index = static_cast<std::uint64_t>(i);
                const std::uint64_t value = index * 131u + (index >> 8) * 17u + 0xABCDu;
                const std::uint8_t expected = static_cast<std::uint8_t>(value & 0xffu);

                if (pageable_host[i] != expected)
                {
                    pageable_ok = false;
                }
                if (static_cast<std::uint8_t *>(pinned_host)[i] != expected)
                {
                    pinned_ok = false;
                }
            }

            if (!pageable_ok || !pinned_ok)
            {
                throw std::runtime_error(
                    "d2h correctness check failed at size " + std::to_string(size) +
                    " (pageable_ok=" + std::to_string(pageable_ok) +
                    " pinned_ok=" + std::to_string(pinned_ok) + ")");
            }

            cudaFreeHost(pinned_host);
            cudaFree(device_src);
            cudaStreamDestroy(stream);
        }
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        int warmup = 50;
        int iterations = 500;
        int runs = 5;
        long poll_bound = 200000;
        std::string experiment = "all"; // kernel|idle|h2d|d2h|h2d-dense|sync-mode|all
        std::string sync_mode_filter = "all"; // stream|event|stream-poll|event-poll|all
        std::string device_schedule; // empty = do not call cudaSetDeviceFlags at all
        // Overrides the relevant hardcoded size list for a single,
        // non-"all" --experiment -- lets a test exercise sizes (0,
        // odd byte counts) the built-in sweeps don't cover, without a
        // second size-list surface per experiment.
        std::vector<std::size_t> sizes_override;

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

            if (argument == "--warmup")
            {
                warmup = std::stoi(require_value(argument));
            }
            else if (argument == "--iterations")
            {
                iterations = std::stoi(require_value(argument));
            }
            else if (argument == "--runs")
            {
                runs = std::stoi(require_value(argument));
            }
            else if (argument == "--poll-bound")
            {
                poll_bound = std::stol(require_value(argument));
            }
            else if (argument == "--experiment")
            {
                experiment = require_value(argument);
            }
            else if (argument == "--sync-mode")
            {
                sync_mode_filter = require_value(argument);
            }
            else if (argument == "--device-schedule")
            {
                device_schedule = require_value(argument);
            }
            else if (argument == "--sizes")
            {
                const std::string value = require_value(argument);
                std::size_t start = 0;

                while (start < value.size())
                {
                    const std::size_t comma = value.find(',', start);
                    const std::size_t end =
                        comma == std::string::npos ? value.size() : comma;

                    sizes_override.push_back(
                        static_cast<std::size_t>(
                            std::stoul(value.substr(start, end - start))));

                    if (comma == std::string::npos)
                    {
                        break;
                    }

                    start = comma + 1;
                }
            }
            else
            {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }

        if (!sizes_override.empty() && experiment == "all")
        {
            throw std::runtime_error(
                "--sizes requires a specific --experiment (not \"all\"), "
                "since each experiment has its own natural size range");
        }

        // CudaSetDeviceFlags() must be called before any CUDA API
        // call that establishes the device context -- this is
        // therefore the very first CUDA call in the process, ahead of
        // even cudaSetDevice(). Opt-in only: with no
        // --device-schedule, this is skipped entirely, preserving the
        // CUDA runtime's own default (which is itself
        // cudaDeviceScheduleAuto, so skipping vs. explicitly passing
        // "auto" should be equivalent -- but only the "skip" path was
        // the CUDA synchronization-audit work's actual, unmodified
        // behavior, so it remains the default here too).
        if (!device_schedule.empty())
        {
            unsigned int flags = cudaDeviceScheduleAuto;

            if (device_schedule == "auto")
            {
                flags = cudaDeviceScheduleAuto;
            }
            else if (device_schedule == "spin")
            {
                flags = cudaDeviceScheduleSpin;
            }
            else if (device_schedule == "blocking-sync")
            {
                flags = cudaDeviceScheduleBlockingSync;
            }
            else
            {
                throw std::runtime_error(
                    "unknown --device-schedule: " + device_schedule);
            }

            TBCCL_CUDA_CHECK(cudaSetDeviceFlags(flags));
        }

        // First-use CUDA context initialization happens here, entirely
        // outside every experiment's timed loop (Part 31).
        TBCCL_CUDA_CHECK(cudaSetDevice(0));
        TBCCL_CUDA_CHECK(cudaFree(nullptr)); // forces lazy context init

        print_csv_header();

        if (experiment == "kernel" || experiment == "all")
        {
            const std::vector<std::size_t> kernel_sizes =
                sizes_override.empty()
                    ? std::vector<std::size_t>{64, 4096, 65536, 262144, 1048576,
                                                4194304, 16777216}
                    : sizes_override;

            run_kernel_sweep(kernel_sizes, warmup, iterations, runs);
        }

        if (experiment == "idle" || experiment == "all")
        {
            run_idle_interval_experiment(1048576, 30);
        }

        if (experiment == "h2d" || experiment == "all")
        {
            const std::vector<std::size_t> h2d_sizes =
                sizes_override.empty()
                    ? std::vector<std::size_t>{64, 256, 4096, 16384, 65536,
                                                262144, 1048576, 4194304}
                    : sizes_override;

            run_h2d_comparison(h2d_sizes, warmup, iterations, runs);
        }

        // A denser sweep specifically bracketing the two approximate
        // crossover regions the CUDA synchronization-audit work
        // observed (~64KiB for pinned, ~256KiB for pageable) -- the
        // union of the plan's two suggested size lists, so both
        // thresholds are bracketed finely in one run using the same
        // (pageable, pinned) comparison at every size.
        if (experiment == "h2d-dense" || experiment == "all")
        {
            const std::vector<std::size_t> dense_sizes = {
                16384, 24576, 32768, 40960, 49152, 57344, 61440, 64512, 65536,
                66560, 73728, 98304, 131072, 163840, 196608, 229376, 245760,
                261120, 262144, 263168, 278528, 327680, 524288};

            run_h2d_comparison(dense_sizes, warmup, iterations, runs);
        }

        // D2H control, at a smaller matching subset.
        if (experiment == "d2h" || experiment == "all")
        {
            const std::vector<std::size_t> d2h_sizes =
                sizes_override.empty()
                    ? std::vector<std::size_t>{4096, 16384, 65536, 131072,
                                                262144, 1048576}
                    : sizes_override;

            run_d2h_comparison(d2h_sizes, warmup, iterations, runs);
        }

        if (experiment == "sync-mode" || experiment == "all")
        {
            std::vector<SyncMode> modes;

            if (sync_mode_filter == "all")
            {
                modes = {SyncMode::Stream, SyncMode::Event, SyncMode::StreamPoll,
                         SyncMode::EventPoll};
            }
            else
            {
                modes = {parse_sync_mode(sync_mode_filter)};
            }

            const std::vector<std::size_t> sync_mode_sizes =
                sizes_override.empty()
                    ? std::vector<std::size_t>{64, 4096, 65536, 262144, 1048576,
                                                4194304}
                    : sizes_override;

            run_sync_mode_comparison(
                sync_mode_sizes, warmup, iterations, runs, poll_bound, modes);
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
