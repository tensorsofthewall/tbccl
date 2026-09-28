// Phase 18 standalone CUDA diagnostic: isolates kernel-launch,
// device-execution, and stream-synchronization costs (Part E), and
// investigates the Phase 17 cuda-pinned H2D latency spikes (Part F).
// No TBCCL networking or TensorBackend involvement -- this is a
// direct CUDA-only microbenchmark, intentionally decoupled from
// benchmarks/tensor/cuda_backend.cu so its timing can't be confused
// with (or accidentally perturbed by) TensorBackend's own machinery.
// fill_pattern_kernel is an intentional duplicate of cuda_backend.cu's
// kernel of the same name, for the same reason tensor_backend.hpp's
// pattern_byte() is duplicated there in the first place (Part D).
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

    void print_csv_header()
    {
        std::cout
            << "experiment,kind,size_bytes,run_index,idle_ms,sample_count,"
               "min_us,median_us,p95_us,p99_us,max_us,gt_500us_count,"
               "gt_1000us_count\n";
    }

    void print_stats_row(
        const std::string &experiment,
        const std::string &kind,
        std::size_t size_bytes,
        int run_index,
        int idle_ms,
        const Stats &stats)
    {
        std::cout
            << experiment << ',' << kind << ',' << size_bytes << ',' << run_index
            << ',' << idle_ms << ',' << stats.count << ',' << stats.min_us << ','
            << stats.median_us << ',' << stats.p95_us << ',' << stats.p99_us << ','
            << stats.max_us << ',' << stats.gt_500us << ',' << stats.gt_1000us
            << '\n';

        std::cerr
            << '[' << experiment << '/' << kind << "] size=" << size_bytes
            << " run=" << run_index << " idle_ms=" << idle_ms
            << " median_us=" << stats.median_us << " p99_us=" << stats.p99_us
            << " max_us=" << stats.max_us << " n>500us=" << stats.gt_500us
            << " n>1000us=" << stats.gt_1000us << '\n';
    }

    // -----------------------------------------------------------------------------
    // Part E: kernel launch / device-execution / stream-sync sweep.
    //
    // Every iteration measures FOUR distinct quantities for the same
    // kernel launch, never conflated (Part 19, Part D item 19):
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
    // Part E item 29: controlled idle-interval experiment. Tests the
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
    // Part F: cuda-pageable vs cuda-pinned H2D comparison. Alternates
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

            cudaFreeHost(pinned_host);
            cudaFree(device_dst_pageable);
            cudaFree(device_dst_pinned);
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
            else
            {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }

        // First-use CUDA context initialization happens here, entirely
        // outside every experiment's timed loop (Part 31).
        TBCCL_CUDA_CHECK(cudaSetDevice(0));
        TBCCL_CUDA_CHECK(cudaFree(nullptr)); // forces lazy context init

        print_csv_header();

        const std::vector<std::size_t> kernel_sizes = {
            64, 4096, 65536, 262144, 1048576, 4194304, 16777216};

        run_kernel_sweep(kernel_sizes, warmup, iterations, runs);

        run_idle_interval_experiment(1048576, 30);

        const std::vector<std::size_t> h2d_sizes = {
            64, 256, 4096, 16384, 65536, 262144, 1048576, 4194304};

        run_h2d_comparison(h2d_sizes, warmup, iterations, runs);

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
