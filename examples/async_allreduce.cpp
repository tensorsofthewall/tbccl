// The standalone reference example for the public
// TBCCL runtime API. Uses ONLY public, installed headers
// (<tbccl/communicator.hpp> and friends) -- no benchmark headers, no
// internal TBCCL classes. This is also the vehicle for the
// central end-to-end proof: a real N=2 heterogeneous SUM AllReduce
// (CUDA on Linux <-> Metal-shared on Mac) over the real Thunderbolt
// link, reached only through Communicator::all_reduce(), reusing the
// exact, unmodified n2_all_reduce_tensor() engine the
// benchmarks already use.
//
// Build with TBCCL installed (see docs/public_api.md) or from this
// source tree directly:
//   --backend host         : portable, no device required.
//   --backend cuda          : requires a CUDA device (Linux) and the
//                             optional CUDA device component.
//   --backend metal-shared   : requires a Metal device (macOS).
//
// Usage (two ranks, two processes):
//   async_allreduce --rank 0 --peers HOST0:PORT0,HOST1:PORT1 --backend cuda
//   async_allreduce --rank 1 --peers HOST0:PORT0,HOST1:PORT1 --backend metal-shared

#include <tbccl/communicator.hpp>

#if defined(TBCCL_EXAMPLE_ENABLE_CUDA)
#include <tbccl/cuda_support.hpp>
#include <cuda_runtime.h>
#endif

#if defined(TBCCL_EXAMPLE_ENABLE_METAL)
#import <Metal/Metal.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
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
    if (colon == std::string::npos) throw std::runtime_error("invalid peer endpoint: " + text);
    return {text.substr(0, colon), static_cast<std::uint16_t>(std::stoul(text.substr(colon + 1)))};
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
    std::string backend = "host"; // host, cuda, metal-shared
    std::size_t count = 1 << 20;  // float32 elements, ~4MiB
    std::size_t rounds = 1;       // >1 reuses the same Communicator
                                   // and buffer for a steady-state timing
                                   // comparison against the old benchmark path
                                   // (round 1 is a cold first-call, not
                                   // representative --.md).
};

Options parse_args(int argc, char **argv)
{
    Options o;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--rank") o.rank = std::stoul(next());
        else if (arg == "--peers") o.peers = parse_peers(next());
        else if (arg == "--backend") o.backend = next();
        else if (arg == "--count") o.count = std::stoull(next());
        else if (arg == "--rounds") o.rounds = std::stoull(next());
        else throw std::runtime_error("unknown argument: " + arg);
    }
    return o;
}

// Represents "unrelated work" an application does while the collective
// is in flight (Part 1's do_other_work()) -- a trivial CPU spin here,
// standing in for whatever real computation a caller would overlap.
void do_other_work()
{
    volatile std::uint64_t acc = 0;
    for (std::uint64_t i = 0; i < 50'000'000ull; ++i) acc += i;
    (void)acc;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_args(argc, argv);
        if (options.peers.size() != 2) throw std::runtime_error("--peers must list exactly 2 endpoints");

#if defined(TBCCL_EXAMPLE_ENABLE_CUDA)
        if (options.backend == "cuda") tbccl::register_cuda_support();
#endif

        tbccl::CommunicatorOptions comm_opts;
        comm_opts.rank = options.rank;
        for (const auto &p : options.peers) comm_opts.peers.push_back({p.host, p.port});

        auto comm = tbccl::Communicator::create(comm_opts);
        std::cout << "[async_allreduce] rank=" << comm->rank() << " world_size=" << comm->world_size()
                  << " backend=" << options.backend << " count=" << options.count << "\n";

        const std::size_t bytes = options.count * sizeof(float);
        const float local_value = static_cast<float>(comm->rank()) + 1.0f;

        std::optional<tbccl::Work> work;
        bool verify_ok = false;
        auto submit_time = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point complete_time;

        if (options.backend == "host")
        {
            std::vector<float> data(options.count);
            tbccl::BufferView view{tbccl::MemoryKind::Host, data.data(), bytes, 0};
            for (std::size_t round = 0; round < options.rounds; ++round)
            {
                data.assign(options.count, local_value);
                const auto round_start = std::chrono::steady_clock::now();
                work = comm->all_reduce(view, view, options.count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                if (round == 0) do_other_work(); // overlapped with the in-flight collective
                work->wait();
                const auto round_end = std::chrono::steady_clock::now();
                if (options.rounds > 1)
                {
                    std::cout << "[async_allreduce] rank=" << comm->rank() << " round=" << round
                              << " round_us=" << std::chrono::duration<double, std::micro>(round_end - round_start).count() << "\n";
                }
                if (round == 0) submit_time = round_start;
            }
            complete_time = std::chrono::steady_clock::now();
            verify_ok = !work->has_error() && data[0] == 3.0f; // rank0(1.0) + rank1(2.0)
        }
#if defined(TBCCL_EXAMPLE_ENABLE_CUDA)
        else if (options.backend == "cuda")
        {
            void *device_ptr = nullptr;
            cudaMalloc(&device_ptr, bytes);
            std::vector<float> host_init(options.count, local_value);
            cudaMemcpy(device_ptr, host_init.data(), bytes, cudaMemcpyHostToDevice);
            tbccl::BufferView view{tbccl::MemoryKind::Cuda, device_ptr, bytes, 0};

            for (std::size_t round = 0; round < options.rounds; ++round)
            {
                const auto round_start = std::chrono::steady_clock::now();
                work = comm->all_reduce(view, view, options.count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                if (round == 0) do_other_work(); // demonstrate overlap on the first round only
                work->wait();
                const auto round_end = std::chrono::steady_clock::now();
                std::cout << "[async_allreduce] rank=" << comm->rank() << " round=" << round
                          << " round_us=" << std::chrono::duration<double, std::micro>(round_end - round_start).count() << "\n";
                if (round == 0) { submit_time = round_start; }
                // Re-seed local input before the NEXT round only (not
                // after the last one) -- all_reduce left the reduced
                // result in device_ptr, and the final verification below
                // must check that actual last-round result, not a
                // re-seeded value.
                if (round + 1 < options.rounds)
                {
                    cudaMemcpy(device_ptr, host_init.data(), bytes, cudaMemcpyHostToDevice);
                }
            }

            std::vector<float> result(options.count);
            cudaMemcpy(result.data(), device_ptr, bytes, cudaMemcpyDeviceToHost);
            complete_time = std::chrono::steady_clock::now();
            verify_ok = !work->has_error() && result[0] == 3.0f; // rank0(1.0) + rank1(2.0)
            cudaFree(device_ptr);
        }
#endif
#if defined(TBCCL_EXAMPLE_ENABLE_METAL)
        else if (options.backend == "metal-shared")
        {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("no Metal device available");
            id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            auto *floats = static_cast<float *>(buffer.contents);
            tbccl::BufferView view{tbccl::MemoryKind::MetalShared, buffer.contents, bytes, 0};

            for (std::size_t round = 0; round < options.rounds; ++round)
            {
                for (std::size_t i = 0; i < options.count; ++i) floats[i] = local_value;
                const auto round_start = std::chrono::steady_clock::now();
                work = comm->all_reduce(view, view, options.count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
                if (round == 0) do_other_work();
                work->wait();
                const auto round_end = std::chrono::steady_clock::now();
                if (options.rounds > 1)
                {
                    std::cout << "[async_allreduce] rank=" << comm->rank() << " round=" << round
                              << " round_us=" << std::chrono::duration<double, std::micro>(round_end - round_start).count() << "\n";
                }
                if (round == 0) submit_time = round_start;
            }
            complete_time = std::chrono::steady_clock::now();
            verify_ok = !work->has_error() && floats[0] == 3.0f;
        }
#endif
        else
        {
            throw std::runtime_error("unsupported or not-compiled-in --backend: " + options.backend);
        }

        const double elapsed_us = std::chrono::duration<double, std::micro>(complete_time - submit_time).count();
        std::cout << "[async_allreduce] rank=" << comm->rank()
                  << " has_error=" << (work ? work->has_error() : true)
                  << " verify_ok=" << verify_ok
                  << " elapsed_us=" << elapsed_us << "\n";

        // comm destructor (end of scope) performs clean shutdown: stops
        // workers, joins threads, releases the transport -- never
        // touches the caller's external buffer.
        return verify_ok ? 0 : 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[async_allreduce] error: " << e.what() << "\n";
        return 1;
    }
}
