// The smallest C++ program that runs a communicator: two ranks as two threads of one process on loopback, one Float32 all_reduce (Sum).
// Real deployments run one rank per process or machine and publish real endpoints (docs/getting-started/first-program.md).
#include <tbccl/communicator.hpp>
#include <tbccl/version.h>

#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

int main()
{
    constexpr std::uint16_t kPort0 = 29711, kPort1 = 29712; // each rank's data endpoint is its control port + 1000: keep those free too
    float result[2] = {0, 0};
    auto run = [&](std::size_t rank) {
        tbccl::CommunicatorOptions opts;
        opts.rank = rank;
        opts.peers = {{"127.0.0.1", kPort0}, {"127.0.0.1", kPort1}};
        auto comm = tbccl::Communicator::create(opts);
        float value = static_cast<float>(rank + 1);
        tbccl::BufferView buf{tbccl::MemoryKind::Host, &value, sizeof(value), 0};
        auto work = comm->all_reduce(buf, buf, 1, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
        work.wait();
        if (work.has_error()) std::fprintf(stderr, "rank %zu: %s\n", rank, work.error().c_str());
        result[rank] = value;
    };
    std::thread t0(run, 0), t1(run, 1);
    t0.join();
    t1.join();
    std::printf("TBCCL %s: all_reduce(Sum) of 1 and 2 -> %g and %g\n", TBCCL_VERSION_STRING, result[0], result[1]);
    return (result[0] == 3.0f && result[1] == 3.0f) ? 0 : 1;
}
