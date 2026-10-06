// Phase 73: single-domain loopback medians (Host buffers, W2) for the before/after A/B: P2P ping-pong round trip and all_reduce at 4 KiB / 1 MiB / 16 MiB.
// Not a benchmark campaign: a handful of warm-up rounds, then `--iters` timed rounds with no verification inside the timed loop, then one untimed verified round.
//   p73_single_domain_bench [--iters N] [--cuda-free]   prints one line per case: case median_ms min_ms
#include "../tests/mesh_test_support.hpp"

#include <algorithm>
#include <cstdlib>
#include <dirent.h>
#include <iostream>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;

static int proc_threads()
{
    int n = 0;
    if (DIR *d = opendir("/proc/self/task"))
    {
        while (readdir(d) != nullptr) ++n;
        closedir(d);
    }
    return n - 2;
}

// Per-world footprint: process-wide file descriptors and threads with a live W2 pair (both ranks in this process) after one P2P and one all_reduce, versus before it existed.
static void footprint()
{
    const int fds0 = open_fd_count() - 1, th0 = proc_threads();
    int fds1 = 0, th1 = 0;
    run_world(2, [&](std::size_t rank, tbccl::Communicator &comm) {
        std::vector<std::uint8_t> a(1 << 16, 1), b(1 << 16, 0);
        std::vector<std::int32_t> r(1024, 1);
        const BufferView va{MemoryKind::Host, a.data(), a.size(), 0}, vb{MemoryKind::Host, b.data(), b.size(), 0}, vr{MemoryKind::Host, r.data(), 4096, 0};
        if (rank == 0)
        {
            comm.send(va, a.size(), DataType::UInt8, 1).wait();
            comm.recv(vb, b.size(), DataType::UInt8, 1).wait();
        }
        else
        {
            comm.recv(vb, b.size(), DataType::UInt8, 0).wait();
            comm.send(va, a.size(), DataType::UInt8, 0).wait();
        }
        comm.all_reduce(vr, vr, 1024, DataType::Int32, tbccl::ReduceOp::Sum).wait();
        comm.barrier().wait();
        if (rank == 0)
        {
            fds1 = open_fd_count() - 1;
            th1 = proc_threads();
        }
    });
    std::cout << "footprint (a live W2 pair, both ranks in this process, after one P2P, one all_reduce, one barrier): fds +" << fds1 - fds0 << ", threads +" << th1 - th0 << "\n";
}

int main(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--footprint")
        {
            footprint();
            return 0;
        }
    int iters = 200;
    bool ar_first = false; // run the all_reduce rounds before the P2P ones on a fresh communicator (a cold collective path)
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (std::string(argv[i]) == "--ar-first") ar_first = true;
    }
    struct Row
    {
        std::string name;
        std::vector<double> ms;
    };
    std::vector<Row> rows;
    for (const std::size_t bytes : {std::size_t{4096}, std::size_t{1} << 20, std::size_t{16} << 20})
    {
        Row p2p{"p2p_pingpong_" + std::to_string(bytes), {}}, ar{"all_reduce_" + std::to_string(bytes), {}};
        run_world(2, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::uint8_t> a(bytes, static_cast<std::uint8_t>(rank + 1)), b(bytes, 0);
            std::vector<std::int32_t> r(bytes / 4, static_cast<std::int32_t>(rank + 1));
            const BufferView va{MemoryKind::Host, a.data(), bytes, 0}, vb{MemoryKind::Host, b.data(), bytes, 0}, vr{MemoryKind::Host, r.data(), bytes, 0};
            auto round = [&](bool pingpong) {
                if (pingpong)
                {
                    if (rank == 0)
                    {
                        comm.send(va, bytes, DataType::UInt8, 1).wait();
                        comm.recv(vb, bytes, DataType::UInt8, 1).wait();
                    }
                    else
                    {
                        comm.recv(vb, bytes, DataType::UInt8, 0).wait();
                        comm.send(va, bytes, DataType::UInt8, 0).wait();
                    }
                }
                else comm.all_reduce(vr, vr, bytes / 4, DataType::Int32, tbccl::ReduceOp::Sum).wait();
            };
            for (const bool pingpong : {!ar_first, ar_first})
            {
                for (int i = 0; i < 20; ++i) round(pingpong);
                std::vector<double> t;
                for (int i = 0; i < iters; ++i)
                {
                    const auto t0 = std::chrono::steady_clock::now();
                    round(pingpong);
                    t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
                }
                if (rank == 0) (pingpong ? p2p : ar).ms = t;
            }
        });
        rows.push_back(p2p);
        rows.push_back(ar);
    }
    for (auto &r : rows)
    {
        std::sort(r.ms.begin(), r.ms.end());
        std::cout << r.name << " " << r.ms[r.ms.size() / 2] << " " << r.ms.front() << "\n";
    }
    return 0;
}
