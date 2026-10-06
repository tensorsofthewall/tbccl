// Phase 73: single-domain loopback medians (Host buffers, W2) for the before/after A/B: P2P ping-pong round trip and all_reduce at 4 KiB / 1 MiB / 16 MiB.
// Not a benchmark campaign: a handful of warm-up rounds, then `--iters` timed rounds with no verification inside the timed loop, then one untimed verified round.
//   p73_single_domain_bench [--iters N] [--cuda-free]   prints one line per case: case median_ms min_ms
#include "../tests/mesh_test_support.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;

int main(int argc, char **argv)
{
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
