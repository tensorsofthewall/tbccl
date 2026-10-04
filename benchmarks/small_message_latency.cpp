// Phase 55: process-per-rank small-message latency benchmark for the N=2 Communicator (Host memory), usable on loopback and across two hosts.
//
//   small_message_latency --rank R --peers <ip0:port0,ip1:port1> --modes p2p,oneway,simul,allgather,allreduce,chain
//                         [--sizes 64,256,...] [--iters N] [--warmup W] [--cpu CORE] [--label NAME]
//
// Rank R listens on peers[R] (legacy two-rank bootstrap); each rank is its own process. Rank 0 prints one JSON line per (mode, size): median / p25 /
// p75 / p95 microseconds over the timed iterations, plus process CPU use (user + system seconds / wall seconds, i.e. busy cores) for the timed loop.
// Nothing is verified inside a timed loop; each (mode, size) ends with one untimed, verified round.
//   p2p        ping-pong round trip (rank 0 send+wait, recv+wait; rank 1 the mirror image)
//   oneway     a stream of back-to-back sends in blocks of 32 (all posted, then waited); time per message as seen by the receiver per block
//   simul      both ranks post a send and a recv at once, then wait (round trip)
//   allgather  N=2 all_gather of `bytes` per rank
//   allreduce  in-place Float32 SUM
//   chain      the Phase 54 decode chain: rank 0 send+wait then all_gather; rank 1 recv+wait then all_gather (one iteration)
// Internal experiment switches are environment variables read by libtbccl (TBCCL_LATENCY_TRACE, TBCCL_DIAG_*), see docs/progress_model.md.

#include <tbccl/communicator.hpp>

#include <sys/resource.h>
#include <sys/time.h>

#ifdef __linux__
#include <sched.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

namespace
{

std::vector<std::string> split(const std::string &s, char sep)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
        if (!item.empty()) out.push_back(item);
    return out;
}

double cpu_seconds()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
}

tbccl::BufferView view(std::vector<std::uint8_t> &v) { return {tbccl::MemoryKind::Host, v.data(), v.size(), -1}; }

void report(const std::string &label, const char *mode, std::size_t bytes, std::vector<double> &us, double cpu_cores)
{
    std::sort(us.begin(), us.end());
    const auto at = [&](double q) { return us[std::min(us.size() - 1, static_cast<std::size_t>(q * us.size()))]; };
    std::printf(
        "{\"label\":\"%s\",\"mode\":\"%s\",\"bytes\":%zu,\"median_us\":%.2f,\"p25_us\":%.2f,\"p75_us\":%.2f,\"p95_us\":%.2f,\"n\":%zu,\"cpu_cores\":%.2f}\n",
        label.c_str(), mode, bytes, at(0.5), at(0.25), at(0.75), at(0.95), us.size(), cpu_cores);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv)
{
    std::size_t rank = 0;
    std::string peers_arg, modes_arg = "p2p", sizes_arg = "64,256,1024,2048,4096,8192,16384,65536", label = "run";
    int iters = 2000, warmup = 200, cpu = -1;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--rank") rank = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--peers") peers_arg = next();
        else if (a == "--modes") modes_arg = next();
        else if (a == "--sizes") sizes_arg = next();
        else if (a == "--iters") iters = std::atoi(next().c_str());
        else if (a == "--warmup") warmup = std::atoi(next().c_str());
        else if (a == "--cpu") cpu = std::atoi(next().c_str());
        else if (a == "--label") label = next();
    }
    tbccl::CommunicatorOptions o;
    o.rank = rank;
    for (const auto &p : split(peers_arg, ','))
    {
        const auto colon = p.rfind(':');
        o.peers.push_back({p.substr(0, colon), static_cast<std::uint16_t>(std::atoi(p.c_str() + colon + 1))});
    }
    if (o.peers.size() != 2)
    {
        std::fprintf(stderr, "need --peers ip0:port0,ip1:port1\n");
        return 2;
    }
    o.bootstrap_timeout = std::chrono::milliseconds(60000);
#ifdef __linux__
    if (cpu >= 0)
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
#endif
    auto comm = tbccl::Communicator::create(o);
    const std::size_t peer = 1 - rank;
    const bool leader = rank == 0;
    constexpr int kBlock = 32;

    for (const auto &mode : split(modes_arg, ','))
    {
        for (const auto &size_text : split(sizes_arg, ','))
        {
            const std::size_t bytes = std::strtoull(size_text.c_str(), nullptr, 10);
            const std::size_t elems = std::max<std::size_t>(bytes / 4, 1);
            if (mode == "allreduce" && bytes < 4) continue;
            std::vector<std::uint8_t> a(bytes, 1), b(bytes, 2), a_in(bytes), b_in(bytes), g0(bytes), g1(bytes);
            std::vector<float> f(elems, 1.0f);
            const auto fview = tbccl::BufferView{tbccl::MemoryKind::Host, f.data(), elems * 4, -1};
            const auto one = [&](int iteration) {
                if (mode == "p2p")
                {
                    if (leader)
                    {
                        comm->send(view(a), bytes, tbccl::DataType::UInt8, peer).wait();
                        comm->recv(view(a_in), bytes, tbccl::DataType::UInt8, peer).wait();
                    }
                    else
                    {
                        comm->recv(view(b_in), bytes, tbccl::DataType::UInt8, peer).wait();
                        comm->send(view(b_in), bytes, tbccl::DataType::UInt8, peer).wait();
                    }
                }
                else if (mode == "simul")
                {
                    auto &out = leader ? a : b;
                    auto &in = leader ? a_in : b_in;
                    auto s = comm->send(view(out), bytes, tbccl::DataType::UInt8, peer);
                    auto r = comm->recv(view(in), bytes, tbccl::DataType::UInt8, peer);
                    s.wait();
                    r.wait();
                }
                else if (mode == "allgather")
                {
                    auto &in = leader ? a : b;
                    comm->all_gather(view(in), {view(g0), view(g1)}).wait();
                }
                else if (mode == "allreduce")
                {
                    comm->all_reduce(fview, fview, elems, tbccl::DataType::Float32, tbccl::ReduceOp::Sum).wait();
                }
                else if (mode == "chain")
                {
                    if (leader)
                    {
                        comm->send(view(a), bytes, tbccl::DataType::UInt8, peer).wait();
                        comm->all_gather(view(a), {view(g0), view(g1)}).wait();
                    }
                    else
                    {
                        comm->recv(view(b_in), bytes, tbccl::DataType::UInt8, peer).wait();
                        comm->all_gather(view(b_in), {view(g0), view(g1)}).wait();
                    }
                }
                (void)iteration;
            };
            const auto block = [&]() { // oneway: kBlock messages from rank 0 to rank 1 posted back to back
                std::vector<tbccl::Work> works;
                works.reserve(kBlock);
                for (int k = 0; k < kBlock; ++k)
                    works.push_back(leader ? comm->send(view(a), bytes, tbccl::DataType::UInt8, peer) : comm->recv(view(b_in), bytes, tbccl::DataType::UInt8, peer));
                for (auto &w : works) w.wait();
            };
            comm->barrier().wait();
            const bool streamed = mode == "oneway";
            const int loops = streamed ? std::max(1, iters / kBlock) : iters;
            const int warm = streamed ? std::max(1, warmup / kBlock) : warmup;
            for (int i = 0; i < warm; ++i) streamed ? block() : one(i);
            comm->barrier().wait();
            std::vector<double> samples;
            samples.reserve(loops);
            const double cpu0 = cpu_seconds();
            const auto wall0 = Clock::now();
            for (int i = 0; i < loops; ++i)
            {
                const auto t0 = Clock::now();
                streamed ? block() : one(i);
                const auto t1 = Clock::now();
                samples.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / (streamed ? kBlock : 1));
            }
            const double wall = std::chrono::duration<double>(Clock::now() - wall0).count();
            const double cores = (cpu_seconds() - cpu0) / std::max(wall, 1e-9);
            comm->barrier().wait();
            // one untimed, verified round for the point-to-point modes
            if (mode == "p2p" && bytes > 0)
            {
                std::fill(a.begin(), a.end(), 0x5a);
                std::fill(a_in.begin(), a_in.end(), 0);
                if (leader)
                {
                    comm->send(view(a), bytes, tbccl::DataType::UInt8, peer).wait();
                    comm->recv(view(a_in), bytes, tbccl::DataType::UInt8, peer).wait();
                    if (std::memcmp(a.data(), a_in.data(), bytes) != 0)
                    {
                        std::fprintf(stderr, "VERIFY FAILED %s %zu\n", mode.c_str(), bytes);
                        return 3;
                    }
                }
                else
                {
                    comm->recv(view(b_in), bytes, tbccl::DataType::UInt8, peer).wait();
                    comm->send(view(b_in), bytes, tbccl::DataType::UInt8, peer).wait();
                }
            }
            if (leader) report(label, mode.c_str(), bytes, samples, cores);
            else if (std::getenv("TBCCL_BENCH_BOTH")) report(label + "/r1", mode.c_str(), bytes, samples, cores);
        }
    }
    comm->barrier().wait();
    return 0;
}
