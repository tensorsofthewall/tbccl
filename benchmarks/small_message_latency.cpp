// process-per-rank small-message latency benchmark for the N=2 Communicator (Host memory), usable on loopback and across two hosts.
//
//   small_message_latency --rank R --peers <ip0:port0,ip1:port1> --modes p2p,oneway,simul,allgather,allreduce,chain
//                         [--sizes 64,256,...] [--iters N] [--warmup W] [--cpu CORE[,CORE...]] [--gap-us N] [--label NAME]
//
// Rank R listens on peers[R] (legacy two-rank bootstrap); each rank is its own process. Rank 0 prints one JSON line per (mode, size): median / p25 /
// p75 / p95 microseconds over the timed iterations, plus process CPU use (user + system seconds / wall seconds, i.e. busy cores) for the timed loop.
// Nothing is verified inside a timed loop; each (mode, size) ends with one untimed, verified round.
//   p2p        ping-pong round trip (rank 0 send+wait, recv+wait; rank 1 the mirror image)
//   oneway     a stream of back-to-back sends in blocks of 32 (all posted, then waited); time per message as seen by the receiver per block
//   simul      both ranks post a send and a recv at once, then wait (round trip)
//   allgather  N=2 all_gather of `bytes` per rank
//   allreduce  in-place Float32 SUM
//   chain      the latency-attribution decode chain: rank 0 send+wait then all_gather; rank 1 recv+wait then all_gather (one iteration)
//   replay     replay a recorded cadence (--profile FILE, this rank's ops: kind, bytes, idle gap before the op; --passes N). A profile comes from
//              tools/cadence_profile.py (a real decode's communication calls); each rank replays its own file, so the waits for the peer's compute
//              reproduce themselves. Reports the per-pass wall time and the communication overhead per step ((wall - sum of gaps) / steps), medians.
// Internal experiment switches are environment variables read by libtbccl (TBCCL_LATENCY_TRACE, TBCCL_DIAG_*), see docs/progress_model.md.

#include <tbccl/communicator.hpp>

#include <sys/resource.h>
#include <sys/time.h>

#ifdef __linux__
#include <sched.h>
#endif

#include <algorithm>
#include <chrono>
#include <fstream>
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

struct ProfileOp
{
    std::string op;
    std::size_t bytes = 0;
    long gap_us = 0;
};

// Reads the fixed-shape JSON written by tools/cadence_profile.py: {"rank":R,"ops":[{"op":"send","bytes":2048,"gap_us":4300},...]}.
std::vector<ProfileOp> read_profile(const std::string &path)
{
    std::ifstream in(path);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<ProfileOp> ops;
    for (std::size_t at = text.find("{\"op\":\""); at != std::string::npos; at = text.find("{\"op\":\"", at + 1))
    {
        ProfileOp p;
        const std::size_t name = at + 7, end = text.find('"', name);
        p.op = text.substr(name, end - name);
        p.bytes = std::strtoull(text.c_str() + text.find("\"bytes\":", end) + 8, nullptr, 10);
        p.gap_us = std::atol(text.c_str() + text.find("\"gap_us\":", end) + 9);
        ops.push_back(p);
    }
    return ops;
}

} // namespace

int main(int argc, char **argv)
{
    std::size_t rank = 0;
    std::string peers_arg, modes_arg = "p2p", sizes_arg = "64,256,1024,2048,4096,8192,16384,65536", label = "run";
    int iters = 2000, warmup = 200;
    std::vector<int> cpus;
    long gap_us = 0;
    std::string profile_path;
    int passes = 20;
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
        else if (a == "--cpu")
            for (const auto &c : split(next(), ',')) cpus.push_back(std::atoi(c.c_str()));
        else if (a == "--label") label = next();
        else if (a == "--gap-us") gap_us = std::atol(next().c_str());
        else if (a == "--profile") profile_path = next();
        else if (a == "--passes") passes = std::atoi(next().c_str());
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
    if (!cpus.empty())
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c : cpus) CPU_SET(c, &set);
        sched_setaffinity(0, sizeof(set), &set); // threads created later (the communicator's workers) inherit the mask
    }
#endif
    auto comm = tbccl::Communicator::create(o);
    const std::size_t peer = 1 - rank;
    const bool leader = rank == 0;
    constexpr int kBlock = 32;

    if (!profile_path.empty())
    {
        const auto ops = read_profile(profile_path);
        if (ops.empty())
        {
            std::fprintf(stderr, "empty or unreadable profile %s\n", profile_path.c_str());
            return 2;
        }
        std::size_t max_bytes = 1, steps = 0;
        long gap_total = 0;
        for (const auto &p : ops)
        {
            max_bytes = std::max(max_bytes, p.bytes);
            gap_total += p.gap_us;
            if (p.op == "all_gather") ++steps;
        }
        if (steps == 0) steps = ops.size();
        std::vector<std::uint8_t> sbuf(max_bytes, 1), rbuf(max_bytes), g0(max_bytes), g1(max_bytes);
        double slept_us = 0; // the sleeps' actual length (their overshoot is the OS timer, not the communication library, so it is not overhead)
        const auto play = [&](const ProfileOp &p) {
            if (p.gap_us > 0)
            {
                const auto s0 = Clock::now();
                std::this_thread::sleep_for(std::chrono::microseconds(p.gap_us));
                slept_us += std::chrono::duration<double, std::micro>(Clock::now() - s0).count();
            }
            const auto sv = tbccl::BufferView{tbccl::MemoryKind::Host, sbuf.data(), p.bytes, -1};
            const auto rv = tbccl::BufferView{tbccl::MemoryKind::Host, rbuf.data(), p.bytes, -1};
            if (p.op == "send") comm->send(sv, p.bytes, tbccl::DataType::UInt8, peer).wait();
            else if (p.op == "recv") comm->recv(rv, p.bytes, tbccl::DataType::UInt8, peer).wait();
            else if (p.op == "all_gather")
                comm->all_gather(sv, {tbccl::BufferView{tbccl::MemoryKind::Host, g0.data(), p.bytes, -1}, tbccl::BufferView{tbccl::MemoryKind::Host, g1.data(), p.bytes, -1}}).wait();
            else if (p.op == "barrier") comm->barrier().wait();
        };
        comm->barrier().wait();
        for (int w = 0; w < std::max(1, passes / 10); ++w)
            for (const auto &p : ops) play(p);
        comm->barrier().wait();
        std::vector<double> wall_us, overhead_us;
        const double cpu0 = cpu_seconds();
        const auto run0 = Clock::now();
        for (int i = 0; i < passes; ++i)
        {
            const double slept0 = slept_us;
            const auto t0 = Clock::now();
            for (const auto &p : ops) play(p);
            const double w = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
            wall_us.push_back(w);
            overhead_us.push_back((w - (slept_us - slept0)) / static_cast<double>(steps));
        }
        const double cores = (cpu_seconds() - cpu0) / std::max(std::chrono::duration<double>(Clock::now() - run0).count(), 1e-9);
        comm->barrier().wait();
        std::sort(wall_us.begin(), wall_us.end());
        std::sort(overhead_us.begin(), overhead_us.end());
        const auto mid = [](const std::vector<double> &v) { return v[v.size() / 2]; };
        if (leader || std::getenv("TBCCL_BENCH_BOTH"))
            std::printf("{\"label\":\"%s%s\",\"mode\":\"replay\",\"ops\":%zu,\"steps\":%zu,\"gap_total_us\":%ld,\"pass_wall_median_us\":%.1f,\"overhead_per_step_median_us\":%.1f,"
                        "\"overhead_per_step_p25_us\":%.1f,\"overhead_per_step_p75_us\":%.1f,\"passes\":%d,\"cpu_cores\":%.3f}\n",
                        label.c_str(), leader ? "" : "/r1", ops.size(), steps, gap_total, mid(wall_us), mid(overhead_us),
                        overhead_us[overhead_us.size() / 4], overhead_us[overhead_us.size() * 3 / 4], passes, cores);
        std::fflush(stdout);
        return 0;
    }

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
                // Optional idle gap between operations (the decode cadence: a rank computes for milliseconds between messages), NOT included in the sample.
                if (gap_us > 0) std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
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
