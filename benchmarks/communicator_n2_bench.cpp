// N=2 Communicator regression micro-benchmark, loopback, Host memory, built from this one source against either the pre-N-rank-runtime or the
// current libtbccl (it only uses the legacy `peers` bootstrap and send/recv/all_reduce). Two threads act as the two ranks. Prints one JSON line
// per case: median / p25 / p75 microseconds over `iters` timed operations after `warmup`. Nothing is verified inside the timed loop.
//   p2p      rank 0 sends `bytes` to rank 1, rank 1 sends them back (one round trip, reported as round-trip time)
//   allreduce  in-place Float32 SUM of `bytes`
//   p2p_x    both ranks post a send and a recv at the same time (round trip); only for payloads that fit the socket buffers (the pre-N-rank-runtime
//            single-FIFO worker deadlocks on larger ones, which is the point of the duplex lanes)
// Usage: communicator_n2_bench <port> <iters> [bytes ...]

#include <tbccl/communicator.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

static tbccl::BufferView view(std::vector<float> &v) { return {tbccl::MemoryKind::Host, v.data(), v.size() * 4, -1}; }

static void report(const char *kind, std::size_t bytes, std::vector<double> &us)
{
    std::sort(us.begin(), us.end());
    std::printf("{\"kind\":\"%s\",\"bytes\":%zu,\"median_us\":%.2f,\"p25_us\":%.2f,\"p75_us\":%.2f,\"n\":%zu}\n", kind, bytes, us[us.size() / 2], us[us.size() / 4],
                us[3 * us.size() / 4], us.size());
}

int main(int argc, char **argv)
{
    const auto port = static_cast<std::uint16_t>(std::atoi(argv[1]));
    const int iters = std::atoi(argv[2]);
    std::vector<std::size_t> sizes;
    for (int i = 3; i < argc; ++i) sizes.push_back(static_cast<std::size_t>(std::atoll(argv[i])));
    const int warmup = std::max(20, iters / 10);

    tbccl::CommunicatorOptions o0;
    o0.rank = 0;
    o0.peers = {{"127.0.0.1", port}, {"127.0.0.1", static_cast<std::uint16_t>(port + 1)}};
    auto o1 = o0;
    o1.rank = 1;

    std::unique_ptr<tbccl::Communicator> c1;
    const auto init_start = Clock::now();
    std::thread t1([&] { c1 = tbccl::Communicator::create(o1); });
    auto c0 = tbccl::Communicator::create(o0);
    t1.join();
    const double init_us = std::chrono::duration<double, std::micro>(Clock::now() - init_start).count();
    std::printf("{\"kind\":\"init\",\"init_us\":%.0f}\n", init_us);

    for (std::size_t bytes : sizes)
    {
        const std::size_t count = bytes / 4;
        std::vector<float> a(count, 1.0f), b(count, 2.0f), a_in(count), b_in(count);
        for (const char *kind : {"p2p", "p2p_x", "allreduce"})
        {
            const bool p2p = kind[0] == 'p';
            const bool both = std::string(kind) == "p2p_x";
            if (both && bytes > (1u << 20)) continue;
            std::vector<double> samples;
            std::thread peer([&] {
                for (int i = 0; i < warmup + iters; ++i)
                {
                    if (p2p)
                    {
                        if (both)
                        {
                            auto r = c1->recv(view(b_in), count, tbccl::DataType::Float32, 0);
                            auto s = c1->send(view(b), count, tbccl::DataType::Float32, 0);
                            r.wait();
                            s.wait();
                        }
                        else
                        {
                            c1->recv(view(b_in), count, tbccl::DataType::Float32, 0).wait();
                            c1->send(view(b_in), count, tbccl::DataType::Float32, 0).wait();
                        }
                    }
                    else
                    {
                        c1->all_reduce(view(b), view(b), count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum).wait();
                    }
                }
            });
            for (int i = 0; i < warmup + iters; ++i)
            {
                const auto t0 = Clock::now();
                if (p2p)
                {
                    if (both)
                    {
                        auto s = c0->send(view(a), count, tbccl::DataType::Float32, 1);
                        auto r = c0->recv(view(a_in), count, tbccl::DataType::Float32, 1);
                        s.wait();
                        r.wait();
                    }
                    else
                    {
                        c0->send(view(a), count, tbccl::DataType::Float32, 1).wait();
                        c0->recv(view(a_in), count, tbccl::DataType::Float32, 1).wait();
                    }
                }
                else
                {
                    c0->all_reduce(view(a), view(a), count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum).wait();
                }
                const auto t1p = Clock::now();
                if (i >= warmup) samples.push_back(std::chrono::duration<double, std::micro>(t1p - t0).count());
                if (!p2p) std::fill(a.begin(), a.end(), 1.0f); // keep values bounded; outside the timed region
            }
            peer.join();
            if (!p2p) std::fill(b.begin(), b.end(), 2.0f);
            report(kind, bytes, samples);
        }
    }
    return 0;
}
