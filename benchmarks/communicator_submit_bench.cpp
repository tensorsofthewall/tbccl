// Diagnostic (not a pass/fail gate): CALLER-SIDE cost of submitting a send, in nanoseconds per call, through the C++ API and through the C ABI wrapper.
//   idle      the lane is empty and the peer is receiving
//   busy      a large send is active on the lane (the peer receives it) when the timed 4 KiB send is submitted behind it
//   deep      1000 large sends are already queued to a peer that posts nothing (the queue is deep and the head is stalled)   [unbounded admission only: before it the 10th post blocked]
//   stalled   world_size 3: 1000 sends are queued to a stalled peer; the timed sends go to ANOTHER peer that is receiving                 [unbounded admission only]
// Usage: communicator_submit_bench <iterations>. Built with -DTBCCL_BENCH_BASELINE=1 against the earlier library only the first two cases (and no C wrapper) exist.
#include "../tests/mesh_test_support.hpp"

#include <tbccl/communicator.hpp>
#ifndef TBCCL_BENCH_BASELINE
#include <tbccl/tbccl.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace mesh_test;

// A reusable (cyclic) barrier: mesh_test::Latch is single-use.
class Gate
{
public:
    explicit Gate(std::size_t n) : n_(n) {}
    void arrive_and_wait()
    {
        std::unique_lock<std::mutex> lock(m_);
        const std::size_t gen = generation_;
        if (++waiting_ == n_) { waiting_ = 0; ++generation_; cv_.notify_all(); }
        else cv_.wait(lock, [&] { return gen != generation_; });
    }

private:
    std::mutex m_;
    std::condition_variable cv_;
    std::size_t n_, waiting_ = 0, generation_ = 0;
};
using Clock = std::chrono::steady_clock;

namespace
{
double ns_since(Clock::time_point t) { return std::chrono::duration<double, std::nano>(Clock::now() - t).count(); }

void report(const char *api, const char *what, std::vector<double> &v)
{
    std::sort(v.begin(), v.end());
    std::printf("{\"api\":\"%s\",\"case\":\"%s\",\"median_ns\":%.0f,\"p25_ns\":%.0f,\"p75_ns\":%.0f,\"p99_ns\":%.0f,\"n\":%zu}\n", api, what, v[v.size() / 2], v[v.size() / 4],
                v[3 * v.size() / 4], v[std::min(v.size() - 1, v.size() * 99 / 100)], v.size());
    std::fflush(stdout);
}

tbccl::BufferView view(std::uint8_t *p, std::size_t n) { return {tbccl::MemoryKind::Host, p, n, -1}; }

// ---- C++ API -----------------------------------------------------------------------------------------------------------------------------------
struct CppApi
{
    static constexpr const char *name = "cpp";
    tbccl::Communicator &c;
    tbccl::Work send(std::uint8_t *p, std::size_t n, std::size_t peer) { return c.send(view(p, n), n, tbccl::DataType::UInt8, peer); }
    tbccl::Work recv(std::uint8_t *p, std::size_t n, std::size_t peer) { return c.recv(view(p, n), n, tbccl::DataType::UInt8, peer); }
    static void wait(tbccl::Work &w) { w.wait(); }
    void abort() { c.abort("bench done"); }
};

void run_case(const char *which, std::size_t world, int iters, bool deep_allowed);

template <typename Api, typename Make> void scenario_body(const char *which, std::size_t rank, std::size_t world, Api &api, int iters, Gate &gate, std::vector<double> &out)
{
    const std::size_t kSmall = 4096, kBig = std::size_t{64} << 20, kDeepBytes = std::size_t{4} << 20;
    std::vector<std::uint8_t> small(kSmall, 1), big(kBig, 2);
    const std::string w = which;
    if (w == "idle")
    {
        for (int i = 0; i < iters + 20; ++i)
        {
            if (rank == 1)
            {
                auto r = api.recv(small.data(), kSmall, 0);
                gate.arrive_and_wait();
                Api::wait(r);
            }
            else
            {
                gate.arrive_and_wait();
                const auto t = Clock::now();
                auto s = api.send(small.data(), kSmall, 1);
                const double ns = ns_since(t);
                Api::wait(s);
                if (i >= 20) out.push_back(ns);
            }
        }
    }
    else if (w == "busy")
    {
        for (int i = 0; i < iters + 5; ++i)
        {
            if (rank == 1)
            {
                auto r1 = api.recv(big.data(), kBig, 0);
                auto r2 = api.recv(small.data(), kSmall, 0);
                gate.arrive_and_wait();
                Api::wait(r1);
                Api::wait(r2);
            }
            else
            {
                auto s1 = api.send(big.data(), kBig, 1);
                gate.arrive_and_wait();
                const auto t = Clock::now();
                auto s2 = api.send(small.data(), kSmall, 1);
                const double ns = ns_since(t);
                Api::wait(s1);
                Api::wait(s2);
                if (i >= 5) out.push_back(ns);
            }
        }
    }
    else if (w == "deep" || w == "stalled")
    {
        constexpr int kDepth = 1000;
        std::vector<std::vector<std::uint8_t>> payload(kDepth, std::vector<std::uint8_t>(1)); // tiny backing; the stall comes from the peer not receiving, so sizes below the socket buffer would just drain
        (void)payload;
        std::vector<std::uint8_t> head(kDeepBytes, 3);
        if (rank == 0)
        {
            std::vector<decltype(api.send(head.data(), 1, 1))> works;
            works.reserve(kDepth + iters);
            for (int i = 0; i < kDepth; ++i)
            {
                const auto t = Clock::now();
                works.push_back(api.send(head.data(), kDeepBytes, 1)); // peer 1 never receives: its queue only grows
                const double ns = ns_since(t);
                if (w == "deep" && i >= 20) out.push_back(ns);
            }
            if (w == "stalled")
            {
                gate.arrive_and_wait(); // rank 2 is receiving
                for (int i = 0; i < iters; ++i)
                {
                    const auto t = Clock::now();
                    auto s = api.send(small.data(), kSmall, 2);
                    out.push_back(ns_since(t));
                    Api::wait(s);
                }
                gate.arrive_and_wait();
            }
            api.abort(); // fail the stalled queue: bounded, nothing touches `head` afterwards
            for (auto &x : works) Api::wait(x);
        }
        else if (w == "stalled" && rank == 2)
        {
            gate.arrive_and_wait();
            for (int i = 0; i < iters; ++i)
            {
                auto r = api.recv(small.data(), kSmall, 0);
                Api::wait(r);
            }
            gate.arrive_and_wait();
        }
        else if (w == "stalled")
        {
            gate.arrive_and_wait();
            gate.arrive_and_wait();
        }
    }
}

void run_cpp(const char *which, std::size_t world, int iters)
{
    auto results = bootstrap(healthy_slots(world), std::chrono::seconds(20));
    Gate gate(world == 2 ? 2 : 3);
    std::vector<std::vector<double>> out(world);
    std::vector<std::thread> threads;
    for (std::size_t r = 0; r < world; ++r)
        threads.emplace_back([&, r] {
            CppApi api{*results[r].comm};
            scenario_body<CppApi, void>(which, r, world, api, iters, gate, out[r]);
        });
    for (auto &t : threads) t.join();
    report("cpp", which, out[0]);
}

#ifndef TBCCL_BENCH_BASELINE
// ---- C ABI ---------------------------------------------------------------------------------------------------------------------------------------
struct CWork
{
    tbcclWork_t w = nullptr;
};
struct CApi
{
    static constexpr const char *name = "c";
    tbcclComm_t c;
    CWork send(std::uint8_t *p, std::size_t n, std::size_t peer)
    {
        tbcclBuffer b{};
        b.struct_size = sizeof(b);
        b.memory_kind = TBCCL_MEMORY_HOST;
        b.device_ordinal = -1;
        b.data = p;
        b.bytes = n;
        CWork w;
        if (tbcclSend(c, &b, static_cast<uint32_t>(peer), nullptr, &w.w) != TBCCL_SUCCESS) std::abort();
        return w;
    }
    CWork recv(std::uint8_t *p, std::size_t n, std::size_t peer)
    {
        tbcclBuffer b{};
        b.struct_size = sizeof(b);
        b.memory_kind = TBCCL_MEMORY_HOST;
        b.device_ordinal = -1;
        b.data = p;
        b.bytes = n;
        CWork w;
        if (tbcclRecv(c, &b, static_cast<uint32_t>(peer), nullptr, &w.w) != TBCCL_SUCCESS) std::abort();
        return w;
    }
    static void wait(CWork &w)
    {
        tbcclResult_t op;
        tbcclWorkWait(w.w, &op);
        tbcclWorkDestroy(w.w);
        w.w = nullptr;
    }
    void abort() { tbcclCommAbort(c, "bench done"); }
};

void run_c(const char *which, std::size_t world, int iters)
{
    tbcclUniqueId id;
    tbcclGetUniqueId(&id);
    std::vector<tbcclEndpointBlob> blobs(world);
    std::vector<tbcclComm_t> comms(world, nullptr);
    Latch published(world);
    Gate gate(world == 2 ? 2 : 3);
    std::vector<std::vector<double>> out(world);
    std::vector<std::thread> threads;
    for (std::size_t r = 0; r < world; ++r)
        threads.emplace_back([&, r] {
            tbcclBootstrap_t bs = nullptr;
            if (tbcclBootstrapBegin(static_cast<uint32_t>(r), static_cast<uint32_t>(world), &id, nullptr, &bs) != TBCCL_SUCCESS) std::abort();
            blobs[r] = tbcclEndpointBlob{};
            blobs[r].struct_size = sizeof(tbcclEndpointBlob);
            tbcclBootstrapGetEndpoint(bs, &blobs[r]);
            published.arrive_and_wait();
            if (tbcclBootstrapComplete(bs, blobs.data(), static_cast<uint32_t>(world), &comms[r]) != TBCCL_SUCCESS) std::abort();
            tbcclBootstrapDestroy(bs);
            CApi api{comms[r]};
            scenario_body<CApi, void>(which, r, world, api, iters, gate, out[r]);
        });
    for (auto &t : threads) t.join();
    report("c", which, out[0]);
    for (auto c : comms) tbcclCommDestroy(c);
}
#endif
} // namespace

int main(int argc, char **argv)
{
    const int iters = argc > 1 ? std::atoi(argv[1]) : 2000;
    for (const char *which : {"idle", "busy"})
    {
        run_cpp(which, 2, iters);
#ifndef TBCCL_BENCH_BASELINE
        run_c(which, 2, iters);
#endif
    }
#ifndef TBCCL_BENCH_BASELINE
    run_cpp("deep", 2, iters);
    run_c("deep", 2, iters);
    run_cpp("stalled", 3, std::min(iters, 500));
    run_c("stalled", 3, std::min(iters, 500));
#endif
    return 0;
}
