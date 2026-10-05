// Phase 55: the internal per-operation latency trace. With TBCCL_LATENCY_TRACE set before first use, every point-to-point operation records ordered
// events (submit, admission, worker dequeue, terminal, waiter wake, wait return, plus the socket-level sites for send and receive). Run with the
// argument `off` the trace must record nothing (the negative control for the gate: a trace that is not off by default would fail it).

#include <tbccl/communicator.hpp>

#include "../src/transport/latency_trace.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
void expect(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error("assertion failed: " + message);
}
constexpr std::uint16_t kPort = 29960;
} // namespace

int main(int argc, char **argv)
{
    const bool off = argc > 1 && std::string(argv[1]) == "off";
    if (!off) ::setenv("TBCCL_LATENCY_TRACE", "/tmp/tbccl_latency_trace_test", 1);
    else ::unsetenv("TBCCL_LATENCY_TRACE");
    try
    {
        tbccl::CommunicatorOptions o0;
        o0.rank = 0;
        o0.peers = {{"127.0.0.1", kPort}, {"127.0.0.1", static_cast<std::uint16_t>(kPort + 1)}};
        auto o1 = o0;
        o1.rank = 1;
        std::unique_ptr<tbccl::Communicator> c1;
        std::thread t1([&] { c1 = tbccl::Communicator::create(o1); });
        auto c0 = tbccl::Communicator::create(o0);
        t1.join();

        std::vector<std::uint8_t> out(2048, 7), in(2048, 0);
        const tbccl::BufferView sv{tbccl::MemoryKind::Host, out.data(), out.size(), -1};
        const tbccl::BufferView rv{tbccl::MemoryKind::Host, in.data(), in.size(), -1};
        for (int i = 0; i < 5; ++i)
        {
            auto r = c1->recv(rv, in.size(), tbccl::DataType::UInt8, 0);
            auto s = c0->send(sv, out.size(), tbccl::DataType::UInt8, 1);
            s.wait();
            r.wait();
        }
        expect(std::memcmp(out.data(), in.data(), out.size()) == 0, "payload intact with the trace compiled in");

        auto &trace = tbccl::detail::LatencyTrace::get();
        const std::size_t p2p_events = trace.recorded();
        constexpr int kGathers = 3;
        std::vector<std::uint8_t> a0(2048), a1(2048), b0(2048), b1(2048), other(2048, 9);
        for (int i = 0; i < kGathers; ++i)
        {
            const auto view = [](std::vector<std::uint8_t> &v) { return tbccl::BufferView{tbccl::MemoryKind::Host, v.data(), v.size(), -1}; };
            auto w1 = c1->all_gather(view(other), {view(b0), view(b1)});
            auto w0 = c0->all_gather(view(out), {view(a0), view(a1)});
            w0.wait();
            w1.wait();
        }
        if (off)
        {
            expect(!trace.on() && trace.recorded() == 0, "trace off: nothing recorded");
            std::cout << "latency_trace_test (off): ok\n";
            return 0;
        }
        expect(trace.on() && trace.recorded() > 0, "trace on: events recorded");
        std::map<std::uint64_t, std::map<std::uint32_t, std::int64_t>> ops;
        for (std::size_t i = 0; i < p2p_events; ++i)
        {
            const auto &e = trace.event(i);
            ops[e.id][e.site] = e.ns;
        }
        expect(ops.size() == 10, "ten operations traced (5 sends + 5 receives), got " + std::to_string(ops.size()));
        int sends = 0, recvs = 0;
        for (const auto &[id, sites] : ops)
        {
            const auto has = [&](std::uint32_t s) { return sites.count(s) != 0; };
            const auto at = [&](std::uint32_t s) { return sites.at(s); };
            for (std::uint32_t s : {0u, 1u, 2u, 8u, 9u, 10u}) expect(has(s), "op " + std::to_string(id) + " has common site " + std::to_string(s));
            expect(at(0) <= at(1) && at(1) <= at(2) && at(2) <= at(8) && at(8) <= at(9) && at(9) <= at(10), "common sites are ordered");
            if (has(3))
            {
                ++sends;
                expect(has(4) && at(2) <= at(3) && at(3) <= at(4) && at(4) <= at(8), "send sites ordered");
            }
            else
            {
                ++recvs;
                expect(has(5) && has(6) && has(7) && at(2) <= at(5) && at(5) <= at(6) && at(6) <= at(7) && at(7) <= at(8), "receive sites ordered");
            }
        }
        expect(sends == 5 && recvs == 5, "five sends and five receives");
        // Phase 56: the collective executor sites. Each all_gather (2 ranks x 3) is one parent with two child transfers; the parent's events and its
        // children's events must correlate through the aux field and be ordered.
        std::map<std::uint64_t, std::map<std::uint32_t, std::int64_t>> all;
        std::map<std::uint64_t, std::vector<std::uint64_t>> posted, observed;
        for (std::size_t i = p2p_events; i < trace.recorded(); ++i)
        {
            const auto &e = trace.event(i);
            all[e.id][e.site] = e.ns;
            if (e.site == tbccl::detail::kLatCollChildPosted) posted[e.id].push_back(e.aux);
            if (e.site == tbccl::detail::kLatCollChildObserved) observed[e.id].push_back(e.aux);
        }
        int parents = 0;
        for (const auto &[id, sites] : all)
        {
            if (!sites.count(tbccl::detail::kLatCollSubmit)) continue;
            ++parents;
            const auto at = [&](std::uint32_t site) { return sites.at(site); };
            for (std::uint32_t site : {11u, 12u, 13u, 8u, 9u, 10u}) expect(sites.count(site) != 0, "collective " + std::to_string(id) + " has site " + std::to_string(site));
            expect(at(11) <= at(12) && at(11) <= at(13) && at(13) <= at(8) && at(8) <= at(9) && at(9) <= at(10), "collective sites are ordered");
            expect(posted[id].size() == 2 && observed[id] == posted[id], "two children posted and observed in the same order");
            for (std::uint64_t child : posted[id])
            {
                const auto &cs = all.at(child);
                expect(cs.count(0) && cs.count(8), "child " + std::to_string(child) + " has its own submit and terminal events");
                expect(at(13) <= cs.at(0) && cs.at(8) <= at(8), "child lies inside the parent's executor window");
            }
        }
        expect(parents == 2 * kGathers, "six collectives traced, got " + std::to_string(parents));
        std::cout << "latency_trace_test: ok\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
