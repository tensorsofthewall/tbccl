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
        if (off)
        {
            expect(!trace.on() && trace.recorded() == 0, "trace off: nothing recorded");
            std::cout << "latency_trace_test (off): ok\n";
            return 0;
        }
        expect(trace.on() && trace.recorded() > 0, "trace on: events recorded");
        std::map<std::uint64_t, std::map<std::uint32_t, std::int64_t>> ops;
        for (std::size_t i = 0; i < trace.recorded(); ++i)
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
        std::cout << "latency_trace_test: ok\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
