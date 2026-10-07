// The loopback algorithm-sweep work diagnostic tool: runs a handful
// of ring ReduceScatter invocations between two ranks and dumps the
// merged ring trace (calling-thread events + the persistent worker's
// own events) to stdout, sorted by timestamp, so the T0-T4 executor
// handoff and the send/recv/reduce stage breakdown can be inspected
// directly. Only meaningful when built with
// -DTBCCL_ENABLE_RING_TRACE=ON; with tracing compiled out, this
// still runs but prints no events (every ring_trace_* call is a
// no-op). Internal/diagnostic only — not installed, not part of the
// public API.

#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include "reduce_scatter_internal.hpp"
#include "ring_executor.hpp"
#include "ring_trace.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    tbccl::PeerEndpoint parse_peer(const std::string &text)
    {
        const std::size_t colon = text.rfind(':');
        tbccl::PeerEndpoint peer;
        peer.host = text.substr(0, colon);
        peer.port =
            static_cast<std::uint16_t>(std::stoul(text.substr(colon + 1)));
        return peer;
    }

    std::vector<tbccl::PeerEndpoint> parse_peers(const std::string &input)
    {
        std::vector<tbccl::PeerEndpoint> peers;
        std::size_t start = 0;

        while (start < input.size())
        {
            const std::size_t comma = input.find(',', start);
            const std::size_t end =
                comma == std::string::npos ? input.size() : comma;
            peers.push_back(parse_peer(input.substr(start, end - start)));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }

        return peers;
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        std::size_t rank = 0;
        std::string peers_text;
        std::string bind_address;
        std::size_t recv_count = 1024; // elements (float32)
        int iterations = 5;

        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--rank") rank = std::stoul(argv[++i]);
            else if (arg == "--peers") peers_text = argv[++i];
            else if (arg == "--bind") bind_address = argv[++i];
            else if (arg == "--recv-count") recv_count = std::stoul(argv[++i]);
            else if (arg == "--iterations") iterations = std::stoi(argv[++i]);
        }

        tbccl::TcpWorldOptions options;
        options.rank = rank;
        options.peers = parse_peers(peers_text);
        options.bind_address = bind_address;

        auto world = tbccl::create_tcp_world(options);

        std::vector<float> send(world->size() * recv_count, 1.0f);
        std::vector<float> recv(recv_count, 0.0f);

        // Warmup: ensure the executor/worker is already up before the
        // traced iterations, matching normal steady-state benchmark
        // methodology (cold-start is a separate concern, not mixed
        // into these traces).
        tbccl::detail::reduce_scatter_ring(
            *world, send.data(), recv.data(), recv_count,
            tbccl::DataType::Float32, tbccl::ReduceOp::Sum);

        std::vector<tbccl::detail::RingTraceEvent> all_events;

        for (int i = 0; i < iterations; ++i)
        {
            tbccl::detail::reduce_scatter_ring(
                *world, send.data(), recv.data(), recv_count,
                tbccl::DataType::Float32, tbccl::ReduceOp::Sum);

            auto caller_events = tbccl::detail::ring_trace_drain_this_thread();

#if defined(TBCCL_ENABLE_RING_TRACE)
            auto &executor =
                tbccl::detail::RingExecutorAccess::get_or_create(*world);
            auto worker_events = executor.drain_worker_trace();
#else
            std::vector<tbccl::detail::RingTraceEvent> worker_events;
#endif

            all_events.insert(
                all_events.end(), caller_events.begin(), caller_events.end());
            all_events.insert(
                all_events.end(), worker_events.begin(), worker_events.end());
        }

        std::sort(
            all_events.begin(), all_events.end(),
            [](const auto &a, const auto &b)
            {
                if (a.operation_id != b.operation_id)
                    return a.operation_id < b.operation_id;
                return a.timestamp_ns < b.timestamp_ns;
            });

        std::cout
            << "operation_id,rank,role,event,timestamp_ns,message_size\n";

        for (const auto &event : all_events)
        {
            std::cout
                << event.operation_id << ',' << event.rank << ','
                << (event.role == tbccl::detail::RingTraceRole::Sender
                        ? "sender"
                        : "receiver")
                << ',' << tbccl::detail::ring_trace_event_name(event.type)
                << ',' << event.timestamp_ns << ',' << event.message_size
                << '\n';
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
