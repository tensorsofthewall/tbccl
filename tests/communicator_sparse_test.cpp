// Phase 51: lazy data channels. For world_size > 2 the data plane starts EMPTY (the control plane is the only full mesh) and a data connection is made the
// first time something needs it. Checked through the private diagnostic debug_connected_data_peers().

#include "mesh_test_support.hpp"

#include "communicator_debug.hpp"

#include <algorithm>
#include <iostream>
#include <set>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;

namespace
{
    BufferView view(std::vector<std::uint8_t> &v) { return BufferView{MemoryKind::Host, v.data(), v.size(), 0}; }

    std::set<std::size_t> peers(const tbccl::Communicator &c)
    {
        auto v = tbccl::detail::debug_connected_data_peers(c);
        return std::set<std::size_t>(v.begin(), v.end());
    }

    void test_no_data_connections_after_init(std::size_t world)
    {
        run_world(world, [world](std::size_t, tbccl::Communicator &comm) {
            expect(peers(comm).empty(), "no data connection exists right after initialization at world_size " + std::to_string(world));
            comm.barrier().wait(); // descriptors and verdicts travel on the CONTROL plane
            expect(peers(comm).empty(), "a barrier does not open any data connection");
        });
        std::cout << "[PASS] world_size=" << world << ": zero data connections after init and after a barrier\n";
    }

    // A reference broadcast from rank 0: rank 0 needs an edge to every rank, every other rank only to rank 0.
    void test_edges_follow_the_algorithm(std::size_t world)
    {
        run_world(world, [world](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::uint8_t> buf(1000, rank == 0 ? 7 : 0);
            auto w = comm.broadcast(view(buf), 0);
            w.wait();
            expect(!w.has_error() && buf[999] == 7, "broadcast");
            const auto p = peers(comm);
            if (rank == 0)
            {
                expect(p.size() == world - 1, "rank 0 connected to every rank");
            }
            else
            {
                expect(p == std::set<std::size_t>{0}, "a leaf only has its edge to rank 0");
            }
        });
        std::cout << "[PASS] world_size=" << world << ": data edges are exactly what the reference broadcast uses\n";
    }

    // P2P between two ranks that have never talked: the data connection is created lazily and works; both orientations; repeated use never adds a second channel.
    void test_lazy_p2p_any_pair(std::size_t world)
    {
        const int fds_before_any = open_fd_count();
        (void)fds_before_any;
        run_world(world, [world](std::size_t rank, tbccl::Communicator &comm) {
            const std::size_t a = 1, b = world - 1; // a < b: b dials a's data listener
            expect(peers(comm).empty(), "starts sparse");
            if (rank == a || rank == b)
            {
                const std::size_t other = rank == a ? b : a;
                std::vector<std::uint8_t> out(2u << 20, static_cast<std::uint8_t>(rank + 1)), in(2u << 20, 0);
                std::vector<tbccl::Work> ws;
                ws.push_back(comm.send(view(out), out.size(), DataType::UInt8, other)); // both sides send first: simultaneous dial / accept
                ws.push_back(comm.recv(view(in), in.size(), DataType::UInt8, other));
                for (auto &w : ws)
                {
                    w.wait();
                    expect(!w.has_error(), "lazy P2P: " + w.error());
                }
                expect(in[0] == static_cast<std::uint8_t>(other + 1) && in.back() == static_cast<std::uint8_t>(other + 1), "payload");
                expect(peers(comm) == std::set<std::size_t>{other}, "exactly one data edge, to the peer that was used");
            }
            else
            {
                expect(peers(comm).empty(), "uninvolved ranks stay sparse");
            }
            // Barriers use the control plane only, so the process-wide descriptor count is quiescent between them.
            comm.barrier().wait();
            const int fds = open_fd_count();
            comm.barrier().wait();
            if (rank == a || rank == b)
            {
                const std::size_t other = rank == a ? b : a;
                for (int i = 0; i < 50; ++i)
                {
                    std::vector<std::uint8_t> small(100, 1), got(100);
                    auto s = comm.send(view(small), 100, DataType::UInt8, other);
                    auto r = comm.recv(view(got), 100, DataType::UInt8, other);
                    s.wait();
                    r.wait();
                }
                expect(peers(comm) == std::set<std::size_t>{other}, "still one edge");
            }
            comm.barrier().wait();
            expect(open_fd_count() == fds, "reuse never opens a second connection");
        });
        std::cout << "[PASS] world_size=" << world << ": lazy P2P between a previously unconnected pair (simultaneous first use), no duplicate channel\n";
    }

    // Every pair sends to every other at once as the FIRST data use: the full mesh is built lazily and concurrently without a deadlock or a duplicate.
    void test_all_to_all_first_use(std::size_t world)
    {
        run_world(world, [world](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::vector<std::uint8_t>> out(world), in(world);
            std::vector<tbccl::Work> ws;
            for (std::size_t p = 0; p < world; ++p)
            {
                if (p == rank) continue;
                out[p].assign(70000, static_cast<std::uint8_t>(10 * rank + p));
                in[p].assign(70000, 0);
                ws.push_back(comm.send(view(out[p]), out[p].size(), DataType::UInt8, p));
                ws.push_back(comm.recv(view(in[p]), in[p].size(), DataType::UInt8, p));
            }
            for (auto &w : ws)
            {
                w.wait();
                expect(!w.has_error(), "all-to-all first use: " + w.error());
            }
            for (std::size_t p = 0; p < world; ++p)
                if (p != rank) expect(in[p][0] == static_cast<std::uint8_t>(10 * p + rank), "payload from " + std::to_string(p));
            expect(peers(comm).size() == world - 1, "every pair connected exactly once");
        });
        std::cout << "[PASS] world_size=" << world << ": concurrent first use of every pair\n";
    }

    // A rank blocked on a lazy edge whose peer never uses it (so never dials) is released by abort, and teardown does not hang.
    void test_abort_while_waiting_for_a_lazy_dial()
    {
        auto results = bootstrap(healthy_slots(4), std::chrono::seconds(10));
        for (auto &r : results) expect(r.comm != nullptr, "bootstrap: " + r.error);
        std::vector<std::uint8_t> buf(4096);
        auto w = results[1].comm->recv(view(buf), buf.size(), DataType::UInt8, 3); // rank 3 (higher) must dial rank 1: it never will
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        expect(!w.is_completed(), "the receive waits for the peer's dial");
        results[1].comm->abort("test abort while waiting for a lazy dial");
        w.wait();
        expect(w.has_error(), "abort fails the waiting receive");
        const auto t0 = std::chrono::steady_clock::now();
        results.clear();
        expect(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5), "teardown is bounded");
        std::cout << "[PASS] abort releases a receive waiting for a lazy data dial; bounded teardown\n";
    }
} // namespace

int main()
{
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(120));
        std::cerr << "[FAIL] watchdog: a sparse-topology test hung\n";
        std::_Exit(2);
    }).detach();
    try
    {
        for (std::size_t world : {std::size_t{3}, std::size_t{4}, std::size_t{8}})
        {
            test_no_data_connections_after_init(world);
            test_edges_follow_the_algorithm(world);
            test_lazy_p2p_any_pair(world);
            test_all_to_all_first_use(world);
        }
        test_abort_while_waiting_for_a_lazy_dial();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All sparse-topology tests passed.\n";
    return 0;
}
