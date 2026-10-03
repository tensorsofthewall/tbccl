// The optimized N>2 collective algorithms, FORCED one at a time (TBCCL_*_ALGORITHM) on Host loopback at world_size 2..8, plus the default selection. Sections
// are added commit by commit with the algorithm they test. Every rank is a thread of this process; ports are kernel-assigned.

#include "mesh_test_support.hpp"

#include "collective_topology.hpp"
#include "communicator_debug.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <set>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;
using tbccl::ReduceOp;

namespace
{
    BufferView view(void *p, std::size_t bytes) { return BufferView{MemoryKind::Host, p, bytes, 0}; }

    struct ForceEnv
    {
        std::string var;
        ForceEnv(const char *v, const char *value) : var(v)
        {
            if (value) setenv(v, value, 1);
        }
        ~ForceEnv() { unsetenv(var.c_str()); }
    };

    std::set<std::size_t> peers(const tbccl::Communicator &c)
    {
        auto v = tbccl::detail::debug_connected_data_peers(c);
        return std::set<std::size_t>(v.begin(), v.end());
    }

    // ---- dissemination barrier -----------------------------------------------------------------------------------------------------------------------

    void test_barrier(std::size_t world, const char *forced)
    {
        ForceEnv env("TBCCL_BARRIER_ALGORITHM", forced);
        std::atomic<int> entered{0}, early{0};
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            for (int round = 0; round < 30; ++round)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(rank * 3 + (round % 3)));
                entered.fetch_add(1);
                auto w = comm.barrier();
                w.wait();
                expect(!w.has_error(), "barrier: " + w.error());
                if (entered.load() < static_cast<int>(world) * (round + 1)) early.fetch_add(1);
                comm.barrier().wait(); // keep the rounds apart
            }
        });
        expect(early.load() == 0, "a rank left a barrier before every rank entered it");
        std::cout << "[PASS] barrier " << (forced ? forced : "default") << " world_size=" << world << " (staggered entrants, 30 rounds)\n";
    }

    void test_dissemination_edges(std::size_t world)
    {
        ForceEnv env("TBCCL_BARRIER_ALGORITHM", "dissemination");
        run_world(world, [world](std::size_t rank, tbccl::Communicator &comm) {
            comm.barrier().wait();
            std::set<std::size_t> want;
            for (std::size_t d = 1; d < world; d <<= 1)
            {
                want.insert((rank + d) % world);
                want.insert((rank + world - d) % world);
            }
            expect(peers(comm) == want, "dissemination barrier uses only the +-2^k edges (rank " + std::to_string(rank) + ")");
        });
        std::cout << "[PASS] dissemination barrier world_size=" << world << " uses only its +-2^k data edges\n";
    }

    // One rank calls a different collective than the rest: every rank must fail promptly, never hang.
    void test_barrier_mismatch(std::size_t world, std::size_t odd_rank, const char *label)
    {
        ForceEnv env("TBCCL_BARRIER_ALGORITHM", "dissemination");
        const auto t0 = std::chrono::steady_clock::now();
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::uint8_t> buf(64, 1);
            try
            {
                tbccl::Work w = rank == odd_rank ? comm.broadcast(view(buf.data(), buf.size()), 0) : comm.barrier();
                w.wait();
                expect(w.has_error(), std::string(label) + ": rank " + std::to_string(rank) + " must fail");
            }
            catch (const std::runtime_error &)
            {
                // the mismatch was already detected elsewhere before this rank got to submit: the call throws on a terminal communicator
            }
            expect(comm.failed(), std::string(label) + ": poisoned");
        });
        expect(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10), std::string(label) + ": no hang");
        std::cout << "[PASS] barrier vs broadcast mismatch (" << label << ") world_size=" << world << ": every rank fails\n";
    }

    // ---- binomial-tree broadcast -----------------------------------------------------------------------------------------------------------------

    std::vector<std::uint8_t> payload(std::size_t seed, std::size_t n)
    {
        std::vector<std::uint8_t> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint8_t>(seed * 37 + i * 11 + (i >> 9)); // every byte value occurs: FP8 NaN/-0 encodings, packed nibbles, anything
        return v;
    }

    void test_tree_broadcast(std::size_t world)
    {
        ForceEnv env("TBCCL_BROADCAST_ALGORITHM", "tree");
        for (std::size_t bytes : {std::size_t{0}, std::size_t{1}, std::size_t{17}, std::size_t{4096}, std::size_t{65536}, std::size_t{1} << 20, (std::size_t{5} << 20) + 3})
        {
            for (std::size_t root = 0; root < world; ++root)
            {
                run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
                    auto buf = rank == root ? payload(root + 1, bytes) : std::vector<std::uint8_t>(bytes, 0xEE);
                    auto w = comm.broadcast(view(buf.data(), buf.size()), root);
                    w.wait();
                    expect(!w.has_error(), "tree broadcast: " + w.error());
                    expect(buf == payload(root + 1, bytes), "tree broadcast payload, rank " + std::to_string(rank) + " root " + std::to_string(root) + " bytes " + std::to_string(bytes));
                });
            }
        }
        std::cout << "[PASS] tree broadcast every root, 0 B .. 5 MiB, arbitrary bytes, world_size=" << world << "\n";
    }

    void test_tree_broadcast_edges(std::size_t world)
    {
        ForceEnv env("TBCCL_BROADCAST_ALGORITHM", "tree");
        for (std::size_t root : {std::size_t{0}, world - 1})
        {
            run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
                std::vector<std::uint8_t> buf(5000, rank == root ? 3 : 0);
                comm.broadcast(view(buf.data(), buf.size()), root).wait();
                std::set<std::size_t> want;
                for (std::size_t c : tbccl::detail::tree_children(rank, root, world)) want.insert(c);
                if (rank != root) want.insert(tbccl::detail::tree_parent(rank, root, world));
                expect(peers(comm) == want, "tree broadcast uses only parent/children edges (rank " + std::to_string(rank) + ", root " + std::to_string(root) + ")");
            });
        }
        std::cout << "[PASS] tree broadcast world_size=" << world << " uses only parent/children data edges\n";
    }

    // ---- ring all-gather -----------------------------------------------------------------------------------------------------------------------

    void test_ring_all_gather(std::size_t world)
    {
        ForceEnv env("TBCCL_ALLGATHER_ALGORITHM", "ring");
        for (std::size_t bytes : {std::size_t{0}, std::size_t{1}, std::size_t{17}, std::size_t{4093}, std::size_t{65537}, (std::size_t{1} << 20) + 5})
        {
            for (bool aliased : {false, true})
            {
                run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
                    auto input = payload(rank + 10, bytes);
                    std::vector<std::vector<std::uint8_t>> out(world, std::vector<std::uint8_t>(bytes, 0xEE));
                    if (aliased) out[rank] = input; // outputs[rank] is the input memory
                    std::vector<BufferView> outs;
                    for (auto &o : out) outs.push_back(view(o.data(), bytes));
                    auto w = comm.all_gather(aliased ? outs[rank] : view(input.data(), bytes), outs);
                    w.wait();
                    expect(!w.has_error(), "ring all_gather: " + w.error());
                    for (std::size_t q = 0; q < world; ++q)
                        expect(out[q] == payload(q + 10, bytes), "ring all_gather slot " + std::to_string(q) + " on rank " + std::to_string(rank) + " bytes " + std::to_string(bytes));
                });
            }
        }
        std::cout << "[PASS] ring all_gather world_size=" << world << ": rank order exact, odd sizes, in-place and out-of-place, arbitrary bytes\n";
    }

    void test_ring_all_gather_edges_and_overlap(std::size_t world)
    {
        ForceEnv env("TBCCL_ALLGATHER_ALGORITHM", "ring");
        const std::size_t bytes = std::size_t{40} << 20; // far beyond the socket buffers: a send that had to finish before the receive was posted would deadlock the ring
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::uint8_t> in(bytes, static_cast<std::uint8_t>(rank + 1));
            std::vector<std::vector<std::uint8_t>> out(world, std::vector<std::uint8_t>(bytes, 0));
            std::vector<BufferView> outs;
            for (auto &o : out) outs.push_back(view(o.data(), bytes));
            auto w = comm.all_gather(view(in.data(), bytes), outs);
            w.wait();
            expect(!w.has_error(), "large ring all_gather: " + w.error());
            for (std::size_t q = 0; q < world; ++q) expect(out[q].front() == q + 1 && out[q].back() == q + 1, "large slot " + std::to_string(q));
            const std::set<std::size_t> want{(rank + 1) % world, (rank + world - 1) % world};
            expect(peers(comm) == want, "ring all_gather uses only its predecessor and successor edges (rank " + std::to_string(rank) + ")");
        }, std::chrono::seconds(60));
        std::cout << "[PASS] ring all_gather world_size=" << world << ": 40 MiB per rank (send/receive overlap), only ring edges\n";
    }

    // The DEFAULT selection (no override): which data edges appear shows which algorithm ran.
    void test_default_selection_edges()
    {
        struct Case { std::size_t world; std::size_t bytes; const char *what; };
        auto edges_of = [](const char *collective, std::size_t world, std::size_t bytes) {
            std::vector<std::set<std::size_t>> edges(world);
            run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
                std::vector<std::uint8_t> buf(bytes, 1);
                const std::size_t count = bytes / 4;
                std::string c = collective;
                if (c == "all_reduce") comm.all_reduce(view(buf.data(), bytes), view(buf.data(), bytes), count, DataType::Float32, ReduceOp::Sum).wait();
                else if (c == "broadcast") comm.broadcast(view(buf.data(), bytes), 0).wait();
                else
                {
                    std::vector<std::vector<std::uint8_t>> out(world, std::vector<std::uint8_t>(bytes));
                    std::vector<BufferView> outs;
                    for (auto &o : out) outs.push_back(view(o.data(), bytes));
                    comm.all_gather(view(buf.data(), bytes), outs).wait();
                }
                auto v = tbccl::detail::debug_connected_data_peers(comm);
                edges[rank] = std::set<std::size_t>(v.begin(), v.end());
            });
            return edges;
        };
        // N=8: 4 KiB all_reduce -> tree (edges parent/children of root 0); 2 MiB -> ring
        auto e = edges_of("all_reduce", 8, 4096);
        for (std::size_t r = 0; r < 8; ++r)
        {
            std::set<std::size_t> want;
            for (std::size_t c : tbccl::detail::tree_children(r, 0, 8)) want.insert(c);
            if (r) want.insert(tbccl::detail::tree_parent(r, 0, 8));
            expect(e[r] == want, "default N=8 4 KiB all_reduce is the binomial tree");
        }
        e = edges_of("all_reduce", 8, 2u << 20);
        for (std::size_t r = 0; r < 8; ++r) expect(e[r] == (std::set<std::size_t>{(r + 1) % 8, (r + 7) % 8}), "default N=8 2 MiB all_reduce is the ring");
        // N=4: small -> recursive doubling (rank XOR 1, XOR 2)
        e = edges_of("all_reduce", 4, 4096);
        for (std::size_t r = 0; r < 4; ++r) expect(e[r] == (std::set<std::size_t>{r ^ 1, r ^ 2}), "default N=4 4 KiB all_reduce is recursive doubling");
        // N=3 large -> ring
        e = edges_of("all_reduce", 3, 1u << 20);
        for (std::size_t r = 0; r < 3; ++r) expect(e[r] == (std::set<std::size_t>{(r + 1) % 3, (r + 2) % 3}), "default N=3 1 MiB all_reduce is the ring (every rank is a neighbour at N=3)");
        // broadcast -> tree, all_gather -> ring (N=8)
        e = edges_of("broadcast", 8, 4096);
        for (std::size_t r = 0; r < 8; ++r)
        {
            std::set<std::size_t> want;
            for (std::size_t c : tbccl::detail::tree_children(r, 0, 8)) want.insert(c);
            if (r) want.insert(tbccl::detail::tree_parent(r, 0, 8));
            expect(e[r] == want, "default broadcast is the binomial tree");
        }
        e = edges_of("all_gather", 8, 4096);
        for (std::size_t r = 0; r < 8; ++r) expect(e[r] == (std::set<std::size_t>{(r + 1) % 8, (r + 7) % 8}), "default all_gather is the ring");
        std::cout << "[PASS] default selection: tree / recursive doubling / ring for all_reduce by size and world, tree broadcast, ring all_gather (shown by the data edges used)\n";
    }

    void test_abort_during_barrier(std::size_t world)
    {
        ForceEnv env("TBCCL_BARRIER_ALGORITHM", "dissemination");
        const auto t0 = std::chrono::steady_clock::now();
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            if (rank == world / 2)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(200)); // never enters the barrier; aborts instead
                comm.abort("rank never reached the barrier");
                return;
            }
            auto w = comm.barrier();
            w.wait();
            expect(w.has_error(), "the barrier must fail on rank " + std::to_string(rank));
        });
        expect(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10), "bounded");
        std::cout << "[PASS] one rank aborting unblocks every dissemination-barrier participant, world_size=" << world << "\n";
    }
} // namespace

int main()
{
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(280));
        std::cerr << "[FAIL] watchdog: an algorithm test hung\n";
        std::_Exit(2);
    }).detach();
    try
    {
        for (std::size_t world : {std::size_t{2}, std::size_t{3}, std::size_t{4}, std::size_t{5}, std::size_t{8}})
        {
            test_barrier(world, nullptr);
            test_barrier(world, "reference");
            if (world > 2) test_barrier(world, "dissemination");
        }
        for (std::size_t world : {std::size_t{3}, std::size_t{4}, std::size_t{5}, std::size_t{8}})
        {
            test_dissemination_edges(world);
            test_abort_during_barrier(world);
        }
        for (std::size_t world : {std::size_t{3}, std::size_t{4}, std::size_t{5}, std::size_t{8}})
        {
            test_tree_broadcast(world);
            test_tree_broadcast_edges(world);
        }
        for (std::size_t world : {std::size_t{3}, std::size_t{4}, std::size_t{5}, std::size_t{8}})
        {
            test_ring_all_gather(world);
            if (world <= 5) test_ring_all_gather_edges_and_overlap(world);
        }
        test_default_selection_edges();
        for (std::size_t world : {std::size_t{3}, std::size_t{5}})
        {
            test_barrier_mismatch(world, 1, "non-coordinator rank differs");
            test_barrier_mismatch(world, 0, "the coordinator differs");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All algorithm tests passed.\n";
    return 0;
}
