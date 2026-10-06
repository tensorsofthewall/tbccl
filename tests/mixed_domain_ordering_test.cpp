// Phase 73: P2P and collective traffic are independent ordering domains on one communicator (loopback, Host buffers; threads or forked processes per rank).
//
// The W2 ordering matrix (same / opposite relative order, 4 families, 3 sizes) lives in tools/p73_mixed_domain_probe.cpp, run with --gate as a second ctest. This file adds:
//   W3 / W4 mixes (all_reduce + P2P pair / ring, all_gather + ring, broadcast + independent pair; the first use of each domain on a pair in either order),
//   application threads (one submits collectives, another P2P), seeded random-jitter stress, collective-mismatch and P2P-size-mismatch semantics with the other domain
//   in flight, abort / destroy / peer death with both domains outstanding (the survivor must fail promptly and never touch a buffer after its Work is terminal),
//   resource lifecycle, and the handshake rules of the second data connection (either arrival order, missing role, wire 3 <-> 4).
// A failing stress iteration prints its seed. A watchdog turns any hang into a failure.

#include "mesh_fork_support.hpp"
#include "mesh_test_support.hpp"

#include "wire_protocol.hpp"

#include <tbccl/communicator.hpp>
#include <tbccl/peer_capabilities.hpp>
#include <tbccl/tcp.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <map>
#include <random>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;
using Bytes = std::vector<std::uint8_t>;

namespace
{

BufferView view(Bytes &v) { return BufferView{MemoryKind::Host, v.data(), v.size(), 0}; }

Bytes pattern(std::size_t from, std::size_t tag, std::size_t n)
{
    Bytes v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint8_t>(from * 131 + tag * 17 + i * 7 + (i >> 8) + 1);
    return v;
}

tbccl::Work isend(tbccl::Communicator &c, Bytes &v, std::size_t peer) { return c.send(view(v), v.size(), DataType::UInt8, peer); }
tbccl::Work irecv(tbccl::Communicator &c, Bytes &v, std::size_t peer) { return c.recv(view(v), v.size(), DataType::UInt8, peer); }

// An int32 SUM all_reduce over `n` bytes; ranks contribute distinct values.
struct Reduce
{
    std::vector<std::int32_t> data;
    explicit Reduce(std::size_t rank, std::size_t n, std::size_t salt = 0) : data(n / 4)
    {
        for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::int32_t>(i * 3 + rank * 100003 + salt * 7919 + 17);
    }
    tbccl::Work post(tbccl::Communicator &c) { return c.all_reduce(BufferView{MemoryKind::Host, data.data(), data.size() * 4, 0}, BufferView{MemoryKind::Host, data.data(), data.size() * 4, 0}, data.size(), DataType::Int32, tbccl::ReduceOp::Sum); }
    void check(std::size_t world, std::size_t salt = 0) const
    {
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            std::int32_t want = 0;
            for (std::size_t r = 0; r < world; ++r) want += static_cast<std::int32_t>(i * 3 + r * 100003 + salt * 7919 + 17);
            expect(data[i] == want, "all_reduce element " + std::to_string(i) + " is " + std::to_string(data[i]) + ", expected " + std::to_string(want));
        }
    }
};

void wait_all(std::vector<tbccl::Work> &works, const std::string &what, std::chrono::seconds limit = std::chrono::seconds(30))
{
    const auto deadline = std::chrono::steady_clock::now() + limit;
    for (std::size_t i = 0; i < works.size(); ++i)
    {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        expect(works[i].wait_for(std::max(left, std::chrono::milliseconds(1))), what + ": Work #" + std::to_string(i) + " did not finish");
        expect(!works[i].has_error(), what + ": Work #" + std::to_string(i) + " failed: " + works[i].error());
    }
}

// ---------------------------------------------------------------------------------------------------------------- W3 / W4

// `coll_first`: whether this rank submits its collective before its P2P operations. Ranks alternate, so neighbours are in the OPPOSITE relative order.
bool coll_first_for(std::size_t rank, bool variant) { return ((rank % 2 == 0) == variant); }

void run_mix(std::size_t world, const std::string &kind, bool variant, std::size_t bytes)
{
    run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
        const bool cf = coll_first_for(rank, variant);
        Reduce red(rank, bytes);
        Bytes bc = rank == 0 ? pattern(0, 5, bytes) : Bytes(bytes, 0xEE);
        Bytes ag_in = pattern(rank, 6, bytes);
        std::vector<Bytes> ag_out(world, Bytes(bytes, 0xEE));
        std::vector<BufferView> ag_views;
        for (auto &o : ag_out) ag_views.push_back(view(o));

        // P2P plan: the peers this rank sends to / receives from.
        std::vector<std::size_t> send_to, recv_from;
        const std::string pattern_name = kind.substr(0, kind.find('/'));
        if (pattern_name == "pair01") // rank 2.. only in the collective
        {
            if (rank < 2) send_to = {1 - rank}, recv_from = {1 - rank};
        }
        else if (pattern_name == "pair12")
        {
            if (rank == 1 || rank == 2) send_to = {3 - rank}, recv_from = {3 - rank};
        }
        else if (pattern_name == "ring")
        {
            send_to = {(rank + 1) % world};
            recv_from = {(rank + world - 1) % world};
        }
        else if (pattern_name == "pairs") // 0<->1, 2<->3, ...
        {
            if (rank < world - world % 2) send_to = {rank ^ 1}, recv_from = {rank ^ 1};
        }
        else if (pattern_name == "tail") // the last two ranks exchange while the rest only collect
        {
            if (rank >= world - 2) send_to = {rank == world - 1 ? world - 2 : world - 1}, recv_from = send_to;
        }
        std::vector<Bytes> tx, rx;
        for (std::size_t p : send_to) tx.push_back(pattern(rank, 20 + p, bytes));
        for (std::size_t i = 0; i < recv_from.size(); ++i) rx.emplace_back(bytes, 0xEE);

        std::vector<tbccl::Work> works;
        const std::string family = kind.substr(kind.find('/') + 1); // "AR" | "AG" | "BC" ; kind is "<p2p pattern>/<family>"
        auto post_c = [&] {
            if (family == "AR") works.push_back(red.post(comm));
            else if (family == "AG") works.push_back(comm.all_gather(view(ag_in), ag_views));
            else works.push_back(comm.broadcast(view(bc), 0));
        };
        auto post_p = [&] {
            for (std::size_t i = 0; i < recv_from.size(); ++i) works.push_back(irecv(comm, rx[i], recv_from[i]));
            for (std::size_t i = 0; i < send_to.size(); ++i) works.push_back(isend(comm, tx[i], send_to[i]));
        };
        if (cf)
        {
            post_c();
            post_p();
        }
        else
        {
            post_p();
            post_c();
        }
        wait_all(works, kind + " rank " + std::to_string(rank));
        if (family == "AR") red.check(world);
        else if (family == "BC") expect(bc == pattern(0, 5, bytes), "broadcast payload");
        else
            for (std::size_t r = 0; r < world; ++r) expect(ag_out[r] == pattern(r, 6, bytes), "all_gather slot " + std::to_string(r));
        for (std::size_t i = 0; i < recv_from.size(); ++i) expect(rx[i] == pattern(recv_from[i], 20 + rank, bytes), "P2P payload from rank " + std::to_string(recv_from[i]));
    });
}

void test_w3_w4()
{
    for (const bool variant : {false, true})
    {
        for (const std::size_t bytes : {std::size_t{4096}, std::size_t{1} << 20})
        {
            // W3: the P2P pair excludes rank 2 (it takes part only in the collective), then the other pair, then a ring.
            for (const char *k : {"pair01/AR", "pair12/AR", "ring/AR", "ring/AG", "pair01/BC", "tail/AG"}) run_mix(3, k, variant, bytes);
            // W4: pairwise P2P + all_reduce, ring + all_gather, an independent pair + broadcast.
            for (const char *k : {"pairs/AR", "ring/AG", "tail/BC", "ring/AR"}) run_mix(4, k, variant, bytes);
        }
    }
    std::cout << "[PASS] W3/W4 mixed domains\n";
}

// ------------------------------------------------------------------------------------------------------ application threads

void test_threads(std::size_t world)
{
    constexpr int kOps = 40;
    for (int start_order = 0; start_order < 2; ++start_order)
    {
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<Reduce> reds;
            for (int i = 0; i < kOps; ++i) reds.emplace_back(rank, 4096, static_cast<std::size_t>(i));
            const bool has_pair = rank < 2;
            std::vector<Bytes> tx, rx;
            for (int i = 0; i < kOps; ++i)
            {
                tx.push_back(pattern(rank, 40 + static_cast<std::size_t>(i), 2048 + static_cast<std::size_t>(i) * 5));
                rx.emplace_back(2048 + static_cast<std::size_t>(i) * 5, 0xEE);
            }
            std::string error_c, error_p;
            auto collectives = [&] {
                try
                {
                    std::vector<tbccl::Work> w;
                    for (int i = 0; i < kOps; ++i) w.push_back(reds[static_cast<std::size_t>(i)].post(comm));
                    wait_all(w, "collective thread");
                    for (int i = 0; i < kOps; ++i) reds[static_cast<std::size_t>(i)].check(world, static_cast<std::size_t>(i));
                }
                catch (const std::exception &e)
                {
                    error_c = e.what();
                }
            };
            auto p2p = [&] {
                try
                {
                    if (!has_pair) return;
                    const std::size_t peer = 1 - rank;
                    std::vector<tbccl::Work> w;
                    for (int i = 0; i < kOps; ++i)
                    {
                        w.push_back(isend(comm, tx[static_cast<std::size_t>(i)], peer));
                        w.push_back(irecv(comm, rx[static_cast<std::size_t>(i)], peer));
                    }
                    wait_all(w, "p2p thread");
                    for (int i = 0; i < kOps; ++i) expect(rx[static_cast<std::size_t>(i)] == pattern(peer, 40 + static_cast<std::size_t>(i), rx[static_cast<std::size_t>(i)].size()), "p2p payload " + std::to_string(i));
                }
                catch (const std::exception &e)
                {
                    error_p = e.what();
                }
            };
            // the start order differs between ranks and between the two passes
            const bool collectives_first = ((rank + static_cast<std::size_t>(start_order)) % 2) == 0;
            std::thread a, b;
            if (collectives_first)
            {
                a = std::thread(collectives);
                b = std::thread(p2p);
            }
            else
            {
                b = std::thread(p2p);
                a = std::thread(collectives);
            }
            a.join();
            b.join();
            expect(error_c.empty(), "collective thread: " + error_c);
            expect(error_p.empty(), "p2p thread: " + error_p);
        });
    }
    std::cout << "[PASS] application threads (collectives on one, P2P on another) world_size=" << world << "\n";
}

// -------------------------------------------------------------------------------------------------------- jitter stress

void test_jitter(std::size_t iterations, std::size_t bytes, unsigned base_seed)
{
    for (std::size_t it = 0; it < iterations; ++it)
    {
        const unsigned seed = base_seed * 1000003u + static_cast<unsigned>(it);
        try
        {
            run_world(2, [&](std::size_t rank, tbccl::Communicator &comm) {
                std::mt19937 rng(seed * 31u + static_cast<unsigned>(rank));
                Reduce red(rank, bytes, it);
                Bytes tx = pattern(rank, 70 + it % 5, bytes), rx(bytes, 0xEE), bc = rank == 0 ? pattern(0, 71, bytes) : Bytes(bytes, 0xEE);
                // 0 all_reduce, 1 send, 2 recv, 3 broadcast. The collectives keep ONE logical order on both ranks (that is the contract); the P2P operations and the
                // interleaving of the two domains are random and differ between the ranks.
                std::vector<int> p2p = {1, 2}, order;
                std::shuffle(p2p.begin(), p2p.end(), rng);
                std::vector<int> coll = {0, 3};
                while (!coll.empty() || !p2p.empty())
                {
                    const bool take_coll = p2p.empty() || (!coll.empty() && rng() % 2 == 0);
                    auto &from = take_coll ? coll : p2p;
                    order.push_back(from.front());
                    from.erase(from.begin());
                }
                std::vector<tbccl::Work> works;
                for (int op : order)
                {
                    if (op == 0) works.push_back(red.post(comm));
                    else if (op == 1) works.push_back(isend(comm, tx, 1 - rank));
                    else if (op == 2) works.push_back(irecv(comm, rx, 1 - rank));
                    else works.push_back(comm.broadcast(view(bc), 0));
                    if (rng() % 3 == 0) std::this_thread::sleep_for(std::chrono::microseconds(rng() % 800));
                }
                wait_all(works, "iteration");
                red.check(2, it);
                expect(rx == pattern(1 - rank, 70 + it % 5, bytes), "p2p payload");
                expect(bc == pattern(0, 71, bytes), "broadcast payload");
            });
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("jitter stress failed at iteration " + std::to_string(it) + " seed " + std::to_string(seed) + " (base " + std::to_string(base_seed) + ", bytes " + std::to_string(bytes) + "): " + e.what());
        }
    }
    std::cout << "[PASS] jitter stress " << iterations << " iterations of " << bytes << " bytes (base seed " << base_seed << ")\n";
}

// -------------------------------------------------------------------------------------------- mismatch / failure semantics

bool contains(const std::string &text, const char *needle) { return text.find(needle) != std::string::npos; }

void test_collective_mismatch_with_p2p_in_flight()
{
    // World size 3 (descriptor-checked collectives): rank 2 calls broadcast while ranks 0 and 1 call all_reduce, with a P2P exchange between 0 and 1 in flight.
    auto results = bootstrap(healthy_slots(3), std::chrono::seconds(20));
    for (auto &r : results) expect(r.comm != nullptr, "bootstrap: " + r.error);
    std::vector<std::string> coll_error(3);
    std::vector<std::thread> threads;
    for (std::size_t rank = 0; rank < 3; ++rank)
        threads.emplace_back([&, rank] {
            auto &comm = *results[rank].comm;
            Reduce red(rank, 4096);
            Bytes bc(4096, 0xEE);
            Bytes tx = pattern(rank, 3, 64 << 10), rx(64 << 10, 0xEE);
            std::vector<tbccl::Work> w;
            if (rank < 2)
            {
                w.push_back(isend(comm, tx, 1 - rank));
                w.push_back(irecv(comm, rx, 1 - rank));
            }
            auto c = rank == 2 ? comm.broadcast(view(bc), 0) : red.post(comm);
            expect(c.wait_for(std::chrono::seconds(20)), "the mismatching collective did not terminate (rank " + std::to_string(rank) + ")");
            expect(c.has_error(), "the mismatching collective reported success (rank " + std::to_string(rank) + ")");
            coll_error[rank] = c.error();
            for (auto &x : w) expect(x.wait_for(std::chrono::seconds(20)), "a P2P Work did not terminate after the mismatch");
            if (w.size() == 2 && !w[1].has_error()) expect(rx == pattern(1 - rank, 3, 64 << 10), "a P2P receive that completed holds wrong data"); // w[1] is the receive
        });
    for (auto &t : threads) t.join();
    bool named = false;
    for (std::size_t r = 0; r < 3; ++r) named = named || contains(coll_error[r], "protocol_mismatch");
    expect(named, "no rank reported protocol_mismatch: " + coll_error[0] + " / " + coll_error[1] + " / " + coll_error[2]);
    results.clear();
    std::cout << "[PASS] collective mismatch is detected with P2P in flight (W3)\n";
}

void test_p2p_size_mismatch_with_collective_active()
{
    auto results = bootstrap(healthy_slots(2), std::chrono::seconds(20));
    for (auto &r : results) expect(r.comm != nullptr, "bootstrap: " + r.error);
    std::string recv_error, coll_error0, coll_error1;
    std::vector<std::thread> threads;
    for (std::size_t rank = 0; rank < 2; ++rank)
        threads.emplace_back([&, rank] {
            auto &comm = *results[rank].comm;
            Reduce red(rank, 1 << 20);
            Bytes tx = pattern(0, 4, 1000), rx(2000, 0xEE);
            tbccl::Work p = rank == 0 ? isend(comm, tx, 1) : irecv(comm, rx, 0); // rank 1 posts a receive twice as large as the message
            auto c = red.post(comm);
            expect(p.wait_for(std::chrono::seconds(20)) && c.wait_for(std::chrono::seconds(20)), "a Work did not terminate");
            if (rank == 1)
            {
                expect(p.has_error() && contains(p.error(), "size mismatch"), "the P2P size mismatch was not reported: " + p.error());
                recv_error = p.error();
                coll_error1 = c.has_error() ? c.error() : "";
            }
            else coll_error0 = c.has_error() ? c.error() : "";
        });
    for (auto &t : threads) t.join();
    // The all_reduce may have finished before the P2P failure; if it did not, it must have failed (never hung, never wrong). Either way the communicator is failed afterwards.
    expect(results[1].comm->failed(), "the communicator survived a P2P size mismatch");
    results.clear();
    std::cout << "[PASS] P2P size mismatch with a collective active: " << recv_error.substr(0, 60) << "...\n";
}

void test_abort_with_both_domains_pending()
{
    auto results = bootstrap(healthy_slots(2), std::chrono::seconds(20));
    for (auto &r : results) expect(r.comm != nullptr, "bootstrap: " + r.error);
    auto &comm = *results[0].comm; // rank 1 never posts anything: both of rank 0's operations are pending
    Reduce red(0, 1 << 20);
    Bytes rx(1 << 20, 0xEE);
    auto c = red.post(comm);
    auto p = irecv(comm, rx, 1);
    std::thread aborter([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        comm.abort("test abort");
    });
    const auto t0 = std::chrono::steady_clock::now();
    expect(c.wait_for(std::chrono::seconds(10)) && p.wait_for(std::chrono::seconds(10)), "a Work stayed pending after abort()");
    aborter.join();
    expect(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5), "abort was not prompt");
    expect(c.has_error() && p.has_error(), "aborted Works must fail");
    const Bytes untouched(1 << 20, 0xEE);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    expect(rx == untouched, "a receive buffer was written after abort");
    results.clear();
    std::cout << "[PASS] abort with a collective and a P2P Work pending\n";
}

void test_destroy_with_both_domains_pending()
{
    auto results = bootstrap(healthy_slots(2), std::chrono::seconds(20));
    for (auto &r : results) expect(r.comm != nullptr, "bootstrap: " + r.error);
    Reduce red(0, 1 << 20);
    Bytes rx(1 << 20, 0xEE);
    auto c = red.post(*results[0].comm);
    auto p = irecv(*results[0].comm, rx, 1);
    const auto t0 = std::chrono::steady_clock::now();
    results[0].comm.reset(); // destroy with both outstanding: aborts them, bounded
    expect(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10), "destroy was not bounded");
    expect(c.is_completed() && p.is_completed() && c.has_error() && p.has_error(), "Work handles must be terminal and queryable after the communicator is gone");
    results.clear();
    std::cout << "[PASS] destroy with a collective and a P2P Work pending\n";
}

void test_peer_death(std::size_t world)
{
    // The last rank dies without a Goodbye (its sockets are closed by the kernel) while the others have a collective AND a P2P Work outstanding.
    const std::size_t dead = world - 1;
    const auto status = run_forked(world, [&](std::size_t rank, tbccl::Communicator &comm) {
        Bytes rx(1 << 20, 0xEE), tx = pattern(rank, 8, 1 << 20);
        Reduce red(rank, 1 << 20);
        if (rank == dead)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200)); // never joins the collective: the survivors' Works are pending when it dies
            _exit(9);
        }
        // survivors: a collective with the doomed rank, and a receive from it (world 2) or from each other (world 3)
        std::vector<tbccl::Work> w;
        w.push_back(red.post(comm));
        w.push_back(world == 2 ? irecv(comm, rx, dead) : irecv(comm, rx, 1 - rank));
        if (world > 2) w.push_back(isend(comm, tx, 1 - rank)); // the survivors also exchange with each other
        const auto t0 = std::chrono::steady_clock::now();
        for (auto &x : w) expect(x.wait_for(std::chrono::seconds(20)), "a Work stayed pending after the peer died");
        expect(w[0].has_error(), "the collective with a dead peer reported success");
        expect(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(15), "failure was not prompt");
        expect(comm.failed(), "the communicator did not fail");
    }, std::chrono::seconds(60));
    expect_children_ok(status, world, dead, "peer death W" + std::to_string(world));
    std::cout << "[PASS] peer death with a collective and a P2P Work outstanding, world_size=" << world << "\n";
}

// ---------------------------------------------------------------------------------------------------------------- lifecycle

int thread_count()
{
    int n = 0;
    if (DIR *d = opendir("/proc/self/task"))
    {
        while (readdir(d) != nullptr) ++n;
        closedir(d);
    }
    return n;
}

void mixed_cycle()
{
    run_world(2, [&](std::size_t rank, tbccl::Communicator &comm) {
        Reduce red(rank, 1 << 16);
        Bytes tx = pattern(rank, 2, 1 << 16), rx(1 << 16, 0xEE);
        std::vector<tbccl::Work> w;
        if (rank == 0)
        {
            w.push_back(red.post(comm));
            w.push_back(isend(comm, tx, 1));
        }
        else
        {
            w.push_back(irecv(comm, rx, 0));
            w.push_back(red.post(comm));
        }
        wait_all(w, "cycle");
    });
}

void test_lifecycle()
{
    for (int i = 0; i < 3; ++i) mixed_cycle(); // warm up
    const int fds0 = open_fd_count(), th0 = thread_count();
    for (int i = 0; i < 10; ++i) mixed_cycle();
    const int fds1 = open_fd_count(), th1 = thread_count();
    expect(fds1 == fds0, "file descriptors leaked over 10 mixed cycles: " + std::to_string(fds0) + " -> " + std::to_string(fds1));
    expect(th1 <= th0 + 1, "threads leaked over 10 mixed cycles: " + std::to_string(th0) + " -> " + std::to_string(th1));
    for (int i = 0; i < 100; ++i)
    {
        auto results = bootstrap(healthy_slots(2), std::chrono::seconds(10));
        for (auto &r : results) expect(r.comm != nullptr, "bootstrap: " + r.error);
    }
    const int fds2 = open_fd_count(), th2 = thread_count();
    expect(fds2 == fds0, "file descriptors leaked over 100 create/destroy cycles: " + std::to_string(fds0) + " -> " + std::to_string(fds2));
    expect(th2 <= th0 + 1, "threads leaked over 100 create/destroy cycles: " + std::to_string(th0) + " -> " + std::to_string(th2));
    std::cout << "[PASS] lifecycle: fds " << fds0 << "->" << fds2 << ", threads " << th0 << "->" << th2 << "\n";
}

// ------------------------------------------------------------------------------------- handshake of the second data connection

struct FakePeer
{
    std::vector<std::unique_ptr<tbccl::Connection>> keep;
};

// A hand-driven rank 1 against a real rank 0: control + capabilities, then the two data roles in the order given (or only some of them).
// Returns rank 0's create() error ("" on success).
std::string bootstrap_against_fake_peer(const std::vector<tbccl::detail::ConnectionRole> &roles, std::chrono::milliseconds timeout, std::uint32_t wire_version = tbccl::kWireProtocolVersion)
{
    auto listeners = tbccl::CommunicatorListeners::bind("127.0.0.1");
    tbccl::RankDirectory dir;
    dir.entries.push_back({0, listeners->control(), listeners->data()});
    dir.entries.push_back({1, {}, {}});
    const auto id = tbccl::CommunicatorId::generate();
    std::string error, peer_error;
    std::unique_ptr<tbccl::Communicator> comm;
    std::thread rank0([&] {
        tbccl::CommunicatorOptions o;
        o.rank = 0;
        o.world_size = 2;
        o.communicator_id = id;
        o.rank_directory = dir;
        o.listeners = std::move(listeners);
        o.bootstrap_timeout = timeout;
        try
        {
            comm = tbccl::Communicator::create(o);
        }
        catch (const std::exception &e)
        {
            error = e.what();
        }
    });
    FakePeer peer;
    try
    {
        auto hello_for = [&](tbccl::detail::ConnectionRole role) {
            tbccl::detail::Hello h;
            h.wire_version = wire_version;
            h.communicator_id = id;
            h.rank = 1;
            h.world_size = 2;
            h.role = role;
            return h;
        };
        auto dial = [&](const tbccl::Endpoint &ep, tbccl::detail::ConnectionRole role) {
            auto c = tbccl::tcp_connect(ep.host, ep.port, {});
            c->set_io_timeout(std::chrono::seconds(5));
            tbccl::detail::dial_handshake(*c, hello_for(role), 0);
            return c;
        };
        auto control = dial(dir.entries[0].control, tbccl::detail::ConnectionRole::Control);
        peer.keep.push_back(std::move(control));
        for (auto role : roles) peer.keep.push_back(dial(dir.entries[0].data, role)); // rank 0 accepts the data connections before the capability exchange
        (void)tbccl::exchange_capabilities(*peer.keep.front(), tbccl::local_capabilities());
    }
    catch (const std::exception &e)
    {
        peer_error = std::string("fake peer: ") + e.what();
    }
    rank0.join();
    comm.reset();
    return error.empty() ? peer_error : error; // rank 0's own verdict wins: the fake peer only sees its socket closed
}

void test_handshake_roles()
{
    using tbccl::detail::ConnectionRole;
    // either arrival order of the two data roles builds the communicator
    expect(bootstrap_against_fake_peer({ConnectionRole::Data, ConnectionRole::CollectiveData}, std::chrono::seconds(10)).empty(), "data then collective-data");
    expect(bootstrap_against_fake_peer({ConnectionRole::CollectiveData, ConnectionRole::Data}, std::chrono::seconds(10)).empty(), "collective-data then data");
    // a peer that connects only one of the roles ends in a bounded timeout that names the missing role
    const auto t0 = std::chrono::steady_clock::now();
    const std::string missing = bootstrap_against_fake_peer({ConnectionRole::Data}, std::chrono::milliseconds(1500));
    expect(contains(missing, "collective-data from rank 1"), "the missing role is not named: " + missing);
    const std::string missing2 = bootstrap_against_fake_peer({ConnectionRole::CollectiveData}, std::chrono::milliseconds(1500));
    expect(contains(missing2, "data from rank 1"), "the missing role is not named: " + missing2);
    expect(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10), "a half-connected peer was not bounded");
    // a duplicate of one role is rejected, a wire-3 dialer is rejected cleanly, and nothing hangs
    const std::string dup = bootstrap_against_fake_peer({ConnectionRole::Data, ConnectionRole::Data}, std::chrono::milliseconds(3000));
    expect(contains(dup, "protocol_mismatch") || contains(dup, "duplicate"), "duplicate role: " + dup);
    const std::string old_peer = bootstrap_against_fake_peer({ConnectionRole::Data}, std::chrono::milliseconds(3000), 3);
    expect(contains(old_peer, "protocol_mismatch") && contains(old_peer, "wire protocol 3"), "a wire-3 peer must be rejected with a wire-protocol mismatch: " + old_peer);
    std::cout << "[PASS] handshake: either role order, missing role named, duplicate role, wire-3 dialer rejected\n";
}

void test_new_dialer_rejected_by_old_listener()
{
    // "old" listener: a raw socket that answers the new dialer's Hello with a wire-3 HelloAck rejecting the version (what a wire-3 build sends).
    auto listener = tbccl::tcp_listen("127.0.0.1", 0, {});
    const std::uint16_t port = listener->local_port();
    std::thread old([&] {
        auto c = listener->accept_for(std::chrono::seconds(5));
        if (!c) return;
        std::uint8_t hello[tbccl::detail::kHelloWireSize];
        c->recv(hello, sizeof(hello));
        std::uint8_t reply[tbccl::detail::kHelloWireSize];
        std::memcpy(reply, hello, sizeof(reply));
        tbccl::detail::put_u32(reply + 4, 3);  // wire version 3
        tbccl::detail::put_u32(reply + 8, 2);  // HelloAck
        tbccl::detail::put_u32(reply + 40, 1); // BadVersion
        tbccl::detail::put_text(reply + 44, 64, "wire protocol 4 != 3");
        c->send(reply, sizeof(reply));
    });
    auto c = tbccl::tcp_connect("127.0.0.1", port, {});
    c->set_io_timeout(std::chrono::seconds(5));
    tbccl::detail::Hello h;
    h.communicator_id = tbccl::CommunicatorId::generate();
    h.rank = 1;
    h.world_size = 2;
    h.role = tbccl::detail::ConnectionRole::Control;
    std::string error;
    try
    {
        tbccl::detail::dial_handshake(*c, h, 0);
    }
    catch (const std::exception &e)
    {
        error = e.what();
    }
    old.join();
    expect(contains(error, "protocol_mismatch") && contains(error, "wire protocol"), "a wire-4 dialer against a wire-3 listener: " + error);
    expect(tbccl::kWireProtocolVersion == 4, "kWireProtocolVersion is expected to be 4 after Phase 73");
    std::cout << "[PASS] wire-4 dialer against a wire-3 listener: " << error.substr(0, 80) << "\n";
}

} // namespace

int main(int argc, char **argv)
{
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::minutes(30));
        std::cerr << "mixed_domain_ordering_test: watchdog expired\n";
        std::_Exit(2);
    }).detach();
    const std::string only = argc > 1 ? argv[1] : "";
    const auto run = [&](const char *name, const std::function<void()> &f) {
        if (!only.empty() && only != name) return;
        f();
    };
    try
    {
        run("w3w4", [] { test_w3_w4(); });
        run("threads", [] { test_threads(2); test_threads(3); });
        run("jitter", [] { test_jitter(1000, 4096, 1); test_jitter(100, 1 << 20, 2); });
        run("mismatch", [] { test_collective_mismatch_with_p2p_in_flight(); test_p2p_size_mismatch_with_collective_active(); });
        run("abort", [] { test_abort_with_both_domains_pending(); test_destroy_with_both_domains_pending(); });
        run("death", [] { test_peer_death(2); test_peer_death(3); });
        run("lifecycle", [] { test_lifecycle(); });
        run("handshake", [] { test_handshake_roles(); test_new_dialer_rejected_by_old_listener(); });
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "mixed_domain_ordering_test: ok\n";
    return 0;
}
