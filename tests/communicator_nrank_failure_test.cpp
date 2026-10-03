// Phase 50: communicator-wide failure and abort at world_size 4 (loopback, dynamic ports). A rank that dies, goes silent or aborts must take the
// whole communicator down: every healthy rank leaves its blocked operation with an error, becomes terminal, and tears down in bounded time.
//
// Abrupt rank death needs a real process: those cases fork one process per rank (the sockets are then closed by the kernel, with no Goodbye).
// The parent pre-binds every rank's listeners, so ports are still dynamic. Cases that need no death use threads.

#include "mesh_test_support.hpp"

#include <tbccl/tcp.hpp>

#include <csignal>
#include <optional>
#include <cstdlib>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;
using tbccl::ReduceOp;
using Clock = std::chrono::steady_clock;

namespace
{

    constexpr std::size_t kWorld = 4;
    constexpr auto kBound = std::chrono::seconds(8);

    bool has(const std::string &text, const std::string &needle) { return text.find(needle) != std::string::npos; }
    BufferView view(void *p, std::size_t bytes) { return BufferView{MemoryKind::Host, p, bytes, 0}; }
    double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

    // Waits for `w` and requires it to have failed, within the bound.
    void expect_failed(tbccl::Work &w, const std::string &label)
    {
        const auto t0 = Clock::now();
        w.wait();
        expect(w.has_error(), label + ": the Work must fail, it succeeded");
        expect(since(t0) < std::chrono::duration<double>(kBound).count(), label + ": left the blocked operation too slowly");
    }

    void expect_terminal(tbccl::Communicator &c, const std::string &label)
    {
        expect(c.aborted() && c.failed(), label + ": communicator is terminal");
        bool threw = false;
        try
        {
            c.barrier();
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        expect(threw, label + ": new operations are rejected");
    }

    // ---- forked world ---------------------------------------------------------------------------------------------------------------------------

    // Runs body(rank, comm) in one child process per rank. A child that returns normally exits 0; a thrown exception exits 1 (message on stderr);
    // the body itself may _exit() to simulate an abrupt death. Returns each child's exit status after at most `limit`.
    std::vector<int> run_forked(const std::function<void(std::size_t, tbccl::Communicator &)> &body, std::chrono::seconds limit = std::chrono::seconds(60))
    {
        const auto id = tbccl::CommunicatorId::generate();
        std::vector<std::shared_ptr<tbccl::CommunicatorListeners>> listeners(kWorld);
        tbccl::RankDirectory dir;
        for (std::size_t r = 0; r < kWorld; ++r)
        {
            tbccl::RankEndpoint e;
            e.rank = r;
            if (tbccl::rank_accepts_connections(r, kWorld))
            {
                listeners[r] = tbccl::CommunicatorListeners::bind("127.0.0.1");
                e.control = listeners[r]->control();
                e.data = listeners[r]->data();
            }
            dir.entries.push_back(e);
        }
        std::cout.flush();
        std::vector<pid_t> pids;
        for (std::size_t r = 0; r < kWorld; ++r)
        {
            const pid_t pid = fork();
            expect(pid >= 0, "fork failed");
            if (pid == 0)
            {
                int code = 0;
                try
                {
                    tbccl::CommunicatorOptions o;
                    o.rank = r;
                    o.world_size = kWorld;
                    o.communicator_id = id;
                    o.rank_directory = dir;
                    o.listeners = listeners[r];
                    o.bootstrap_timeout = std::chrono::seconds(15);
                    auto comm = tbccl::Communicator::create(o);
                    body(r, *comm);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "  child rank " << r << ": " << e.what() << "\n";
                    code = 1;
                }
                std::cerr.flush();
                _exit(code);
            }
            pids.push_back(pid);
        }
        listeners.clear();
        std::vector<int> status(kWorld, -1);
        const auto deadline = Clock::now() + limit;
        std::size_t done = 0;
        while (done < kWorld && Clock::now() < deadline)
        {
            for (std::size_t r = 0; r < kWorld; ++r)
            {
                if (status[r] != -1) continue;
                int st = 0;
                if (waitpid(pids[r], &st, WNOHANG) == pids[r])
                {
                    status[r] = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
                    ++done;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        for (std::size_t r = 0; r < kWorld; ++r)
        {
            if (status[r] == -1)
            {
                kill(pids[r], SIGKILL);
                waitpid(pids[r], nullptr, 0);
                status[r] = 999; // hung
            }
        }
        return status;
    }

    void expect_children_ok(const std::vector<int> &status, std::size_t dead_rank, const std::string &label)
    {
        for (std::size_t r = 0; r < kWorld; ++r)
        {
            if (r == dead_rank) continue;
            expect(status[r] == 0, label + ": healthy rank " + std::to_string(r) + " exit status " + std::to_string(status[r]) + (status[r] == 999 ? " (HUNG)" : ""));
        }
    }

    void test_rank_dies_during_barrier()
    {
        const auto status = run_forked([](std::size_t rank, tbccl::Communicator &comm) {
            if (rank == 2)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                _exit(0); // abrupt: no Goodbye, sockets closed by the kernel
            }
            const auto t0 = Clock::now();
            auto w = comm.barrier();
            expect_failed(w, "barrier");
            expect(has(w.error(), "rank 2") || has(w.error(), "aborted") || has(w.error(), "closed"), "the error names the failure: " + w.error());
            expect_terminal(comm, "barrier");
            expect(since(t0) < 8, "bounded");
        });
        expect_children_ok(status, 2, "rank 2 exits during barrier");
        std::cout << "[PASS] rank 2 exits during barrier: ranks 0, 1, 3 all leave it with an error\n";
    }

    void test_rank_dies_during_all_reduce()
    {
        const auto status = run_forked([](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::int32_t> data(16u << 20, static_cast<std::int32_t>(rank)); // 64 MiB
            if (rank == 2)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                _exit(0);
            }
            auto w = comm.all_reduce(view(data.data(), data.size() * 4), view(data.data(), data.size() * 4), data.size(), DataType::Int32, ReduceOp::Sum);
            expect_failed(w, "all_reduce");
            expect_terminal(comm, "all_reduce");
        });
        expect_children_ok(status, 2, "rank 2 exits during all_reduce");
        std::cout << "[PASS] rank 2 exits during all_reduce: ranks 0, 1, 3 all leave it with an error\n";
    }

    // Rank 2 is alive but never participates (sockets open, nothing read or sent). Rank 0 gives up and aborts; the abort must reach ranks 1 and 3,
    // which are blocked on rank 2 and know nothing of rank 0's decision.
    void test_rank_silent_during_p2p_ring()
    {
        const std::size_t bytes = 64u << 20;
        const auto status = run_forked([bytes](std::size_t rank, tbccl::Communicator &comm) {
            const std::size_t next = (rank + 1) % kWorld, prev = (rank + kWorld - 1) % kWorld;
            if (rank == 2)
            {
                std::this_thread::sleep_for(std::chrono::seconds(4)); // silent
                return;
            }
            std::vector<std::uint8_t> out(bytes, static_cast<std::uint8_t>(rank)), in(bytes, 0);
            auto s = comm.send(view(out.data(), bytes), bytes, DataType::UInt8, next);
            auto r = comm.recv(view(in.data(), bytes), bytes, DataType::UInt8, prev);
            if (rank == 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(600));
                comm.abort("rank 2 is silent");
                s.wait();
                r.wait();
                expect(comm.aborted(), "rank 0 is terminal");
                return;
            }
            // Rank 1 (its send goes to the silent rank 2) and rank 3 (its recv comes from it) cannot finish by themselves.
            const auto t0 = Clock::now();
            tbccl::Work &blocked = rank == 1 ? s : r;
            expect_failed(blocked, rank == 1 ? "send to silent rank 2" : "recv from silent rank 2");
            s.wait();
            r.wait(); // nothing may stay running against the buffers
            expect_terminal(comm, "silent ring");
            expect(has(blocked.error(), "rank 0") || has(blocked.error(), "rank 2") || has(blocked.error(), "aborted"), "the error says why: " + blocked.error());
            expect(since(t0) < 6, "bounded");
        });
        expect_children_ok(status, 2, "rank 2 silent during the P2P ring");
        std::cout << "[PASS] rank 2 silent during the P2P ring: rank 0's abort reaches ranks 1 and 3, all leave the blocked operation\n";
    }

    // ---- thread worlds ----------------------------------------------------------------------------------------------------------------------------

    void test_explicit_abort_with_several_works_outstanding()
    {
        const auto t_start = Clock::now();
        run_world(kWorld, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::uint8_t> a(1 << 20), b(1 << 20), c(1 << 20);
            std::vector<tbccl::Work> works;
            if (rank == 1)
            {
                works.push_back(comm.recv(view(a.data(), a.size()), a.size(), DataType::UInt8, 0));
                works.push_back(comm.recv(view(b.data(), b.size()), b.size(), DataType::UInt8, 2));
                works.push_back(comm.recv(view(c.data(), c.size()), c.size(), DataType::UInt8, 3));
                std::this_thread::sleep_for(std::chrono::milliseconds(300));
                comm.abort("explicit abort from rank 1");
                expect(comm.aborted() && comm.abort_reason() == "explicit abort from rank 1", "rank 1 keeps its own reason");
            }
            else
            {
                // Rank 1 never enters the barrier the others wait in. Rank 0 posts only the barrier: it is the coordinator, and a P2P recv from rank 1 would be a different thing.
                if (rank != 0) works.push_back(comm.recv(view(a.data(), a.size()), a.size(), DataType::UInt8, 1));
                works.push_back(comm.barrier());
            }
            for (std::size_t i = 0; i < works.size(); ++i)
            {
                works[i].wait();
                expect(works[i].has_error(), "rank " + std::to_string(rank) + " work " + std::to_string(i) + " must fail");
            }
            if (rank != 1)
            {
                // The first reason wins: either rank 1's own explanation (the Abort frame) or the transport error its closing sockets caused on a data lane, whichever a rank sees first.
                expect(has(comm.abort_reason(), "explicit abort from rank 1") || has(comm.abort_reason(), "closed") || has(comm.abort_reason(), "aborted"), "healthy ranks keep a reason: " + comm.abort_reason());
                expect_terminal(comm, "explicit abort");
            }
        });
        expect(since(t_start) < 10, "teardown is bounded");
        std::cout << "[PASS] rank 1 aborts with several Works outstanding: every rank's Works fail, all terminal, teardown bounded\n";
    }

    void test_destroy_with_outstanding_work()
    {
        const auto t_start = Clock::now();
        auto results = bootstrap(healthy_slots(kWorld), std::chrono::seconds(10));
        for (auto &r : results) expect(r.comm != nullptr, "bootstrap: " + r.error);
        std::vector<std::uint8_t> buf(1 << 20);
        std::vector<tbccl::Work> works;
        works.push_back(results[1].comm->recv(view(buf.data(), buf.size()), buf.size(), DataType::UInt8, 0));
        works.push_back(results[1].comm->barrier());
        results[1].comm.reset(); // destroyed while busy: aborts, does not wait for peers that will never come
        for (auto &w : works)
        {
            w.wait();
            expect(w.has_error(), "outstanding Work fails when its communicator is destroyed");
        }
        // The peers learn through the abort frame, not by timing out.
        const auto t0 = Clock::now();
        for (std::size_t r : {0u, 2u, 3u})
            while (!results[r].comm->aborted() && since(t0) < 5) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        for (std::size_t r : {0u, 2u, 3u}) expect(results[r].comm->aborted(), "rank " + std::to_string(r) + " learned about the destroyed communicator");
        results.clear();
        expect(since(t_start) < 10, "teardown is bounded");
        std::cout << "[PASS] communicator destroyed with outstanding work: Works fail, peers abort, teardown bounded\n";
    }

    // A rank that finishes and is destroyed cleanly must not be mistaken for a failure by its idle peers.
    void test_clean_departure_is_not_an_abort()
    {
        auto results = bootstrap(healthy_slots(kWorld), std::chrono::seconds(10));
        for (auto &r : results) expect(r.comm != nullptr, "bootstrap: " + r.error);
        results[3].comm.reset(); // idle, healthy: sends Goodbye
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        for (std::size_t r = 0; r < 3; ++r) expect(!results[r].comm->aborted() && !results[r].comm->failed(), "rank " + std::to_string(r) + " stays healthy after a clean departure");
        std::cout << "[PASS] a clean destruction (Goodbye) does not abort the remaining ranks\n";
    }

    // Regression (found by torch-tbccl's N=2 all_gather test): a rank whose Work just completed and which destroys its communicator at once must NOT be
    // treated as "destroyed with outstanding operations" (that sends Abort to the peers, who may still be receiving its last bytes). Every rank rotates
    // through being the early finisher, for every collective, at world_size 2, 3 and 4.
    void test_early_finisher_never_aborts_a_peer()
    {
        for (std::size_t world : {std::size_t{2}, std::size_t{3}, std::size_t{4}})
        {
            for (int kind = 0; kind < 3; ++kind)
            {
                for (std::size_t early = 0; early < world; ++early)
                {
                    for (int rep = 0; rep < 60; ++rep)
                    {
                        auto results = bootstrap(healthy_slots(world), std::chrono::seconds(10));
                        for (auto &r : results) expect(r.comm != nullptr, "bootstrap: " + r.error);
                        std::vector<std::string> errors(world);
                        std::vector<std::thread> threads;
                        for (std::size_t r = 0; r < world; ++r)
                        {
                            threads.emplace_back([&, r] {
                                try
                                {
                                    const std::size_t bytes = 4u << 20;
                                    std::vector<std::uint8_t> buf(bytes, static_cast<std::uint8_t>(r)), out(bytes);
                                    std::vector<std::vector<std::uint8_t>> gathered(world, std::vector<std::uint8_t>(bytes));
                                    auto &comm = *results[r].comm;
                                    std::optional<tbccl::Work> wo;
                                    if (kind == 0)
                                    {
                                        wo.emplace(comm.broadcast(view(buf.data(), bytes), 0));
                                    }
                                    else if (kind == 1)
                                    {
                                        std::vector<BufferView> outs;
                                        for (auto &g : gathered) outs.push_back(view(g.data(), bytes));
                                        wo.emplace(comm.all_gather(view(buf.data(), bytes), outs));
                                    }
                                    else
                                    {
                                        std::vector<std::int32_t> v(bytes / 4, 1);
                                        wo.emplace(comm.all_reduce(view(v.data(), bytes), view(v.data(), bytes), v.size(), DataType::Int32, ReduceOp::Sum));
                                        auto &w = *wo;
                                        w.wait();
                                        if (r == early) results[r].comm.reset();
                                        else std::this_thread::sleep_for(std::chrono::milliseconds(20));
                                        expect(!w.has_error(), "all_reduce failed: " + w.error());
                                        expect(v[0] == static_cast<std::int32_t>(world) && v.back() == static_cast<std::int32_t>(world), "all_reduce result");
                                        return;
                                    }
                                    auto &w = *wo;
                                    w.wait();
                                    if (r == early) results[r].comm.reset(); // destroyed the moment its own Work completed
                                    else std::this_thread::sleep_for(std::chrono::milliseconds(20));
                                    expect(!w.has_error(), std::string("collective failed on rank ") + std::to_string(r) + ": " + w.error());
                                    if (kind == 0) expect(buf[0] == 0 && buf[bytes - 1] == 0, "broadcast payload");
                                    else
                                        for (std::size_t q = 0; q < world; ++q) expect(gathered[q][0] == q && gathered[q][bytes - 1] == q, "all_gather payload");
                                }
                                catch (const std::exception &e)
                                {
                                    errors[r] = e.what();
                                }
                            });
                        }
                        for (auto &t : threads) t.join();
                        for (std::size_t r = 0; r < world; ++r)
                            expect(errors[r].empty(), "world " + std::to_string(world) + " kind " + std::to_string(kind) + " early rank " + std::to_string(early) + " rep " + std::to_string(rep) + ": rank " + std::to_string(r) + ": " + errors[r]);
                    }
                }
            }
        }
        std::cout << "[PASS] a rank that destroys its communicator right after its own Work completed never aborts a peer (broadcast, all_gather, all_reduce; world_size 2, 3, 4)\n";
    }

    void test_repeated_abort_loop()
    {
        for (int i = 0; i < 15; ++i)
        {
            auto results = bootstrap(healthy_slots(kWorld), std::chrono::seconds(10));
            for (auto &r : results) expect(r.comm != nullptr, "bootstrap: " + r.error);
            std::vector<std::uint8_t> buf(4096);
            auto w = results[0].comm->recv(view(buf.data(), buf.size()), buf.size(), DataType::UInt8, 1);
            if (i % 2) results[static_cast<std::size_t>(i) % kWorld].comm->abort("loop");
            results.clear(); // destroy all, aborted or busy or healthy
            w.wait();
        }
        std::cout << "[PASS] 15 create / abort / destroy cycles at world_size 4 without a hang\n";
    }

} // namespace

int main()
{
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(300));
        std::cerr << "[FAIL] watchdog: a failure test hung\n";
        std::_Exit(2);
    }).detach();
    try
    {
        test_clean_departure_is_not_an_abort();
        test_early_finisher_never_aborts_a_peer();
        test_explicit_abort_with_several_works_outstanding();
        test_destroy_with_outstanding_work();
        test_repeated_abort_loop();
        test_rank_dies_during_barrier();
        test_rank_dies_during_all_reduce();
        test_rank_silent_during_p2p_ring();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All N-rank failure tests passed.\n";
    return 0;
}
