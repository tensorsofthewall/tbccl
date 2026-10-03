// Failure behaviour of the optimized algorithms. A rank that dies (forked processes, abrupt exit, no Goodbye) or aborts while a ring / tree / dissemination collective is
// running must take the communicator down as a unit: every healthy rank leaves the blocked operation with an error, no Work becomes terminal while a transport thread can
// still touch its buffer (the buffers are freed right after the Work: ASan/TSan would flag any later access), and teardown is bounded.

#include "mesh_fork_support.hpp"
#include "collective_topology.hpp"

#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;
using tbccl::ReduceOp;

namespace
{
    BufferView view(void *p, std::size_t bytes) { return BufferView{MemoryKind::Host, p, bytes, 0}; }
    double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

    struct ForceEnv
    {
        std::string var;
        ForceEnv(const char *v, const char *value) : var(v) { setenv(v, value, 1); }
        ~ForceEnv() { unsetenv(var.c_str()); }
    };

    void expect_failed(tbccl::Work &w, const std::string &label)
    {
        const auto t0 = Clock::now();
        w.wait();
        expect(w.has_error(), label + ": the Work must fail, it succeeded");
        expect(since(t0) < 10, label + ": left the blocked operation too slowly");
    }

    // The dying rank participates in the collective (so it is "mid-algorithm") and a helper thread kills the whole process after `delay_ms`.
    void die_after(int delay_ms)
    {
        std::thread([delay_ms] {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            _exit(0);
        }).detach();
    }

    void test_ring_all_reduce_rank_dies(int delay_ms, const char *phase)
    {
        ForceEnv env("TBCCL_ALLREDUCE_ALGORITHM", "ring");
        const std::size_t world = 4, bytes = std::size_t{256} << 20;
        const auto status = run_forked(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<float> data(bytes / 4, static_cast<float>(rank));
            if (rank == 2) die_after(delay_ms);
            auto w = comm.all_reduce(view(data.data(), bytes), view(data.data(), bytes), data.size(), DataType::Float32, ReduceOp::Sum);
            if (rank == 2) { w.wait(); return; } // only reached if it somehow finished before dying
            expect_failed(w, "ring all_reduce");
            expect(comm.failed() && comm.aborted(), "terminal");
        }, std::chrono::seconds(90));
        expect_children_ok(status, world, 2, std::string("rank 2 dies during the ring all_reduce (") + phase + ")");
        std::cout << "[PASS] rank 2 dies " << delay_ms << " ms into a 256 MiB ring all_reduce (" << phase << "): every healthy rank leaves with an error\n";
    }

    void test_tree_rank_dies(const char *collective, std::size_t victim)
    {
        const std::size_t world = 8, bytes = std::size_t{256} << 20;
        ForceEnv env(std::string(collective) == "broadcast" ? "TBCCL_BROADCAST_ALGORITHM" : "TBCCL_ALLREDUCE_ALGORITHM", "tree");
        const auto status = run_forked(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<float> data(bytes / 4, static_cast<float>(rank));
            if (rank == victim) die_after(4);
            std::optional<tbccl::Work> posted;
            try
            {
                posted.emplace(std::string(collective) == "broadcast" ? comm.broadcast(view(data.data(), bytes), 0)
                                                                      : comm.all_reduce(view(data.data(), bytes), view(data.data(), bytes), data.size(), DataType::Float32, ReduceOp::Sum));
            }
            catch (const std::runtime_error &e)
            {
                // The victim may already be gone when this rank gets to submit: the communicator is then terminal and the call throws.
                expect(rank != victim && comm.failed(), std::string("unexpected submission failure: ") + e.what());
                return;
            }
            tbccl::Work &w = *posted;
            if (rank == victim) { w.wait(); return; }
            const bool broadcast = std::string(collective) == "broadcast";
            // Ranks upstream of the dead node may legitimately finish a broadcast (the root has no reason to fail once its sends are delivered); the dead node's descendants
            // cannot, and EVERY healthy rank's communicator must become terminal (the Abort frame reaches it whether or not its own Work was affected).
            const bool downstream = broadcast && tbccl::detail::tree_parent(rank, 0, world) == victim;
            if (!broadcast || downstream) expect_failed(w, std::string("tree ") + collective);
            else w.wait();
            const auto t0 = Clock::now();
            while (!comm.aborted() && since(t0) < 8) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            expect(comm.aborted() && comm.failed(), "terminal");
        }, std::chrono::seconds(90));
        expect_children_ok(status, world, victim, std::string("rank ") + std::to_string(victim) + " dies during the tree " + collective);
        std::cout << "[PASS] internal tree node (rank " << victim << ", parent of two ranks at N=8) dies 4 ms into a 256 MiB tree " << collective << ": every healthy rank leaves with an error\n";
    }

    void test_dissemination_rank_dies()
    {
        ForceEnv env("TBCCL_BARRIER_ALGORITHM", "dissemination");
        const std::size_t world = 5;
        const auto status = run_forked(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            if (rank == 3) die_after(80);
            const auto t0 = Clock::now();
            for (;;)
            {
                auto w = comm.barrier();
                w.wait();
                if (w.has_error()) break; // the death was noticed
                if (rank != 3 && since(t0) > 20) throw std::runtime_error("the barrier loop was never interrupted"); // rank 3 just keeps looping until it is killed
            }
            expect(comm.failed(), "terminal");
        }, std::chrono::seconds(60));
        expect_children_ok(status, world, 3, "rank 3 dies in a barrier loop");
        std::cout << "[PASS] rank 3 dies in the middle of a dissemination-barrier loop at N=5: every healthy rank leaves with an error\n";
    }

    // Explicit abort while a ring / tree / ring-all_gather operation is active (threads, so ASan/TSan watch the buffers): every Work fails, no later access to the freed buffers.
    void test_abort_during(const char *label, const char *variable, const char *algorithm, const std::function<tbccl::Work(tbccl::Communicator &, std::size_t rank, std::vector<std::uint8_t> &, std::vector<std::vector<std::uint8_t>> &)> &start)
    {
        ForceEnv env(variable, algorithm);
        for (std::size_t world : {std::size_t{3}, std::size_t{4}, std::size_t{5}})
        {
            const auto t0 = Clock::now();
            Latch allocated(world);
            run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
                {
                    std::vector<std::uint8_t> buf(std::size_t{96} << 20, static_cast<std::uint8_t>(rank));
                    std::vector<std::vector<std::uint8_t>> extra;
                    allocated.arrive_and_wait(); // everyone has its buffers: the operation is posted at (about) the same time on every rank
                    auto w = start(comm, rank, buf, extra);
                    if (rank == world / 2)
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(8));
                        comm.abort("explicit abort during the " + std::string(label));
                    }
                    w.wait();
                    expect(w.has_error(), "rank " + std::to_string(rank) + ": the Work must fail");
                    // buf and extra are destroyed here: TBCCL must not touch them any more
                }
                expect(comm.aborted(), "terminal");
            });
            expect(since(t0) < 20, "bounded");
        }
        std::cout << "[PASS] explicit abort during " << label << " (world_size 3, 4, 5): every Work fails, bounded, buffers released while TBCCL is quiescent\n";
    }

    void test_repeated_abort_destroy_loop()
    {
        ForceEnv env("TBCCL_ALLREDUCE_ALGORITHM", "ring");
        std::mt19937 gen(5);
        for (int i = 0; i < 20; ++i)
        {
            auto results = bootstrap(healthy_slots(4), std::chrono::seconds(10));
            for (auto &r : results) expect(r.comm != nullptr, "bootstrap: " + r.error);
            std::vector<std::vector<float>> data(4, std::vector<float>(std::size_t{8} << 20, 1.0f));
            std::vector<tbccl::Work> works;
            for (std::size_t r = 0; r < 4; ++r)
                works.push_back(results[r].comm->all_reduce(view(data[r].data(), data[r].size() * 4), view(data[r].data(), data[r].size() * 4), data[r].size(), DataType::Float32, ReduceOp::Sum));
            std::this_thread::sleep_for(std::chrono::microseconds(gen() % 4000));
            if (i % 2) results[gen() % 4].comm->abort("loop");
            results.clear(); // destroy every communicator: aborted, busy, or already finished
            for (auto &w : works) w.wait();
        }
        std::cout << "[PASS] 20 ring all_reduce start / abort / destroy cycles at world_size 4 without a hang\n";
    }
} // namespace

int main()
{
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(600));
        std::cerr << "[FAIL] watchdog: an algorithm failure test hung\n";
        std::_Exit(2);
    }).detach();
    try
    {
        test_ring_all_reduce_rank_dies(15, "during reduce-scatter");
        test_ring_all_reduce_rank_dies(150, "late: reduce-scatter / all-gather");
        test_tree_rank_dies("broadcast", 4);
        test_tree_rank_dies("all_reduce", 4);
        test_dissemination_rank_dies();
        const std::size_t kLarge = std::size_t{96} << 20;
        test_abort_during("a ring all_reduce", "TBCCL_ALLREDUCE_ALGORITHM", "ring", [&](tbccl::Communicator &c, std::size_t, std::vector<std::uint8_t> &buf, std::vector<std::vector<std::uint8_t>> &) {
            return c.all_reduce(view(buf.data(), kLarge), view(buf.data(), kLarge), kLarge, DataType::UInt8, ReduceOp::Sum);
        });
        test_abort_during("a tree all_reduce", "TBCCL_ALLREDUCE_ALGORITHM", "tree", [&](tbccl::Communicator &c, std::size_t, std::vector<std::uint8_t> &buf, std::vector<std::vector<std::uint8_t>> &) {
            return c.all_reduce(view(buf.data(), kLarge), view(buf.data(), kLarge), kLarge, DataType::UInt8, ReduceOp::Sum);
        });
        test_abort_during("a tree broadcast", "TBCCL_BROADCAST_ALGORITHM", "tree", [&](tbccl::Communicator &c, std::size_t, std::vector<std::uint8_t> &buf, std::vector<std::vector<std::uint8_t>> &) {
            return c.broadcast(view(buf.data(), kLarge), 0);
        });
        test_abort_during("a ring all_gather", "TBCCL_ALLGATHER_ALGORITHM", "ring", [&](tbccl::Communicator &c, std::size_t rank, std::vector<std::uint8_t> &buf, std::vector<std::vector<std::uint8_t>> &extra) {
            (void)rank;
            extra.assign(c.world_size(), std::vector<std::uint8_t>(std::size_t{24} << 20));
            std::vector<BufferView> outs;
            for (auto &o : extra) outs.push_back(view(o.data(), o.size()));
            return c.all_gather(view(buf.data(), std::size_t{24} << 20), outs);
        });
        test_repeated_abort_destroy_loop();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All algorithm failure tests passed.\n";
    return 0;
}
