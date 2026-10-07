// Local (loopback, one process, one thread per rank) A/B benchmark of the N>2 collective algorithms, forced one at a time through the TBCCL_*_ALGORITHM
// debug overrides. Diagnostic only: loopback numbers drive a generic selector heuristic, they are NOT Thunderbolt measurements.
//
// Method: ranks are threads released together by a spinning barrier before every iteration (so start times are aligned and the iteration latency is the MAX over ranks of
// start-to-completion); warm-up is untimed; nothing is verified inside the timed loop (a verified round follows, untimed); algorithms are interleaved across repetitions
// so slow drift hits all of them equally. One JSON line per (collective, world, algorithm, bytes, repetition) with the median/p25/p75 iteration latency in microseconds.
//
// Usage: communicator_algo_bench [--worlds 3,4,8] [--reps 5] [--colls barrier,broadcast,all_gather,all_reduce] [--max-bytes 16777216] [--min-bytes 64]

#include <tbccl/communicator.hpp>

#include "../src/core/communicator_debug.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using namespace tbccl;

namespace
{
    class SpinBarrier
    {
    public:
        explicit SpinBarrier(std::size_t n) : n_(n) {}
        void wait()
        {
            const std::size_t gen = generation_.load(std::memory_order_acquire);
            if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_)
            {
                arrived_.store(0, std::memory_order_relaxed);
                generation_.fetch_add(1, std::memory_order_release);
            }
            else
            {
                while (generation_.load(std::memory_order_acquire) == gen) std::this_thread::yield();
            }
        }

    private:
        std::size_t n_;
        std::atomic<std::size_t> arrived_{0}, generation_{0};
    };

    std::vector<std::string> split(const std::string &s)
    {
        std::vector<std::string> out;
        std::stringstream ss(s);
        std::string item;
        while (std::getline(ss, item, ',')) out.push_back(item);
        return out;
    }

    struct Case
    {
        std::string collective;
        std::vector<std::string> algorithms; // values of the override variable; "" = the planner default
        const char *variable;
    };

    bool is_pow2(std::size_t n) { return n && !(n & (n - 1)); }

    void run_case(std::size_t world, const std::string &collective, const char *variable, const std::string &algorithm, std::size_t bytes, int rep)
    {
        if (algorithm.empty()) unsetenv(variable);
        else setenv(variable, algorithm.c_str(), 1);

        std::vector<std::shared_ptr<CommunicatorListeners>> ls(world);
        RankDirectory dir;
        for (std::size_t r = 0; r < world; ++r)
        {
            RankEndpoint e;
            e.rank = r;
            if (rank_accepts_connections(r, world))
            {
                ls[r] = CommunicatorListeners::bind("127.0.0.1");
                e.control = ls[r]->control();
                e.data = ls[r]->data();
            }
            dir.entries.push_back(e);
        }
        const auto id = CommunicatorId::generate();
        std::vector<std::unique_ptr<Communicator>> comms(world);
        {
            std::vector<std::thread> init;
            for (std::size_t r = 0; r < world; ++r)
                init.emplace_back([&, r] {
                    CommunicatorOptions o;
                    o.rank = r;
                    o.world_size = world;
                    o.communicator_id = id;
                    o.rank_directory = dir;
                    o.listeners = ls[r];
                    comms[r] = Communicator::create(o);
                });
            for (auto &t : init) t.join();
        }

        const int warmup = bytes <= (64u << 10) ? 30 : 8;
        const int iters = bytes <= (64u << 10) ? 300 : bytes <= (1u << 20) ? 100 : bytes <= (4u << 20) ? 30 : 12;
        SpinBarrier barrier(world);
        std::vector<std::vector<double>> durations(world, std::vector<double>(iters, 0.0));
        std::atomic<int> failures{0};

        std::vector<std::thread> threads;
        for (std::size_t r = 0; r < world; ++r)
        {
            threads.emplace_back([&, r] {
                Communicator &comm = *comms[r];
                const std::size_t count = std::max<std::size_t>(1, bytes / 4);
                std::vector<float> buf(count, 1.0f);
                std::vector<std::vector<float>> gathered;
                std::vector<BufferView> outs;
                if (collective == "all_gather")
                {
                    gathered.assign(world, std::vector<float>(count, 0.0f));
                    for (auto &g : gathered) outs.push_back({MemoryKind::Host, g.data(), count * 4, 0});
                }
                auto op = [&]() -> Work {
                    BufferView v{MemoryKind::Host, buf.data(), count * 4, 0};
                    if (collective == "barrier") return comm.barrier();
                    if (collective == "broadcast") return comm.broadcast(v, 0);
                    if (collective == "all_gather") return comm.all_gather(v, outs);
                    return comm.all_reduce(v, v, count, DataType::Float32, ReduceOp::Sum);
                };
                for (int i = 0; i < warmup + iters; ++i)
                {
                    std::fill(buf.begin(), buf.end(), 1.0f); // untimed refill: all_reduce values stay bounded
                    barrier.wait();
                    const auto t0 = Clock::now();
                    Work w = op();
                    w.wait();
                    const auto t1 = Clock::now();
                    if (w.has_error()) failures.fetch_add(1);
                    if (i >= warmup) durations[r][i - warmup] = std::chrono::duration<double, std::micro>(t1 - t0).count();
                }
            });
        }
        for (auto &t : threads) t.join();
        if (failures.load() != 0)
        {
            std::fprintf(stderr, "operation failed: %s %s world=%zu bytes=%zu\n", collective.c_str(), algorithm.c_str(), world, bytes);
            std::exit(1);
        }
        std::vector<double> iteration(iters);
        for (int i = 0; i < iters; ++i)
        {
            double worst = 0;
            for (std::size_t r = 0; r < world; ++r) worst = std::max(worst, durations[r][i]);
            iteration[i] = worst;
        }
        std::sort(iteration.begin(), iteration.end());
        std::printf(
            "{\"collective\":\"%s\",\"world\":%zu,\"algorithm\":\"%s\",\"bytes\":%zu,\"rep\":%d,\"median_us\":%.2f,\"p25_us\":%.2f,\"p75_us\":%.2f,\"n\":%d}\n", collective.c_str(), world,
            algorithm.empty() ? "default" : algorithm.c_str(), bytes, rep, iteration[iters / 2], iteration[iters / 4], iteration[3 * iters / 4], iters);
        std::fflush(stdout);
    }
} // namespace

int main(int argc, char **argv)
{
    std::vector<std::size_t> worlds = {3, 4, 8};
    std::vector<std::string> colls = {"barrier", "broadcast", "all_gather", "all_reduce"};
    int reps = 5;
    std::size_t min_bytes = 64, max_bytes = 16u << 20;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--worlds") { worlds.clear(); for (auto &w : split(argv[i + 1])) worlds.push_back(std::stoul(w)); }
        else if (a == "--colls") colls = split(argv[i + 1]);
        else if (a == "--reps") reps = std::atoi(argv[i + 1]);
        else if (a == "--min-bytes") min_bytes = std::stoul(argv[i + 1]);
        else if (a == "--max-bytes") max_bytes = std::stoul(argv[i + 1]);
    }
    for (std::size_t world : worlds)
    {
        for (const auto &coll : colls)
        {
            const char *variable = coll == "barrier" ? "TBCCL_BARRIER_ALGORITHM" : coll == "broadcast" ? "TBCCL_BROADCAST_ALGORITHM" : coll == "all_gather" ? "TBCCL_ALLGATHER_ALGORITHM" : "TBCCL_ALLREDUCE_ALGORITHM";
            std::vector<std::string> algos;
            if (coll == "barrier") algos = {"reference", "dissemination"};
            else if (coll == "broadcast") algos = {"reference", "tree"};
            else if (coll == "all_gather") algos = {"reference", "ring"};
            else
            {
                algos = {"reference", "tree", "ring"};
                if (is_pow2(world)) algos.push_back("recursive");
            }
            std::vector<std::size_t> sizes;
            if (coll == "barrier") sizes = {0};
            else for (std::size_t b = min_bytes; b <= max_bytes; b *= 4) sizes.push_back(b);
            for (std::size_t bytes : sizes)
                for (int rep = 0; rep < reps; ++rep)
                    for (const auto &algo : algos) run_case(world, coll, variable, algo, bytes, rep);
        }
    }
    return 0;
}
