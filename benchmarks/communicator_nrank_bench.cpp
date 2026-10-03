// Local (loopback, one process, one thread per rank) diagnostic of the REFERENCE N-rank collectives: communicator initialization time and
// per-collective latency for world_size 2, 3 and 4. Diagnostic only: the reference algorithms are not optimized (the N>2 collective-selection work)
// and nothing here is a network measurement. Usage: communicator_nrank_bench [iters]

#include <tbccl/communicator.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using namespace tbccl;

static double us_since(Clock::time_point t) { return std::chrono::duration<double, std::micro>(Clock::now() - t).count(); }

int main(int argc, char **argv)
{
    const int iters = argc > 1 ? std::atoi(argv[1]) : 300;
    for (std::size_t world = 2; world <= 4; ++world)
    {
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
        const auto t0 = Clock::now();
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
        std::printf("{\"world\":%zu,\"kind\":\"init\",\"us\":%.0f}\n", world, us_since(t0));

        auto run = [&](const char *kind, std::size_t bytes, const std::function<Work(std::size_t, std::vector<float> &)> &op) {
            std::vector<double> samples;
            std::vector<std::thread> threads;
            std::atomic<int> turn{0};
            for (std::size_t r = 0; r < world; ++r)
                threads.emplace_back([&, r] {
                    std::vector<float> buf(std::max<std::size_t>(1, bytes / 4), 1.0f);
                    for (int i = 0; i < 20 + iters; ++i)
                    {
                        const auto t = Clock::now();
                        op(r, buf).wait();
                        if (r == 0 && i >= 20) samples.push_back(us_since(t));
                        std::fill(buf.begin(), buf.end(), 1.0f);
                    }
                });
            for (auto &t : threads) t.join();
            std::sort(samples.begin(), samples.end());
            std::printf("{\"world\":%zu,\"kind\":\"%s\",\"bytes\":%zu,\"median_us\":%.1f,\"p25_us\":%.1f,\"p75_us\":%.1f}\n", world, kind, bytes, samples[samples.size() / 2],
                        samples[samples.size() / 4], samples[3 * samples.size() / 4]);
        };
        run("barrier", 0, [&](std::size_t r, std::vector<float> &) { return comms[r]->barrier(); });
        for (std::size_t bytes : {std::size_t{4096}, std::size_t{65536}, std::size_t{1048576}})
        {
            run("broadcast", bytes, [&](std::size_t r, std::vector<float> &b) { return comms[r]->broadcast({MemoryKind::Host, b.data(), b.size() * 4, 0}, 0); });
            run("all_reduce", bytes, [&](std::size_t r, std::vector<float> &b) {
                BufferView v{MemoryKind::Host, b.data(), b.size() * 4, 0};
                return comms[r]->all_reduce(v, v, b.size(), DataType::Float32, ReduceOp::Sum);
            });
        }
        std::vector<std::vector<float>> gathered(world);
        run("all_gather", 4096, [&](std::size_t r, std::vector<float> &b) {
            static thread_local std::vector<std::vector<float>> out;
            out.assign(world, std::vector<float>(b.size()));
            std::vector<BufferView> outs;
            for (auto &o : out) outs.push_back({MemoryKind::Host, o.data(), o.size() * 4, 0});
            return comms[r]->all_gather({MemoryKind::Host, b.data(), b.size() * 4, 0}, outs);
        });
    }
    return 0;
}
