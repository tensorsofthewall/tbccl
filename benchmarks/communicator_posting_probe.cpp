// the N>2 collective-selection work probe (grouped-operations audit, docs/grouped_operations_audit.md): can a rank POST an arbitrary pattern of asynchronous sends and receives
// without a grouped-call API? Two ranks each post K sends of `bytes` bytes to the other BEFORE posting any receive, then K receives, then wait for everything (the classic
// Isend...Irecv pattern that deadlocks when a post blocks). Prints one line: completed or "BLOCKED" (a watchdog fires). Usage: communicator_posting_probe <K> <bytes> [world]

#include <tbccl/communicator.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <thread>
#include <vector>

using namespace tbccl;

int main(int argc, char **argv)
{
    const int k = std::atoi(argv[1]);
    const std::size_t bytes = std::stoul(argv[2]);
    const std::size_t world = argc > 3 ? std::stoul(argv[3]) : 2;
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
    std::atomic<int> done{0};
    std::vector<std::thread> threads;
    for (std::size_t r = 0; r < 2; ++r)
        threads.emplace_back([&, r] {
            const std::size_t peer = 1 - r;
            std::vector<std::vector<unsigned char>> out(k, std::vector<unsigned char>(bytes, 1)), in(k, std::vector<unsigned char>(bytes, 0));
            std::vector<Work> works;
            for (int i = 0; i < k; ++i) works.push_back(comms[r]->send({MemoryKind::Host, out[i].data(), bytes, 0}, bytes, DataType::UInt8, peer));
            for (int i = 0; i < k; ++i) works.push_back(comms[r]->recv({MemoryKind::Host, in[i].data(), bytes, 0}, bytes, DataType::UInt8, peer));
            for (auto &w : works) w.wait();
            done.fetch_add(1);
        });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(12);
    while (done.load() < 2 && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (done.load() == 2)
    {
        for (auto &t : threads) t.join();
        std::printf("K=%d bytes=%zu: completed\n", k, bytes);
        return 0;
    }
    std::printf("K=%d bytes=%zu: BLOCKED (a post or a wait never returned within 12 s)\n", k, bytes);
    std::fflush(stdout);
    std::_Exit(3);
}
