#pragma once

// multi-rank Communicator test scaffolding. Every rank is a thread of this process with its own Communicator over real loopback sockets.
// Ports are never hard-coded: each rank pre-binds CommunicatorListeners on port 0 (the kernel picks), publishes the actual control/data
// endpoints into the shared directory, and only then calls Communicator::create(). Tests built on this need no ctest port lock.

#include <tbccl/communicator.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <dirent.h>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace mesh_test
{

    inline void expect(bool condition, const std::string &message)
    {
        if (!condition) throw std::runtime_error("assertion failed: " + message);
    }

    // Counts this process's open file descriptors (Linux/macOS), to prove sockets are closed exactly once and never leak.
    inline int open_fd_count()
    {
        int n = 0;
#ifdef __APPLE__
        const char *dir = "/dev/fd";
#else
        const char *dir = "/proc/self/fd";
#endif
        if (DIR *d = opendir(dir))
        {
            while (readdir(d) != nullptr) ++n;
            closedir(d);
        }
        return n;
    }

    class Latch
    {
    public:
        explicit Latch(std::size_t count) : count_(count) {}
        void arrive_and_wait()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (--count_ == 0) cv_.notify_all();
            else cv_.wait(lock, [&] { return count_ == 0; });
        }

    private:
        std::mutex mutex_;
        std::condition_variable cv_;
        std::size_t count_;
    };

    // What one thread claims about itself. For a healthy world rank == index, world == n and the id is shared.
    struct Slot
    {
        std::size_t rank = 0;
        std::size_t world = 0;
        tbccl::CommunicatorId id;
    };

    struct RankResult
    {
        std::unique_ptr<tbccl::Communicator> comm;
        std::string error; // empty on success
    };

    // Bootstraps one Communicator per slot concurrently. A slot that fails reports its error text. The directory is indexed by claimed rank
    // (a duplicate claim overwrites the earlier endpoint); ranks nobody claims get an unreachable placeholder endpoint.
    inline std::vector<RankResult> bootstrap(const std::vector<Slot> &slots, std::chrono::milliseconds timeout = std::chrono::seconds(10))
    {
        std::size_t max_world = 0;
        for (const auto &s : slots) max_world = std::max(max_world, s.world);
        std::vector<std::shared_ptr<tbccl::CommunicatorListeners>> listeners(slots.size());
        std::vector<tbccl::RankDirectory> directories(slots.size());
        // Per-slot directories: slot i sees its own world size, and the endpoints published by slots claiming each rank.
        std::vector<tbccl::RankEndpoint> published(max_world);
        std::vector<bool> have(max_world, false);
        for (std::size_t i = 0; i < slots.size(); ++i)
        {
            const auto &s = slots[i];
            if (tbccl::rank_accepts_connections(s.rank, s.world))
            {
                listeners[i] = tbccl::CommunicatorListeners::bind("127.0.0.1");
                published[s.rank] = {s.rank, listeners[i]->control(), listeners[i]->data()};
                have[s.rank] = true;
            }
        }
        for (std::size_t i = 0; i < slots.size(); ++i)
        {
            const auto &s = slots[i];
            for (std::size_t r = 0; r < s.world; ++r)
            {
                tbccl::RankEndpoint e;
                if (r < max_world && have[r]) e = published[r];
                else e = {r, {"127.0.0.1", static_cast<std::uint16_t>(r + 1)}, {"127.0.0.1", static_cast<std::uint16_t>(r + 100)}};
                e.rank = r;
                if (!tbccl::rank_accepts_connections(r, s.world)) e = {r, {}, {}};
                directories[i].entries.push_back(e);
            }
        }
        std::vector<RankResult> results(slots.size());
        std::vector<std::thread> threads;
        for (std::size_t i = 0; i < slots.size(); ++i)
        {
            threads.emplace_back([&, i] {
                tbccl::CommunicatorOptions o;
                o.rank = slots[i].rank;
                o.world_size = slots[i].world;
                o.communicator_id = slots[i].id;
                o.rank_directory = directories[i];
                o.listeners = listeners[i];
                o.bootstrap_timeout = timeout;
                try
                {
                    results[i].comm = tbccl::Communicator::create(o);
                }
                catch (const std::exception &e)
                {
                    results[i].error = e.what();
                }
            });
        }
        for (auto &t : threads) t.join();
        return results;
    }

    inline std::vector<Slot> healthy_slots(std::size_t world, const tbccl::CommunicatorId &id = tbccl::CommunicatorId::generate())
    {
        std::vector<Slot> slots;
        for (std::size_t r = 0; r < world; ++r) slots.push_back({r, world, id});
        return slots;
    }

    // Runs `body(rank, comm)` on one thread per rank of a healthy world, then destroys every communicator after ALL bodies finished (so an
    // early finisher never tears down a socket a peer still uses). Rethrows the first failure after joining everything.
    inline void run_world(std::size_t world, const std::function<void(std::size_t, tbccl::Communicator &)> &body, std::chrono::milliseconds timeout = std::chrono::seconds(20))
    {
        auto results = bootstrap(healthy_slots(world), timeout);
        for (std::size_t r = 0; r < world; ++r) expect(results[r].comm != nullptr, "rank " + std::to_string(r) + " bootstrap failed: " + results[r].error);
        Latch done(world);
        std::vector<std::string> errors(world);
        std::vector<std::thread> threads;
        for (std::size_t r = 0; r < world; ++r)
        {
            threads.emplace_back([&, r] {
                try
                {
                    body(r, *results[r].comm);
                }
                catch (const std::exception &e)
                {
                    errors[r] = e.what();
                }
                done.arrive_and_wait();
            });
        }
        for (auto &t : threads) t.join();
        results.clear();
        for (std::size_t r = 0; r < world; ++r)
            if (!errors[r].empty()) throw std::runtime_error("rank " + std::to_string(r) + ": " + errors[r]);
    }

} // namespace mesh_test
