#pragma once

// Phase 50/51: forked-process rank harness for failure tests. One child process per rank (the sockets of an abruptly dying rank are closed by the kernel, with no Goodbye);
// the parent pre-binds every rank's listeners so ports stay dynamic. A child returns normally -> exit 0; a thrown exception -> exit 1 (message on stderr); the body may _exit().

#include "mesh_test_support.hpp"

#include <tbccl/tcp.hpp>

#include <csignal>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

namespace mesh_test
{
    using Clock = std::chrono::steady_clock;


// Runs body(rank, comm) in one child process per rank. A child that returns normally exits 0; a thrown exception exits 1 (message on stderr);
// the body itself may _exit() to simulate an abrupt death. Returns each child's exit status after at most `limit`.
inline std::vector<int> run_forked(std::size_t kWorld, const std::function<void(std::size_t, tbccl::Communicator &)> &body, std::chrono::seconds limit = std::chrono::seconds(60))
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

inline void expect_children_ok(const std::vector<int> &status, std::size_t kWorld, std::size_t dead_rank, const std::string &label)
{
    for (std::size_t r = 0; r < kWorld; ++r)
    {
        if (r == dead_rank) continue;
        expect(status[r] == 0, label + ": healthy rank " + std::to_string(r) + " exit status " + std::to_string(status[r]) + (status[r] == 999 ? " (HUNG)" : ""));
    }
}


} // namespace mesh_test
