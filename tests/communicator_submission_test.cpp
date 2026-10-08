// Submission is nonblocking. A send/recv/collective call returns a Work without waiting for socket progress, a matching operation on the peer, staging, or
// a bounded lane. These tests do NOT rely on timing: transport progress is gated explicitly (debug_set_progress_paused), every submission call must return
// while the gate is closed, and only after every rank has signalled ALL_POSTED is progress released. Before unbounded admission the 10th large send to one
// peer blocked the caller inside TensorCommWorker::enqueue, so each of these tests would have hung at the post (the watchdog reports that).

#include "mesh_test_support.hpp"

#include "communicator_debug.hpp"

#include <tbccl/error.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;
using tbccl::ReduceOp;
using tbccl::detail::debug_set_progress_paused;

namespace
{
    using Clock = std::chrono::steady_clock;

    BufferView view(std::uint8_t *p, std::size_t n) { return BufferView{MemoryKind::Host, p, n, 0}; }
    BufferView view(std::vector<std::uint8_t> &v) { return view(v.data(), v.size()); }

    double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

    int thread_count()
    {
        std::ifstream f("/proc/self/status");
        std::string line;
        while (std::getline(f, line))
            if (line.rfind("Threads:", 0) == 0) return std::atoi(line.c_str() + 8);
        return -1;
    }

    void wait_all(std::vector<tbccl::Work> &works, const std::string &what)
    {
        for (std::size_t i = 0; i < works.size(); ++i)
        {
            works[i].wait();
            expect(!works[i].has_error(), what + ": operation " + std::to_string(i) + " failed: " + works[i].error());
        }
    }

    // Distinct, position-dependent content so FIFO order and integrity are both checked.
    void fill(std::uint8_t *p, std::size_t n, unsigned seed)
    {
        for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<std::uint8_t>(seed * 31u + i * 7u + (i >> 8));
    }
    bool check(const std::uint8_t *p, std::size_t n, unsigned seed)
    {
        for (std::size_t i = 0; i < n; ++i)
            if (p[i] != static_cast<std::uint8_t>(seed * 31u + i * 7u + (i >> 8))) return false;
        return true;
    }

    // Fires only if the test it guards is still running after `seconds`: a submission call (or a wait) that never returns.
    struct Watchdog
    {
        Watchdog(int seconds, const char *what) : state_(std::make_shared<State>())
        {
            std::thread([state = state_, seconds, what] {
                std::unique_lock<std::mutex> lock(state->mutex);
                if (state->cv.wait_for(lock, std::chrono::seconds(seconds), [&] { return state->finished; })) return;
                std::fprintf(stderr, "[FAIL] watchdog: %s never finished (a submission call or a wait is blocking)\n", what);
                std::_Exit(2);
            }).detach();
        }
        ~Watchdog()
        {
            {
                std::lock_guard<std::mutex> lock(state_->mutex);
                state_->finished = true;
            }
            state_->cv.notify_all();
        }

    private:
        struct State
        {
            std::mutex mutex;
            std::condition_variable cv;
            bool finished = false;
        };
        std::shared_ptr<State> state_;
    };

    // ---------------------------------------------------------------------------------------------------------------------------------------
    // Rank 1's progress is held; rank 0 submits 100 large sends and every call must return. Only then is the peer released and everything drains.
    // ---------------------------------------------------------------------------------------------------------------------------------------
    void test_gated_one_sided(std::size_t count, std::size_t bytes, const char *label)
    {
        Watchdog dog(120, label);
        Latch posted(2);
        run_world(2, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::vector<std::uint8_t>> bufs(count, std::vector<std::uint8_t>(bytes));
            std::vector<tbccl::Work> works;
            if (rank == 1) debug_set_progress_paused(comm, true); // the peer is deliberately kept out of transport progress
            if (rank == 0) debug_set_progress_paused(comm, true);
            if (rank == 0)
            {
                for (std::size_t i = 0; i < count; ++i)
                {
                    fill(bufs[i].data(), bytes, static_cast<unsigned>(i));
                    works.push_back(comm.send(view(bufs[i]), bytes, DataType::UInt8, 1));
                }
                // POSTING_COMPLETE: every submission call returned while the peer made no progress at all.
                expect(works.size() == count, "all sends submitted");
                for (auto &w : works) expect(!w.is_completed(), "no send can complete while progress is gated");
            }
            posted.arrive_and_wait();
            if (rank == 1)
            {
                for (std::size_t i = 0; i < count; ++i) works.push_back(comm.recv(view(bufs[i]), bytes, DataType::UInt8, 0));
            }
            debug_set_progress_paused(comm, false); // release the gate only now
            wait_all(works, label);
            if (rank == 1)
                for (std::size_t i = 0; i < count; ++i) expect(check(bufs[i].data(), bytes, static_cast<unsigned>(i)), std::string(label) + ": FIFO/integrity of message " + std::to_string(i));
        });
        std::cout << "[PASS] " << label << ": " << count << " x " << bytes << " B sends all submitted while the peer's progress was gated, then delivered in order\n";
    }

    // ---------------------------------------------------------------------------------------------------------------------------------------
    // The classic post-everything-then-wait exchange (the main GroupStart/GroupEnd correctness test): each rank submits N sends THEN N receives with
    // progress gated on both ranks, signals ALL_POSTED, and only after both have, progress is released.
    // ---------------------------------------------------------------------------------------------------------------------------------------
    void test_symmetric(std::size_t count, std::size_t bytes, bool gated, const char *label)
    {
        Watchdog dog(180, label);
        Latch posted(2);
        run_world(2, [&](std::size_t rank, tbccl::Communicator &comm) {
            const std::size_t peer = 1 - rank;
            std::vector<std::vector<std::uint8_t>> out(count, std::vector<std::uint8_t>(bytes)), in(count, std::vector<std::uint8_t>(bytes, 0));
            if (gated) debug_set_progress_paused(comm, true);
            std::vector<tbccl::Work> works;
            for (std::size_t i = 0; i < count; ++i)
            {
                fill(out[i].data(), bytes, static_cast<unsigned>(i * 2 + rank));
                works.push_back(comm.send(view(out[i]), bytes, DataType::UInt8, peer));
            }
            for (std::size_t i = 0; i < count; ++i) works.push_back(comm.recv(view(in[i]), bytes, DataType::UInt8, peer));
            posted.arrive_and_wait(); // ALL_POSTED on both ranks
            if (gated) debug_set_progress_paused(comm, false);
            wait_all(works, label);
            for (std::size_t i = 0; i < count; ++i) expect(check(in[i].data(), bytes, static_cast<unsigned>(i * 2 + peer)), std::string(label) + ": message " + std::to_string(i));
        });
        std::cout << "[PASS] " << label << ": " << count << " sends then " << count << " receives posted on both ranks" << (gated ? " (progress gated until ALL_POSTED)" : "") << ", all completed\n";
    }

    // ---------------------------------------------------------------------------------------------------------------------------------------
    // N=4: hundreds of operations without waiting: one peer repeatedly, alternating peers, simultaneous send+recv, a ring.
    // ---------------------------------------------------------------------------------------------------------------------------------------
    void test_n4_patterns(bool gated)
    {
        Watchdog dog(180, "n4 patterns");
        constexpr std::size_t kWorld = 4, kRounds = 150, kBytes = 4096;
        Latch posted(kWorld);
        run_world(kWorld, [&](std::size_t rank, tbccl::Communicator &comm) {
            const std::size_t next = (rank + 1) % kWorld, prev = (rank + kWorld - 1) % kWorld, across = (rank + 2) % kWorld;
            struct Op { bool send; std::size_t peer; std::vector<std::uint8_t> buf; unsigned seed; };
            std::vector<Op> ops;
            // The same logical order on both ends of every pair: for round i the sender's n-th message to a peer is the receiver's n-th receive from it.
            for (std::size_t i = 0; i < kRounds; ++i)
            {
                ops.push_back({true, next, std::vector<std::uint8_t>(kBytes), static_cast<unsigned>(i * 8 + rank)});          // one peer repeatedly (ring)
                ops.push_back({false, prev, std::vector<std::uint8_t>(kBytes), static_cast<unsigned>(i * 8 + prev)});
                ops.push_back({true, (i % 2) ? next : prev, std::vector<std::uint8_t>(kBytes), static_cast<unsigned>(i * 8 + 4 + rank)});   // alternating peers
                ops.push_back({false, (i % 2) ? prev : next, std::vector<std::uint8_t>(kBytes), static_cast<unsigned>(i * 8 + 4 + ((i % 2) ? prev : next))});
                // the diagonal pairs (0,2) and (1,3): the lower rank sends, the higher receives
                if (i % 5 == 0) ops.push_back({rank < across, across, std::vector<std::uint8_t>(kBytes), static_cast<unsigned>(1000 + i)});
            }
            if (gated) debug_set_progress_paused(comm, true);
            std::vector<tbccl::Work> works;
            for (auto &op : ops)
            {
                if (op.send)
                {
                    fill(op.buf.data(), kBytes, op.seed);
                    works.push_back(comm.send(view(op.buf), kBytes, DataType::UInt8, op.peer));
                }
                else works.push_back(comm.recv(view(op.buf), kBytes, DataType::UInt8, op.peer));
            }
            posted.arrive_and_wait();
            if (gated) debug_set_progress_paused(comm, false);
            wait_all(works, "n4 patterns");
            for (auto &op : ops)
                if (!op.send) expect(check(op.buf.data(), kBytes, op.seed), "n4 patterns: received content (peer " + std::to_string(op.peer) + ")");
        });
        std::cout << "[PASS] N=4: ~" << kRounds * 5 << " operations per rank across one peer, alternating peers, simultaneous send/recv and a ring, none waited on while posting" << (gated ? " (progress gated)" : "") << "\n";
    }

    // ---------------------------------------------------------------------------------------------------------------------------------------
    // A stalled peer must not stall another peer: rank 0 queues operations to rank 1 (whose progress is held) and still completes its exchange with rank 2.
    // ---------------------------------------------------------------------------------------------------------------------------------------
    void test_fairness()
    {
        Watchdog dog(120, "fairness");
        Latch posted(3), neighbour_done(3); // rank 1 stays stalled until the healthy exchange is done
        run_world(3, [&](std::size_t rank, tbccl::Communicator &comm) {
            constexpr std::size_t kStalled = 150, kBytes = 1 << 20, kSmall = 4096;
            std::vector<std::vector<std::uint8_t>> bufs;
            std::vector<tbccl::Work> to_stalled, normal;
            if (rank == 1) debug_set_progress_paused(comm, true); // rank 1 is the stalled peer
            if (rank == 0)
            {
                bufs.assign(kStalled, std::vector<std::uint8_t>(kBytes, 5));
                for (std::size_t i = 0; i < kStalled; ++i) to_stalled.push_back(comm.send(view(bufs[i]), kBytes, DataType::UInt8, 1));
                expect(to_stalled.size() == kStalled, "all sends to the stalled peer were admitted");
                for (auto &w : to_stalled) expect(!w.is_completed(), "nothing completes towards the stalled peer");
            }
            posted.arrive_and_wait(); // rank 0's queue to rank 1 is deep before the healthy exchange starts
            if (rank != 1)
            {
                // rank 0 <-> rank 2: both directions, must complete although rank 0 has a deep queue towards rank 1
                const std::size_t peer = rank == 0 ? 2 : 0;
                std::vector<std::uint8_t> out(kSmall), in(kSmall, 0);
                fill(out.data(), kSmall, static_cast<unsigned>(rank + 40));
                auto s = comm.send(view(out), kSmall, DataType::UInt8, peer);
                auto r = comm.recv(view(in), kSmall, DataType::UInt8, peer);
                const auto t0 = Clock::now();
                s.wait();
                r.wait();
                expect(!s.has_error() && !r.has_error(), "exchange with the healthy peer");
                expect(check(in.data(), kSmall, static_cast<unsigned>(peer + 40)), "healthy exchange content");
                expect(since(t0) < 20, "the healthy exchange was not held up by the stalled peer");
                for (auto &w : to_stalled) expect(!w.is_completed(), "the stalled peer's queue is still pending after the healthy exchange");
            }
            neighbour_done.arrive_and_wait();
            if (rank == 1)
            {
                std::vector<std::vector<std::uint8_t>> sink(kStalled, std::vector<std::uint8_t>(kBytes));
                for (std::size_t i = 0; i < kStalled; ++i) normal.push_back(comm.recv(view(sink[i]), kBytes, DataType::UInt8, 0));
                debug_set_progress_paused(comm, false);
                wait_all(normal, "stalled peer drains");
                for (std::size_t i = 0; i < kStalled; ++i) expect(sink[i][0] == 5 && sink[i][kBytes - 1] == 5, "queued data delivered after the stall");
            }
            if (rank == 0) wait_all(to_stalled, "sends to the formerly stalled peer");
        });
        std::cout << "[PASS] a stalled peer holds 150 queued 1 MiB sends while the exchange with another peer completes; the queue drains after release\n";
    }

    // ---------------------------------------------------------------------------------------------------------------------------------------
    // Many collectives queued before any wait: submission stays asynchronous and the executor preserves the sequence.
    // ---------------------------------------------------------------------------------------------------------------------------------------
    void test_collectives_queued(std::size_t world)
    {
        Watchdog dog(120, "queued collectives");
        Latch posted(world);
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            constexpr std::size_t kRounds = 12;
            debug_set_progress_paused(comm, true); // data lanes are gated: the collectives cannot move payload yet, only be admitted
            std::vector<std::vector<float>> reduce(kRounds, std::vector<float>(1000)), bcast(kRounds, std::vector<float>(500));
            std::vector<tbccl::Work> works;
            for (std::size_t i = 0; i < kRounds; ++i)
            {
                std::fill(reduce[i].begin(), reduce[i].end(), static_cast<float>(rank + 1 + i));
                std::fill(bcast[i].begin(), bcast[i].end(), rank == i % world ? static_cast<float>(100 + i) : 0.0f);
                works.push_back(comm.barrier());
                works.push_back(comm.all_reduce({MemoryKind::Host, reduce[i].data(), reduce[i].size() * 4, 0}, {MemoryKind::Host, reduce[i].data(), reduce[i].size() * 4, 0}, reduce[i].size(), DataType::Float32, ReduceOp::Sum));
                works.push_back(comm.broadcast({MemoryKind::Host, bcast[i].data(), bcast[i].size() * 4, 0}, i % world));
            }
            posted.arrive_and_wait();
            debug_set_progress_paused(comm, false);
            wait_all(works, "queued collectives");
            for (std::size_t i = 0; i < kRounds; ++i)
            {
                float want = 0;
                for (std::size_t r = 0; r < world; ++r) want += static_cast<float>(r + 1 + i);
                expect(reduce[i][0] == want && reduce[i][999] == want, "queued all_reduce " + std::to_string(i));
                expect(bcast[i][0] == static_cast<float>(100 + i) && bcast[i][499] == static_cast<float>(100 + i), "queued broadcast " + std::to_string(i));
            }
        });
        std::cout << "[PASS] world_size=" << world << ": 36 collectives admitted before any wait, executed in submission order\n";
    }

    // ---------------------------------------------------------------------------------------------------------------------------------------
    // Abort / destroy with a deep queue.
    // ---------------------------------------------------------------------------------------------------------------------------------------
    void test_abort_deep_queue(bool gated)
    {
        Watchdog dog(120, "abort with a deep queue");
        const int fds_before = open_fd_count(), threads_before = thread_count();
        {
            auto results = bootstrap(healthy_slots(2), std::chrono::seconds(20));
            expect(results[0].comm && results[1].comm, "bootstrap");
            auto &comm = *results[0].comm;
            constexpr std::size_t kOps = 400, kBytes = 1 << 20;
            std::vector<std::vector<std::uint8_t>> bufs(kOps, std::vector<std::uint8_t>(kBytes, 9));
            if (gated) debug_set_progress_paused(comm, true);
            std::vector<tbccl::Work> works;
            for (std::size_t i = 0; i < kOps; ++i)
            {
                if (i % 2 == 0) works.push_back(comm.send(view(bufs[i]), kBytes, DataType::UInt8, 1));
                else works.push_back(comm.recv(view(bufs[i]), kBytes, DataType::UInt8, 1));
            }
            if (!gated) std::this_thread::sleep_for(std::chrono::milliseconds(50)); // let the first send become active and block in the kernel
            const auto t0 = Clock::now();
            comm.abort("deep queue abort");
            std::size_t failed = 0;
            for (auto &w : works)
            {
                w.wait(); // every operation becomes terminal
                if (w.has_error())
                {
                    ++failed;
                    // Ungated, the first send is already running when abort() is called; the peer can react to the abort broadcast and close its sockets before the local transfer
// is interrupted, and that operation then ends with the transport error. Operations that never started (all of them when gated) must carry the Aborted code.
                    expect(w.error_code() == tbccl::ErrorCode::Aborted || (!gated && w.error_code() == tbccl::ErrorCode::TransportError), "a failed operation carries the Aborted code");
                }
                else expect(!gated, "with progress gated no operation can have completed before the abort"); // ungated: small early sends may already have been delivered
            }
            expect(failed > 0 && (gated ? failed == works.size() : works.back().has_error()), "the queued operations fail");
            expect(since(t0) < 10, "abort of a deep queue is bounded");
            // every Work is terminal: no transport thread touches `bufs` any more, so they can be freed (ASan checks this)
            bufs.clear();
            bufs.shrink_to_fit();
            results.clear(); // destroys both communicators
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        expect(open_fd_count() == fds_before, "no file descriptor leaked: " + std::to_string(open_fd_count()) + " vs " + std::to_string(fds_before));
        expect(thread_count() <= threads_before, "no thread leaked: " + std::to_string(thread_count()) + " vs " + std::to_string(threads_before));
        std::cout << "[PASS] abort with 400 queued operations" << (gated ? " (none started)" : " (one active, blocked in the kernel)") << ": all Works Aborted, bounded, no fd/thread leak\n";
    }

    void test_destroy_deep_queue()
    {
        Watchdog dog(120, "destroying a deep queue");
        const int fds_before = open_fd_count(), threads_before = thread_count();
        {
            auto results = bootstrap(healthy_slots(2), std::chrono::seconds(20));
            expect(results[0].comm && results[1].comm, "bootstrap");
            constexpr std::size_t kOps = 300, kBytes = 1 << 20;
            std::vector<std::vector<std::uint8_t>> bufs(kOps, std::vector<std::uint8_t>(kBytes, 3));
            std::vector<tbccl::Work> works;
            debug_set_progress_paused(*results[0].comm, true);
            for (std::size_t i = 0; i < kOps; ++i) works.push_back(results[0].comm->send(view(bufs[i]), kBytes, DataType::UInt8, 1));
            const auto t0 = Clock::now();
            results[0].comm.reset(); // destroy with everything still queued: must abort the peers and fail the queue, not wait for it
            expect(since(t0) < 10, "destruction with a deep queue is bounded");
            for (auto &w : works)
            {
                expect(w.wait_for(std::chrono::seconds(5)), "every operation is terminal once the communicator is gone");
                expect(w.has_error(), "...and failed");
            }
            bufs.clear();
            results.clear();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        expect(open_fd_count() == fds_before, "no file descriptor leaked");
        expect(thread_count() <= threads_before, "no thread leaked");
        std::cout << "[PASS] destroying a communicator with 300 queued operations: bounded, all Works failed, no fd/thread leak\n";
    }

    // Work handle destroyed immediately after submission: the operation continues; the caller keeps the buffer alive until it is terminal.
    void test_work_handle_destroyed_early()
    {
        Watchdog dog(60, "early work destruction");
        Latch done(2);
        run_world(2, [&](std::size_t rank, tbccl::Communicator &comm) {
            constexpr std::size_t kBytes = 1 << 20;
            std::vector<std::uint8_t> buf(kBytes, 0);
            if (rank == 0)
            {
                fill(buf.data(), kBytes, 77);
                {
                    tbccl::Work w = comm.send(view(buf), kBytes, DataType::UInt8, 1);
                } // the handle is gone; the send continues, and `buf` must stay valid and unmodified until the peer has it
            }
            else
            {
                {
                    tbccl::Work w = comm.recv(view(buf), kBytes, DataType::UInt8, 0);
                }
            }
            // The only way to know the operation finished without a handle: a later operation on the same FIFO lane, or the peer's confirmation.
            std::vector<std::uint8_t> token(8, rank == 0 ? 1 : 2), back(8, 0);
            auto s = comm.send(view(token), 8, DataType::UInt8, 1 - rank);
            auto r = comm.recv(view(back), 8, DataType::UInt8, 1 - rank);
            s.wait();
            r.wait();
            done.arrive_and_wait();
            if (rank == 1) expect(check(buf.data(), kBytes, 77), "a send whose Work handle was destroyed was still delivered intact");
        });
        std::cout << "[PASS] destroying a Work handle neither cancels the operation nor frees the buffer obligation\n";
    }
} // namespace

int main()
{
    std::cout << std::unitbuf;
    try
    {
        test_gated_one_sided(100, 8 << 20, "gated one-sided (100 x 8 MiB)");
        test_symmetric(100, 1 << 20, true, "symmetric 100 sends then 100 receives (1 MiB), gated");
        test_symmetric(100, 8 << 20, false, "symmetric 100 sends then 100 receives (8 MiB), ungated");
        test_symmetric(1000, 256, true, "symmetric 1000 small operations, gated");
        test_symmetric(1000, 256, false, "symmetric 1000 small operations, ungated");
        test_n4_patterns(true);
        test_n4_patterns(false);
        test_fairness();
        test_collectives_queued(2);
        test_collectives_queued(4);
        test_abort_deep_queue(true);
        test_abort_deep_queue(false);
        test_destroy_deep_queue();
        test_work_handle_destroyed_early();
        std::cout << "All submission tests passed.\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
