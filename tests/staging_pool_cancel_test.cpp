// StagingPool cancellation: acquire()/release() keep their behaviour, and cancel() ends the pool's use for good (wakes every waiter, no slot is handed out afterwards).
// Every scenario has a bounded wait; a thread that is still blocked at the deadline fails the test (and ends the process) instead of hanging it.

#include <tbccl/async_transfer.hpp>
#include <tbccl/error.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace
{
using namespace std::chrono_literals;

[[noreturn]] void die(const std::string &what)
{
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    std::fflush(stderr);
    std::_Exit(1);
}

void expect(bool condition, const std::string &what)
{
    if (!condition) die(what);
}

// Runs `body` on a thread and waits at most `limit` for it to finish; a body that is still running (a thread stuck in acquire()) ends the process with a message.
template <typename F>
void bounded(const char *name, std::chrono::milliseconds limit, F body)
{
    std::promise<void> done;
    auto future = done.get_future();
    std::thread t([&] {
        body();
        done.set_value();
    });
    if (future.wait_for(limit) != std::future_status::ready)
    {
        std::fprintf(stderr, "STALLED: %s did not finish within %lld ms\n", name, static_cast<long long>(limit.count()));
        std::fflush(stderr);
        std::_Exit(124);
    }
    t.join();
}

// Starts a thread that calls pool.acquire() and reports how it ended. `started` is set just before the call.
struct Waiter
{
    std::atomic<bool> started{false};
    std::atomic<int> outcome{0}; // 0 = still inside acquire(), 1 = got a slot, 2 = threw Aborted, 3 = threw something else
    std::size_t slot = 0;
    std::thread thread;

    void run(tbccl::StagingPool &pool)
    {
        thread = std::thread([this, &pool] {
            started = true;
            try
            {
                slot = pool.acquire();
                outcome = 1;
            }
            catch (const tbccl::Error &e)
            {
                outcome = e.code() == tbccl::ErrorCode::Aborted ? 2 : 3;
            }
            catch (...)
            {
                outcome = 3;
            }
        });
    }
    bool wait_outcome(std::chrono::milliseconds limit)
    {
        const auto deadline = std::chrono::steady_clock::now() + limit;
        while (outcome.load() == 0 && std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_for(1ms);
        return outcome.load() != 0;
    }
};

void wait_started(Waiter &w)
{
    while (!w.started.load()) std::this_thread::yield();
    std::this_thread::sleep_for(30ms); // let the thread reach its wait; both orders (blocked first, or cancelled first) must end the same way
}

void test_acquire_release()
{
    tbccl::StagingPool pool(64, 2);
    expect(!pool.cancelled(), "a new pool is active");
    const std::size_t a = pool.acquire(), b = pool.acquire();
    expect(a != b && a < 2 && b < 2, "two distinct slots");
    expect(pool.state(a) == tbccl::StagingSlotState::SourcePreparing, "an acquired slot is in use");
    pool.release(a);
    expect(pool.state(a) == tbccl::StagingSlotState::Free, "a released slot is free");
    expect(pool.acquire() == a, "a released slot can be acquired again");
    pool.release(a);
    pool.release(b);
    std::puts("PASS acquire/release");
}

void test_blocked_acquire_wakes_on_release()
{
    tbccl::StagingPool pool(64, 1);
    const std::size_t held = pool.acquire();
    Waiter w;
    w.run(pool);
    wait_started(w);
    expect(w.outcome.load() == 0, "the waiter must block while no slot is free");
    pool.release(held);
    expect(w.wait_outcome(5000ms), "the waiter must wake after release");
    w.thread.join();
    expect(w.outcome.load() == 1 && w.slot == held, "the woken waiter gets the released slot");
    std::puts("PASS blocked acquire wakes after release");
}

void test_blocked_acquire_wakes_on_cancel()
{
    tbccl::StagingPool pool(64, 2);
    pool.acquire();
    pool.acquire();
    Waiter w;
    w.run(pool);
    wait_started(w);
    expect(w.outcome.load() == 0, "the waiter must block while every slot is in use");
    pool.cancel();
    expect(w.wait_outcome(5000ms), "the waiter must wake after cancel");
    w.thread.join();
    expect(w.outcome.load() == 2, "a cancelled acquire throws Error(Aborted)");
    expect(pool.cancelled(), "the pool reports cancelled");
    std::puts("PASS blocked acquire wakes after cancel");
}

void test_acquire_after_cancel_fails_even_with_free_slots()
{
    tbccl::StagingPool pool(64, 2);
    pool.cancel();
    bounded("acquire after cancel", 5000ms, [&] {
        for (int i = 0; i < 3; ++i)
        {
            bool threw = false;
            try
            {
                pool.acquire();
            }
            catch (const tbccl::Error &e)
            {
                threw = e.code() == tbccl::ErrorCode::Aborted;
            }
            expect(threw, "acquire() after cancel() must throw Aborted promptly");
        }
    });
    expect(pool.state(0) == tbccl::StagingSlotState::Free && pool.state(1) == tbccl::StagingSlotState::Free, "no slot became owned after cancel");
    std::puts("PASS acquire after cancel fails promptly and takes no slot");
}

void test_cancel_idempotent_and_release_safe()
{
    tbccl::StagingPool pool(64, 2);
    const std::size_t owned = pool.acquire();
    pool.cancel();
    pool.cancel();
    pool.cancel();
    expect(pool.cancelled(), "cancelled after repeated cancel()");
    pool.release(owned); // a slot already owned when the pool was cancelled can still be given back
    expect(pool.state(owned) == tbccl::StagingSlotState::Free, "release after cancel returns the slot");
    expect(pool.cancelled(), "release must not reactivate a cancelled pool");
    bool threw = false;
    try
    {
        pool.acquire();
    }
    catch (const tbccl::Error &e)
    {
        threw = e.code() == tbccl::ErrorCode::Aborted;
    }
    expect(threw, "a cancelled pool stays cancelled after release()");
    std::puts("PASS cancel is idempotent; release after cancel is safe and does not reactivate");
}

void test_all_waiters_wake()
{
    constexpr int kWaiters = 6;
    tbccl::StagingPool pool(64, 2);
    pool.acquire();
    pool.acquire();
    std::vector<std::unique_ptr<Waiter>> waiters;
    for (int i = 0; i < kWaiters; ++i)
    {
        waiters.push_back(std::make_unique<Waiter>());
        waiters.back()->run(pool);
    }
    for (auto &w : waiters) wait_started(*w);
    pool.cancel();
    for (auto &w : waiters)
    {
        expect(w->wait_outcome(5000ms), "every waiter must wake after cancel");
        w->thread.join();
        expect(w->outcome.load() == 2, "every waiter sees the cancellation");
    }
    std::puts("PASS all blocked waiters wake after one cancel");
}

void test_cancel_races_with_release()
{
    // cancel() and release() racing with several acquirers: nobody may stay blocked, and no slot is handed out after cancel() has returned.
    for (int round = 0; round < 200; ++round)
    {
        tbccl::StagingPool pool(64, 1);
        const std::size_t held = pool.acquire();
        std::vector<std::unique_ptr<Waiter>> waiters;
        for (int i = 0; i < 3; ++i)
        {
            waiters.push_back(std::make_unique<Waiter>());
            waiters.back()->run(pool);
        }
        std::thread releaser([&] { pool.release(held); });
        std::thread canceller([&] { pool.cancel(); });
        releaser.join();
        canceller.join();
        int got = 0;
        for (auto &w : waiters)
        {
            expect(w->wait_outcome(5000ms), "a waiter stayed blocked after cancel/release");
            w->thread.join();
            got += w->outcome.load() == 1;
            expect(w->outcome.load() == 1 || w->outcome.load() == 2, "unexpected acquire outcome");
        }
        expect(got <= 1, "a single-slot pool handed out more than one slot");
        // after both calls have returned, nothing more can be acquired
        bool threw = false;
        try
        {
            pool.acquire();
        }
        catch (const tbccl::Error &)
        {
            threw = true;
        }
        expect(threw, "acquire() after cancel() returned must throw");
    }
    std::puts("PASS cancel racing with release (200 rounds)");
}

void test_destruction_after_users_stopped()
{
    auto pool = std::make_unique<tbccl::StagingPool>(1 << 16, 2);
    Waiter w1, w2;
    pool->acquire();
    pool->acquire();
    w1.run(*pool);
    w2.run(*pool);
    wait_started(w1);
    wait_started(w2);
    pool->cancel();
    expect(w1.wait_outcome(5000ms) && w2.wait_outcome(5000ms), "waiters wake before destruction");
    w1.thread.join(); // users have stopped touching the pool: only now may it be destroyed
    w2.thread.join();
    pool.reset();
    std::puts("PASS pool destroyed after its users stopped");
}
} // namespace

int main()
{
    // the whole program is bounded as well
    std::thread([] {
        std::this_thread::sleep_for(120s);
        std::fprintf(stderr, "STALLED: staging_pool_cancel_test exceeded 120 s\n");
        std::fflush(stderr);
        std::_Exit(124);
    }).detach();
    test_acquire_release();
    test_blocked_acquire_wakes_on_release();
    test_blocked_acquire_wakes_on_cancel();
    test_acquire_after_cancel_fails_even_with_free_slots();
    test_cancel_idempotent_and_release_safe();
    test_all_waiters_wake();
    test_cancel_races_with_release();
    test_destruction_after_users_stopped();
    std::puts("staging_pool_cancel_test passed");
    return 0;
}
