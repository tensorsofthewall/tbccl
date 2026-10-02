// communicator-wide abort. Every scenario has a live-but-silent (or dead) peer, and an in-process deadline: if
// abort/destruction does not return, the test aborts the process (outer ctest TIMEOUT is the second net).

#include <tbccl/communicator.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace tbccl;
using clk = std::chrono::steady_clock;
using std::chrono::milliseconds;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

namespace
{
void expect(bool c, const std::string &m) { if (!c) throw std::runtime_error("assertion failed: " + m); }

bool port_free(std::uint16_t port)
{
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(port);
    const bool ok = ::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)) == 0;
    ::close(fd);
    return ok;
}
std::uint16_t next_port()
{
    static std::uint16_t p = 33200;
    for (;;)
    {
        p = static_cast<std::uint16_t>(p + 4);
        if (port_free(p) && port_free(p + 1) && port_free(p + 1000) && port_free(p + 1001)) return p;
    }
}

// Fails the whole process if `f` does not finish within `limit` (a hung abort is the bug under test).
template <typename F> auto bounded(const char *what, milliseconds limit, F f)
{
    auto fut = std::async(std::launch::async, f);
    if (fut.wait_for(limit) != std::future_status::ready)
    {
        std::fprintf(stderr, "[HANG] %s did not return within %lld ms\n", what, (long long)limit.count());
        std::_Exit(3);
    }
    return fut.get();
}

// rank 0 runs on the test thread; rank 1 ("peer") runs `peer_fn` on its own thread and is released by `release`.
struct Pair
{
    std::unique_ptr<Communicator> c0, c1;
    Pair()
    {
        const auto port = next_port();
        CommunicatorOptions o0;
        o0.rank = 0;
        o0.peers = {{"127.0.0.1", port}, {"127.0.0.1", static_cast<std::uint16_t>(port + 1)}};
        auto o1 = o0;
        o1.rank = 1;
        std::exception_ptr e;
        std::thread t([&] { try { c1 = Communicator::create(o1); } catch (...) { e = std::current_exception(); } });
        c0 = Communicator::create(o0);
        t.join();
        if (e) std::rethrow_exception(e);
    }
};

struct Silent // keeps the peer communicator alive (socket open, not reading) until release()
{
    std::mutex m; std::condition_variable cv; bool go = false;
    void release() { { std::lock_guard<std::mutex> l(m); go = true; } cv.notify_all(); }
    void park() { std::unique_lock<std::mutex> l(m); cv.wait(l, [&] { return go; }); }
};

BufferView host(std::vector<float> &v) { return {MemoryKind::Host, v.data(), v.size() * 4, -1}; }
Work ar(Communicator &c, std::vector<float> &v) { return c.all_reduce(host(v), host(v), v.size(), DataType::Float32, ReduceOp::Sum); }
void healthy_allreduce(Pair &p)
{
    std::vector<float> a(1024, 1), b(1024, 2);
    std::thread t([&] { ar(*p.c1, b).wait(); });
    auto w = ar(*p.c0, a);
    w.wait();
    t.join();
    expect(!w.has_error() && a[0] == 3.0f && b[0] == 3.0f, "healthy control allreduce");
}
void terminal_error(Work &w, const std::string &what)
{
    expect(w.is_completed(), what + ": completed");
    expect(w.has_error(), what + ": has_error");
    expect(w.error().find("abort") != std::string::npos, what + ": error mentions abort: " + w.error());
    w.wait();                       // repeated wait is stable (does not hang, does not change state)
    w.wait();
    expect(w.has_error() && w.error() == w.error(), what + ": stable error");
}
void print_lat(const char *what, double a, double b) { std::printf("  %-34s abort()=%.2f ms  destroy=%.2f ms\n", what, a, b); }

void test_silent_allreduce()
{
    Pair p; healthy_allreduce(p);
    Silent s; std::thread keep([&] { s.park(); });
    std::vector<float> a(1 << 18, 1);
    auto w = ar(*p.c0, a);
    std::this_thread::sleep_for(milliseconds(300));
    expect(!w.is_completed(), "work pending on silent peer");
    auto t0 = clk::now();
    bounded("abort", milliseconds(5000), [&] { p.c0->abort("test"); return 0; });
    auto t1 = clk::now();
    terminal_error(w, "allreduce");
    expect(p.c0->aborted() && p.c0->abort_reason() == "test", "state + first reason");
    bounded("destroy", milliseconds(5000), [&] { p.c0.reset(); return 0; });
    print_lat("silent all_reduce", ms(t0, t1), ms(t1, clk::now()));
    s.release(); keep.join();
}

void test_silent_broadcast_and_allgather()
{
    for (int variant = 0; variant < 3; ++variant)
    {
        Pair p; healthy_allreduce(p);
        Silent s; std::thread keep([&] { s.park(); });
        std::vector<float> a(1 << 18, 1), o0(1 << 18), o1(1 << 18);
        Work w = variant == 0   ? p.c0->broadcast(host(a), 1)                       // receiver waiting for root
               : variant == 1   ? p.c0->broadcast(host(a), 0)                       // root sending to a silent peer
                                : p.c0->all_gather(host(a), {host(o0), host(o1)});
        std::this_thread::sleep_for(milliseconds(200));
        auto t0 = clk::now();
        bounded("abort", milliseconds(5000), [&] { p.c0->abort("bc/ag"); return 0; });
        auto t1 = clk::now();
        // a small root broadcast fits in socket buffers and can legitimately finish before the abort; both are valid
        expect(w.is_completed(), "terminal after abort");
        if (w.has_error()) terminal_error(w, "collective");
        bounded("destroy", milliseconds(5000), [&] { p.c0.reset(); return 0; });
        print_lat(variant == 0 ? "silent broadcast (receiver)" : variant == 1 ? "silent broadcast (root)" : "silent all_gather", ms(t0, t1), ms(t1, clk::now()));
        s.release(); keep.join();
    }
}

void test_p2p_recv_and_send()
{
    for (int send = 0; send < 2; ++send)
    {
        Pair p; healthy_allreduce(p);
        Silent s; std::thread keep([&] { s.park(); });
        // 256 MiB far exceeds loopback socket buffers, so a sender to a non-reading peer must block.
        std::vector<float> buf(send ? (64u << 20) : 1024);
        BufferView v = host(buf);
        Work w = send ? p.c0->send(v, buf.size(), DataType::Float32, 1) : p.c0->recv(v, buf.size(), DataType::Float32, 1);
        std::this_thread::sleep_for(milliseconds(300));
        expect(!w.is_completed(), send ? "send blocked on backpressure" : "recv blocked");
        auto t0 = clk::now();
        bounded("abort", milliseconds(5000), [&] { p.c0->abort("p2p"); return 0; });
        auto t1 = clk::now();
        terminal_error(w, send ? "send" : "recv");
        bounded("destroy", milliseconds(5000), [&] { p.c0.reset(); return 0; });
        print_lat(send ? "blocked P2P send" : "blocked P2P recv", ms(t0, t1), ms(t1, clk::now()));
        s.release(); keep.join();
    }
}

void test_queued_mixed_and_dropped()
{
    Pair p; healthy_allreduce(p);
    Silent s; std::thread keep([&] { s.park(); });
    std::vector<float> a(1 << 16, 1), b(1 << 16, 1), c(1 << 16, 1), d(256, 1);
    Work w0 = ar(*p.c0, a);                                    // active, blocked
    Work w1 = ar(*p.c0, b);                                    // queued collective
    Work w2 = p.c0->broadcast(host(c), 1);                     // queued collective
    Work w3 = p.c0->recv(host(d), d.size(), DataType::Float32, 1); // queued P2P
    std::vector<float> e(1024, 1);
    (void)ar(*p.c0, e);                                        // dropped Work handle
    std::this_thread::sleep_for(milliseconds(200));
    expect(!w0.is_completed() && !w1.is_completed(), "pending");
    bounded("abort", milliseconds(5000), [&] { p.c0->abort("queued"); return 0; });
    terminal_error(w0, "W0"); terminal_error(w1, "W1"); terminal_error(w2, "W2"); terminal_error(w3, "W3");
    bounded("destroy", milliseconds(5000), [&] { p.c0.reset(); return 0; });
    s.release(); keep.join();
}

void test_multi_abort_and_idempotent()
{
    Pair p; healthy_allreduce(p);
    Silent s; std::thread keep([&] { s.park(); });
    std::vector<float> a(1 << 16, 1);
    Work w = ar(*p.c0, a);
    std::vector<std::thread> ts;
    for (int i = 0; i < 8; ++i) ts.emplace_back([&, i] { p.c0->abort("caller" + std::to_string(i)); });
    bounded("8 concurrent aborts", milliseconds(5000), [&] { for (auto &t : ts) t.join(); return 0; });
    p.c0->abort("again"); p.c0->abort();
    const auto r = p.c0->abort_reason();
    expect(r.rfind("caller", 0) == 0, "first reason wins and is one of the callers: " + r);
    terminal_error(w, "multi-abort");
    bounded("destroy", milliseconds(5000), [&] { p.c0.reset(); return 0; });
    s.release(); keep.join();
}

void test_zero_ops_and_after_completion_and_submit_after_abort()
{
    { Pair p; p.c0->abort("zero ops"); p.c1->abort("zero ops"); }
    Pair p; healthy_allreduce(p);
    std::vector<float> a(1024, 1), b(1024, 2);
    std::thread t([&] { ar(*p.c1, b).wait(); });
    Work done = ar(*p.c0, a); done.wait(); t.join();
    p.c0->abort("after completion");
    expect(!done.has_error(), "completed Work stays successful");
    auto throws = [&](const std::function<void()> &f, const char *w) {
        try { f(); } catch (const std::exception &e) { expect(std::string(e.what()).find("abort") != std::string::npos, std::string(w) + ": " + e.what()); return; }
        throw std::runtime_error(std::string("expected throw after abort: ") + w);
    };
    throws([&] { ar(*p.c0, a); }, "all_reduce");
    throws([&] { p.c0->broadcast(host(a), 0); }, "broadcast");
    throws([&] { p.c0->all_gather(host(a), {host(a), host(a)}); }, "all_gather");
    throws([&] { p.c0->send(host(a), a.size(), DataType::Float32, 1); }, "send");
    throws([&] { p.c0->recv(host(a), a.size(), DataType::Float32, 1); }, "recv");
}

void test_validation_does_not_poison_and_timeout_semantics()
{
    Pair p; healthy_allreduce(p);
    std::vector<float> a(1024, 1), b(1024, 2);
    bool threw = false;
    try { p.c0->broadcast(host(a), 7); } catch (const std::exception &) { threw = true; }
    expect(threw && !p.c0->aborted(), "invalid root does not poison");
    threw = false;
    try { p.c0->all_reduce(host(a), host(a), 4096, DataType::Float32, ReduceOp::Sum); } catch (const std::exception &) { threw = true; }
    expect(threw && !p.c0->aborted(), "bad count does not poison");
    std::thread t([&] { std::this_thread::sleep_for(milliseconds(400)); ar(*p.c1, b).wait(); });
    Work w = ar(*p.c0, a);
    std::this_thread::sleep_for(milliseconds(150));
    expect(!w.is_completed() && !p.c0->aborted(), "slow peer: pending, communicator healthy (waiting is not aborting)");
    w.wait(); t.join();
    expect(!w.has_error() && a[0] == 3.0f, "operation completes after the delayed peer participates");
}

void test_peer_death_and_remote_abort_observed()
{
    { // abrupt peer death: first failure poisons, queued/future operations fail promptly
        Pair p; healthy_allreduce(p);
        std::vector<float> a(1 << 16, 1);
        Work w = ar(*p.c0, a);
        Work w2 = ar(*p.c0, a);
        p.c1.reset();
        bounded("peer-death work", milliseconds(5000), [&] { w.wait(); w2.wait(); return 0; });
        expect(w.has_error() && w2.has_error(), "both fail");
        expect(p.c0->aborted(), "transport failure poisoned the communicator");
        bool threw = false;
        try { ar(*p.c0, a); } catch (const std::exception &) { threw = true; }
        expect(threw, "later submission rejected");
    }
    { // remote abort: peer sees the connection terminate when it next does I/O and poisons itself
        Pair p; healthy_allreduce(p);
        p.c0->abort("remote");
        std::vector<float> b(1024, 1);
        Work w = ar(*p.c1, b);
        bounded("remote-abort observation", milliseconds(5000), [&] { w.wait(); return 0; });
        expect(w.has_error() && p.c1->aborted(), "peer observed termination and poisoned itself: " + w.error());
    }
}

void test_repeated_create_abort_destroy()
{
    for (int i = 0; i < 20; ++i)
    {
        Pair p; healthy_allreduce(p);
        Silent s; std::thread keep([&] { s.park(); });
        std::vector<float> a(4096, 1);
        Work w = ar(*p.c0, a);
        if (i % 2) p.c0->abort("loop");        // explicit abort, or destructor-triggered abort
        bounded("loop destroy", milliseconds(5000), [&] { p.c0.reset(); return 0; });
        expect(w.is_completed() && w.has_error(), "outstanding Work failed");
        s.release(); keep.join();
    }
}
} // namespace

int main(int argc, char **argv)
{
    struct T { const char *n; void (*f)(); } tests[] = {
        {"silent all_reduce", test_silent_allreduce},
        {"silent broadcast / all_gather", test_silent_broadcast_and_allgather},
        {"blocked P2P recv and send", test_p2p_recv_and_send},
        {"queued mixed + dropped Work", test_queued_mixed_and_dropped},
        {"multiple concurrent aborts", test_multi_abort_and_idempotent},
        {"zero ops / after completion / submit after abort", test_zero_ops_and_after_completion_and_submit_after_abort},
        {"validation does not poison; wait is not abort", test_validation_does_not_poison_and_timeout_semantics},
        {"peer death poisons; remote abort observed", test_peer_death_and_remote_abort_observed},
        {"repeated create/abort/destroy", test_repeated_create_abort_destroy},
    };
    try
    {
        for (const auto &t : tests)
        {
            if (argc > 1 && std::string(t.n).find(argv[1]) == std::string::npos) continue;
            std::cerr << "[RUN ] " << t.n << std::endl;
            t.f();
            std::cout << "[PASS] " << t.n << "\n";
        }
    }
    catch (const std::exception &e) { std::cerr << "[FAIL] " << e.what() << "\n"; return 1; }
    std::cout << "All communicator abort tests passed.\n";
    return 0;
}
