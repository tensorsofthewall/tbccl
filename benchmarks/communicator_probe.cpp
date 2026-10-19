// Communicator probe: process-per-rank measurement of the modern N=2 Communicator on host memory (two hosts or loopback).
//
//   communicator_probe --rank R --peers <ip0:port0,ip1:port1> --ops p2p,simul,stream01,stream10,allreduce,allgather,broadcast0,broadcast1,barrier
//                      [--sizes 64,4096,...] [--iters N] [--label NAME] [--samples PATH]
//
// Why a new probe: small_message_latency has no broadcast or barrier, verifies only p2p, and does not separate submission from completion.
// Every (op, size) point here
//   * starts with an UNTIMED correctness round that checks the real result contents (and a barrier checks that it really waits for the late rank),
//   * is timed with the caller-side split: submit_us (the API call returns a Work), wait_us (Work::wait), total_us,
//   * ends with a second untimed verification, so a corruption that only appears after many iterations is caught,
//   * reports both ranks (each rank prints its own JSON line; clocks are never compared across hosts).
// Definitions (all on the printing rank's own steady clock):
//   p2p        round trip: rank 0 send+wait then recv+wait; rank 1 recv+wait then send+wait. total = one full round trip as seen by that rank.
//   simul      both ranks post a send and a recv, then wait both (one exchange).
//   streamXY   rank X sends `window` messages of `bytes` back to back to rank Y, then waits for its sends; Y receives them all and returns a 1-byte
//              acknowledgement. Reported per batch on the sender: sender_complete (all sends terminal), acked (ack received) and acked minus half a
//              calibrated 64-byte round trip. The receiver reports its own first-post-to-last-complete time. Payload only (no framing) is counted.
//   allreduce  in-place Float32 SUM of bytes/4 elements (the buffer is restored before each timed call, untimed).
//   allgather  every rank contributes `bytes`, both receive 2*bytes (traffic: bytes each way).
//   broadcastN root N sends `bytes` to the other rank.
//   barrier    Communicator::barrier().
// The operation list never combines an unsupported operation and size. Internal trace sites are enabled by the library with TBCCL_LATENCY_TRACE.

#include <tbccl/communicator.hpp>

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;

namespace
{

std::vector<std::string> split(const std::string &s, char sep)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep))
        if (!item.empty()) out.push_back(item);
    return out;
}

double us_between(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

struct Usage
{
    double cpu_user = 0, cpu_sys = 0;
    long nvcsw = 0, nivcsw = 0;
    long maxrss_kb = 0;
};

Usage usage()
{
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    Usage u;
    u.cpu_user = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6;
    u.cpu_sys = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
    u.nvcsw = ru.ru_nvcsw;
    u.nivcsw = ru.ru_nivcsw;
    u.maxrss_kb = ru.ru_maxrss;
    return u;
}

BufferView host(void *p, std::size_t bytes) { return {MemoryKind::Host, p, bytes, -1}; }

struct Series
{
    std::vector<double> total, submit, wait, extra1, extra2; // extra1/2: stream-only columns
    void reserve(std::size_t n) { total.reserve(n); submit.reserve(n); wait.reserve(n); }
};

double pct(std::vector<double> v, double q)
{
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<std::size_t>(q * v.size()))];
}

std::string stats_json(const char *name, const std::vector<double> &v)
{
    if (v.empty()) return std::string();
    double sum = 0;
    for (double x : v) sum += x;
    char buf[400];
    std::snprintf(buf, sizeof buf, "\"%s\":{\"median\":%.2f,\"p25\":%.2f,\"p75\":%.2f,\"p95\":%.2f,\"p99\":%.2f,\"min\":%.2f,\"max\":%.2f,\"mean\":%.2f}", name, pct(v, 0.5),
                  pct(v, 0.25), pct(v, 0.75), pct(v, 0.95), pct(v, 0.99), pct(v, 0.0), pct(v, 0.9999), sum / v.size());
    return buf;
}

[[noreturn]] void fail(const std::string &what)
{
    std::fprintf(stderr, "VERIFY FAILED %s\n", what.c_str());
    std::fflush(stderr);
    std::_Exit(3);
}

// Deterministic content, distinct per rank, per message and per byte, so a swapped, truncated or stale buffer is detected.
std::uint8_t pat(std::size_t rank, std::size_t msg, std::size_t j) { return static_cast<std::uint8_t>(rank * 37 + msg * 31 + j * 7 + (j >> 8) * 13 + 1); }

void fill_pat(std::vector<std::uint8_t> &v, std::size_t rank, std::size_t msg)
{
    for (std::size_t j = 0; j < v.size(); ++j) v[j] = pat(rank, msg, j);
}

void check_pat(const std::vector<std::uint8_t> &v, std::size_t rank, std::size_t msg, const std::string &what)
{
    for (std::size_t j = 0; j < v.size(); ++j)
        if (v[j] != pat(rank, msg, j)) fail(what + " first mismatch at byte " + std::to_string(j));
}

struct Context
{
    tbccl::Communicator &comm;
    std::size_t rank;
    std::size_t peer;
    bool leader;
    std::string label;
    std::FILE *samples = nullptr;
    double half_rtt_us = 0; // calibrated from a 64-byte p2p, used only to correct the stream ack leg
};

int iterations_for(std::size_t bytes, int base)
{
    int n = base;
    if (bytes > 64 * 1024) n = base / 3;
    if (bytes >= 4 * 1024 * 1024) n = base / 12;
    if (bytes >= 16 * 1024 * 1024) n = base / 30;
    return std::max(n, 12);
}

void emit(Context &c, const char *op, std::size_t bytes, double traffic_bytes, const Series &s, const Usage &u0, const Usage &u1, double wall_s, const std::string &note = std::string())
{
    const double cores = (u1.cpu_user + u1.cpu_sys - u0.cpu_user - u0.cpu_sys) / std::max(wall_s, 1e-9);
    const double med = pct(s.total, 0.5);
    std::string j = "{\"label\":\"" + c.label + "\",\"rank\":" + std::to_string(c.rank) + ",\"op\":\"" + op + "\",\"bytes\":" + std::to_string(bytes) + ",\"n\":" + std::to_string(s.total.size());
    j += "," + stats_json("total_us", s.total);
    if (!s.submit.empty()) j += "," + stats_json("submit_us", s.submit);
    if (!s.wait.empty()) j += "," + stats_json("wait_us", s.wait);
    if (!s.extra1.empty()) j += "," + stats_json("extra1_us", s.extra1);
    if (!s.extra2.empty()) j += "," + stats_json("extra2_us", s.extra2);
    char buf[420];
    std::snprintf(buf, sizeof buf,
                  ",\"traffic_bytes_per_op\":%.0f,\"gbit_s_at_median\":%.3f,\"cpu_cores\":%.3f,\"cpu_user_s\":%.4f,\"cpu_sys_s\":%.4f,\"nvcsw\":%ld,\"nivcsw\":%ld,\"wall_s\":%.3f,\"maxrss_kb\":%ld",
                  traffic_bytes, med > 0 ? traffic_bytes * 8.0 / (med * 1e3) : 0.0, cores, u1.cpu_user - u0.cpu_user, u1.cpu_sys - u0.cpu_sys, u1.nvcsw - u0.nvcsw, u1.nivcsw - u0.nivcsw,
                  wall_s, u1.maxrss_kb);
    j += buf;
    if (!note.empty()) j += ",\"note\":\"" + note + "\"";
    j += "}";
    std::printf("%s\n", j.c_str());
    std::fflush(stdout);
    if (c.samples)
    {
        for (std::size_t i = 0; i < s.total.size(); ++i)
            std::fprintf(c.samples, "%s,%zu,%zu,%.3f,%.3f,%.3f\n", op, bytes, c.rank, s.total[i], i < s.submit.size() ? s.submit[i] : 0.0, i < s.wait.size() ? s.wait[i] : 0.0);
        std::fflush(c.samples);
    }
}

// Runs warmup + timed iterations of `one`, which fills a Series row (total, submit, wait).
using One = std::function<void(Series &, bool timed)>;

void run_point(Context &c, const char *op, std::size_t bytes, double traffic_bytes, int iters, const One &one, const std::function<void()> &verify, const std::string &note = std::string())
{
    verify();
    c.comm.barrier().wait();
    Series s;
    s.reserve(iters);
    const int warm = std::max(iters / 10, 5);
    Series discard;
    for (int i = 0; i < warm; ++i) one(discard, false);
    c.comm.barrier().wait();
    const Usage u0 = usage();
    const auto w0 = Clock::now();
    for (int i = 0; i < iters; ++i) one(s, true);
    const double wall = us_between(w0, Clock::now()) * 1e-6;
    const Usage u1 = usage();
    c.comm.barrier().wait();
    verify();
    emit(c, op, bytes, traffic_bytes, s, u0, u1, wall, note);
}

void point_p2p(Context &c, std::size_t bytes, int base)
{
    std::vector<std::uint8_t> out(bytes), in(bytes);
    fill_pat(out, c.rank, 0);
    const auto one = [&](Series &s, bool timed) {
        const auto t0 = Clock::now();
        double submit = 0, wait = 0;
        const auto step = [&](bool send, std::vector<std::uint8_t> &buf) {
            const auto a = Clock::now();
            auto w = send ? c.comm.send(host(buf.data(), bytes), bytes, DataType::UInt8, c.peer) : c.comm.recv(host(buf.data(), bytes), bytes, DataType::UInt8, c.peer);
            const auto b = Clock::now();
            w.wait();
            const auto d = Clock::now();
            submit += us_between(a, b);
            wait += us_between(b, d);
        };
        if (c.leader)
        {
            step(true, out);
            step(false, in);
        }
        else
        {
            step(false, in);
            step(true, in); // echo what was received
        }
        if (timed)
        {
            s.total.push_back(us_between(t0, Clock::now()));
            s.submit.push_back(submit);
            s.wait.push_back(wait);
        }
    };
    const auto verify = [&] {
        std::fill(in.begin(), in.end(), 0);
        if (c.leader)
        {
            c.comm.send(host(out.data(), bytes), bytes, DataType::UInt8, c.peer).wait();
            c.comm.recv(host(in.data(), bytes), bytes, DataType::UInt8, c.peer).wait();
            check_pat(in, c.rank, 0, "p2p echo " + std::to_string(bytes));
        }
        else
        {
            c.comm.recv(host(in.data(), bytes), bytes, DataType::UInt8, c.peer).wait();
            check_pat(in, c.peer, 0, "p2p received " + std::to_string(bytes));
            c.comm.send(host(in.data(), bytes), bytes, DataType::UInt8, c.peer).wait();
        }
    };
    run_point(c, "p2p", bytes, 2.0 * bytes, iterations_for(bytes, base), one, verify, "round trip; traffic is bytes each way");
}

void point_simul(Context &c, std::size_t bytes, int base)
{
    std::vector<std::uint8_t> out(bytes), in(bytes);
    fill_pat(out, c.rank, 1);
    const auto one = [&](Series &s, bool timed) {
        const auto t0 = Clock::now();
        auto sw = c.comm.send(host(out.data(), bytes), bytes, DataType::UInt8, c.peer);
        auto rw = c.comm.recv(host(in.data(), bytes), bytes, DataType::UInt8, c.peer);
        const auto t1 = Clock::now();
        sw.wait();
        rw.wait();
        const auto t2 = Clock::now();
        if (timed)
        {
            s.total.push_back(us_between(t0, t2));
            s.submit.push_back(us_between(t0, t1));
            s.wait.push_back(us_between(t1, t2));
        }
    };
    const auto verify = [&] {
        std::fill(in.begin(), in.end(), 0);
        auto sw = c.comm.send(host(out.data(), bytes), bytes, DataType::UInt8, c.peer);
        auto rw = c.comm.recv(host(in.data(), bytes), bytes, DataType::UInt8, c.peer);
        sw.wait();
        rw.wait();
        check_pat(in, c.peer, 1, "simul received " + std::to_string(bytes));
    };
    run_point(c, "simul", bytes, 2.0 * bytes, iterations_for(bytes, base), one, verify, "bidirectional exchange; traffic is bytes each way concurrently");
}

void point_stream(Context &c, std::size_t sender, std::size_t bytes, int base)
{
    const bool is_sender = c.rank == sender;
    const std::size_t window = std::clamp<std::size_t>((64u << 20) / std::max<std::size_t>(bytes, 1), 4, 64);
    std::vector<std::vector<std::uint8_t>> bufs(window, std::vector<std::uint8_t>(bytes));
    std::uint8_t ack_out = 0xA5, ack_in = 0;
    if (is_sender)
        for (std::size_t k = 0; k < window; ++k) fill_pat(bufs[k], sender, k);
    // a batch: sender posts `window` sends, waits, then waits for the ack; receiver posts `window` recvs and (after all complete) sends the ack.
    const auto batch = [&](Series *s, std::size_t count, bool verify_content) {
        if (is_sender)
        {
            const auto t0 = Clock::now();
            std::vector<tbccl::Work> works;
            works.reserve(count + 1);
            auto ack = c.comm.recv(host(&ack_in, 1), 1, DataType::UInt8, c.peer);
            for (std::size_t k = 0; k < count; ++k) works.push_back(c.comm.send(host(bufs[k].data(), bytes), bytes, DataType::UInt8, c.peer));
            const auto t1 = Clock::now();
            for (auto &w : works) w.wait();
            const auto t2 = Clock::now();
            ack.wait();
            const auto t3 = Clock::now();
            if (s)
            {
                s->submit.push_back(us_between(t0, t1));
                s->wait.push_back(us_between(t1, t2));
                s->total.push_back(us_between(t0, t3)); // acked
                s->extra1.push_back(us_between(t0, t2)); // sender complete
                s->extra2.push_back(us_between(t0, t3) - c.half_rtt_us); // acked, corrected by half a calibrated small round trip
            }
        }
        else
        {
            const auto t0 = Clock::now();
            std::vector<tbccl::Work> works;
            works.reserve(count);
            for (std::size_t k = 0; k < count; ++k) works.push_back(c.comm.recv(host(bufs[k].data(), bytes), bytes, DataType::UInt8, c.peer));
            const auto t1 = Clock::now();
            for (auto &w : works) w.wait();
            const auto t2 = Clock::now();
            c.comm.send(host(&ack_out, 1), 1, DataType::UInt8, c.peer).wait();
            if (verify_content)
                for (std::size_t k = 0; k < count; ++k) check_pat(bufs[k], sender, k, "stream message " + std::to_string(k) + " of " + std::to_string(bytes));
            if (s)
            {
                s->submit.push_back(us_between(t0, t1));
                s->wait.push_back(us_between(t1, t2));
                s->total.push_back(us_between(t0, t2)); // receiver: first post to last complete (own clock)
            }
        }
    };
    const int batches = std::max(iterations_for(bytes, base) / static_cast<int>(std::max<std::size_t>(window / 4, 1)), 8);
    const auto one = [&](Series &s, bool timed) { batch(timed ? &s : nullptr, window, false); };
    const auto verify = [&] {
        if (!is_sender) for (auto &b : bufs) std::fill(b.begin(), b.end(), 0);
        batch(nullptr, std::min<std::size_t>(window, 4), true);
    };
    char note[200];
    std::snprintf(note, sizeof note, "one-way stream %zu->%zu window=%zu; sender total=acked, extra1=sender complete, extra2=acked minus calibrated half 64B RTT; receiver total=first post to last complete", sender,
                  1 - sender, window);
    const char *name = sender == 0 ? "stream01" : "stream10";
    run_point(c, name, bytes, static_cast<double>(window) * bytes, batches, one, verify, note);
}

void point_allreduce(Context &c, std::size_t bytes, int base)
{
    const std::size_t elems = bytes / 4;
    std::vector<float> pristine(elems), f(elems);
    for (std::size_t i = 0; i < elems; ++i) pristine[i] = static_cast<float>((c.rank + 1) * ((i % 251) + 1));
    const auto one = [&](Series &s, bool timed) {
        std::memcpy(f.data(), pristine.data(), bytes);
        const auto t0 = Clock::now();
        auto w = c.comm.all_reduce(host(f.data(), bytes), host(f.data(), bytes), elems, DataType::Float32, tbccl::ReduceOp::Sum);
        const auto t1 = Clock::now();
        w.wait();
        const auto t2 = Clock::now();
        if (timed)
        {
            s.total.push_back(us_between(t0, t2));
            s.submit.push_back(us_between(t0, t1));
            s.wait.push_back(us_between(t1, t2));
        }
    };
    const auto verify = [&] {
        std::memcpy(f.data(), pristine.data(), bytes);
        c.comm.all_reduce(host(f.data(), bytes), host(f.data(), bytes), elems, DataType::Float32, tbccl::ReduceOp::Sum).wait();
        for (std::size_t i = 0; i < elems; ++i)
            if (f[i] != static_cast<float>(3 * ((i % 251) + 1))) fail("allreduce element " + std::to_string(i) + " of " + std::to_string(elems));
    };
    run_point(c, "allreduce", bytes, 2.0 * bytes, iterations_for(bytes, base), one, verify, "in-place float32 SUM; reduce-to-root then broadcast back: bytes each way, sequential");
}

void point_allgather(Context &c, std::size_t bytes, int base)
{
    std::vector<std::uint8_t> in(bytes), g0(bytes), g1(bytes);
    fill_pat(in, c.rank, 2);
    const auto one = [&](Series &s, bool timed) {
        const auto t0 = Clock::now();
        auto w = c.comm.all_gather(host(in.data(), bytes), {host(g0.data(), bytes), host(g1.data(), bytes)});
        const auto t1 = Clock::now();
        w.wait();
        const auto t2 = Clock::now();
        if (timed)
        {
            s.total.push_back(us_between(t0, t2));
            s.submit.push_back(us_between(t0, t1));
            s.wait.push_back(us_between(t1, t2));
        }
    };
    const auto verify = [&] {
        std::fill(g0.begin(), g0.end(), 0);
        std::fill(g1.begin(), g1.end(), 0);
        c.comm.all_gather(host(in.data(), bytes), {host(g0.data(), bytes), host(g1.data(), bytes)}).wait();
        check_pat(g0, 0, 2, "allgather slot 0 " + std::to_string(bytes));
        check_pat(g1, 1, 2, "allgather slot 1 " + std::to_string(bytes));
    };
    run_point(c, "allgather", bytes, 2.0 * bytes, iterations_for(bytes, base), one, verify, "bytes per rank; bytes each way, ordered (rank 0 sends first)");
}

void point_broadcast(Context &c, std::size_t root, std::size_t bytes, int base)
{
    std::vector<std::uint8_t> buf(bytes);
    const bool is_root = c.rank == root;
    if (is_root) fill_pat(buf, root, 3);
    const auto one = [&](Series &s, bool timed) {
        const auto t0 = Clock::now();
        auto w = c.comm.broadcast(host(buf.data(), bytes), root);
        const auto t1 = Clock::now();
        w.wait();
        const auto t2 = Clock::now();
        if (timed)
        {
            s.total.push_back(us_between(t0, t2));
            s.submit.push_back(us_between(t0, t1));
            s.wait.push_back(us_between(t1, t2));
        }
    };
    const auto verify = [&] {
        if (!is_root) std::fill(buf.begin(), buf.end(), 0);
        c.comm.broadcast(host(buf.data(), bytes), root).wait();
        check_pat(buf, root, 3, "broadcast from " + std::to_string(root) + " " + std::to_string(bytes));
    };
    run_point(c, root == 0 ? "broadcast0" : "broadcast1", bytes, static_cast<double>(bytes), iterations_for(bytes, base), one, verify, "one-way payload from the root");
}

void point_barrier(Context &c, int base)
{
    const auto one = [&](Series &s, bool timed) {
        const auto t0 = Clock::now();
        auto w = c.comm.barrier();
        const auto t1 = Clock::now();
        w.wait();
        const auto t2 = Clock::now();
        if (timed)
        {
            s.total.push_back(us_between(t0, t2));
            s.submit.push_back(us_between(t0, t1));
            s.wait.push_back(us_between(t1, t2));
        }
    };
    const auto verify = [&] {
        // the late rank enters 30 ms late; the early rank's barrier must not return before that (checked on the early rank's own clock)
        for (std::size_t late = 0; late < 2; ++late)
        {
            if (c.rank == late) std::this_thread::sleep_for(std::chrono::milliseconds(30));
            const auto t0 = Clock::now();
            c.comm.barrier().wait();
            const double waited = us_between(t0, Clock::now());
            if (c.rank != late && waited < 25000.0) fail("barrier returned after " + std::to_string(waited) + " us although rank " + std::to_string(late) + " entered 30 ms late");
        }
    };
    run_point(c, "barrier", 0, 0.0, base, one, verify, "no payload");
}

} // namespace

int main(int argc, char **argv)
{
    std::size_t rank = 0;
    std::string peers_arg, ops_arg = "p2p", sizes_arg = "64,4096,65536,262144,1048576,4194304,16777216", label = "run", samples_path;
    int iters = 1500;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--rank") rank = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--peers") peers_arg = next();
        else if (a == "--ops") ops_arg = next();
        else if (a == "--sizes") sizes_arg = next();
        else if (a == "--iters") iters = std::atoi(next().c_str());
        else if (a == "--label") label = next();
        else if (a == "--samples") samples_path = next();
    }
    tbccl::CommunicatorOptions o;
    o.rank = rank;
    for (const auto &p : split(peers_arg, ','))
    {
        const auto colon = p.rfind(':');
        o.peers.push_back({p.substr(0, colon), static_cast<std::uint16_t>(std::atoi(p.c_str() + colon + 1))});
    }
    if (o.peers.size() != 2)
    {
        std::fprintf(stderr, "need --peers ip0:port0,ip1:port1\n");
        return 2;
    }
    o.bootstrap_timeout = std::chrono::milliseconds(60000);
    const Usage init0 = usage();
    const auto t_init = Clock::now();
    auto comm = tbccl::Communicator::create(o);
    const double init_us = us_between(t_init, Clock::now());
    const Usage init1 = usage();
    Context c{*comm, rank, 1 - rank, rank == 0, label};
    if (!samples_path.empty()) c.samples = std::fopen((samples_path + ".r" + std::to_string(rank) + ".csv").c_str(), "w");
    std::printf("{\"label\":\"%s\",\"rank\":%zu,\"op\":\"init\",\"init_us\":%.0f,\"cpu_s\":%.4f,\"note\":\"Communicator::create including the bootstrap handshake and waiting for the peer\"}\n", label.c_str(), rank, init_us,
                init1.cpu_user + init1.cpu_sys - init0.cpu_user - init0.cpu_sys);
    std::fflush(stdout);
    comm->barrier().wait();

    // calibrate a small round trip (used only to correct the stream acknowledgement leg); untimed for reporting
    {
        std::vector<std::uint8_t> a(64, 1), b(64);
        std::vector<double> rtt;
        for (int i = 0; i < 400; ++i)
        {
            const auto t0 = Clock::now();
            if (c.leader)
            {
                comm->send(host(a.data(), 64), 64, DataType::UInt8, c.peer).wait();
                comm->recv(host(b.data(), 64), 64, DataType::UInt8, c.peer).wait();
            }
            else
            {
                comm->recv(host(b.data(), 64), 64, DataType::UInt8, c.peer).wait();
                comm->send(host(b.data(), 64), 64, DataType::UInt8, c.peer).wait();
            }
            if (i >= 100) rtt.push_back(us_between(t0, Clock::now()));
        }
        c.half_rtt_us = pct(rtt, 0.5) / 2.0;
        std::printf("{\"label\":\"%s\",\"rank\":%zu,\"op\":\"calibration\",\"rtt64_median_us\":%.2f}\n", label.c_str(), rank, pct(rtt, 0.5));
        std::fflush(stdout);
    }

    std::vector<std::size_t> sizes;
    for (const auto &t : split(sizes_arg, ',')) sizes.push_back(std::strtoull(t.c_str(), nullptr, 10));
    for (const auto &op : split(ops_arg, ','))
    {
        if (op == "barrier")
        {
            point_barrier(c, iters);
            continue;
        }
        for (std::size_t bytes : sizes)
        {
            if (op == "p2p") point_p2p(c, bytes, iters);
            else if (op == "simul") point_simul(c, bytes, iters);
            else if (op == "stream01") point_stream(c, 0, bytes, iters);
            else if (op == "stream10") point_stream(c, 1, bytes, iters);
            else if (op == "allreduce")
            {
                if (bytes >= 4 && bytes % 4 == 0) point_allreduce(c, bytes, iters);
            }
            else if (op == "allgather") point_allgather(c, bytes, iters);
            else if (op == "broadcast0") point_broadcast(c, 0, bytes, iters);
            else if (op == "broadcast1") point_broadcast(c, 1, bytes, iters);
            else
            {
                std::fprintf(stderr, "unknown op %s\n", op.c_str());
                return 2;
            }
        }
    }
    comm->barrier().wait();
    if (c.samples) std::fclose(c.samples);
    return 0;
}
