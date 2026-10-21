// Failure handling of the staged (chunked) pipeline of TensorCommWorker, with a fake transport and a fake staging backend that the test controls.
//
// The staged pipeline has two stages that share a StagingPool of `depth` slots: the staging stage and the network stage. When one stage fails, the other must terminate, whatever
// the interleaving of the two. The failure transition of a stage (src/core/tensor_comm_worker.cpp, fail_transfer) reports each of its steps to a test-only probe
// (src/core/tensor_comm_worker_probe.hpp); these tests use it to stall the failing stage after a chosen step while the other stage runs on, which is exactly the interleaving
// that used to leave a stage asleep in StagingPool::acquire():
//   send:    the network stage fails, its slot goes back to the pool, the staging stage takes it, publishes one more chunk and blocks in acquire() with every slot occupied;
//   receive: the same with the roles swapped (a failed destination commit, the network stage receiving into the freed slot).
// Every scenario is bounded: a request that has not finished at the deadline ends the process with a diagnostic and status 124, so a regression is a failing test and not a hang.
//
//   tensor_comm_worker_failure_test [--only <substring of a scenario name>]

#include "../src/core/tensor_comm_worker_probe.hpp"

#include <tbccl/async_transfer.hpp>
#include <tbccl/error.hpp>
#include <tbccl/transport.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <dirent.h>
#endif

namespace
{
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kChunk = 4096;

std::string g_scenario = "(none)";

void dump_threads()
{
#ifdef __linux__
    std::fprintf(stderr, "threads (tid state comm wchan):\n");
    if (DIR *d = ::opendir("/proc/self/task"))
    {
        while (dirent *e = ::readdir(d))
        {
            if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
            auto line = [&](const char *file) {
                std::string path = std::string("/proc/self/task/") + e->d_name + "/" + file, out;
                if (FILE *f = std::fopen(path.c_str(), "r"))
                {
                    char buf[256];
                    if (std::fgets(buf, sizeof buf, f)) out = buf;
                    std::fclose(f);
                }
                while (!out.empty() && (out.back() == '\n')) out.pop_back();
                return out;
            };
            const std::string stat = line("stat");
            const auto rp = stat.rfind(')');
            std::fprintf(stderr, "  %s %c %s %s\n", e->d_name, rp != std::string::npos && rp + 2 < stat.size() ? stat[rp + 2] : '?', line("comm").c_str(), line("wchan").c_str());
        }
        ::closedir(d);
    }
#endif
}

[[noreturn]] void die(const std::string &what)
{
    std::fprintf(stderr, "FAIL [%s]: %s\n", g_scenario.c_str(), what.c_str());
    std::fflush(stderr);
    std::_Exit(1);
}

void expect(bool condition, const std::string &what)
{
    if (!condition) die(what);
}

// Ends the process if a scenario takes longer than `limit`. Disarmed by the destructor.
class Deadline
{
public:
    explicit Deadline(std::chrono::milliseconds limit) : thread_([this, limit] {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, limit, [&] { return done_; }))
        {
            std::fprintf(stderr, "STALLED [%s]: not finished within %lld ms (a pipeline stage is waiting for a slot or for the other stage)\n", g_scenario.c_str(),
                         static_cast<long long>(limit.count()));
            dump_threads();
            std::fflush(stderr);
            std::_Exit(124);
        }
    })
    {
    }
    ~Deadline()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool done_ = false;
    std::thread thread_;
};

class Gate
{
public:
    void open()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            open_ = true;
        }
        cv_.notify_all();
    }
    bool wait(std::chrono::milliseconds limit)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, limit, [&] { return open_; });
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool open_ = false;
};

class Counter
{
public:
    void add()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++n_;
        }
        cv_.notify_all();
    }
    std::size_t value()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return n_;
    }
    bool wait_at_least(std::size_t n, std::chrono::milliseconds limit)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, limit, [&] { return n_ >= n; });
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::size_t n_ = 0;
};

std::uint8_t pattern(std::size_t request, std::size_t offset) { return static_cast<std::uint8_t>(request * 31 + offset * 7 + (offset >> 8) * 3 + 1); }

// One scenario's behaviour. Chunk indices are counted per request.
struct Plan
{
    std::size_t request_id = 1;
    int fail_stage_at = -1;       // send: stage_source_chunk throws at this chunk index; recv: not used
    int fail_commit_at = -1;      // recv: commit_destination_chunk throws at this chunk index
    int fail_send_at = -1;        // transport->send throws at this call index
    int fail_recv_at = -1;        // transport->recv throws at this call index
    Gate *hold_send = nullptr;    // transport->send waits here before it (possibly) fails
    Gate *hold_commit = nullptr;  // commit_destination_chunk waits here before it (possibly) fails
};

class FakeBackend final : public tbccl::AsyncMemoryBackend
{
public:
    FakeBackend(std::size_t bytes, Plan &plan) : data_(bytes, 0), plan_(plan) {}
    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
    {
        const int index = static_cast<int>(chunk.offset / kChunk);
        staged.add();
        {
            std::lock_guard<std::mutex> lock(m_);
            staging_pointers.insert(staging);
        }
        if (index == plan_.fail_stage_at) throw tbccl::Error(tbccl::ErrorCode::DeviceError, "injected staging failure");
        auto *p = static_cast<std::uint8_t *>(staging);
        for (std::size_t j = 0; j < chunk.size; ++j) p[j] = pattern(plan_.request_id, chunk.offset + j);
    }
    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override
    {
        const int index = static_cast<int>(chunk.offset / kChunk);
        committed_calls.add();
        {
            std::lock_guard<std::mutex> lock(m_);
            staging_pointers.insert(const_cast<void *>(staging));
        }
        if (plan_.hold_commit && index == 0) plan_.hold_commit->wait(30s);
        if (index == plan_.fail_commit_at) throw tbccl::Error(tbccl::ErrorCode::DeviceError, "injected commit failure");
        std::memcpy(data_.data() + chunk.offset, staging, chunk.size);
    }
    bool destination_matches(std::size_t request_id) const
    {
        for (std::size_t j = 0; j < data_.size(); ++j)
            if (data_[j] != pattern(request_id, j)) return false;
        return true;
    }
    Counter staged, committed_calls;
    std::set<void *> staging_pointers;

private:
    std::vector<std::uint8_t> data_;
    Plan &plan_;
    std::mutex m_;
};

class FakeTransport final : public tbccl::Transport
{
public:
    explicit FakeTransport(Plan &plan) : plan_(plan) {}
    void send(const void *data, std::size_t bytes) override
    {
        const int index = static_cast<int>(send_calls_.value());
        send_calls_.add();
        if (plan_.hold_send && index == 0) plan_.hold_send->wait(30s);
        if (index == plan_.fail_send_at) throw tbccl::Error(tbccl::ErrorCode::TransportError, "injected send failure");
        const auto *p = static_cast<const std::uint8_t *>(data);
        std::lock_guard<std::mutex> lock(m_);
        sent_.insert(sent_.end(), p, p + bytes);
    }
    void recv(void *data, std::size_t bytes) override
    {
        const int index = static_cast<int>(recv_calls.value());
        recv_calls.add();
        if (index == plan_.fail_recv_at) throw tbccl::Error(tbccl::ErrorCode::TransportError, "injected recv failure");
        auto *p = static_cast<std::uint8_t *>(data);
        const std::size_t offset = static_cast<std::size_t>(index) * kChunk;
        for (std::size_t j = 0; j < bytes; ++j) p[j] = pattern(plan_.request_id, offset + j);
    }
    tbccl::TransportCapabilities capabilities() const noexcept override { return {}; }
    std::string peer_name() const override { return "fake"; }
    bool sent_matches(std::size_t request_id, std::size_t bytes) const
    {
        std::lock_guard<std::mutex> lock(m_);
        if (sent_.size() != bytes) return false;
        for (std::size_t j = 0; j < bytes; ++j)
            if (sent_[j] != pattern(request_id, j)) return false;
        return true;
    }
    Counter &send_calls() { return send_calls_; }
    Counter recv_calls;

private:
    Plan &plan_;
    Counter send_calls_;
    mutable std::mutex m_;
    std::vector<std::uint8_t> sent_;
};

tbccl::TransferRequest make_request(tbccl::TransferDirection dir, FakeBackend &backend, FakeTransport &transport, std::size_t chunks, std::size_t id)
{
    tbccl::TransferRequest r;
    r.transfer_id = id;
    r.direction = dir;
    r.backend = &backend;
    r.transport = &transport;
    r.total_bytes = chunks * kChunk;
    r.chunk_hint = kChunk;
    return r;
}

void check_failed(tbccl::TransferWork &work, tbccl::ErrorCode code, const char *original_text)
{
    expect(work.wait_for(10s), "the request did not reach a terminal state");
    expect(work.is_completed(), "the request is terminal");
    expect(work.has_error(), "a failed request must report an error");
    expect(work.error_code() == code, "the original error code must be preserved");
    expect(work.error().find(original_text) != std::string::npos, "the original error text must be preserved, got: " + work.error());
    expect(work.error().find("staging pool cancelled") == std::string::npos, "the cancellation must not replace the original error");
}

// --- the probe: stalls the failing stage after a chosen step ----------------------------------------------------------------------------------------------------------------
struct ProbeState
{
    std::atomic<bool> armed{false};
    const char *stage = "";
    const char *step = "";
    Gate reached;
    Gate release;
    std::atomic<int> hits{0};
} g_probe;

void probe_fn(const char *stage, const char *step)
{
    if (!g_probe.armed.load()) return;
    if (std::strcmp(stage, g_probe.stage) != 0 || std::strcmp(step, g_probe.step) != 0) return;
    g_probe.hits.fetch_add(1);
    g_probe.reached.open();
    g_probe.release.wait(30s);
}

// How many chunks the other stage may get through before the failing stage is released. With the failure transition as designed the pool is cancelled before the slot goes back,
// so the staging stage can never take the freed slot; the bounded wait only lets a stage that is able to run do so (it returns at once when the count is reached).
bool let_other_stage_run(Counter &progress, std::size_t target, std::chrono::milliseconds limit) { return progress.wait_at_least(target, limit); }

// --- scenarios --------------------------------------------------------------------------------------------------------------------------------------------------------------
// Send. depth 2, 6 chunks: the staging stage fills both slots with chunks 0 and 1 and blocks in acquire() for chunk 2 while the network stage is inside send() of chunk 0.
// send() of chunk 0 then fails; the failing network stage is stalled after `step`.
void scenario_send_network_failure(const char *step)
{
    g_scenario = std::string("send network failure, stalled after ") + step;
    Deadline deadline(40s);
    Plan plan;
    Gate hold_send;
    plan.hold_send = &hold_send;
    plan.fail_send_at = 0;
    FakeBackend backend(6 * kChunk, plan);
    FakeTransport transport(plan);
    tbccl::TensorCommWorker worker(2, 8);
    g_probe.stage = "send.network";
    g_probe.step = step;
    g_probe.hits = 0;
    g_probe.armed = true;
    tbccl::detail::set_pipeline_failure_probe(&probe_fn);
    auto work = worker.enqueue(make_request(tbccl::TransferDirection::Send, backend, transport, 6, 1));
    expect(backend.staged.wait_at_least(2, 10s), "the staging stage should have filled both slots");
    expect(transport.send_calls().wait_at_least(1, 10s), "the network stage should be inside send()");
    hold_send.open(); // send() of chunk 0 now throws
    expect(g_probe.reached.wait(10s), std::string("the failure transition never reached step ") + step);
    // Let the staging stage run on while the failing stage is stalled. Before the fix, it took the freed slot, published chunk 2 and blocked in acquire() for chunk 3.
    let_other_stage_run(backend.staged, 3, std::string(step) == "slot_released" ? 700ms : 100ms);
    g_probe.release.open();
    check_failed(work, tbccl::ErrorCode::TransportError, "injected send failure");
    worker.wait_idle();
    const auto stats = worker.stats();
    expect(stats.submitted == 1 && stats.failed == 1 && stats.completed == 0, "exactly one terminal completion, a failure");
    expect(backend.staged.value() <= 3, "no chunk beyond the ones the pipeline could stage may be processed");
    tbccl::detail::set_pipeline_failure_probe(nullptr);
    g_probe.armed = false;
    std::printf("PASS %s\n", g_scenario.c_str());
}

// Receive. depth 2, 6 chunks: the network stage receives chunks 0 and 1 into both slots and blocks in acquire() for chunk 2 while the staging stage is inside the commit of
// chunk 0, which fails; the failing staging stage is stalled after `step`.
void scenario_recv_staging_failure(const char *step)
{
    g_scenario = std::string("recv commit failure, stalled after ") + step;
    Deadline deadline(40s);
    Plan plan;
    plan.request_id = 2;
    Gate hold_commit;
    plan.hold_commit = &hold_commit;
    plan.fail_commit_at = 0;
    FakeBackend backend(6 * kChunk, plan);
    FakeTransport transport(plan);
    tbccl::TensorCommWorker worker(2, 8);
    g_probe.stage = "recv.staging";
    g_probe.step = step;
    g_probe.hits = 0;
    g_probe.armed = true;
    tbccl::detail::set_pipeline_failure_probe(&probe_fn);
    auto work = worker.enqueue(make_request(tbccl::TransferDirection::Recv, backend, transport, 6, 2));
    expect(transport.recv_calls.wait_at_least(2, 10s), "the network stage should have received chunks 0 and 1");
    expect(backend.committed_calls.wait_at_least(1, 10s), "the staging stage should be inside the commit of chunk 0");
    hold_commit.open(); // commit of chunk 0 now throws
    expect(g_probe.reached.wait(10s), std::string("the failure transition never reached step ") + step);
    // Before the fix, the network stage took the freed slot, received chunk 2 and blocked in acquire() for chunk 3.
    let_other_stage_run(transport.recv_calls, 3, std::string(step) == "slot_released" ? 700ms : 100ms);
    g_probe.release.open();
    check_failed(work, tbccl::ErrorCode::DeviceError, "injected commit failure");
    worker.wait_idle();
    const auto stats = worker.stats();
    expect(stats.submitted == 1 && stats.failed == 1 && stats.completed == 0, "exactly one terminal completion, a failure");
    tbccl::detail::set_pipeline_failure_probe(nullptr);
    g_probe.armed = false;
    std::printf("PASS %s\n", g_scenario.c_str());
}

// --- injected failures at the first / a middle / the last chunk, for every component and depth ------------------------------------------------------------------------------
struct Injection
{
    const char *name;
    tbccl::TransferDirection dir;
    tbccl::ErrorCode code;
    const char *text;
    std::function<void(Plan &, int)> set;
};

// A probe that throws at one step of the failure transition (every stage). The transition must survive it: the request still ends, with the original error, and the next request works.
const char *g_throw_step = nullptr;
void throwing_probe(const char *, const char *step)
{
    if (g_throw_step != nullptr && std::strcmp(step, g_throw_step) == 0) throw std::runtime_error("injected exception inside the failure transition");
}

void scenario_injection(const Injection &inj, std::size_t depth, int at, std::size_t chunks, const char *throw_at_step = nullptr)
{
    g_scenario = std::string(inj.name) + " at chunk " + std::to_string(at) + " of " + std::to_string(chunks) + ", depth " + std::to_string(depth);
    if (throw_at_step != nullptr)
    {
        g_scenario += std::string(", exception thrown inside the transition at ") + throw_at_step;
        g_throw_step = throw_at_step;
        tbccl::detail::set_pipeline_failure_probe(&throwing_probe);
    }
    Deadline deadline(40s);
    Plan plan;
    plan.request_id = 3;
    inj.set(plan, at);
    FakeBackend backend(chunks * kChunk, plan);
    FakeTransport transport(plan);
    tbccl::TensorCommWorker worker(depth, 8);
    auto work = worker.enqueue(make_request(inj.dir, backend, transport, chunks, 3));
    check_failed(work, inj.code, inj.text);
    worker.wait_idle();
    auto stats = worker.stats();
    expect(stats.submitted == 1 && stats.failed == 1 && stats.completed == 0, "exactly one terminal completion, a failure");
    if (throw_at_step != nullptr)
    {
        tbccl::detail::set_pipeline_failure_probe(nullptr);
        g_throw_step = nullptr;
    }

    // an unrelated request after the failure: same geometry, no failure, data intact (a cancelled pool must not be reused)
    Plan good;
    good.request_id = 4;
    FakeBackend backend2(chunks * kChunk, good);
    FakeTransport transport2(good);
    auto work2 = worker.enqueue(make_request(inj.dir, backend2, transport2, chunks, 4));
    expect(work2.wait_for(10s), "the request after a failure did not finish");
    expect(!work2.has_error(), "the request after a failure must succeed, got: " + work2.error());
    if (inj.dir == tbccl::TransferDirection::Send) expect(transport2.sent_matches(4, chunks * kChunk), "data of the request after a failure");
    else expect(backend2.destination_matches(4), "data of the request after a failure");
    worker.wait_idle();
    stats = worker.stats();
    expect(stats.submitted == 2 && stats.failed == 1 && stats.completed == 1, "stats after a failure and a success");
    std::printf("PASS %s\n", g_scenario.c_str());
}

// --- successful transfers: geometry, ordering, pool reuse ---------------------------------------------------------------------------------------------------------------------
void scenario_success(tbccl::TransferDirection dir, std::size_t depth, std::size_t chunks, std::size_t total_extra)
{
    g_scenario = std::string(dir == tbccl::TransferDirection::Send ? "send" : "recv") + " success, depth " + std::to_string(depth) + ", " + std::to_string(chunks) + " chunks +" + std::to_string(total_extra);
    Deadline deadline(40s);
    tbccl::TensorCommWorker worker(depth, 8);
    std::set<void *> first_pointers;
    for (std::size_t round = 0; round < 3; ++round)
    {
        Plan plan;
        plan.request_id = 10 + round;
        const std::size_t bytes = chunks * kChunk + total_extra; // a short last chunk when total_extra != 0
        FakeBackend backend(bytes, plan);
        FakeTransport transport(plan);
        auto r = make_request(dir, backend, transport, chunks, plan.request_id);
        r.total_bytes = bytes;
        auto work = worker.enqueue(r);
        expect(work.wait_for(10s), "a successful request did not finish");
        expect(!work.has_error(), "a successful request reported an error: " + work.error());
        if (dir == tbccl::TransferDirection::Send) expect(transport.sent_matches(plan.request_id, bytes), "sent bytes in order and complete");
        else expect(backend.destination_matches(plan.request_id), "received bytes in order and complete");
        // Which slots a request touches depends on when the other stage returns them, so the sets need not be equal; a reused pool must however hand out buffers of the earlier
        // request. (A pool allocated while the old one is alive would have entirely different addresses.)
        if (round == 0) first_pointers = backend.staging_pointers;
        else
        {
            bool overlap = false;
            for (void *pointer : backend.staging_pointers) overlap = overlap || first_pointers.count(pointer) != 0;
            expect(overlap, "the staging buffers of a same-shaped request must be reused (cached pool)");
        }
    }
    worker.wait_idle(); // the statistics are recorded just after a Work becomes terminal
    const auto stats = worker.stats();
    expect(stats.submitted == 3 && stats.completed == 3 && stats.failed == 0, "three completions");
    std::printf("PASS %s\n", g_scenario.c_str());
}

// --- abort while the pipeline is full ------------------------------------------------------------------------------------------------------------------------------------------
// The transport is interrupted by its owner (here: the held send() throws once the abort has been issued, as Transport::abort makes a real send do).
void scenario_abort_with_full_pipeline()
{
    g_scenario = "abort with a full pipeline";
    Deadline deadline(40s);
    Plan plan;
    Gate hold_send;
    plan.hold_send = &hold_send;
    plan.fail_send_at = 0;
    FakeBackend backend(8 * kChunk, plan);
    FakeTransport transport(plan);
    tbccl::TensorCommWorker worker(2, 8);
    auto work = worker.enqueue(make_request(tbccl::TransferDirection::Send, backend, transport, 8, 5));
    expect(backend.staged.wait_at_least(2, 10s), "both slots should fill");
    expect(transport.send_calls().wait_at_least(1, 10s), "the network stage should be inside send()");
    worker.abort("test abort");
    worker.abort("test abort again"); // idempotent
    hold_send.open();                 // the interrupted send() fails
    expect(work.wait_for(10s), "the aborted request did not reach a terminal state");
    expect(work.has_error() && work.error_code() == tbccl::ErrorCode::Aborted, "an aborted request reports Aborted, got: " + work.error());
    worker.wait_idle();
    const auto stats = worker.stats();
    expect(stats.submitted == 1 && stats.failed == 1 && stats.completed == 0, "exactly one terminal completion");
    std::printf("PASS %s\n", g_scenario.c_str());
}
} // namespace

int main(int argc, char **argv)
{
    std::string only;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) only = argv[++i];
        else
        {
            std::fprintf(stderr, "usage: tensor_comm_worker_failure_test [--only <substring>]\n");
            return 2;
        }
    }
    auto wanted = [&](const std::string &name) { return only.empty() || name.find(only) != std::string::npos; };
    int ran = 0;
    const char *steps[] = {"begin", "recorded", "cancelled", "slot_released"};
    for (const char *step : steps)
    {
        if (wanted(std::string("send network failure, stalled after ") + step)) { scenario_send_network_failure(step); ++ran; }
        if (wanted(std::string("recv commit failure, stalled after ") + step)) { scenario_recv_staging_failure(step); ++ran; }
    }

    const Injection injections[] = {
        {"send: transport failure", tbccl::TransferDirection::Send, tbccl::ErrorCode::TransportError, "injected send failure", [](Plan &p, int at) { p.fail_send_at = at; }},
        {"send: source staging failure", tbccl::TransferDirection::Send, tbccl::ErrorCode::DeviceError, "injected staging failure", [](Plan &p, int at) { p.fail_stage_at = at; }},
        {"recv: transport failure", tbccl::TransferDirection::Recv, tbccl::ErrorCode::TransportError, "injected recv failure", [](Plan &p, int at) { p.fail_recv_at = at; }},
        {"recv: destination commit failure", tbccl::TransferDirection::Recv, tbccl::ErrorCode::DeviceError, "injected commit failure", [](Plan &p, int at) { p.fail_commit_at = at; }},
    };
    for (const auto &inj : injections)
        for (std::size_t depth : {1u, 2u, 4u})
            for (int at : {0, 3, 5})
            {
                const std::string name = std::string(inj.name) + " at chunk " + std::to_string(at) + " of 6, depth " + std::to_string(depth);
                if (wanted(name)) { scenario_injection(inj, depth, at, 6); ++ran; }
            }

    for (const auto &inj : injections)
        for (const char *step : steps)
        {
            const std::string name = std::string(inj.name) + " at chunk 3 of 6, depth 2, exception thrown inside the transition at " + step;
            if (wanted(name)) { scenario_injection(inj, 2, 3, 6, step); ++ran; }
        }

    for (auto dir : {tbccl::TransferDirection::Send, tbccl::TransferDirection::Recv})
        for (std::size_t depth : {1u, 2u, 3u})
            for (std::size_t chunks : {1u, 2u, 3u, 7u})
                for (std::size_t extra : {0u, 100u})
                {
                    const std::string name = std::string(dir == tbccl::TransferDirection::Send ? "send" : "recv") + " success, depth " + std::to_string(depth) + ", " + std::to_string(chunks) + " chunks +" + std::to_string(extra);
                    if (wanted(name)) { scenario_success(dir, depth, chunks, extra); ++ran; }
                }

    if (wanted("abort with a full pipeline")) { scenario_abort_with_full_pipeline(); ++ran; }
    if (ran == 0)
    {
        std::fprintf(stderr, "no scenario matches '%s'\n", only.c_str());
        return 2;
    }
    std::printf("tensor_comm_worker_failure_test passed (%d scenarios)\n", ran);
    return 0;
}
