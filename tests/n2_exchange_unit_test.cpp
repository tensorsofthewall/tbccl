// The N=2 all_gather exchange (detail::run_n2_exchange) over a controlled transport and a real duplex TensorCommWorker.
//
// Proof of duplex execution without timing thresholds: the transport's send() cannot finish until a recv() has started, and recv() cannot finish until a send() has
// started (a rendezvous). An exchange that runs its two legs one after the other never gets past the first leg; that is detected by a bounded wait that records a
// timeout, not by measuring how long anything took. The same harness run over a sequential schedule is the negative control.
//
// Failure injection checks, for every scenario: the exchange throws the first cause (not the Aborted it triggered), the communicator-level abort callback ran, and
// NO transfer is still inside the transport when run_n2_exchange returns (the providers' memory may be released by the caller at that point).
// Run once per chunk setting (TBCCL_STAGED_CHUNK_BYTES unset, and a small value that splits the staged payloads into several chunks).

#include "communicator_debug.hpp"

#include <tbccl/async_transfer.hpp>
#include <tbccl/error.hpp>
#include <tbccl/transport.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{
using namespace std::chrono_literals;

std::string g_scenario = "startup";

[[noreturn]] void die(const std::string &what)
{
    std::fprintf(stderr, "FAIL [%s]: %s\n", g_scenario.c_str(), what.c_str());
    std::fflush(stderr);
    std::_Exit(1);
}
void expect(bool ok, const std::string &what)
{
    if (!ok) die(what);
}

// Ends the process if a scenario is still running after `limit` (a hang is a failure, never a stuck test).
class Deadline
{
public:
    explicit Deadline(std::chrono::milliseconds limit) : thread_([this, limit] {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!cv_.wait_for(lock, limit, [&] { return done_; }))
        {
            std::fprintf(stderr, "STALLED [%s]: not finished within %lld ms\n", g_scenario.c_str(), static_cast<long long>(limit.count()));
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

constexpr std::size_t kBytes = 5 * 4096 + 17; // several chunks (and a partial last one) when TBCCL_STAGED_CHUNK_BYTES=4096

std::uint8_t pat(std::size_t seed, std::size_t j) { return static_cast<std::uint8_t>(seed * 37 + j * 7 + (j >> 9) * 5 + 3); }

struct Script
{
    std::chrono::milliseconds rendezvous_limit{5000}; // how long a leg waits for the other leg to have started
    bool rendezvous = true;                            // send/recv cannot complete before the other has started
    bool recv_preloaded = false;                       // recv returns data at once (as if the peer had already sent)
    bool fail_send = false;                            // send() throws TransportError at once
    bool fail_recv = false;                            // recv() throws TransportError at once
    bool hold_send = false;                            // send() blocks until the transport is aborted
    bool hold_recv = false;                            // recv() blocks until the transport is aborted
};

// A loopback "wire": bytes sent are what a recv returns. send/recv may rendezvous (see the file comment) or be scripted to fail or block.
class ScriptedTransport final : public tbccl::Transport
{
public:
    explicit ScriptedTransport(Script script) : script_(script) {}

    void send(const void *data, std::size_t bytes) override
    {
        Active active(*this);
        std::unique_lock<std::mutex> lock(m_);
        send_entered_ = true;
        cv_.notify_all();
        if (script_.fail_send) throw tbccl::Error(tbccl::ErrorCode::TransportError, "injected send failure");
        if (script_.hold_send) hold(lock);
        if (script_.rendezvous && !cv_.wait_for(lock, script_.rendezvous_limit, [&] { return recv_entered_ || aborted_; }))
        {
            rendezvous_timeouts_.fetch_add(1);
            throw tbccl::Error(tbccl::ErrorCode::Timeout, "send never saw a receive start (the legs ran one after the other)");
        }
        throw_if_aborted();
        const auto *p = static_cast<const std::uint8_t *>(data);
        wire_.insert(wire_.end(), p, p + bytes);
        cv_.notify_all();
    }

    void recv(void *data, std::size_t bytes) override
    {
        Active active(*this);
        std::unique_lock<std::mutex> lock(m_);
        recv_entered_ = true;
        cv_.notify_all();
        if (script_.fail_recv) throw tbccl::Error(tbccl::ErrorCode::TransportError, "injected recv failure");
        if (script_.hold_recv) hold(lock);
        if (script_.recv_preloaded)
        {
            auto *p = static_cast<std::uint8_t *>(data);
            for (std::size_t j = 0; j < bytes; ++j) p[j] = pat(preload_seed_, preload_offset_ + j);
            preload_offset_ += bytes;
            return;
        }
        if (script_.rendezvous && !cv_.wait_for(lock, script_.rendezvous_limit, [&] { return send_entered_ || aborted_; }))
        {
            rendezvous_timeouts_.fetch_add(1);
            throw tbccl::Error(tbccl::ErrorCode::Timeout, "recv never saw a send start (the legs ran one after the other)");
        }
        if (!cv_.wait_for(lock, script_.rendezvous_limit, [&] { return wire_.size() >= bytes || aborted_; })) throw tbccl::Error(tbccl::ErrorCode::Timeout, "recv: no data");
        throw_if_aborted();
        auto *p = static_cast<std::uint8_t *>(data);
        for (std::size_t j = 0; j < bytes; ++j)
        {
            p[j] = wire_.front();
            wire_.pop_front();
        }
    }

    tbccl::TransportCapabilities capabilities() const noexcept override { return {}; }
    std::string peer_name() const override { return "scripted"; }

    void abort(const std::string &reason) override
    {
        {
            std::lock_guard<std::mutex> lock(m_);
            aborted_ = true;
            reason_ = reason;
        }
        cv_.notify_all();
    }

    // Fresh script and empty wire for the next exchange on the same worker (a prior exchange must be finished).
    void reconfigure(Script script)
    {
        std::lock_guard<std::mutex> lock(m_);
        script_ = script;
        wire_.clear();
        send_entered_ = recv_entered_ = aborted_ = false;
        preload_offset_ = 0;
    }
    bool wait_both_entered(std::chrono::milliseconds limit)
    {
        std::unique_lock<std::mutex> lock(m_);
        return cv_.wait_for(lock, limit, [&] { return send_entered_ && recv_entered_; });
    }
    bool was_aborted()
    {
        std::lock_guard<std::mutex> lock(m_);
        return aborted_;
    }
    int active() const { return active_.load(); }
    int rendezvous_timeouts() const { return rendezvous_timeouts_.load(); }
    void set_preload_seed(std::size_t seed) { preload_seed_ = seed; }

private:
    struct Active
    {
        explicit Active(ScriptedTransport &t) : t_(t) { t_.active_.fetch_add(1); }
        ~Active() { t_.active_.fetch_sub(1); }
        ScriptedTransport &t_;
    };
    // A blocked leg takes a moment to unwind once the transport is aborted, so a run_n2_exchange that returned without waiting for it is caught every time
    // (not only when the scheduler happens to be slow). The delay widens a window; no assertion depends on how long anything took.
    void hold(std::unique_lock<std::mutex> &lock)
    {
        cv_.wait(lock, [&] { return aborted_; });
        lock.unlock();
        std::this_thread::sleep_for(50ms);
        lock.lock();
        throw_if_aborted();
    }
    void throw_if_aborted()
    {
        if (aborted_) throw tbccl::Error(tbccl::ErrorCode::Aborted, "aborted: communicator aborted (" + reason_ + ")");
    }

    Script script_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::uint8_t> wire_;
    bool send_entered_ = false, recv_entered_ = false, aborted_ = false;
    std::string reason_;
    std::atomic<int> active_{0};
    std::atomic<int> rendezvous_timeouts_{0};
    std::size_t preload_seed_ = 0, preload_offset_ = 0;
};

// Host memory the transport can read and write in place (the direct path: no staging).
class DirectBackend final : public tbccl::AsyncMemoryBackend
{
public:
    explicit DirectBackend(std::vector<std::uint8_t> &buffer) : buffer_(buffer) {}
    void stage_source_chunk(const tbccl::Chunk &, void *) override { std::abort(); }
    void commit_destination_chunk(const tbccl::Chunk &, const void *) override { std::abort(); }
    bool supports_direct_transport_access() const noexcept override { return true; }
    const void *direct_source_data() const noexcept override { return buffer_.data(); }
    void *direct_destination_data() noexcept override { return buffer_.data(); }

private:
    std::vector<std::uint8_t> &buffer_;
};

// Memory that must be staged (like device memory): the worker copies chunks through its staging pool. Can fail at the source copy or the destination commit.
class StagedBackend final : public tbccl::AsyncMemoryBackend
{
public:
    explicit StagedBackend(std::vector<std::uint8_t> &buffer) : buffer_(buffer) {}
    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
    {
        if (fail_stage || (fail_stage_last && chunk.offset + chunk.size == kBytes)) throw tbccl::Error(tbccl::ErrorCode::DeviceError, "injected staging failure");
        std::memcpy(staging, buffer_.data() + chunk.offset, chunk.size);
    }
    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override
    {
        if (fail_commit || (fail_commit_last && chunk.offset + chunk.size == kBytes)) throw tbccl::Error(tbccl::ErrorCode::DeviceError, "injected commit failure");
        std::memcpy(buffer_.data() + chunk.offset, staging, chunk.size);
    }
    bool fail_stage = false, fail_commit = false;           // fail on the first chunk
    bool fail_stage_last = false, fail_commit_last = false; // fail only on the final (partial) chunk, after earlier chunks succeeded

private:
    std::vector<std::uint8_t> &buffer_;
};


struct Rig
{
    // `send_staged` / `recv_staged` choose the provider kind of each direction independently (a direct host buffer or memory that must be staged, like device
    // memory); the mixed combinations are the heterogeneous case: one leg is chunked through a staging pool while the other is read or written in place.
    Rig(Script script, bool send_staged, bool recv_staged)
        : transport(script), worker(2, tbccl::TensorCommWorker::kUnboundedAdmission, /*duplex=*/true), source(kBytes), dest(kBytes, 0)
    {
        for (std::size_t j = 0; j < kBytes; ++j) source[j] = pat(1, j);
        if (send_staged) staged_send = std::make_unique<StagedBackend>(source);
        else direct_send = std::make_unique<DirectBackend>(source);
        if (recv_staged) staged_recv = std::make_unique<StagedBackend>(dest);
        else direct_recv = std::make_unique<DirectBackend>(dest);
        transport.set_preload_seed(7);
    }
    Rig(Script script, bool staged) : Rig(script, staged, staged) {}
    tbccl::AsyncMemoryBackend &send_backend() { return staged_send ? static_cast<tbccl::AsyncMemoryBackend &>(*staged_send) : *direct_send; }
    tbccl::AsyncMemoryBackend &recv_backend() { return staged_recv ? static_cast<tbccl::AsyncMemoryBackend &>(*staged_recv) : *direct_recv; }
    // What the communicator's abort does to the exchange's transfers: interrupt the worker's lanes and the transport.
    std::function<void(const std::string &)> fatal_callback()
    {
        return [this](const std::string &why) {
            fatal_calls.fetch_add(1);
            worker.abort(why);
            transport.abort(why);
        };
    }
    // Runs the exchange; returns the exception's code (Success when it returned normally) and text.
    std::pair<tbccl::ErrorCode, std::string> run() { return run_with(fatal_callback()); }
    std::pair<tbccl::ErrorCode, std::string> run_with(const std::function<void(const std::string &)> &fatal)
    {
        try
        {
            tbccl::detail::run_n2_exchange(worker, transport, send_backend(), recv_backend(), kBytes, "all_gather", fatal);
        }
        catch (const tbccl::Error &e)
        {
            return {e.code(), e.what()};
        }
        catch (const std::bad_alloc &)
        {
            return {tbccl::ErrorCode::ResourceExhausted, "bad_alloc"};
        }
        return {tbccl::ErrorCode::Success, ""};
    }
    // Everything the exchange started is finished: nothing is inside the transport and both lanes are idle.
    void expect_quiescent()
    {
        expect(transport.active() == 0, "a transfer is still inside the transport after the exchange returned");
        worker.wait_idle();
        expect(!worker.busy(), "the worker still reports in-flight transfers");
    }
    bool dest_matches_source() const { return dest == source; }
    bool dest_matches_preload() const
    {
        for (std::size_t j = 0; j < kBytes; ++j)
            if (dest[j] != pat(7, j)) return false;
        return true;
    }

    ScriptedTransport transport;
    tbccl::TensorCommWorker worker;
    std::vector<std::uint8_t> source, dest;
    std::unique_ptr<DirectBackend> direct_send, direct_recv;
    std::unique_ptr<StagedBackend> staged_send, staged_recv;
    std::atomic<int> fatal_calls{0};
};

void scenario(const char *name, const std::function<void()> &body)
{
    g_scenario = name;
    Deadline deadline(60s);
    body();
    std::printf("PASS %s\n", name);
}

// --- the duplex proof and its negative control ---------------------------------------------------------------------------------------------------------------------------

void duplex_exchange_completes(bool staged)
{
    Rig rig(Script{}, staged);
    const auto result = rig.run();
    expect(result.first == tbccl::ErrorCode::Success, "the exchange must complete when each leg needs the other to have started: " + result.second);
    expect(rig.transport.rendezvous_timeouts() == 0, "a leg waited for the other in vain");
    expect(rig.dest_matches_source(), "the received bytes must equal the sent bytes");
    expect(rig.fatal_calls.load() == 0, "a successful exchange must not abort");
    rig.expect_quiescent();
}

// Negative control: the schedule this change replaced (shared lane, one leg waited for before the other is posted) cannot get past the rendezvous.
void sequential_schedule_is_detected(bool staged)
{
    Script script;
    script.rendezvous_limit = 400ms;
    Rig rig(script, staged);
    const auto make = [&](tbccl::TransferDirection d) {
        tbccl::TransferRequest r;
        r.transfer_id = d == tbccl::TransferDirection::Send ? 1 : 2;
        r.direction = d;
        r.backend = d == tbccl::TransferDirection::Send ? &rig.send_backend() : &rig.recv_backend();
        r.transport = &rig.transport;
        r.total_bytes = kBytes;
        r.shared_lane = true;
        return r;
    };
    auto first = rig.worker.enqueue(make(tbccl::TransferDirection::Send));
    first.wait();
    expect(first.has_error() && first.error_code() == tbccl::ErrorCode::Timeout, "the sequential schedule must be unable to finish its first leg");
    auto second = rig.worker.enqueue(make(tbccl::TransferDirection::Recv));
    second.wait();
    expect(rig.transport.rendezvous_timeouts() >= 1, "the rendezvous must have recorded the sequential schedule");
}

// --- failure injection ----------------------------------------------------------------------------------------------------------------------------------------------------

void expect_cause(const std::pair<tbccl::ErrorCode, std::string> &result, tbccl::ErrorCode code, const char *leg)
{
    expect(result.first == code, "wrong error code, got text: " + result.second);
    expect(result.second.find("all_gather") != std::string::npos, "the error must name the collective: " + result.second);
    if (leg) expect(result.second.find(leg) != std::string::npos, std::string("the error must come from the ") + leg + " leg: " + result.second);
}

void send_fails_first_recv_blocked(bool staged)
{
    Script s;
    s.fail_send = true;
    s.hold_recv = true;
    Rig rig(s, staged);
    const auto result = rig.run();
    expect_cause(result, tbccl::ErrorCode::TransportError, "send");
    expect(result.second.find("injected send failure") != std::string::npos, "the original send error text is kept: " + result.second);
    expect(rig.fatal_calls.load() == 1, "the failure aborts the communicator exactly once");
    rig.expect_quiescent();
}

void recv_fails_first_send_blocked(bool staged)
{
    Script s;
    s.fail_recv = true;
    s.hold_send = true;
    Rig rig(s, staged);
    const auto result = rig.run();
    expect_cause(result, tbccl::ErrorCode::TransportError, "recv");
    expect(result.second.find("injected recv failure") != std::string::npos, "the original recv error text is kept: " + result.second);
    expect(rig.fatal_calls.load() == 1, "the failure aborts the communicator exactly once");
    rig.expect_quiescent();
}

void both_fail(bool staged)
{
    Script s;
    s.fail_send = s.fail_recv = true;
    Rig rig(s, staged);
    const auto result = rig.run();
    expect(result.first == tbccl::ErrorCode::TransportError, "a cause, never the Aborted of the abort it triggered: " + result.second);
    expect(result.second.find("injected") != std::string::npos, "an original error text is reported: " + result.second);
    expect(rig.fatal_calls.load() >= 1, "the failure aborts the communicator");
    rig.expect_quiescent();
}

// Failure inside a staged copy, with the other direction direct or staged. `last` fails only the final (partial) chunk, after earlier chunks went through.
void staged_source_copy_fails(bool recv_staged, bool last)
{
    Script s;
    s.rendezvous = false;
    s.hold_recv = true; // the receive is blocked in the transport until the failure aborts it
    Rig rig(s, /*send_staged=*/true, recv_staged);
    (last ? rig.staged_send->fail_stage_last : rig.staged_send->fail_stage) = true;
    const auto result = rig.run();
    expect_cause(result, tbccl::ErrorCode::DeviceError, "send");
    expect(rig.fatal_calls.load() == 1, "one abort");
    rig.expect_quiescent();
}

void staged_destination_commit_fails(bool send_staged, bool last)
{
    Script s;
    s.rendezvous = false;
    s.recv_preloaded = true;
    s.hold_send = true;
    Rig rig(s, send_staged, /*recv_staged=*/true);
    (last ? rig.staged_recv->fail_commit_last : rig.staged_recv->fail_commit) = true;
    const auto result = rig.run();
    expect_cause(result, tbccl::ErrorCode::DeviceError, "recv");
    expect(rig.fatal_calls.load() == 1, "one abort");
    rig.expect_quiescent();
}

// Every provider combination (direct/direct, staged/staged and the two mixed ones, i.e. one leg chunked and the other not) completes with the right bytes.
void hetero_exchange_completes(bool send_staged, bool recv_staged)
{
    Rig rig(Script{}, send_staged, recv_staged);
    const auto result = rig.run();
    expect(result.first == tbccl::ErrorCode::Success, result.second);
    expect(rig.dest_matches_source(), "contents (partial final chunk included)");
    expect(rig.transport.rendezvous_timeouts() == 0, "no leg waited in vain");
    rig.expect_quiescent();
}

// Phase 1B.5 guarantee through the exchange: a failed staged leg cancels and discards its pool, and the same worker (both lanes) serves a clean exchange afterwards.
// Here the abort callback only counts, so the worker is NOT aborted and must recover by itself.
void worker_recovers_after_staged_failure(bool fail_source)
{
    Script s;
    s.rendezvous = false;
    s.recv_preloaded = true; // the other leg completes on its own
    Rig rig(s, /*send_staged=*/true, /*recv_staged=*/true);
    if (fail_source) rig.staged_send->fail_stage_last = true;
    else rig.staged_recv->fail_commit_last = true;
    std::atomic<int> calls{0};
    const auto count_only = [&](const std::string &) { calls.fetch_add(1); };
    const auto first = rig.run_with(count_only);
    expect(first.first == tbccl::ErrorCode::DeviceError, "the staged failure is the reported cause: " + first.second);
    expect(calls.load() == 1, "the failure requests the abort once");
    rig.expect_quiescent();
    rig.staged_send->fail_stage_last = false;
    rig.staged_recv->fail_commit_last = false;
    Script clean;
    clean.rendezvous = false;
    clean.recv_preloaded = true;
    rig.transport.reconfigure(clean);
    std::fill(rig.dest.begin(), rig.dest.end(), 0);
    const auto second = rig.run_with(count_only);
    expect(second.first == tbccl::ErrorCode::Success, "the worker must serve a clean exchange after a staged failure: " + second.second);
    expect(rig.dest_matches_preload(), "contents of the second exchange");
    expect(calls.load() == 1, "the clean exchange must not abort");
    rig.expect_quiescent();
}

void abort_before_posting(bool staged)
{
    Rig rig(Script{}, staged);
    rig.worker.abort("aborted before the exchange");
    const auto result = rig.run();
    expect(result.first == tbccl::ErrorCode::Aborted, "an aborted worker rejects the exchange with Aborted: " + result.second);
    rig.expect_quiescent();
}

void abort_with_both_legs_in_flight(bool staged)
{
    Script s;
    s.rendezvous = false;
    s.hold_send = s.hold_recv = true;
    Rig rig(s, staged);
    std::thread aborter([&] {
        expect(rig.transport.wait_both_entered(20s), "both legs must be in the transport before the abort");
        rig.fatal_callback()("communicator abort"); // what Communicator::abort does to this peer's worker and transport
    });
    const auto result = rig.run();
    aborter.join();
    expect(result.first == tbccl::ErrorCode::Aborted, "an abort with both legs in flight ends the exchange with Aborted: " + result.second);
    rig.expect_quiescent();
}

// The second admission throws after the first leg was admitted and is inside the transport: the admitted leg must not be abandoned.
void partial_admission(bool staged)
{
    Script s;
    s.rendezvous = false;
    s.hold_send = true; // the admitted send is blocked in the transport when the second admission fails
    Rig rig(s, staged);
    tbccl::detail::g_n2_exchange_fail_second_enqueue = 1;
    const auto result = rig.run();
    tbccl::detail::g_n2_exchange_fail_second_enqueue = 0;
    expect(result.first == tbccl::ErrorCode::ResourceExhausted, "the admission failure is what the caller sees: " + result.second);
    expect(rig.fatal_calls.load() == 1, "a partially admitted exchange aborts the communicator (the peer would wait for the missing leg)");
    rig.expect_quiescent(); // the admitted send was interrupted AND finished before run_n2_exchange returned
}

// An abort callback that throws must not make the exchange return early: the admitted leg is still waited for and the exchange's own error propagates.
void throwing_abort_callback_keeps_waiting(bool partial_admission)
{
    Script s;
    s.rendezvous = false;
    s.hold_send = true;
    s.hold_recv = !partial_admission;
    if (!partial_admission) s.fail_send = true; // the send fails at once while the receive is blocked
    s.hold_send = partial_admission;
    Rig rig(s, false);
    const auto throwing = [&](const std::string &why) {
        rig.fatal_calls.fetch_add(1);
        rig.worker.abort(why); // the interruption still happens, then the callback fails
        rig.transport.abort(why);
        throw std::runtime_error("abort callback failed");
    };
    if (partial_admission) tbccl::detail::g_n2_exchange_fail_second_enqueue = 1;
    const auto result = rig.run_with(throwing);
    tbccl::detail::g_n2_exchange_fail_second_enqueue = 0;
    if (partial_admission) expect(result.first == tbccl::ErrorCode::ResourceExhausted, "the admission error, not the callback's, propagates: " + result.second);
    else expect_cause(result, tbccl::ErrorCode::TransportError, "send");
    rig.expect_quiescent();
}

// Fault injection on the abort path itself. Whatever fails while the exchange is being aborted (the abort request throws before it interrupts anything, the
// reason string cannot be built, an exception strikes right after both legs were admitted), a leg that is blocked in the transport must still be interrupted and
// waited for, and the original problem must be what propagates. Legs are blocked "until aborted", so a missing interruption is a bounded stall, not a pass.
enum class Trigger { PartialAdmission, SendFails, RecvFails, AfterAdmission };

void abort_path_fault(Trigger trigger, bool callback_throws_first, bool message_fails, bool staged)
{
    Script s;
    s.rendezvous = false;
    tbccl::ErrorCode expected = tbccl::ErrorCode::TransportError;
    const char *leg = nullptr;
    switch (trigger)
    {
    case Trigger::PartialAdmission: s.hold_send = true; expected = tbccl::ErrorCode::ResourceExhausted; break;
    case Trigger::SendFails: s.fail_send = true; s.hold_recv = true; leg = "send"; break;
    case Trigger::RecvFails: s.fail_recv = true; s.hold_send = true; leg = "recv"; break;
    case Trigger::AfterAdmission: s.hold_send = s.hold_recv = true; expected = tbccl::ErrorCode::ResourceExhausted; break;
    }
    Rig rig(s, staged);
    std::atomic<int> callback_calls{0};
    const auto callback = [&](const std::string &why) {
        callback_calls.fetch_add(1);
        if (callback_throws_first) throw std::runtime_error("abort request failed before interrupting anything"); // nothing is interrupted by the callback
        rig.worker.abort(why);
        rig.transport.abort(why);
    };
    if (trigger == Trigger::PartialAdmission) tbccl::detail::g_n2_exchange_fail_second_enqueue = 1;
    if (trigger == Trigger::AfterAdmission) tbccl::detail::g_n2_exchange_fail_after_admission = 1;
    if (message_fails) tbccl::detail::g_n2_exchange_fail_abort_message = 1;
    const auto result = rig.run_with(callback);
    tbccl::detail::g_n2_exchange_fail_second_enqueue = 0;
    tbccl::detail::g_n2_exchange_fail_after_admission = 0;
    tbccl::detail::g_n2_exchange_fail_abort_message = 0;
    expect(result.first == expected, "the original problem propagates, got: " + result.second);
    if (leg) expect_cause(result, expected, leg);
    expect(rig.transport.was_aborted(), "the blocked leg must have been interrupted");
    expect(callback_calls.load() == (message_fails ? 0 : 1), "the abort request is made exactly once unless its reason could not be built");
    rig.expect_quiescent();
}

void repeated_exchanges_keep_both_lanes_usable()
{
    Rig rig(Script{}, false);
    for (int i = 0; i < 300; ++i)
    {
        const auto result = rig.run();
        expect(result.first == tbccl::ErrorCode::Success, "iteration " + std::to_string(i) + ": " + result.second);
        expect(rig.dest_matches_source(), "iteration " + std::to_string(i) + ": contents");
        std::fill(rig.dest.begin(), rig.dest.end(), 0);
    }
    expect(rig.transport.rendezvous_timeouts() == 0, "no leg ever waited in vain");
    rig.expect_quiescent();
}
} // namespace

int main()
{
    std::thread([] {
        std::this_thread::sleep_for(600s);
        std::fprintf(stderr, "STALLED: n2_exchange_unit_test exceeded 600 s\n");
        std::fflush(stderr);
        std::_Exit(124);
    }).detach();
    for (const bool staged : {false, true})
    {
        const std::string tag = staged ? " (staged)" : " (direct)";
        const auto name = [&](const char *n) { return std::string(n) + tag; };
        scenario(name("duplex exchange completes only if both legs run together").c_str(), [&] { duplex_exchange_completes(staged); });
        scenario(name("negative control: a sequential schedule is detected").c_str(), [&] { sequential_schedule_is_detected(staged); });
        scenario(name("send fails first while recv is blocked").c_str(), [&] { send_fails_first_recv_blocked(staged); });
        scenario(name("recv fails first while send is blocked").c_str(), [&] { recv_fails_first_send_blocked(staged); });
        scenario(name("both legs fail").c_str(), [&] { both_fail(staged); });
        scenario(name("abort before posting").c_str(), [&] { abort_before_posting(staged); });
        scenario(name("abort with both legs in flight").c_str(), [&] { abort_with_both_legs_in_flight(staged); });
        scenario(name("second admission fails after the first leg is in flight").c_str(), [&] { partial_admission(staged); });
    }
    for (const bool send_staged : {false, true})
        for (const bool recv_staged : {false, true})
        {
            const std::string tag = std::string(" (send ") + (send_staged ? "staged" : "direct") + ", recv " + (recv_staged ? "staged" : "direct") + ")";
            scenario(("heterogeneous exchange completes" + tag).c_str(), [&] { hetero_exchange_completes(send_staged, recv_staged); });
        }
    for (const bool other_staged : {false, true})
        for (const bool last : {false, true})
        {
            const std::string tag = std::string(other_staged ? " (other leg staged" : " (other leg direct") + (last ? ", final chunk)" : ", first chunk)");
            scenario(("failure during the staged source copy" + tag).c_str(), [&] { staged_source_copy_fails(other_staged, last); });
            scenario(("failure during the staged destination commit" + tag).c_str(), [&] { staged_destination_commit_fails(other_staged, last); });
        }
    scenario("worker recovers after a staged source failure (pool discarded)", [] { worker_recovers_after_staged_failure(true); });
    scenario("worker recovers after a staged commit failure (pool discarded)", [] { worker_recovers_after_staged_failure(false); });
    scenario("throwing abort callback during a partial admission", [] { throwing_abort_callback_keeps_waiting(true); });
    scenario("throwing abort callback after a leg failed", [] { throwing_abort_callback_keeps_waiting(false); });
    for (const bool staged : {false, true})
        for (const auto trigger : {Trigger::PartialAdmission, Trigger::SendFails, Trigger::RecvFails, Trigger::AfterAdmission})
            for (const bool throws_first : {false, true})
                for (const bool message_fails : {false, true})
                {
                    if (throws_first && message_fails) continue; // the callback is never reached
                    const char *names[] = {"partial admission", "send fails", "recv fails", "exception after admission"};
                    const std::string name = std::string("abort path fault: ") + names[static_cast<int>(trigger)] + (throws_first ? ", abort request throws before interrupting" : "") +
                                             (message_fails ? ", abort reason cannot be built" : "") + (staged ? " (staged)" : " (direct)");
                    scenario(name.c_str(), [&] { abort_path_fault(trigger, throws_first, message_fails, staged); });
                }
    scenario("300 repeated exchanges", repeated_exchanges_keep_both_lanes_usable);
    std::puts("n2_exchange_unit_test passed");
    return 0;
}
