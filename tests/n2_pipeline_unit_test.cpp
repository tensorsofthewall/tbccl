// Both ranks of the chunk-pipelined two-rank all_reduce (detail::run_n2_all_reduce_pipeline) over a controlled transport, fake backends and a real duplex TensorCommWorker.
//
// The transport models the peer as seen from the rank under test. Seen from the root, the peer (the non-root) "sends" a known contribution and the result bytes the root sends
// back are recorded. Seen from the non-root, the peer (the root) returns the reduced result of exactly the contribution bytes it has been sent so far. Concurrency is proven with
// causal gates, never with timing thresholds:
//   * overlap: the receive of the LAST contribution chunk (root) / the send of the LAST chunk (non-root) cannot complete until the FIRST result chunk has been sent / received.
//     A schedule that runs the legs one after the other (the sequential algorithm, run here as the negative control) never gets past the gate; that is a bounded wait that
//     records a timeout.
//   * no leg queued behind the other: against a peer that sends everything before it receives anything (every root send waits for the whole contribution) or one that returns
//     nothing before it has everything (a sequential root), the pipelined rank still finishes.
//   * in-place ownership (D2): the non-root's transport asserts that a receive only ever starts on a range whose send has already finished.
// Failure injection checks, for every scenario: the error is the first cause (not the Aborted it triggered) and names the failing leg and chunk, the communicator-level abort
// callback ran exactly once, NO transfer is still inside the transport when the call returns, and nothing touches the buffers afterwards.
// Run once per TBCCL_STAGED_CHUNK_BYTES setting (unset, and 4096 which makes every chunk one staged chunk).

#include "communicator_debug.hpp"

#include <tbccl/async_transfer.hpp>
#include <tbccl/error.hpp>
#include <tbccl/hetero_allreduce.hpp>
#include <tbccl/transport.hpp>

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

constexpr std::size_t kChunk = 4096;                       // pipeline chunk (a multiple of lcm(4, 64))
constexpr std::size_t kBytes = 5 * kChunk + 1024;          // six chunks, the last one partial (and element aligned)
constexpr std::size_t kChunks = 6;
constexpr std::size_t kElem = sizeof(std::int32_t);
constexpr std::size_t kCount = kBytes / kElem;
constexpr std::size_t kNone = static_cast<std::size_t>(-1);

std::uint8_t pat(std::size_t seed, std::size_t j) { return static_cast<std::uint8_t>(seed * 37 + j * 7 + (j >> 9) * 5 + 3); }

std::int32_t word(const std::uint8_t *p, std::size_t index)
{
    std::int32_t w;
    std::memcpy(&w, p + index * kElem, kElem);
    return w;
}

// The test patterns are arbitrary bytes: add as unsigned so an overflow wraps (the reduction's defined behaviour) instead of being undefined.
std::int32_t wrapping_sum(std::int32_t a, std::int32_t b) { return static_cast<std::int32_t>(static_cast<std::uint32_t>(a) + static_cast<std::uint32_t>(b)); }

struct Script
{
    std::chrono::milliseconds limit{5000};   // how long a gated leg waits for its condition
    bool gate_last_on_first_result = false;  // root: the receive of the last contribution chunk completes only after the first result chunk was sent;
                                             // non-root: the send of the last chunk completes only after the first result chunk was received
    bool peer_is_sequential = false;         // root: a result send completes only after every contribution byte was received (the peer sends everything before it
                                             // receives); non-root: no result byte is returned before the whole contribution was sent (a sequential root)
    std::size_t fail_recv_at = kNone;        // the receive covering this byte offset throws TransportError
    std::size_t fail_send_at = kNone;        // likewise for a result send
    std::size_t hold_recv_at = kNone;        // the receive covering this offset blocks until the transport is aborted
    std::size_t hold_send_at = kNone;
};

class PipeTransport final : public tbccl::Transport
{
public:
    PipeTransport(Script script, bool root) : script_(script), root_(root) {}

    void send(const void *data, std::size_t bytes) override
    {
        Active active(*this);
        std::unique_lock<std::mutex> lock(m_);
        const std::size_t at = sent_.size();
        if (covers(script_.fail_send_at, at, bytes)) throw tbccl::Error(tbccl::ErrorCode::TransportError, "injected send failure");
        if (covers(script_.hold_send_at, at, bytes)) hold(lock);
        if (root_ && script_.peer_is_sequential && !cv_.wait_for(lock, script_.limit, [&] { return received_ >= kBytes || aborted_; }))
        {
            timeouts_.fetch_add(1);
            throw tbccl::Error(tbccl::ErrorCode::Timeout, "a result send waited for the whole contribution in vain (a receive was queued behind it)");
        }
        if (!root_ && script_.gate_last_on_first_result && at + bytes > (kChunks - 1) * kChunk &&
            !cv_.wait_for(lock, script_.limit, [&] { return received_ >= kChunk || aborted_; }))
        {
            timeouts_.fetch_add(1);
            throw tbccl::Error(tbccl::ErrorCode::Timeout, "the last chunk was sent before any result chunk came back (the legs ran one after the other)");
        }
        throw_if_aborted();
        const auto *p = static_cast<const std::uint8_t *>(data);
        sent_.insert(sent_.end(), p, p + bytes);
        completed_send_ = sent_.size();
        cv_.notify_all();
    }

    void recv(void *data, std::size_t bytes) override
    {
        Active active(*this);
        std::unique_lock<std::mutex> lock(m_);
        const std::size_t at = received_;
        // D2: a rank that receives in place must not start receiving into a range before the send that reads it has finished.
        if (!root_ && at + bytes > completed_send_) die("a receive started on a range whose send had not finished (in-place ownership), range " + std::to_string(at) + "+" + std::to_string(bytes) + ", sent " + std::to_string(completed_send_));
        if (covers(script_.fail_recv_at, at, bytes)) throw tbccl::Error(tbccl::ErrorCode::TransportError, "injected recv failure");
        if (covers(script_.hold_recv_at, at, bytes)) hold(lock);
        if (root_)
        {
            if (script_.gate_last_on_first_result && at + bytes > (kChunks - 1) * kChunk &&
                !cv_.wait_for(lock, script_.limit, [&] { return sent_.size() >= kChunk || aborted_; }))
            {
                timeouts_.fetch_add(1);
                throw tbccl::Error(tbccl::ErrorCode::Timeout, "the last contribution chunk arrived before any result chunk was sent (the legs ran one after the other)");
            }
            throw_if_aborted();
            auto *p = static_cast<std::uint8_t *>(data);
            for (std::size_t j = 0; j < bytes; ++j) p[j] = pat(7, at + j);
        }
        else
        {
            // The model root returns the reduced result of exactly the contribution bytes it has been sent (all of them, for a sequential root).
            const std::size_t need = script_.peer_is_sequential ? kBytes : at + bytes;
            if (!cv_.wait_for(lock, script_.limit, [&] { return sent_.size() >= need || aborted_; }))
            {
                timeouts_.fetch_add(1);
                throw tbccl::Error(tbccl::ErrorCode::Timeout, "a result was awaited for a contribution that was never sent (a send was queued behind a receive)");
            }
            throw_if_aborted();
            auto *p = static_cast<std::uint8_t *>(data);
            for (std::size_t j = 0; j < bytes; j += kElem)
            {
                const std::size_t i = (at + j) / kElem;
                std::uint8_t peer[kElem]; // the root's own contribution
                for (std::size_t b = 0; b < kElem; ++b) peer[b] = pat(1, i * kElem + b);
                const std::int32_t sum = wrapping_sum(word(sent_.data(), i), word(peer, 0));
                std::memcpy(p + j, &sum, kElem);
            }
        }
        received_ += bytes;
        cv_.notify_all();
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

    void reconfigure(Script script)
    {
        std::lock_guard<std::mutex> lock(m_);
        script_ = script;
        sent_.clear();
        received_ = completed_send_ = 0;
        aborted_ = false;
    }
    // Blocks until this rank has received (root: the contribution; non-root: the result) at least `bytes`.
    bool wait_received_at_least(std::size_t bytes, std::chrono::milliseconds limit)
    {
        std::unique_lock<std::mutex> lock(m_);
        return cv_.wait_for(lock, limit, [&] { return received_ >= bytes; });
    }
    std::vector<std::uint8_t> sent()
    {
        std::lock_guard<std::mutex> lock(m_);
        return sent_;
    }
    int active() const { return active_.load(); }
    int timeouts() const { return timeouts_.load(); }

private:
    struct Active
    {
        explicit Active(PipeTransport &t) : t_(t) { t_.active_.fetch_add(1); }
        ~Active() { t_.active_.fetch_sub(1); }
        PipeTransport &t_;
    };
    static bool covers(std::size_t point, std::size_t at, std::size_t bytes) { return point != kNone && point >= at && point < at + bytes; }
    // A blocked leg takes a moment to unwind once the transport is aborted, so a call that returned without waiting for it is caught every time. The delay widens a window;
    // no assertion depends on how long anything took.
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
    const bool root_;
    std::mutex m_;
    std::condition_variable cv_;
    std::vector<std::uint8_t> sent_;
    std::size_t received_ = 0, completed_send_ = 0;
    bool aborted_ = false;
    std::string reason_;
    std::atomic<int> active_{0};
    std::atomic<int> timeouts_{0};
};

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

// Memory that must be staged (like device memory). `fail_stage_at` / `fail_commit_at` fail the copy that covers that absolute byte offset.
class StagedBackend final : public tbccl::AsyncMemoryBackend
{
public:
    explicit StagedBackend(std::vector<std::uint8_t> &buffer) : buffer_(buffer) {}
    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
    {
        if (fail_stage_at != kNone && fail_stage_at >= chunk.offset && fail_stage_at < chunk.offset + chunk.size) throw tbccl::Error(tbccl::ErrorCode::DeviceError, "injected staging failure");
        std::memcpy(staging, buffer_.data() + chunk.offset, chunk.size);
    }
    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override
    {
        if (fail_commit_at != kNone && fail_commit_at >= chunk.offset && fail_commit_at < chunk.offset + chunk.size) throw tbccl::Error(tbccl::ErrorCode::DeviceError, "injected commit failure");
        std::memcpy(buffer_.data() + chunk.offset, staging, chunk.size);
    }
    std::size_t fail_stage_at = kNone, fail_commit_at = kNone;

private:
    std::vector<std::uint8_t> &buffer_;
};

// The sequential algorithm's reduce interface over the same reduction (negative control).
class WholeReduce final : public tbccl::LocalReduceBackend
{
public:
    explicit WholeReduce(std::function<void(std::size_t, std::size_t)> f) : f_(std::move(f)) {}
    void reduce_sum(std::size_t count, tbccl::DataType) override { f_(0, count); }

private:
    std::function<void(std::size_t, std::size_t)> f_;
};

enum class Fault
{
    None,
    ReduceThrowsError,    // tbccl::Error(DeviceError)
    ReduceThrowsStd,      // std::runtime_error
};

struct Rig
{
    // `a_staged` / `b_staged`: the provider kind of the receive landing zone and of the result source (root: scratch, primary); the non-root has one in-place buffer (a_staged).
    Rig(Script script, bool root_, bool a_staged, bool b_staged)
        : root(root_), transport(script, root_), worker(2, tbccl::TensorCommWorker::kUnboundedAdmission, /*duplex=*/true), scratch(kBytes, 0), primary(kBytes), original(kBytes)
    {
        for (std::size_t j = 0; j < kBytes; ++j) original[j] = primary[j] = pat(root ? 1 : 7, j);
        if (a_staged) scratch_staged_backend = std::make_unique<StagedBackend>(scratch);
        else scratch_direct = std::make_unique<DirectBackend>(scratch);
        if (b_staged) primary_staged_backend = std::make_unique<StagedBackend>(primary);
        else primary_direct = std::make_unique<DirectBackend>(primary);
    }
    Rig(Script script, bool root_, bool staged) : Rig(script, root_, staged, staged) {}

    tbccl::AsyncMemoryBackend &scratch_backend() { return scratch_staged_backend ? static_cast<tbccl::AsyncMemoryBackend &>(*scratch_staged_backend) : *scratch_direct; }
    tbccl::AsyncMemoryBackend &primary_backend() { return primary_staged_backend ? static_cast<tbccl::AsyncMemoryBackend &>(*primary_staged_backend) : *primary_direct; }
    // The failure-injection handles of the buffer a leg reads (commit goes to the receive landing zone; stage reads the send source). The non-root uses `primary` for both.
    StagedBackend &recv_staged() { return root ? *scratch_staged_backend : *primary_staged_backend; }
    StagedBackend &send_staged() { return *primary_staged_backend; }

    std::function<void(const std::string &)> fatal_callback()
    {
        return [this](const std::string &why) {
            fatal_calls.fetch_add(1);
            worker.abort(why);
            transport.abort(why);
        };
    }

    // primary[off..] += scratch[off..] as Int32. Checks the dependency D1 on the way in: the range must already hold the whole peer contribution.
    void reduce(std::size_t byte_offset, std::size_t elements)
    {
        std::size_t k;
        {
            std::lock_guard<std::mutex> lock(order_mutex);
            k = reduced_chunks.size();
            reduced_chunks.push_back(byte_offset);
        }
        for (std::size_t j = 0; j < elements * kElem; ++j)
            if (scratch[byte_offset + j] != pat(7, byte_offset + j)) die("chunk reduced before its contribution was completely received (offset " + std::to_string(byte_offset) + ")");
        if (fault != Fault::None && k == fault_chunk)
        {
            if (fault == Fault::ReduceThrowsError) throw tbccl::Error(tbccl::ErrorCode::DeviceError, "injected reduce failure");
            throw std::runtime_error("injected reduce failure");
        }
        for (std::size_t i = 0; i < elements; ++i)
        {
            const std::int32_t sum = wrapping_sum(word(primary.data() + byte_offset, i), word(scratch.data() + byte_offset, i));
            std::memcpy(primary.data() + byte_offset + i * kElem, &sum, kElem);
        }
    }

    // Runs this rank; returns the exception's code (Success when it returned normally; InternalError for a non-tbccl exception) and text.
    std::pair<tbccl::ErrorCode, std::string> run(const std::function<void(const std::string &)> *fatal_override = nullptr)
    {
        const auto fatal = fatal_override ? *fatal_override : fatal_callback();
        try
        {
            if (root)
                tbccl::detail::run_n2_all_reduce_pipeline(
                    worker, transport, true, scratch_backend(), primary_backend(), kBytes, kChunk, kElem, [this](std::size_t off, std::size_t n) { reduce(off, n); }, "all_reduce", fatal);
            else
                tbccl::detail::run_n2_all_reduce_pipeline(worker, transport, false, primary_backend(), primary_backend(), kBytes, kChunk, kElem, nullptr, "all_reduce", fatal);
        }
        catch (const tbccl::Error &e)
        {
            return {e.code(), e.what()};
        }
        catch (const std::bad_alloc &)
        {
            return {tbccl::ErrorCode::ResourceExhausted, "bad_alloc"};
        }
        catch (const std::exception &e)
        {
            return {tbccl::ErrorCode::InternalError, e.what()};
        }
        return {tbccl::ErrorCode::Success, ""};
    }

    std::vector<std::uint8_t> expected_result() const
    {
        std::vector<std::uint8_t> out(kBytes);
        for (std::size_t i = 0; i < kCount; ++i)
        {
            std::uint8_t mine[kElem], peer[kElem];
            for (std::size_t b = 0; b < kElem; ++b)
            {
                mine[b] = pat(root ? 1 : 7, i * kElem + b);
                peer[b] = pat(root ? 7 : 1, i * kElem + b);
            }
            const std::int32_t sum = wrapping_sum(word(mine, 0), word(peer, 0));
            std::memcpy(out.data() + i * kElem, &sum, kElem);
        }
        return out;
    }
    void expect_success_contents()
    {
        const auto want = expected_result();
        if (root)
        {
            expect(transport.sent() == want, "the bytes sent back must be the reduced result, in order");
            expect(reduced_chunks.size() == kChunks, "every chunk is reduced exactly once");
            for (std::size_t k = 0; k < reduced_chunks.size(); ++k) expect(reduced_chunks[k] == k * kChunk, "chunks are reduced in order");
        }
        else
        {
            expect(transport.sent() == original, "the contribution must reach the wire unchanged (it was not overwritten by the result)");
        }
        expect(primary == want, "the rank's own buffer must hold the reduced result");
    }
    // Everything the call started is finished: nothing is inside the transport and both lanes are idle. Then nothing may touch the buffers any more.
    void expect_quiescent()
    {
        expect(transport.active() == 0, "a transfer is still inside the transport after the call returned");
        worker.wait_idle();
        expect(!worker.busy(), "the worker still reports in-flight transfers");
        std::fill(scratch.begin(), scratch.end(), std::uint8_t{0xA5});
        std::fill(primary.begin(), primary.end(), std::uint8_t{0xA5});
        std::this_thread::sleep_for(60ms);
        for (std::size_t j = 0; j < kBytes; ++j)
            if (scratch[j] != 0xA5 || primary[j] != 0xA5) die("a worker touched the buffers after the call returned (offset " + std::to_string(j) + ")");
    }

    const bool root;
    PipeTransport transport;
    tbccl::TensorCommWorker worker;
    std::vector<std::uint8_t> scratch, primary, original;
    std::unique_ptr<DirectBackend> scratch_direct, primary_direct;
    std::unique_ptr<StagedBackend> scratch_staged_backend, primary_staged_backend;
    std::atomic<int> fatal_calls{0};
    std::mutex order_mutex;
    std::vector<std::size_t> reduced_chunks;
    Fault fault = Fault::None;
    std::size_t fault_chunk = 0;
};

void scenario(const char *name, const std::function<void()> &body)
{
    g_scenario = name;
    Deadline deadline(90s);
    body();
    std::printf("PASS %s\n", name);
}

// --- geometry -------------------------------------------------------------------------------------------------------------------------------------------------------------

void geometry()
{
    using tbccl::DataType;
    const auto chunk = [](std::size_t setting, std::size_t bytes, std::size_t count, DataType t) { return tbccl::detail::n2_pipeline_chunk_bytes(setting, bytes, count, t); };
    expect(chunk(0, 1 << 20, 1 << 18, DataType::Float32) == 0, "disabled");
    expect(chunk(65536, 0, 0, DataType::Float32) == 0, "no elements");
    expect(chunk(65536, 1 << 20, 0, DataType::Float32) == 0, "count 0 keeps the sequential schedule even for a large buffer");
    expect(chunk(65536, (1 << 20) + 8, 1 << 18, DataType::Float32) == 0, "a buffer larger than count * element size keeps the sequential schedule");
    expect(chunk(65536, 65536, 16384, DataType::Float32) == 0, "one chunk is not pipelined");
    expect(chunk(131072, 65536, 16384, DataType::Float32) == 0, "two chunks of 32 KiB are not pipelined (fewer than four)");
    expect(chunk(131072, 131072, 32768, DataType::Float32) == 0, "a payload under 256 KiB keeps the sequential schedule");
    expect(chunk(131072, 262143 + 1, 65536, DataType::Float32) == 32768, "256 KiB: eight chunks of 32 KiB");
    expect(chunk(131072, 262144, 65536, DataType::Float32) == 32768, "an eighth of the payload, floored at 32 KiB");
    expect(chunk(131072, 1 << 20, 1 << 18, DataType::Float32) == 131072, "an eighth of 1 MiB is the setting");
    expect(chunk(131072, 16 << 20, 4 << 20, DataType::Float32) == 131072, "the setting is the largest chunk");
    expect(chunk(65536, 1 << 20, 1 << 18, DataType::Float32) == 65536, "capped at the setting");
    expect(chunk(8192, 1 << 20, 1 << 18, DataType::Float32) == 8192, "an explicit small setting is honoured (the floor applies only to the automatic reduction)");
    expect(chunk(65536 + 100, 1 << 20, 1 << 18, DataType::Float32) == 65600, "a multiple of 64 bytes (a whole number of elements)");
    expect(chunk(4097, 1 << 20, 1 << 18, DataType::Float32) == 4096, "rounded down to the unit");
    expect(chunk(4096, 1 << 20, 1 << 17, DataType::Float64) == 4096, "eight-byte elements: the unit is still 64 bytes");
    expect(chunk(4096, 1 << 20, (1 << 17) - 1, DataType::Float64) == 0, "count * element size must equal the buffer");
    expect(chunk(4096, 7 * 65536, 7 * 65536, DataType::UInt8) == 4096, "one-byte elements");
    expect(chunk(4096, std::size_t{1} << 26, std::size_t{1} << 26, DataType::UInt8) == 4096, "16384 chunks are allowed");
    expect(chunk(4096, std::size_t{1} << 30, std::size_t{1} << 30, DataType::UInt8) == 65536, "the chunk count is capped at 16384 however small the setting");
    expect(chunk(4096, (std::size_t{1} << 30) + 64, (std::size_t{1} << 30) + 64, DataType::UInt8) == 65600, "the cap rounds up to the unit");
    expect(chunk(65536, 262144, 262144, DataType::UInt8) == 32768, "256 KiB is the smallest pipelined payload: eight chunks");
        expect(chunk(65536, 262144 - 64, 262144 - 64, DataType::UInt8) == 0, "just under 256 KiB: sequential");
}

// --- success ---------------------------------------------------------------------------------------------------------------------------------------------------------------

void success(bool root, bool a_staged, bool b_staged)
{
    Rig rig(Script{}, root, a_staged, b_staged);
    const auto result = rig.run();
    expect(result.first == tbccl::ErrorCode::Success, "the pipelined all_reduce must complete: " + result.second);
    expect(rig.fatal_calls.load() == 0, "a successful call must not abort");
    rig.expect_success_contents();
    rig.expect_quiescent();
}

// The overlap proof: the last chunk of one direction is held back until the first result chunk of the other has crossed.
void overlap_is_possible(bool root, bool staged)
{
    Script s;
    s.gate_last_on_first_result = true;
    Rig rig(s, root, staged);
    const auto result = rig.run();
    expect(result.first == tbccl::ErrorCode::Success, "the directions must overlap: " + result.second);
    expect(rig.transport.timeouts() == 0, "a gate waited in vain");
    rig.expect_success_contents();
    rig.expect_quiescent();
}

// Negative control: the sequential algorithm (root: receive everything, reduce, send everything; non-root: send everything, then receive) cannot pass the same gate.
void sequential_schedule_is_detected(bool root, bool staged)
{
    Script s;
    s.gate_last_on_first_result = true;
    s.limit = 400ms;
    Rig rig(s, root, staged);
    WholeReduce reduce([&rig](std::size_t off, std::size_t n) { rig.reduce(off, n); });
    bool threw = false;
    try
    {
        if (root) tbccl::n2_all_reduce_tensor(rig.transport, rig.worker, rig.scratch_backend(), rig.primary_backend(), &reduce, 0, 0, kBytes, 0, kCount, tbccl::DataType::Int32);
        else tbccl::n2_all_reduce_tensor(rig.transport, rig.worker, rig.primary_backend(), rig.primary_backend(), nullptr, 1, 0, kBytes, 0, kCount, tbccl::DataType::Int32);
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    expect(threw, "the sequential schedule must be unable to finish with the gated peer");
    expect(rig.transport.timeouts() >= 1, "the gate must have recorded the sequential schedule");
}

// A peer on the sequential schedule: it sends everything before it receives anything (root's view: every result send waits for the whole contribution), or returns nothing
// before it has everything (non-root's view). The pipelined rank must keep going: no leg may queue behind the other.
void against_a_sequential_peer(bool root, bool staged)
{
    Script s;
    s.peer_is_sequential = true;
    Rig rig(s, root, staged);
    const auto result = rig.run();
    expect(result.first == tbccl::ErrorCode::Success, "no leg may queue behind the other: " + result.second);
    expect(rig.transport.timeouts() == 0, "a leg waited for the other in vain");
    rig.expect_success_contents();
    rig.expect_quiescent();
}

// --- failure injection -----------------------------------------------------------------------------------------------------------------------------------------------

std::string leg(const char *what, std::size_t chunk) { return std::string(what) + " chunk " + std::to_string(chunk); }

void expect_cause(const std::pair<tbccl::ErrorCode, std::string> &result, tbccl::ErrorCode code, const std::string &where)
{
    expect(result.first == code, "wrong error code, text: " + result.second);
    expect(result.second.find("all_reduce") != std::string::npos, "the error must name the collective: " + result.second);
    if (!where.empty()) expect(result.second.find(where) != std::string::npos, "the error must name '" + where + "': " + result.second);
}

void recv_fails(bool root, std::size_t chunk, bool staged)
{
    Script s;
    s.fail_recv_at = chunk * kChunk;
    Rig rig(s, root, staged);
    const auto result = rig.run();
    expect_cause(result, tbccl::ErrorCode::TransportError, leg("recv", chunk));
    expect(result.second.find("injected recv failure") != std::string::npos, "the original text is kept: " + result.second);
    expect(rig.fatal_calls.load() == 1, "the failure aborts the communicator exactly once");
    if (root) expect(rig.reduced_chunks.size() <= chunk, "nothing at or after the failed chunk may be reduced");
    rig.expect_quiescent();
}

void send_fails(bool root, std::size_t chunk, bool staged)
{
    Script s;
    s.fail_send_at = chunk * kChunk;
    Rig rig(s, root, staged);
    const auto result = rig.run();
    expect_cause(result, tbccl::ErrorCode::TransportError, leg("send", chunk));
    expect(result.second.find("injected send failure") != std::string::npos, "the original text is kept: " + result.second);
    expect(rig.fatal_calls.load() == 1, "the failure aborts the communicator exactly once");
    rig.expect_quiescent();
}

// One direction fails while a leg of the other is blocked until the abort: the failure must be noticed without waiting for that leg.
void send_fails_while_a_recv_is_blocked(bool root, bool staged)
{
    Script s;
    if (root) s.fail_send_at = kChunk, s.hold_recv_at = 4 * kChunk;
    else s.fail_send_at = 3 * kChunk, s.hold_recv_at = 0;
    Rig rig(s, root, staged);
    const auto result = rig.run();
    expect_cause(result, tbccl::ErrorCode::TransportError, leg("send", root ? 1 : 3));
    expect(rig.fatal_calls.load() == 1, "the failure aborts the communicator exactly once");
    rig.expect_quiescent();
}

void recv_fails_while_a_send_is_blocked(bool root, bool staged)
{
    Script s;
    if (root) s.fail_recv_at = 3 * kChunk, s.hold_send_at = kChunk;
    else s.fail_recv_at = kChunk, s.hold_send_at = 4 * kChunk;
    Rig rig(s, root, staged);
    const auto result = rig.run();
    expect_cause(result, tbccl::ErrorCode::TransportError, leg("recv", root ? 3 : 1));
    expect(rig.fatal_calls.load() == 1, "the failure aborts the communicator exactly once");
    rig.expect_quiescent();
}

void both_fail(bool root, bool staged)
{
    Script s;
    if (root) s.fail_recv_at = 2 * kChunk, s.fail_send_at = 0;
    else s.fail_send_at = 2 * kChunk, s.fail_recv_at = 0;
    Rig rig(s, root, staged);
    const auto result = rig.run();
    expect(result.first == tbccl::ErrorCode::TransportError, "a cause, never the Aborted of the abort it triggered: " + result.second);
    expect(result.second.find("injected") != std::string::npos, "an original error text is reported: " + result.second);
    expect(rig.fatal_calls.load() >= 1, "the failure aborts the communicator");
    rig.expect_quiescent();
}

void reduce_fails(std::size_t chunk, Fault fault, bool staged)
{
    Rig rig(Script{}, true, staged);
    rig.fault = fault;
    rig.fault_chunk = chunk;
    const auto result = rig.run();
    expect(result.second.find("injected reduce failure") != std::string::npos, "the reduction's own error is rethrown: " + result.second);
    expect(result.first == (fault == Fault::ReduceThrowsError ? tbccl::ErrorCode::DeviceError : tbccl::ErrorCode::InternalError), "the exception type is preserved");
    expect(rig.fatal_calls.load() == 1, "a reduction failure aborts the communicator exactly once");
    expect(rig.reduced_chunks.size() == chunk + 1, "no chunk after the failed one may be reduced");
    rig.expect_quiescent();
}

// The staged copy of a chunk fails (the first and the final partial chunk), on the receive side (commit) and the send side (stage out of the source).
void staged_commit_fails(bool root, std::size_t chunk)
{
    Rig rig(Script{}, root, true, true);
    rig.recv_staged().fail_commit_at = chunk * kChunk;
    const auto result = rig.run();
    expect_cause(result, tbccl::ErrorCode::DeviceError, leg("recv", chunk));
    expect(rig.fatal_calls.load() == 1, "the failure aborts the communicator exactly once");
    rig.expect_quiescent();
}

void staged_source_fails(bool root, std::size_t chunk)
{
    Rig rig(Script{}, root, true, true);
    rig.send_staged().fail_stage_at = chunk * kChunk;
    const auto result = rig.run();
    expect_cause(result, tbccl::ErrorCode::DeviceError, leg("send", chunk));
    expect(rig.fatal_calls.load() == 1, "the failure aborts the communicator exactly once");
    rig.expect_quiescent();
}

// The communicator's abort arrives while the exchange is still in progress (a result / contribution chunk is held): the call ends with Aborted and nothing stays behind.
void abort_during_communication(bool root, bool staged)
{
    Script s;
    s.hold_recv_at = (root ? 4 : 2) * kChunk;
    Rig rig(s, root, staged);
    std::pair<tbccl::ErrorCode, std::string> result;
    std::thread caller([&] { result = rig.run(); });
    expect(rig.transport.wait_received_at_least(s.hold_recv_at, 20s), "the first chunks never arrived");
    rig.fatal_callback()("test abort during communication");
    caller.join();
    expect(result.first == tbccl::ErrorCode::Aborted, "an external abort ends the call with Aborted: " + result.second);
    rig.expect_quiescent();
}

// An abort that arrives while a chunk is being reduced.
void abort_during_reduction(bool staged)
{
    Rig rig(Script{}, true, staged);
    std::mutex m;
    std::condition_variable cv;
    bool in_reduce = false, release = false;
    std::pair<tbccl::ErrorCode, std::string> result;
    std::thread caller([&] {
        const auto fatal = rig.fatal_callback();
        try
        {
            tbccl::detail::run_n2_all_reduce_pipeline(
                rig.worker, rig.transport, true, rig.scratch_backend(), rig.primary_backend(), kBytes, kChunk, kElem,
                [&](std::size_t off, std::size_t n) {
                    if (off == kChunk)
                    {
                        std::unique_lock<std::mutex> lock(m);
                        in_reduce = true;
                        cv.notify_all();
                        cv.wait(lock, [&] { return release; });
                        throw tbccl::Error(tbccl::ErrorCode::Aborted, "aborted: reduction interrupted");
                    }
                    rig.reduce(off, n);
                },
                "all_reduce", fatal);
        }
        catch (const tbccl::Error &e)
        {
            result = {e.code(), e.what()};
        }
    });
    {
        std::unique_lock<std::mutex> lock(m);
        expect(cv.wait_for(lock, 20s, [&] { return in_reduce; }), "the reduction never started");
    }
    rig.fatal_callback()("test abort during reduction");
    {
        std::lock_guard<std::mutex> lock(m);
        release = true;
    }
    cv.notify_all();
    caller.join();
    expect(result.first == tbccl::ErrorCode::Aborted, "the abort during a reduction ends the call with Aborted: " + result.second);
    rig.expect_quiescent();
}

// --- admission and cleanup failures ----------------------------------------------------------------------------------------------------------------------------------

// The n-th chunk admission (1-based; the direction admitted up front first, then the other in order) fails as if allocation failed.
void admission_fails(bool root, int nth, bool staged)
{
    Rig rig(Script{}, root, staged);
    tbccl::detail::g_n2_pipeline_fail_admission.store(nth);
    const auto result = rig.run();
    tbccl::detail::g_n2_pipeline_fail_admission.store(0);
    expect(result.first == tbccl::ErrorCode::ResourceExhausted, "an admission failure is reported as such: " + result.second);
    expect(rig.fatal_calls.load() == 1, "a partial admission aborts the communicator exactly once (the peer would wait for the missing chunks)");
    rig.expect_quiescent();
}

// The abort notification itself misbehaves: the callback throws before it interrupts anything, or the reason cannot be built. The transfers must still be interrupted and
// waited for. `trigger` selects the failure that needs the abort.
enum class Trigger { RecvFails, SendFails, ReduceFails, AdmissionFails };
void abort_path_fault(bool root, Trigger trigger, bool callback_throws, bool reason_fails, bool staged)
{
    Script s;
    // Another leg is blocked until the abort arrives, so a notification that never interrupts it would hang the call.
    if (root)
    {
        if (trigger == Trigger::RecvFails) s.fail_recv_at = 2 * kChunk, s.hold_send_at = 0;
        if (trigger == Trigger::SendFails) s.fail_send_at = kChunk, s.hold_recv_at = 4 * kChunk;
        if (trigger == Trigger::ReduceFails) s.hold_recv_at = (kChunks - 1) * kChunk;
        if (trigger == Trigger::AdmissionFails) s.hold_recv_at = 0;
    }
    else
    {
        if (trigger == Trigger::RecvFails) s.fail_recv_at = 2 * kChunk, s.hold_send_at = (kChunks - 1) * kChunk;
        if (trigger == Trigger::SendFails) s.fail_send_at = kChunk, s.hold_recv_at = 0;
        if (trigger == Trigger::AdmissionFails) s.hold_send_at = 0;
    }
    Rig rig(s, root, staged);
    if (trigger == Trigger::ReduceFails)
    {
        rig.fault = Fault::ReduceThrowsError;
        rig.fault_chunk = 2;
    }
    std::atomic<int> callback_calls{0};
    const std::function<void(const std::string &)> throwing = [&](const std::string &) {
        callback_calls.fetch_add(1);
        throw std::runtime_error("abort callback failed before it interrupted anything");
    };
    const std::function<void(const std::string &)> normal = rig.fatal_callback();
    if (reason_fails) tbccl::detail::g_n2_exchange_fail_abort_message.store(1);
    if (trigger == Trigger::AdmissionFails) tbccl::detail::g_n2_pipeline_fail_admission.store(3);
    const auto result = rig.run(callback_throws ? &throwing : &normal);
    tbccl::detail::g_n2_exchange_fail_abort_message.store(0);
    tbccl::detail::g_n2_pipeline_fail_admission.store(0);
    expect(result.first != tbccl::ErrorCode::Success, "the failure must be reported");
    if (trigger == Trigger::AdmissionFails) expect(result.first == tbccl::ErrorCode::ResourceExhausted, "the admission failure stays the reported cause: " + result.second);
    if (callback_throws && !reason_fails) expect(callback_calls.load() == 1, "the callback was tried once");
    if (reason_fails) expect(rig.fatal_calls.load() == 0 && callback_calls.load() == 0, "the callback is not reached when the reason cannot be built");
    rig.expect_quiescent();
}

// Anything that throws after the first admission (here: right after every up-front chunk was admitted) still ends with every chunk finished.
void failure_after_admission(bool root, bool staged)
{
    Rig rig(Script{}, root, staged);
    tbccl::detail::g_n2_exchange_fail_after_admission.store(1);
    const auto result = rig.run();
    tbccl::detail::g_n2_exchange_fail_after_admission.store(0);
    expect(result.first == tbccl::ErrorCode::ResourceExhausted, "the unexpected failure is reported: " + result.second);
    expect(rig.fatal_calls.load() == 1, "the call aborts the communicator");
    rig.expect_quiescent();
}

// The worker serves the next call after a failed one.
void worker_recovers(bool root, bool staged)
{
    Script bad;
    bad.fail_recv_at = 2 * kChunk;
    Rig rig(bad, root, staged);
    std::atomic<int> calls{0};
    const std::function<void(const std::string &)> count_only = [&](const std::string &) { calls.fetch_add(1); }; // the worker's lanes stay usable (no worker abort)
    expect(rig.run(&count_only).first == tbccl::ErrorCode::TransportError, "the first call fails");
    expect(calls.load() == 1, "the failure requests the abort once");
    rig.expect_quiescent();
    rig.transport.reconfigure(Script{});
    rig.primary = rig.original;
    std::fill(rig.scratch.begin(), rig.scratch.end(), std::uint8_t{0});
    rig.reduced_chunks.clear();
    const auto result = rig.run(&count_only);
    expect(result.first == tbccl::ErrorCode::Success, "the worker must serve the next call: " + result.second);
    expect(calls.load() == 1, "the clean call must not abort");
    rig.expect_success_contents();
}

void repeated(bool root, bool gated)
{
    for (int i = 0; i < 200; ++i)
    {
        Script s;
        s.gate_last_on_first_result = gated;
        Rig rig(s, root, i % 2 == 1);
        const auto result = rig.run();
        expect(result.first == tbccl::ErrorCode::Success, "repeated call " + std::to_string(i) + ": " + result.second);
        rig.expect_success_contents();
    }
}
} // namespace

int main()
{
    scenario("geometry", geometry);
    for (bool root : {true, false})
    {
        const std::string who = root ? "root: " : "non-root: ";
        for (bool a : {false, true})
            for (bool b : {false, true})
            {
                if (!root && a != b) continue; // the non-root has one in-place buffer
                scenario((who + "success a_staged=" + std::to_string(a) + " b_staged=" + std::to_string(b)).c_str(), [&] { success(root, a, b); });
            }
        for (bool staged : {false, true})
        {
            const std::string tag = who + (staged ? "(staged) " : "(direct) ");
            scenario((tag + "overlap is possible").c_str(), [&] { overlap_is_possible(root, staged); });
            scenario((tag + "negative control: the sequential schedule is stopped by the overlap gate").c_str(), [&] { sequential_schedule_is_detected(root, staged); });
            scenario((tag + "finishes against a peer on the sequential schedule").c_str(), [&] { against_a_sequential_peer(root, staged); });
            for (std::size_t k : {std::size_t{0}, std::size_t{3}, kChunks - 1})
            {
                scenario((tag + "recv fails at chunk " + std::to_string(k)).c_str(), [&] { recv_fails(root, k, staged); });
                scenario((tag + "send fails at chunk " + std::to_string(k)).c_str(), [&] { send_fails(root, k, staged); });
                if (root) scenario((tag + "reduce throws tbccl::Error at chunk " + std::to_string(k)).c_str(), [&] { reduce_fails(k, Fault::ReduceThrowsError, staged); });
            }
            if (root) scenario((tag + "reduce throws std::runtime_error at chunk 2").c_str(), [&] { reduce_fails(2, Fault::ReduceThrowsStd, staged); });
            scenario((tag + "a send fails while a receive is blocked").c_str(), [&] { send_fails_while_a_recv_is_blocked(root, staged); });
            scenario((tag + "a receive fails while a send is blocked").c_str(), [&] { recv_fails_while_a_send_is_blocked(root, staged); });
            scenario((tag + "both directions fail").c_str(), [&] { both_fail(root, staged); });
            scenario((tag + "abort during communication").c_str(), [&] { abort_during_communication(root, staged); });
            if (root) scenario((tag + "abort during reduction").c_str(), [&] { abort_during_reduction(staged); });
            for (int nth : {1, 3, static_cast<int>(kChunks), static_cast<int>(kChunks) + 1, 2 * static_cast<int>(kChunks)})
                scenario((tag + "admission of request " + std::to_string(nth) + " fails").c_str(), [&] { admission_fails(root, nth, staged); });
            scenario((tag + "failure right after admission").c_str(), [&] { failure_after_admission(root, staged); });
            scenario((tag + "worker recovers after a failure").c_str(), [&] { worker_recovers(root, staged); });
            const std::pair<Trigger, const char *> triggers[] = {{Trigger::RecvFails, "recv fails"}, {Trigger::SendFails, "send fails"}, {Trigger::ReduceFails, "reduce fails"}, {Trigger::AdmissionFails, "admission fails"}};
            for (const auto &t : triggers)
            {
                if (!root && t.first == Trigger::ReduceFails) continue;
                scenario((tag + "abort path fault: " + t.second + ", callback throws before interrupting").c_str(), [&] { abort_path_fault(root, t.first, true, false, staged); });
                scenario((tag + "abort path fault: " + t.second + ", reason cannot be built").c_str(), [&] { abort_path_fault(root, t.first, false, true, staged); });
                scenario((tag + "abort path fault: " + t.second + ", callback throws and reason cannot be built").c_str(), [&] { abort_path_fault(root, t.first, true, true, staged); });
            }
        }
        scenario((who + "staged commit fails at the first chunk").c_str(), [&] { staged_commit_fails(root, 0); });
        scenario((who + "staged commit fails at the final partial chunk").c_str(), [&] { staged_commit_fails(root, kChunks - 1); });
        scenario((who + "staged source copy fails at the first chunk").c_str(), [&] { staged_source_fails(root, 0); });
        scenario((who + "staged source copy fails at the final partial chunk").c_str(), [&] { staged_source_fails(root, kChunks - 1); });
        scenario((who + "200 repeated calls").c_str(), [&] { repeated(root, false); });
        scenario((who + "200 repeated calls, overlap gated").c_str(), [&] { repeated(root, true); });
    }
    std::printf("n2_pipeline_unit_test passed\n");
    return 0;
}
