#pragma once

// Phase 55: internal per-operation latency trace for small-message investigation. Off by default and free when off (one cached bool test per
// site). With TBCCL_LATENCY_TRACE=<path> every point-to-point operation records monotonic-nanosecond events into a preallocated ring and the
// process writes `<path>.<pid>` (CSV: ns,id,site,aux) at exit. No public API, no wire change: an operation is identified by a process-local id
// assigned at admission; correlating a send in one process with the receive in another is done offline by FIFO order per direction (loopback
// shares CLOCK_MONOTONIC, so the stage deltas are directly comparable there).
//
// Sites (the plan's T0..T10): 0 submit entered, 1 admission enqueued, 2 worker dequeued, 3 send syscall entered, 4 send syscall returned,
// 5 receive syscall returned (first bytes), 6 frame header validated, 7 destination ready (payload complete), 8 Work terminal,
// 9 waiter awakened, 10 wait returned.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

namespace tbccl::detail
{

enum LatSite : std::uint32_t
{
    kLatSubmitEnter = 0,
    kLatEnqueued = 1,
    kLatDequeued = 2,
    kLatSendEnter = 3,
    kLatSendReturn = 4,
    kLatRecvFirstBytes = 5,
    kLatRecvHeaderOk = 6,
    kLatPayloadDone = 7,
    kLatTerminal = 8,
    kLatWaiterAwake = 9,
    kLatWaitReturn = 10,
};

struct LatEvent
{
    std::int64_t ns;
    std::uint64_t id;
    std::uint32_t site;
    std::uint32_t aux;
};

inline std::int64_t lat_now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

class LatencyTrace
{
public:
    static LatencyTrace &get()
    {
        static LatencyTrace instance;
        return instance;
    }

    bool on() const noexcept { return on_; }

    std::size_t recorded() const noexcept { return std::min(count_.load(std::memory_order_acquire), buffer_.size()); }
    const LatEvent &event(std::size_t i) const noexcept { return buffer_[i]; }

    std::uint64_t new_id() noexcept { return next_id_.fetch_add(1, std::memory_order_relaxed); }

    void record(std::uint32_t site, std::uint64_t id, std::uint32_t aux = 0) noexcept { record_at(lat_now_ns(), site, id, aux); }

    void record_at(std::int64_t ns, std::uint32_t site, std::uint64_t id, std::uint32_t aux = 0) noexcept
    {
        const std::size_t slot = count_.fetch_add(1, std::memory_order_relaxed);
        if (slot < buffer_.size()) buffer_[slot] = LatEvent{ns, id, site, aux};
    }

    ~LatencyTrace()
    {
        if (!on_) return;
        const std::string path = std::string(std::getenv("TBCCL_LATENCY_TRACE")) + "." + std::to_string(::getpid());
        if (std::FILE *f = std::fopen(path.c_str(), "w"))
        {
            const std::size_t n = std::min(count_.load(), buffer_.size());
            for (std::size_t i = 0; i < n; ++i)
                std::fprintf(f, "%lld,%llu,%u,%u\n", static_cast<long long>(buffer_[i].ns), static_cast<unsigned long long>(buffer_[i].id), buffer_[i].site, buffer_[i].aux);
            std::fclose(f);
        }
    }

private:
    LatencyTrace() : on_(std::getenv("TBCCL_LATENCY_TRACE") != nullptr)
    {
        if (on_) buffer_.resize(std::size_t{1} << 21);
    }

    bool on_;
    std::atomic<std::uint64_t> next_id_{1};
    std::atomic<std::size_t> count_{0};
    std::vector<LatEvent> buffer_;
};

inline bool lat_on() noexcept
{
    static const bool on = LatencyTrace::get().on();
    return on;
}

// The id of the request the calling thread is executing (set by the worker around one request; 0 = none). The transport layer reads it so the
// socket-level sites need no knowledge of the transfer machinery above.
inline thread_local std::uint64_t tl_lat_current_id = 0;
inline thread_local std::int64_t tl_lat_submit_ns = 0;

inline void lat_event(std::uint32_t site, std::uint64_t id, std::uint32_t aux = 0) noexcept
{
    if (lat_on() && id != 0) LatencyTrace::get().record(site, id, aux);
}

} // namespace tbccl::detail
