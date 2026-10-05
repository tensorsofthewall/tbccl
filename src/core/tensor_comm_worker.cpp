#include "../transport/latency_trace.hpp"
#include "../transport/progress_knobs.hpp"
#include <stdexcept>
#include <tbccl/async_transfer.hpp>
#include <tbccl/error.hpp>

#include <tbccl/transport.hpp>

#include <algorithm>
#include <limits>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace tbccl
{

namespace
{

    // Optional per-stage timing, disabled by default (a single getenv()
    // at first use, cached -- the "disabled by default or
    // compiled/activated only under benchmark flag" requirement).
    // Prints directly to stderr rather than threading a diagnostic
    // return value through TransferWork's public API, since this is a
    // one-sided producer of timing data for humans/scripts reading a
    // captured log, not a value any caller consumes programmatically.
    bool timing_enabled()
    {
        static const bool enabled = (std::getenv("TBCCL_ASYNC_TIMING") != nullptr);
        return enabled;
    }

    // diagnostic-only control to test the "idle staging-thread affects
    // network-thread scheduling" hypothesis. When set, Impl skips
    // creating staging_thread entirely -- ONLY safe for workloads that
    // exclusively use the direct path (chunk_hint==0 &&
    // backend->supports_direct_transport_access()), since the staged
    // path's submit_job()/wait_job() would otherwise block forever
    // with no staging thread to service it. Not part of the public
    // API; gated off by default, for a controlled A/B only (no
    // production topology change without justification).
    bool no_staging_thread_enabled()
    {
        static const bool enabled = (std::getenv("TBCCL_ASYNC_NO_STAGING_THREAD") != nullptr);
        return enabled;
    }

    // Framed transfers (TransferRequest::framed). 16 bytes: u32 magic, u32 reserved (0), u64 payload length, big endian.
    constexpr std::uint32_t kFrameMagic = 0x54424D50U; // "TBMP"
    constexpr std::size_t kFrameHeaderBytes = 16;

    void encode_frame_header(std::uint8_t (&out)[kFrameHeaderBytes], std::uint64_t length)
    {
        for (int i = 0; i < 4; ++i) out[i] = static_cast<std::uint8_t>(kFrameMagic >> (24 - 8 * i));
        for (int i = 4; i < 8; ++i) out[i] = 0;
        for (int i = 0; i < 8; ++i) out[8 + i] = static_cast<std::uint8_t>(length >> (56 - 8 * i));
    }

    // Throws "protocol_mismatch: ..." unless the header is a frame of exactly `expected` bytes.
    void check_frame_header(const std::uint8_t (&in)[kFrameHeaderBytes], std::uint64_t expected)
    {
        std::uint32_t magic = 0;
        for (int i = 0; i < 4; ++i) magic = (magic << 8) | in[i];
        if (magic != kFrameMagic)
            throw Error(ErrorCode::ProtocolMismatch, "protocol_mismatch: the peer's point-to-point message has no valid frame header (the other rank is not sending a framed message here)");
        std::uint64_t length = 0;
        for (int i = 0; i < 8; ++i) length = (length << 8) | in[8 + i];
        if (length != expected)
            throw Error(ErrorCode::ProtocolMismatch, 
                "protocol_mismatch: point-to-point size mismatch: the peer sent " + std::to_string(length) + " bytes but this rank posted a receive for " +
                std::to_string(expected) + " bytes");
    }

    double now_us()
    {
        return std::chrono::duration<double, std::micro>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }

} // namespace

// ---------------------------------------------------------------------
// TransferWork
// ---------------------------------------------------------------------

struct TransferWork::State
{
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    bool error_flag = false;
    ErrorCode error_code = ErrorCode::Success;
    std::string error_message;
    std::uint64_t lat_id = 0; // the latency-audit work trace id (0 = untraced)
    std::atomic<bool> done_flag{false}; // experiments: lets a polling waiter see the terminal transition without the mutex
};

TransferWork::TransferWork() : state_(std::make_shared<State>())
{
    state_->lat_id = std::exchange(detail::tl_lat_next_work_id, 0); // A collective Work is traced under the id its submitter reserved
}

void TransferWork::wait()
{
    if (const long poll_us = detail::tl_is_executor_thread ? detail::progress().exec_poll_us : detail::progress().wait_poll_us; poll_us > 0)
    {
        const std::int64_t deadline = detail::progress_now_ns() + poll_us * 1000;
        while (!state_->done_flag.load(std::memory_order_acquire) && detail::progress_now_ns() < deadline) detail::cpu_relax();
    }
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->cv.wait(lock, [&]() { return state_->done; });
    if (detail::lat_on())
    {
        detail::lat_event(detail::kLatWaiterAwake, state_->lat_id);
        detail::lat_event(detail::kLatWaitReturn, state_->lat_id);
    }
}

bool TransferWork::is_completed() const
{
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->done;
}

bool TransferWork::has_error() const
{
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->error_flag;
}

std::string TransferWork::error() const
{
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->error_message;
}

ErrorCode TransferWork::error_code() const
{
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->error_code;
}

bool TransferWork::wait_for(std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(state_->mutex);
    return state_->cv.wait_for(lock, timeout, [&]() { return state_->done; });
}

namespace detail
{

    void TransferWorkAccess::complete_ok(const std::shared_ptr<TransferWork::State> &state)
    {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->done) return; // exactly one terminal transition
            state->done = true;
            state->done_flag.store(true, std::memory_order_release);
            detail::lat_event(detail::kLatTerminal, state->lat_id);
        }
        state->cv.notify_all();
    }

    void TransferWorkAccess::complete_error(
        const std::shared_ptr<TransferWork::State> &state,
        ErrorCode code,
        const std::string &message)
    {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->done) return;
            state->done = true;
            state->done_flag.store(true, std::memory_order_release);
            state->error_flag = true;
            state->error_code = code;
            state->error_message = message;
        }
        state->cv.notify_all();
    }

    TransferWork TransferWorkAccess::make()
    {
        return TransferWork();
    }

} // namespace detail

namespace
{

    // Per-request handoff between the staging thread and the network
    // thread for one TransferRequest's chunks. Stack-allocated inside
    // process_request() -- both threads only ever touch it while that
    // call's submit_job()/wait_job() pair brackets their concurrent
    // work, so it never outlives the call that owns it (same lifetime
    // discipline as RingExecutor's RingSession, independently
    // implemented --.md's RingExecutor-reuse decision for why this
    // isn't literal code sharing).
    struct ChunkProgress
    {
        explicit ChunkProgress(std::size_t chunk_count)
            : slot_for(chunk_count, static_cast<std::size_t>(-1))
        {
        }

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<std::size_t> slot_for;
        std::size_t ready_count = 0;
        bool failed = false;
        ErrorCode error_code = ErrorCode::InternalError;
        std::string error_message;
    };

    void mark_failed(ChunkProgress &progress, const std::exception &error)
    {
        {
            std::lock_guard<std::mutex> lock(progress.mutex);
            if (!progress.failed)
            {
                progress.failed = true;
                progress.error_code = error_code_of(error);
                progress.error_message = error.what();
            }
        }
        progress.cv.notify_all();
    }

    // Staging thread's role for a Send request: acquire a slot, ask the
    // backend to fill it from the source tensor, publish it as ready.
    //
    // Optional per-chunk timeline timing (same
    // TBCCL_ASYNC_TIMING gate as the direct path's instrumentation) --
    // this is what lets a benchmark prove device-copy/network overlap
    // from something other than aggregate throughput, item
    // 73's explicit requirement ("do not claim pipeline overlap from
    // throughput alone").
    void staging_produce_send(
        const TransferRequest &request,
        const std::vector<Chunk> &chunks,
        StagingPool &pool,
        ChunkProgress &progress)
    {
        const bool timing = timing_enabled();
        for (std::size_t i = 0; i < chunks.size(); ++i)
        {
            std::size_t slot;
            try
            {
                slot = pool.acquire();
            }
            catch (const std::exception &error)
            {
                mark_failed(progress, error);
                return;
            }

            if (timing)
            {
                std::fprintf(stderr,
                    "[tbccl_chunk_timing] stage=source_stage_start chunk=%zu transfer_id=%llu t_us=%.1f\n",
                    i, static_cast<unsigned long long>(request.transfer_id), now_us());
            }
            try
            {
                request.backend->stage_source_chunk(chunks[i], pool.data(slot));
            }
            catch (const std::exception &error)
            {
                pool.release(slot);
                mark_failed(progress, error);
                return;
            }
            if (timing)
            {
                std::fprintf(stderr,
                    "[tbccl_chunk_timing] stage=source_stage_end chunk=%zu transfer_id=%llu t_us=%.1f\n",
                    i, static_cast<unsigned long long>(request.transfer_id), now_us());
            }

            {
                std::lock_guard<std::mutex> lock(progress.mutex);
                if (progress.failed)
                {
                    pool.release(slot);
                    return;
                }
                progress.slot_for[i] = slot;
                progress.ready_count = i + 1;
            }
            progress.cv.notify_all();
        }
    }

    // Network thread's role for a Send request: wait for each chunk in
    // order, send it, release the slot.
    void network_consume_send(
        const TransferRequest &request,
        const std::vector<Chunk> &chunks,
        StagingPool &pool,
        ChunkProgress &progress)
    {
        const bool timing = timing_enabled();
        for (std::size_t i = 0; i < chunks.size(); ++i)
        {
            std::size_t slot;
            {
                std::unique_lock<std::mutex> lock(progress.mutex);
                progress.cv.wait(
                    lock,
                    [&]() { return progress.ready_count > i || progress.failed; });
                if (progress.failed)
                {
                    return;
                }
                slot = progress.slot_for[i];
            }

            if (timing)
            {
                std::fprintf(stderr,
                    "[tbccl_chunk_timing] stage=network_start chunk=%zu transfer_id=%llu t_us=%.1f\n",
                    i, static_cast<unsigned long long>(request.transfer_id), now_us());
            }
            try
            {
                request.transport->send(pool.data(slot), chunks[i].size);
            }
            catch (const std::exception &error)
            {
                pool.release(slot);
                mark_failed(progress, error);
                return;
            }
            if (timing)
            {
                std::fprintf(stderr,
                    "[tbccl_chunk_timing] stage=network_end chunk=%zu transfer_id=%llu t_us=%.1f\n",
                    i, static_cast<unsigned long long>(request.transfer_id), now_us());
            }

            pool.release(slot);
        }
    }

    // Network thread's role for a Recv request: acquire a slot, receive
    // into it, publish it as ready.
    void network_produce_recv(
        const TransferRequest &request,
        const std::vector<Chunk> &chunks,
        StagingPool &pool,
        ChunkProgress &progress)
    {
        const bool timing = timing_enabled();
        for (std::size_t i = 0; i < chunks.size(); ++i)
        {
            std::size_t slot;
            try
            {
                slot = pool.acquire();
            }
            catch (const std::exception &error)
            {
                mark_failed(progress, error);
                return;
            }

            if (timing)
            {
                std::fprintf(stderr,
                    "[tbccl_chunk_timing] stage=network_start chunk=%zu transfer_id=%llu t_us=%.1f\n",
                    i, static_cast<unsigned long long>(request.transfer_id), now_us());
            }
            try
            {
                request.transport->recv(pool.data(slot), chunks[i].size);
            }
            catch (const std::exception &error)
            {
                pool.release(slot);
                mark_failed(progress, error);
                return;
            }
            if (timing)
            {
                std::fprintf(stderr,
                    "[tbccl_chunk_timing] stage=network_end chunk=%zu transfer_id=%llu t_us=%.1f\n",
                    i, static_cast<unsigned long long>(request.transfer_id), now_us());
            }

            {
                std::lock_guard<std::mutex> lock(progress.mutex);
                if (progress.failed)
                {
                    pool.release(slot);
                    return;
                }
                progress.slot_for[i] = slot;
                progress.ready_count = i + 1;
            }
            progress.cv.notify_all();
        }
    }

    // Staging thread's role for a Recv request: wait for each received
    // chunk in order, commit it into the destination tensor, release
    // the slot.
    void staging_consume_recv(
        const TransferRequest &request,
        const std::vector<Chunk> &chunks,
        StagingPool &pool,
        ChunkProgress &progress)
    {
        const bool timing = timing_enabled();
        for (std::size_t i = 0; i < chunks.size(); ++i)
        {
            std::size_t slot;
            {
                std::unique_lock<std::mutex> lock(progress.mutex);
                progress.cv.wait(
                    lock,
                    [&]() { return progress.ready_count > i || progress.failed; });
                if (progress.failed)
                {
                    return;
                }
                slot = progress.slot_for[i];
            }

            if (timing)
            {
                std::fprintf(stderr,
                    "[tbccl_chunk_timing] stage=dest_stage_start chunk=%zu transfer_id=%llu t_us=%.1f\n",
                    i, static_cast<unsigned long long>(request.transfer_id), now_us());
            }
            try
            {
                request.backend->commit_destination_chunk(chunks[i], pool.data(slot));
            }
            catch (const std::exception &error)
            {
                pool.release(slot);
                mark_failed(progress, error);
                return;
            }
            if (timing)
            {
                std::fprintf(stderr,
                    "[tbccl_chunk_timing] stage=dest_stage_end chunk=%zu transfer_id=%llu t_us=%.1f\n",
                    i, static_cast<unsigned long long>(request.transfer_id), now_us());
            }

            pool.release(slot);
        }
    }

} // namespace

// ---------------------------------------------------------------------
// TensorCommWorker::Impl
// ---------------------------------------------------------------------

struct TensorCommWorker::Impl
{
    std::size_t pipeline_depth;
    std::size_t queue_depth;

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<std::pair<TransferRequest, std::shared_ptr<TransferWork::State>>> queue;

    std::mutex job_mutex;
    std::condition_variable job_cv;
    bool job_ready = false;
    bool job_done = true;
    std::function<void()> job;

    std::atomic<bool> stopping{false};
    bool paused = false; // guarded by queue_mutex (test hook)
    std::atomic<std::size_t> queued{0};      // experiments: mirror of queue.size() readable without the lock
    std::atomic<std::int64_t> last_done_ns{0}; // experiments: when this lane last finished a request
    bool inline_busy = false;                // guarded by queue_mutex: a caller-thread send (direct-send control) owns the lane

    // Requests queued or running on this lane, per direction (0 = Send, 1 = Recv). Incremented by enqueue(), decremented right BEFORE a request's
    // Work becomes terminal, so a caller that waited for a request sees its lane as free again.
    std::atomic<int> inflight[2] = {{0}, {0}};
    static int dir_index(TransferDirection d) { return d == TransferDirection::Send ? 0 : 1; }
    void retire(TransferDirection d) { inflight[dir_index(d)].fetch_sub(1, std::memory_order_acq_rel); }

    // Terminal abort. Written once under queue_mutex (so enqueue/dequeue observe it consistently), then read
    // lock-free. `active` is true while the network thread owns a dequeued request.
    std::atomic<bool> aborted{false};
    std::string abort_reason;
    bool active = false;
    std::function<void(const std::string &)> on_fatal;

    std::mutex stats_mutex;
    TensorCommWorker::Stats stats;

    // A fresh StagingPool per request was measured to cost an order of
    // magnitude more than a reused one for large buffers --
    // first-touch page faults on freshly allocated memory, not the
    // memcpy bandwidth itself. Only network_loop's single thread (via
    // process_request) ever touches this cache, so it needs no
    // separate lock -- it is not shared with staging_loop/job_ handoff
    // state. A request whose (chunk_capacity, depth) doesn't match the
    // cached pool still pays a fresh allocation (unavoidable -- the
    // pool's buffers are sized to a specific chunk_capacity), but the
    // common case of repeated same-shape transfers (the normal
    // steady-state loop a caller runs) now reuses one already-touched
    // pool instead of re-paying page-fault cost every single time.
    std::unique_ptr<StagingPool> cached_pool;
    std::size_t cached_chunk_capacity = 0;
    std::size_t cached_depth = 0;

    // Declared last: both threads' functions touch every member above
    // this point, and std::thread launches immediately -- see
    // ring_executor.hpp's identical reasoning for why worker_ (here,
    // both threads) must be constructed only after all state it reads
    // is already alive.
    std::thread staging_thread;
    std::thread network_thread;

    // The staging thread is created on the first request that needs the staged path (never for a Host-only
    // communicator, which is always direct), so an N-rank communicator with two lanes per peer does not pay for threads
    // it never uses. Only the network thread creates it, and ~Impl reads it after joining the network thread.
    const bool staging_allowed = !no_staging_thread_enabled();

    void ensure_staging_thread()
    {
        if (!staging_thread.joinable()) staging_thread = std::thread([this]() { staging_loop(); });
    }

    Impl(std::size_t pipeline_depth_, std::size_t queue_depth_)
        : pipeline_depth(std::max<std::size_t>(1, pipeline_depth_)),
          queue_depth(queue_depth_ == TensorCommWorker::kUnboundedAdmission ? std::numeric_limits<std::size_t>::max() : std::max<std::size_t>(1, queue_depth_))
    {
        network_thread = std::thread([this]() { network_loop(); });
    }

    ~Impl()
    {
        stopping.store(true);
        queue_cv.notify_all();
        job_cv.notify_all();

        if (network_thread.joinable())
        {
            network_thread.join();
        }
        if (staging_thread.joinable())
        {
            staging_thread.join();
        }
    }

    std::string aborted_message()
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        return "aborted: communicator aborted" + (abort_reason.empty() ? "" : " (" + abort_reason + ")");
    }

    // Fails every queued (never started, so never touching user memory) request. The queue is moved out under the
    // lock and completed outside it.
    void drain_queue_failed()
    {
        std::deque<std::pair<TransferRequest, std::shared_ptr<TransferWork::State>>> doomed;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            doomed.swap(queue);
        }
        queue_cv.notify_all(); // wakes enqueue() callers blocked on capacity
        if (doomed.empty()) return;
        const std::string message = aborted_message();
        for (auto &item : doomed)
        {
            retire(item.first.direction);
            detail::TransferWorkAccess::complete_error(item.second, ErrorCode::Aborted, message);
            record_stat(false);
        }
    }

    void request_abort(const std::string &reason)
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (aborted.load()) return;
            abort_reason = reason;
            aborted.store(true);
        }
        drain_queue_failed();
    }

    // Runs `staging_job` on the persistent staging thread, blocking the
    // caller (the network thread) until a previous job (if any) has
    // finished -- same single-job-at-a-time submit/wait discipline as
    // RingExecutor::submit()/wait_for_job(), reimplemented here rather
    // than shared because RingExecutor is scoped to src/collectives and
    // tied to ring-collective tracing.
    void submit_job(std::function<void()> staging_job)
    {
        {
            std::unique_lock<std::mutex> lock(job_mutex);
            job_cv.wait(lock, [&]() { return job_done; });
            job = std::move(staging_job);
            job_ready = true;
            job_done = false;
        }
        job_cv.notify_all();
    }

    void wait_job()
    {
        std::unique_lock<std::mutex> lock(job_mutex);
        job_cv.wait(lock, [&]() { return job_done; });
    }

    void staging_loop()
    {
        while (true)
        {
            std::function<void()> local_job;
            {
                std::unique_lock<std::mutex> lock(job_mutex);
                job_cv.wait(
                    lock,
                    [&]() { return job_ready || stopping.load(); });

                if (!job_ready)
                {
                    // stopping with no job in flight -- only path that
                    // exits this thread.
                    return;
                }

                local_job = std::move(job);
                job_ready = false;
            }

            local_job();

            {
                std::lock_guard<std::mutex> lock(job_mutex);
                job_done = true;
            }
            job_cv.notify_all();
        }
    }

    void network_loop()
    {
        while (true)
        {
            std::pair<TransferRequest, std::shared_ptr<TransferWork::State>> item;
            if (const long spin_us = detail::progress().tx_spin_us; spin_us > 0)
            {
                const std::int64_t deadline = last_done_ns.load(std::memory_order_relaxed) + spin_us * 1000;
                if (deadline > spin_us * 1000) // a request finished before: spin for its successor, then block
                    while (queued.load(std::memory_order_acquire) == 0 && !stopping.load(std::memory_order_relaxed) && !aborted.load(std::memory_order_relaxed) &&
                           detail::progress_now_ns() < deadline)
                        detail::cpu_relax();
            }
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(
                    lock,
                    [&]() { return (!queue.empty() && !paused && !inline_busy) || stopping.load(); });

                if (queue.empty())
                {
                    // stopping with the queue fully drained -- only
                    // path that exits this thread. A pending queue is
                    // always drained first, even mid-shutdown (
                    // item 173: shutdown must not deadlock, but it also
                    // must not silently drop already-enqueued work).
                    return;
                }

                item = std::move(queue.front());
                queue.pop_front();
                queued.fetch_sub(1, std::memory_order_release);
                active = true;
            }
            queue_cv.notify_all(); // wakes an enqueue() blocked on capacity

            if (aborted.load())
            {
                retire(item.first.direction);
                detail::TransferWorkAccess::complete_error(item.second, ErrorCode::Aborted, aborted_message());
                record_stat(false);
                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    active = false;
                }
                queue_cv.notify_all();
                continue;
            }

            if (timing_enabled())
            {
                std::fprintf(stderr, "[tbccl_timing] dequeued_us=%.1f transfer_id=%llu\n",
                             now_us(), static_cast<unsigned long long>(item.first.transfer_id));
            }

            detail::tl_lat_current_id = detail::lat_on() ? item.second->lat_id : 0;
            detail::lat_event(detail::kLatDequeued, detail::tl_lat_current_id);
            process_request(item.first, item.second);
            detail::tl_lat_current_id = 0;
            last_done_ns.store(detail::progress_now_ns(), std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                active = false;
            }
            queue_cv.notify_all();
        }
    }

    // The direct path. Runs entirely on the network thread (the
    // caller of process_request, i.e. network_loop) -- no staging
    // thread involvement, no StagingPool, no TBCCL-owned memcpy.
    // Measured to remove the two-thread handoff/synchronization cost
    // that dominated the staged path's regression for host<->host
    // transfers.
    void process_request_direct(
        const TransferRequest &request,
        const std::shared_ptr<TransferWork::State> &state)
    {
        try
        {
            const bool timing = timing_enabled();
            if (timing)
            {
                std::fprintf(stderr, "[tbccl_timing] transport_call_start_us=%.1f transfer_id=%llu\n",
                             now_us(), static_cast<unsigned long long>(request.transfer_id));
            }
            if (request.direction == TransferDirection::Send)
            {
                const void *source = request.backend->direct_source_data();
                if (request.framed)
                {
                    std::uint8_t header[kFrameHeaderBytes];
                    encode_frame_header(header, request.total_bytes);
                    request.transport->send_framed(header, sizeof(header), source, request.total_bytes);
                }
                else
                {
                    request.transport->send(source, request.total_bytes);
                }
            }
            else
            {
                void *destination = request.backend->direct_destination_data();
                if (request.framed)
                {
                    std::uint8_t header[kFrameHeaderBytes];
                    const std::uint64_t expected = request.total_bytes;
                    request.transport->recv_framed(header, sizeof(header), destination, request.total_bytes, [expected](const void *h) {
                        check_frame_header(*reinterpret_cast<const std::uint8_t(*)[kFrameHeaderBytes]>(h), expected);
                    });
                }
                else
                {
                    request.transport->recv(destination, request.total_bytes);
                }
            }
            if (timing)
            {
                std::fprintf(stderr, "[tbccl_timing] transport_call_end_us=%.1f transfer_id=%llu\n",
                             now_us(), static_cast<unsigned long long>(request.transfer_id));
            }
            retire(request.direction);
            detail::TransferWorkAccess::complete_ok(state);
            if (timing)
            {
                std::fprintf(stderr, "[tbccl_timing] complete_ok_returned_us=%.1f transfer_id=%llu\n",
                             now_us(), static_cast<unsigned long long>(request.transfer_id));
            }
            record_stat(true);
        }
        catch (const std::exception &error)
        {
            fatal(error.what());
            retire(request.direction);
            detail::TransferWorkAccess::complete_error(state, error_code_of(error), error.what());
            record_stat(false);
        }
    }

    void process_request(
        const TransferRequest &request,
        const std::shared_ptr<TransferWork::State> &state)
    {
        // The design's pseudocode, exactly: direct path only when the memory
        // is directly transport-accessible AND the caller has not
        // explicitly asked for chunking/pipelining (chunk_hint == 0).
        // A caller that explicitly sets chunk_hint (wanting pipelined
        // overlap even for host memory, e.g. to hide compute behind
        // transfer -- see async_overlap_bench.cpp) still gets the
        // staged path, matching the async tensor-transfer work's
        // existing chunk/depth semantics exactly.
        if (request.chunk_hint == 0 && request.backend->supports_direct_transport_access())
        {
            process_request_direct(request, state);
            return;
        }

        if (!staging_allowed)
        {
            // Safety net: TBCCL_ASYNC_NO_STAGING_THREAD was set but this
            // request needs the staged path, which would otherwise
            // deadlock forever waiting for a staging thread that was
            // never created.
            retire(request.direction);
            detail::TransferWorkAccess::complete_error(
                state, ErrorCode::Unsupported, "unsupported: staged path requested but staging thread disabled "
                       "(TBCCL_ASYNC_NO_STAGING_THREAD diagnostic mode)");
            record_stat(false);
            return;
        }

        try
        {
            if (request.framed)
            {
                // Staged path: the header travels on its own before the chunks, and is checked before any user memory is written.
                std::uint8_t header[kFrameHeaderBytes];
                if (request.direction == TransferDirection::Send)
                {
                    encode_frame_header(header, request.total_bytes);
                    request.transport->send(header, sizeof(header));
                }
                else
                {
                    request.transport->recv(header, sizeof(header));
                    check_frame_header(header, request.total_bytes);
                }
            }
            const auto chunks =
                plan_chunks(request.total_bytes, request.chunk_hint, request.alignment);

            if (chunks.empty())
            {
                retire(request.direction);
                detail::TransferWorkAccess::complete_ok(state);
                record_stat(true);
                return;
            }

            std::size_t chunk_capacity = 0;
            for (const auto &chunk : chunks)
            {
                chunk_capacity = std::max(chunk_capacity, chunk.size);
            }

            const std::size_t depth =
                std::max<std::size_t>(1, std::min(pipeline_depth, chunks.size()));

            if (!cached_pool || cached_chunk_capacity != chunk_capacity || cached_depth != depth)
            {
                cached_pool = std::make_unique<StagingPool>(chunk_capacity, depth);
                cached_chunk_capacity = chunk_capacity;
                cached_depth = depth;
            }
            StagingPool &pool = *cached_pool;
            ChunkProgress progress(chunks.size());
            ensure_staging_thread();

            if (request.direction == TransferDirection::Send)
            {
                submit_job(
                    [&]() { staging_produce_send(request, chunks, pool, progress); });
                network_consume_send(request, chunks, pool, progress);
                wait_job();
            }
            else
            {
                submit_job(
                    [&]() { staging_consume_recv(request, chunks, pool, progress); });
                network_produce_recv(request, chunks, pool, progress);
                wait_job();
            }

            if (progress.failed)
            {
                fatal(progress.error_message);
                retire(request.direction);
                detail::TransferWorkAccess::complete_error(state, progress.error_code, progress.error_message);
                record_stat(false);
                return;
            }

            retire(request.direction);
            detail::TransferWorkAccess::complete_ok(state);
            record_stat(true);
        }
        catch (const std::exception &error)
        {
            fatal(error.what());
            retire(request.direction);
            detail::TransferWorkAccess::complete_error(state, error_code_of(error), error.what());
            record_stat(false);
        }
    }

    // A transport/protocol/device failure after a transfer has started leaves the peers' stream position unknown:
    // poison the communicator (idempotent, non-blocking, safe from this thread).
    void fatal(const std::string &message)
    {
        if (on_fatal) on_fatal(message);
    }

    void record_stat(bool ok)
    {
        std::lock_guard<std::mutex> lock(stats_mutex);
        if (ok)
        {
            ++stats.completed;
        }
        else
        {
            ++stats.failed;
        }
    }
};

// ---------------------------------------------------------------------
// TensorCommWorker
// ---------------------------------------------------------------------

TensorCommWorker::TensorCommWorker(std::size_t pipeline_depth, std::size_t queue_depth, bool duplex)
    : impl_(std::make_unique<Impl>(pipeline_depth, queue_depth)),
      recv_impl_(duplex ? std::make_unique<Impl>(pipeline_depth, queue_depth) : nullptr)
{
}

TensorCommWorker::~TensorCommWorker() = default;

TransferWork TensorCommWorker::enqueue(TransferRequest request)
{
    // Duplex routing (the N-rank runtime work). Each direction stays FIFO because all of its outstanding requests share one lane, and a lane never
    // serves both directions at once, so a receive waiting for data never delays a send (and vice versa). When only one direction is in flight (the
    // common, sequential case) everything runs on the first lane, exactly as the single-lane worker did; the second lane (and its thread) is woken
    // only for genuine send/recv concurrency.
    std::unique_lock<std::mutex> route_lock(route_mutex_, std::defer_lock);
    Impl *chosen = impl_.get();
    if (recv_impl_ && !request.shared_lane)
    {
        route_lock.lock();
        const int me = Impl::dir_index(request.direction), other = 1 - me;
        Impl *lanes[2] = {impl_.get(), recv_impl_.get()};
        chosen = nullptr;
        for (Impl *l : lanes)
            if (l->inflight[me].load(std::memory_order_acquire) > 0) chosen = l; // sticky: keep this direction's FIFO on one lane
        if (!chosen)
        {
            for (Impl *l : lanes)
            {
                if (l->inflight[other].load(std::memory_order_acquire) == 0)
                {
                    chosen = l; // a lane the other direction is not using
                    break;
                }
            }
        }
        if (!chosen) chosen = impl_.get();
    }
    Impl &lane = *chosen;
    const TransferDirection request_direction = request.direction; // `request` is moved into the queue below
    lane.inflight[Impl::dir_index(request.direction)].fetch_add(1, std::memory_order_acq_rel);
    TransferWork work;
    if (detail::lat_on())
    {
        auto &trace = detail::LatencyTrace::get();
        work.state_->lat_id = trace.new_id();
        detail::tl_lat_last_enqueued_id = work.state_->lat_id;
        const std::uint32_t aux = (request.direction == TransferDirection::Send ? 0u : 1u << 31) | static_cast<std::uint32_t>(request.total_bytes & 0x7fffffff);
        trace.record_at(detail::tl_lat_submit_ns != 0 ? detail::tl_lat_submit_ns : detail::lat_now_ns(), detail::kLatSubmitEnter, work.state_->lat_id, aux);
    }

    if (timing_enabled())
    {
        std::fprintf(stderr, "[tbccl_timing] enqueue_start_us=%.1f transfer_id=%llu\n",
                     now_us(), static_cast<unsigned long long>(request.transfer_id));
    }

    try
    {
        std::unique_lock<std::mutex> lock(lane.queue_mutex);
        if (lane.queue_depth != std::numeric_limits<std::size_t>::max())
        {
            // blocking backpressure for standalone users; the Communicator's lanes never take this branch
            lane.queue_cv.wait(
                lock,
                [&]() { return lane.aborted.load() || lane.queue.size() < lane.queue_depth; });
        }
        if (lane.aborted.load())
        {
            throw Error(ErrorCode::Aborted,
                "aborted: communicator aborted" +
                (lane.abort_reason.empty() ? "" : " (" + lane.abort_reason + ")"));
        }

        const bool inline_send = detail::progress().direct_send_max > 0 && request.direction == TransferDirection::Send &&
                                 request.total_bytes <= static_cast<std::size_t>(detail::progress().direct_send_max);
        const bool inline_exec = detail::tl_is_executor_thread && detail::progress().exec_inline_max > 0 &&
                                 request.total_bytes <= static_cast<std::size_t>(detail::progress().exec_inline_max);
        if (((inline_send && request.framed) || inline_exec) && request.chunk_hint == 0 && request.backend->supports_direct_transport_access() &&
            lane.queue.empty() && !lane.active && !lane.inline_busy && !lane.paused && !lane.aborted.load())
        {
            lane.inline_busy = true;
            lock.unlock();
            const std::uint64_t saved = detail::tl_lat_current_id;
            detail::tl_lat_current_id = work.state_->lat_id;
            lane.process_request_direct(request, work.state_);
            detail::tl_lat_current_id = saved;
            lock.lock();
            lane.inline_busy = false;
            lock.unlock();
            lane.queue_cv.notify_all();
            detail::lat_event(detail::kLatEnqueued, work.state_->lat_id);
            return work;
        }
        lane.queue.emplace_back(std::move(request), work.state_);
        lane.queued.fetch_add(1, std::memory_order_release);
    }
    catch (...)
    {
        lane.retire(request_direction); // not admitted: the lane's in-flight count must not leak
        throw;
    }

    {
        std::lock_guard<std::mutex> lock(lane.stats_mutex);
        ++lane.stats.submitted;
    }

    lane.queue_cv.notify_all();
    detail::lat_event(detail::kLatEnqueued, work.state_->lat_id);

    return work;
}

void TensorCommWorker::set_progress_paused(bool paused)
{
    for (Impl *lane : {impl_.get(), recv_impl_.get()})
    {
        if (lane == nullptr) continue;
        {
            std::lock_guard<std::mutex> lock(lane->queue_mutex);
            lane->paused = paused;
        }
        lane->queue_cv.notify_all();
    }
}

void TensorCommWorker::abort(const std::string &reason)
{
    impl_->request_abort(reason);
    if (recv_impl_) recv_impl_->request_abort(reason);
}

bool TensorCommWorker::busy() const
{
    // In-flight counts are decremented BEFORE a request's Work becomes terminal, so a caller that has seen every Work complete never finds this
    // worker busy (queue/active flags are cleared slightly later). Communicator's destructor relies on that: "busy" there means abort the peers.
    for (Impl *lane : {impl_.get(), recv_impl_.get()})
    {
        if (lane == nullptr) continue;
        if (lane->inflight[0].load(std::memory_order_acquire) > 0 || lane->inflight[1].load(std::memory_order_acquire) > 0) return true;
    }
    return false;
}

void TensorCommWorker::wait_idle()
{
    for (Impl *lane : {impl_.get(), recv_impl_.get()})
    {
        if (lane == nullptr) continue;
        std::unique_lock<std::mutex> lock(lane->queue_mutex);
        lane->queue_cv.wait(lock, [&]() { return !lane->active && lane->queue.empty(); });
    }
}

void TensorCommWorker::set_fatal_handler(std::function<void(const std::string &)> handler)
{
    if (recv_impl_) recv_impl_->on_fatal = handler;
    impl_->on_fatal = std::move(handler);
}

TensorCommWorker::Stats TensorCommWorker::stats() const
{
    Stats total;
    for (Impl *lane : {impl_.get(), recv_impl_.get()})
    {
        if (lane == nullptr) continue;
        std::lock_guard<std::mutex> lock(lane->stats_mutex);
        total.submitted += lane->stats.submitted;
        total.completed += lane->stats.completed;
        total.failed += lane->stats.failed;
    }
    return total;
}

} // namespace tbccl
