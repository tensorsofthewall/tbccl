#include <stdexcept>
#include <tbccl/async_transfer.hpp>

#include <tbccl/transport.hpp>

#include <algorithm>
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
    std::string error_message;
};

TransferWork::TransferWork() : state_(std::make_shared<State>()) {}

void TransferWork::wait()
{
    std::unique_lock<std::mutex> lock(state_->mutex);
    state_->cv.wait(lock, [&]() { return state_->done; });
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

namespace detail
{

    void TransferWorkAccess::complete_ok(const std::shared_ptr<TransferWork::State> &state)
    {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->done) return; // exactly one terminal transition
            state->done = true;
        }
        state->cv.notify_all();
    }

    void TransferWorkAccess::complete_error(
        const std::shared_ptr<TransferWork::State> &state,
        const std::string &message)
    {
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            if (state->done) return;
            state->done = true;
            state->error_flag = true;
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
        std::string error_message;
    };

    void mark_failed(ChunkProgress &progress, const std::string &message)
    {
        {
            std::lock_guard<std::mutex> lock(progress.mutex);
            if (!progress.failed)
            {
                progress.failed = true;
                progress.error_message = message;
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
                mark_failed(progress, error.what());
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
                mark_failed(progress, error.what());
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
                mark_failed(progress, error.what());
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
                mark_failed(progress, error.what());
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
                mark_failed(progress, error.what());
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
                mark_failed(progress, error.what());
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
          queue_depth(std::max<std::size_t>(1, queue_depth_))
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
            detail::TransferWorkAccess::complete_error(item.second, message);
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
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(
                    lock,
                    [&]() { return !queue.empty() || stopping.load(); });

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
                active = true;
            }
            queue_cv.notify_all(); // wakes an enqueue() blocked on capacity

            if (aborted.load())
            {
                detail::TransferWorkAccess::complete_error(item.second, aborted_message());
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

            process_request(item.first, item.second);
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
                request.transport->send(source, request.total_bytes);
            }
            else
            {
                void *destination = request.backend->direct_destination_data();
                request.transport->recv(destination, request.total_bytes);
            }
            if (timing)
            {
                std::fprintf(stderr, "[tbccl_timing] transport_call_end_us=%.1f transfer_id=%llu\n",
                             now_us(), static_cast<unsigned long long>(request.transfer_id));
            }
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
            detail::TransferWorkAccess::complete_error(state, error.what());
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
            detail::TransferWorkAccess::complete_error(
                state, "staged path requested but staging thread disabled "
                       "(TBCCL_ASYNC_NO_STAGING_THREAD diagnostic mode)");
            record_stat(false);
            return;
        }

        try
        {
            const auto chunks =
                plan_chunks(request.total_bytes, request.chunk_hint, request.alignment);

            if (chunks.empty())
            {
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
                detail::TransferWorkAccess::complete_error(state, progress.error_message);
                record_stat(false);
                return;
            }

            detail::TransferWorkAccess::complete_ok(state);
            record_stat(true);
        }
        catch (const std::exception &error)
        {
            fatal(error.what());
            detail::TransferWorkAccess::complete_error(state, error.what());
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
    Impl &lane = (recv_impl_ && request.direction == TransferDirection::Recv) ? *recv_impl_ : *impl_;
    TransferWork work;

    if (timing_enabled())
    {
        std::fprintf(stderr, "[tbccl_timing] enqueue_start_us=%.1f transfer_id=%llu\n",
                     now_us(), static_cast<unsigned long long>(request.transfer_id));
    }

    {
        std::unique_lock<std::mutex> lock(lane.queue_mutex);
        lane.queue_cv.wait(
            lock,
            [&]() { return lane.aborted.load() || lane.queue.size() < lane.queue_depth; });
        if (lane.aborted.load())
        {
            throw std::runtime_error(
                "aborted: communicator aborted" +
                (lane.abort_reason.empty() ? "" : " (" + lane.abort_reason + ")"));
        }

        lane.queue.emplace_back(std::move(request), work.state_);
    }

    {
        std::lock_guard<std::mutex> lock(lane.stats_mutex);
        ++lane.stats.submitted;
    }

    lane.queue_cv.notify_all();

    return work;
}

void TensorCommWorker::abort(const std::string &reason)
{
    impl_->request_abort(reason);
    if (recv_impl_) recv_impl_->request_abort(reason);
}

bool TensorCommWorker::busy() const
{
    for (Impl *lane : {impl_.get(), recv_impl_.get()})
    {
        if (lane == nullptr) continue;
        std::lock_guard<std::mutex> lock(lane->queue_mutex);
        if (lane->active || !lane->queue.empty()) return true;
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
