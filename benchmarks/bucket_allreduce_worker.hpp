#pragma once

// A minimal persistent collective progress worker,
// benchmark-support code (not a core library/public API -- the bucketed all-reduce
// overlap experiment asks whether async collective submission is useful, it does not yet
// prove a stable public collective API). Lets a producer (CUDA compute)
// submit bucket AllReduce jobs without blocking on each one's completion,
// while guaranteeing only ONE n2_all_reduce_tensor() call is ever
// in-flight at a time (no wire-level AllReduce multiplexing
// yet). This is purely a FIFO queue + one thread driving the
// EXISTING, unmodified n2_all_reduce_tensor() -- it is not a second
// network/transport implementation.
//
// Header-only (all methods inline in-class), matching this being
// benchmark/test-shared support code rather than a library target.

#include <tbccl/hetero_allreduce.hpp>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace tbccl_bench
{

// Lightweight completion handle, matching tbccl::TransferWork's shape
// (wait()/is_completed()/has_error()/error()) -- no
// cancellation/future semantics beyond this.
class BucketAllReduceWork
{
public:
    void wait()
    {
        std::unique_lock<std::mutex> lock(state_->mutex);
        state_->cv.wait(lock, [this] { return state_->completed; });
    }

    bool is_completed() const
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->completed;
    }

    bool has_error() const
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->completed && !state_->error_message.empty();
    }

    std::string error() const
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->error_message;
    }

private:
    friend class BucketAllReduceWorker;

    struct State
    {
        std::mutex mutex;
        std::condition_variable cv;
        bool completed = false;
        std::string error_message; // empty == ok
    };

    std::shared_ptr<State> state_ = std::make_shared<State>();
};

// One job: everything n2_all_reduce_tensor() needs, plus a bucket index
// for diagnostic logging. Non-owning pointers -- all referenced objects
// (transport, worker, backends) must outlive the job's execution, exactly
// matching TransferRequest's own buffer-lifetime contract (one persistent
// tensor/backend per bucket makes this trivial for the benchmark).
struct BucketAllReduceJob
{
    tbccl::Transport *transport = nullptr;
    tbccl::TensorCommWorker *worker = nullptr;
    tbccl::AsyncMemoryBackend *recv_backend = nullptr;
    tbccl::AsyncMemoryBackend *send_backend = nullptr;
    tbccl::LocalReduceBackend *reduce_backend = nullptr; // non-null iff root
    std::size_t rank = 0;
    std::size_t root = 0;
    std::size_t total_bytes = 0;
    std::size_t chunk_hint = 0;
    std::size_t count = 0;
    tbccl::DataType datatype = tbccl::DataType::Float32;
    std::size_t bucket_index = 0; // diagnostic only
};

// One persistent thread, a bounded FIFO queue, exactly one
// n2_all_reduce_tensor() call executing at a time. the failure policy:
// on the first job that throws, that job's Work reports the error, the
// worker aborts (stops pulling further jobs and fails every
// already-queued or future job immediately with a "worker aborted"
// error) rather than attempting to continue -- the wire protocol state
// after a failed AllReduce leg cannot be assumed recoverable, so
// continuing risks a cross-rank deadlock, not just a local error.
class BucketAllReduceWorker
{
public:
    explicit BucketAllReduceWorker(std::size_t queue_depth = 8)
        : queue_depth_(queue_depth)
    {
        thread_ = std::thread([this] { run(); });
    }

    ~BucketAllReduceWorker()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_requested_ = true;
        }
        queue_cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    BucketAllReduceWorker(const BucketAllReduceWorker &) = delete;
    BucketAllReduceWorker &operator=(const BucketAllReduceWorker &) = delete;

    // Blocks only on queue capacity, never on the job's own collective
    // completion -- matching TensorCommWorker's own enqueue() contract.
    // Returns immediately with a live BucketAllReduceWork once queued
    // (or immediately-failed, if the worker has already aborted).
    BucketAllReduceWork enqueue(BucketAllReduceJob job)
    {
        BucketAllReduceWork work;

        std::unique_lock<std::mutex> lock(mutex_);

        if (aborted_)
        {
            lock.unlock();
            complete_error(work, "worker aborted after a prior job failed");
            return work;
        }

        queue_cv_.wait(lock, [this]
                       { return queue_.size() < queue_depth_ || aborted_ || shutdown_requested_; });

        if (aborted_)
        {
            lock.unlock();
            complete_error(work, "worker aborted after a prior job failed");
            return work;
        }
        if (shutdown_requested_)
        {
            lock.unlock();
            complete_error(work, "worker is shutting down");
            return work;
        }

        queue_.push_back(QueuedJob{std::move(job), work.state_});
        lock.unlock();
        queue_cv_.notify_all();
        return work;
    }

private:
    struct QueuedJob
    {
        BucketAllReduceJob job;
        std::shared_ptr<BucketAllReduceWork::State> state;
    };

    static void complete_ok(BucketAllReduceWork &work)
    {
        std::lock_guard<std::mutex> lock(work.state_->mutex);
        work.state_->completed = true;
        work.state_->cv.notify_all();
    }

    static void complete_error(BucketAllReduceWork &work, const std::string &message)
    {
        std::lock_guard<std::mutex> lock(work.state_->mutex);
        work.state_->completed = true;
        work.state_->error_message = message;
        work.state_->cv.notify_all();
    }

    static void complete_state_ok(std::shared_ptr<BucketAllReduceWork::State> &state)
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->completed = true;
        state->cv.notify_all();
    }

    static void complete_state_error(
        std::shared_ptr<BucketAllReduceWork::State> &state, const std::string &message)
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->completed = true;
        state->error_message = message;
        state->cv.notify_all();
    }

    void run()
    {
        for (;;)
        {
            QueuedJob item;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                queue_cv_.wait(lock, [this]
                               { return !queue_.empty() || shutdown_requested_; });
                if (queue_.empty() && shutdown_requested_) return;
                if (queue_.empty()) continue;
                item = std::move(queue_.front());
                queue_.pop_front();
            }
            queue_cv_.notify_all();

            if (aborted_)
            {
                complete_state_error(item.state, "worker aborted after a prior job failed");
                continue;
            }

            try
            {
                tbccl::n2_all_reduce_tensor(
                    *item.job.transport, *item.job.worker, *item.job.recv_backend,
                    *item.job.send_backend, item.job.reduce_backend, item.job.rank,
                    item.job.root, item.job.total_bytes, item.job.chunk_hint,
                    item.job.count, item.job.datatype);
                complete_state_ok(item.state);
            }
            catch (const std::exception &error)
            {
                complete_state_error(item.state, error.what());
                std::lock_guard<std::mutex> lock(mutex_);
                aborted_ = true;
                // Drain and fail every job already queued behind this one
                // -- do not attempt to execute them.
                while (!queue_.empty())
                {
                    complete_state_error(queue_.front().state,
                                         "worker aborted after a prior job failed");
                    queue_.pop_front();
                }
            }
        }
    }

    std::size_t queue_depth_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable queue_cv_;
    std::deque<QueuedJob> queue_;
    bool shutdown_requested_ = false;
    bool aborted_ = false;
};

} // namespace tbccl_bench
