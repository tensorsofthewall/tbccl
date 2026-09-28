#include "ring_executor.hpp"

#include <mutex>
#include <utility>

namespace tbccl::detail
{

    RingExecutor::RingExecutor()
        // stats_.worker_start_count is set to 1 up front, in the
        // member initializer, rather than assigned in the constructor
        // body — the constructor body runs only after every member
        // (including worker_, declared last precisely so it is
        // constructed last) is already fully initialized, by which
        // point the newly-launched worker thread may already be
        // running concurrently and reading stats_ via stats(); a
        // body-assignment here would itself be a race against that.
        : stats_{1, 0, 0}, worker_([this]() { worker_loop(); })
    {
    }

    RingExecutor::~RingExecutor() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }

        cv_.notify_all();
        worker_.join();
    }

    void RingExecutor::submit(std::function<void()> job)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = std::move(job);
            job_ready_ = true;
            job_done_ = false;
            ++stats_.submitted_jobs;
        }

        cv_.notify_all();
    }

    void RingExecutor::wait_for_job()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() { return job_done_; });
    }

    void RingExecutor::worker_loop()
    {
        while (true)
        {
            std::function<void()> job;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&]() { return job_ready_ || stopping_; });

                if (stopping_ && !job_ready_)
                {
                    return;
                }

                job = std::move(job_);
                job_ready_ = false;
            }

            try
            {
                // The job constructed by RingExecutor::execute()
                // already catches every exception from the caller's
                // `sender` callable internally, so this should be
                // unreachable in practice — but a worker-thread
                // exception must never escape regardless of what
                // future job producers do.
                job();
            }
            catch (...)
            {
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                job_done_ = true;
                job_ = nullptr;
                ++stats_.completed_jobs;
            }

            cv_.notify_all();
        }
    }

    RingExecutorStats RingExecutor::stats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    RingExecutor &RingExecutorAccess::get_or_create(World &world)
    {
        if (!world.ring_executor_)
        {
            world.ring_executor_ = std::make_unique<RingExecutor>();
        }

        return *world.ring_executor_;
    }

    RingExecutorStats RingExecutorAccess::stats(const World &world)
    {
        if (!world.ring_executor_)
        {
            return RingExecutorStats{};
        }

        return world.ring_executor_->stats();
    }

} // namespace tbccl::detail
