// Phase 34 Part F/AP: RawIoWorker, extracted from
// async_raw_transport_bench.cpp into its own header so it can be unit
// tested directly (tests/raw_io_worker_test.cpp) as well as used by the
// benchmark. Diagnostic/benchmark-only code -- NOT part of the async
// substrate's public library surface, and deliberately does NOT use
// TensorCommWorker/StagingPool/AsyncMemoryBackend/ChunkPlan/
// TransferWork. It exists purely to answer "does moving a Transport
// call onto a minimal persistent background std::thread reproduce the
// Phase 33 regression on its own?"
#pragma once

#include <tbccl/transport.hpp>

#include <condition_variable>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#if defined(__linux__)
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif

namespace tbccl_bench
{

// One persistent thread, one outstanding request at a time,
// condition_variable handoff. No staging, no memcpy, no chunking, no
// second idle thread.
class RawIoWorker
{
public:
    enum class Op { Send, Recv };

#if defined(__APPLE__)
    // requested_qos, when set, is applied via
    // pthread_set_qos_class_self_np() from inside the worker's own
    // thread body before it processes any request (Part Q: a
    // controlled Mac QoS A/B). Default-constructed (nullopt) means
    // "leave the new thread at whatever QoS pthread_create gives it."
    explicit RawIoWorker(tbccl::Transport *transport,
                          std::optional<qos_class_t> requested_qos = std::nullopt)
        : transport_(transport), requested_qos_(requested_qos)
#else
    explicit RawIoWorker(tbccl::Transport *transport) : transport_(transport)
#endif
    {
        std::unique_lock<std::mutex> lock(mutex_);
        thread_ = std::thread([this] { run(); });
        started_cv_.wait(lock, [this] { return started_; });
    }

    ~RawIoWorker()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_request_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    RawIoWorker(const RawIoWorker &) = delete;
    RawIoWorker &operator=(const RawIoWorker &) = delete;

    void run_op(Op op, void *data, std::size_t bytes)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        op_ = op;
        data_ = data;
        bytes_ = bytes;
        has_request_ = true;
        error_.clear();
        cv_request_.notify_one();
        cv_done_.wait(lock, [this] { return !has_request_; });
        if (!error_.empty())
        {
            throw std::runtime_error(error_);
        }
    }

    std::thread::native_handle_type native_handle() { return thread_.native_handle(); }

#if defined(__linux__)
    pid_t linux_tid() const { return linux_tid_; }
#elif defined(__APPLE__)
    pthread_t pthread_handle() const { return pthread_self_of_worker_; }
#endif

private:
    void run()
    {
#if defined(__linux__)
        linux_tid_ = static_cast<pid_t>(::syscall(SYS_gettid));
#elif defined(__APPLE__)
        pthread_self_of_worker_ = pthread_self();
        if (requested_qos_.has_value())
        {
            pthread_set_qos_class_self_np(*requested_qos_, 0);
        }
#endif
        {
            std::lock_guard<std::mutex> lock(mutex_);
            started_ = true;
        }
        started_cv_.notify_one();

        std::unique_lock<std::mutex> lock(mutex_);
        while (true)
        {
            cv_request_.wait(lock, [this] { return has_request_ || stop_; });
            if (stop_ && !has_request_) return;
            const Op op = op_;
            void *data = data_;
            const std::size_t bytes = bytes_;
            lock.unlock();
            std::string local_error;
            try
            {
                if (op == Op::Send) transport_->send(data, bytes);
                else transport_->recv(data, bytes);
            }
            catch (const std::exception &e)
            {
                local_error = e.what();
            }
            lock.lock();
            error_ = local_error;
            has_request_ = false;
            cv_done_.notify_one();
        }
    }

    tbccl::Transport *transport_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_request_;
    std::condition_variable cv_done_;
    std::condition_variable started_cv_;
    bool started_ = false;
    bool has_request_ = false;
    bool stop_ = false;
    Op op_ = Op::Send;
    void *data_ = nullptr;
    std::size_t bytes_ = 0;
    std::string error_;
#if defined(__linux__)
    pid_t linux_tid_ = 0;
#elif defined(__APPLE__)
    pthread_t pthread_self_of_worker_{};
    std::optional<qos_class_t> requested_qos_;
#endif
};

} // namespace tbccl_bench
