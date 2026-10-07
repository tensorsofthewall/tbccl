#include <tbccl/hetero_allreduce.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace tbccl
{
namespace
{
    std::atomic<std::uint64_t> g_transfer_id_counter{1};

    std::uint64_t next_transfer_id()
    {
        return g_transfer_id_counter.fetch_add(1, std::memory_order_relaxed);
    }

    void wait_or_throw(TransferWork &work, const char *what)
    {
        work.wait();
        if (work.has_error())
        {
            throw std::runtime_error(
                std::string("n2_all_reduce_tensor: ") + what +
                " failed: " + work.error());
        }
    }

    // Optional stage-decomposition timing, disabled by default -- same
    // pattern as TBCCL_ASYNC_TIMING (a
    // single getenv() at first use, cached; prints directly to stderr
    // for a human/script-readable log, not threaded through any public
    // return value). Never adds cost to the normal hot path.
    bool timing_enabled()
    {
        static const bool enabled = (std::getenv("TBCCL_ALLREDUCE_TIMING") != nullptr);
        return enabled;
    }

    using Clock = std::chrono::steady_clock;

    double elapsed_us(Clock::time_point start)
    {
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    }
} // namespace

void n2_all_reduce_tensor(
    Transport &transport,
    TensorCommWorker &worker,
    AsyncMemoryBackend &recv_backend,
    AsyncMemoryBackend &send_backend,
    LocalReduceBackend *reduce_backend,
    std::size_t rank,
    std::size_t root,
    std::size_t total_bytes,
    std::size_t chunk_hint,
    std::size_t count,
    DataType datatype)
{
    if (root >= 2)
    {
        throw std::runtime_error(
            "n2_all_reduce_tensor: root must be 0 or 1 (N=2 only), got " +
            std::to_string(root));
    }

    if (rank == root && reduce_backend == nullptr)
    {
        throw std::runtime_error(
            "n2_all_reduce_tensor: reduce_backend is required on the root "
            "rank");
    }

    const bool timing = timing_enabled();
    const auto total_start = Clock::now();

    if (rank == root)
    {
        // ROOT: RECEIVE_PEER -> REDUCE_LOCAL -> SEND_RESULT -> COMPLETE.
        TransferRequest recv_request;
        recv_request.transfer_id = next_transfer_id();
        recv_request.direction = TransferDirection::Recv;
        recv_request.backend = &recv_backend;
        recv_request.transport = &transport;
        recv_request.total_bytes = total_bytes;
        recv_request.chunk_hint = chunk_hint;
        recv_request.shared_lane = true; // Keep the sequential single-FIFO behaviour on a duplex worker

        const auto recv_start = Clock::now();
        TransferWork recv_work = worker.enqueue(recv_request);
        wait_or_throw(recv_work, "reduce-to-root receive");
        const double recv_us = timing ? elapsed_us(recv_start) : 0.0;

        const auto reduce_start = Clock::now();
        reduce_backend->reduce_sum(count, datatype);
        const double reduce_us = timing ? elapsed_us(reduce_start) : 0.0;

        TransferRequest send_request;
        send_request.transfer_id = next_transfer_id();
        send_request.direction = TransferDirection::Send;
        send_request.backend = &send_backend;
        send_request.transport = &transport;
        send_request.total_bytes = total_bytes;
        send_request.chunk_hint = chunk_hint;

        const auto send_start = Clock::now();
        TransferWork send_work = worker.enqueue(send_request);
        wait_or_throw(send_work, "broadcast-back send");
        const double send_us = timing ? elapsed_us(send_start) : 0.0;

        if (timing)
        {
            std::fprintf(
                stderr,
                "[tbccl_allreduce_timing] role=root rank=%zu root=%zu "
                "bytes=%zu recv_leg_us=%.1f reduce_us=%.1f send_leg_us=%.1f "
                "total_us=%.1f\n",
                rank, root, total_bytes, recv_us, reduce_us, send_us,
                elapsed_us(total_start));
        }
    }
    else
    {
        // NON-ROOT: SEND_LOCAL -> RECEIVE_RESULT -> COMPLETE.
        TransferRequest send_request;
        send_request.transfer_id = next_transfer_id();
        send_request.direction = TransferDirection::Send;
        send_request.backend = &send_backend;
        send_request.transport = &transport;
        send_request.total_bytes = total_bytes;
        send_request.chunk_hint = chunk_hint;

        const auto send_start = Clock::now();
        TransferWork send_work = worker.enqueue(send_request);
        wait_or_throw(send_work, "reduce-to-root send");
        const double send_us = timing ? elapsed_us(send_start) : 0.0;

        TransferRequest recv_request;
        recv_request.transfer_id = next_transfer_id();
        recv_request.direction = TransferDirection::Recv;
        recv_request.backend = &recv_backend;
        recv_request.transport = &transport;
        recv_request.total_bytes = total_bytes;
        recv_request.chunk_hint = chunk_hint;
        recv_request.shared_lane = true; // Keep the sequential single-FIFO behaviour on a duplex worker

        const auto recv_start = Clock::now();
        TransferWork recv_work = worker.enqueue(recv_request);
        wait_or_throw(recv_work, "broadcast-back receive");
        const double recv_us = timing ? elapsed_us(recv_start) : 0.0;

        if (timing)
        {
            std::fprintf(
                stderr,
                "[tbccl_allreduce_timing] role=non-root rank=%zu root=%zu "
                "bytes=%zu send_leg_us=%.1f recv_leg_us=%.1f total_us=%.1f\n",
                rank, root, total_bytes, send_us, recv_us,
                elapsed_us(total_start));
        }
    }
}

} // namespace tbccl
