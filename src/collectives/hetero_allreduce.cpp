#include <tbccl/hetero_allreduce.hpp>

#include <atomic>
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

        TransferWork recv_work = worker.enqueue(recv_request);
        wait_or_throw(recv_work, "reduce-to-root receive");

        reduce_backend->reduce_sum(count, datatype);

        TransferRequest send_request;
        send_request.transfer_id = next_transfer_id();
        send_request.direction = TransferDirection::Send;
        send_request.backend = &send_backend;
        send_request.transport = &transport;
        send_request.total_bytes = total_bytes;
        send_request.chunk_hint = chunk_hint;

        TransferWork send_work = worker.enqueue(send_request);
        wait_or_throw(send_work, "broadcast-back send");
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

        TransferWork send_work = worker.enqueue(send_request);
        wait_or_throw(send_work, "reduce-to-root send");

        TransferRequest recv_request;
        recv_request.transfer_id = next_transfer_id();
        recv_request.direction = TransferDirection::Recv;
        recv_request.backend = &recv_backend;
        recv_request.transport = &transport;
        recv_request.total_bytes = total_bytes;
        recv_request.chunk_hint = chunk_hint;

        TransferWork recv_work = worker.enqueue(recv_request);
        wait_or_throw(recv_work, "broadcast-back receive");
    }
}

} // namespace tbccl
