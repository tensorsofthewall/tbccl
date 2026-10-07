// TBCCL C ABI v1: P2P and collective submission. Every call validates at the ABI level, translates POD to the C++ API, and returns a Work handle; none of
// them waits for transport progress. P2P is byte-based; collectives follow docs/c_abi_v1.md.

#include "c_internal.hpp"

#include <tbccl/reduction.hpp>

#include <vector>

using namespace tbccl;
using namespace tbccl::capi;

namespace
{
// Reserves the heap slot for a Work handle BEFORE the operation is submitted, so that an allocation failure can never strand a submitted operation without
// a handle the caller could wait on.
class WorkSlot
{
public:
    WorkSlot() : mem_(::operator new(sizeof(tbcclWork_st))) {}
    WorkSlot(const WorkSlot &) = delete;
    WorkSlot &operator=(const WorkSlot &) = delete;
    ~WorkSlot() { if (mem_ != nullptr) ::operator delete(mem_); }
    tbcclWork_t emplace(Work work) noexcept
    {
        tbcclWork_t handle = new (mem_) tbcclWork_st{std::move(work)};
        mem_ = nullptr;
        return handle;
    }

private:
    void *mem_;
};

bool peer_ok(const tbcclComm_st &c, uint32_t peer) { return peer < c.comm->world_size() && peer != c.comm->rank(); }
} // namespace

extern "C"
{

tbcclResult_t TBCCL_CALL tbcclSend(tbcclComm_t comm, const tbcclBuffer *buffer, uint32_t peer, const tbcclExecContext *ctx, tbcclWork_t *work)
{
    return guard([&]() -> tbcclResult_t {
        if (work == nullptr) return TBCCL_INVALID_ARGUMENT;
        *work = nullptr;
        if (comm == nullptr) return TBCCL_INVALID_ARGUMENT;
        BufferView view;
        ExecutionContext exec;
        if (tbcclResult_t r = to_cpp(buffer, view)) return r;
        if (tbcclResult_t r = to_cpp(ctx, exec)) return r;
        if (!peer_ok(*comm, peer)) return TBCCL_INVALID_ARGUMENT;
        WorkSlot slot;
        *work = slot.emplace(comm->comm->send(view, view.bytes, DataType::UInt8, peer, exec)); // byte-based: UInt8 x bytes
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclRecv(tbcclComm_t comm, const tbcclBuffer *buffer, uint32_t peer, const tbcclExecContext *ctx, tbcclWork_t *work)
{
    return guard([&]() -> tbcclResult_t {
        if (work == nullptr) return TBCCL_INVALID_ARGUMENT;
        *work = nullptr;
        if (comm == nullptr) return TBCCL_INVALID_ARGUMENT;
        BufferView view;
        ExecutionContext exec;
        if (tbcclResult_t r = to_cpp(buffer, view)) return r;
        if (tbcclResult_t r = to_cpp(ctx, exec)) return r;
        if (!peer_ok(*comm, peer)) return TBCCL_INVALID_ARGUMENT;
        WorkSlot slot;
        *work = slot.emplace(comm->comm->recv(view, view.bytes, DataType::UInt8, peer, exec));
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclBarrier(tbcclComm_t comm, tbcclWork_t *work)
{
    return guard([&]() -> tbcclResult_t {
        if (work == nullptr) return TBCCL_INVALID_ARGUMENT;
        *work = nullptr;
        if (comm == nullptr) return TBCCL_INVALID_ARGUMENT;
        WorkSlot slot;
        *work = slot.emplace(comm->comm->barrier());
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclBroadcast(tbcclComm_t comm, const tbcclBuffer *buffer, uint32_t root, const tbcclExecContext *ctx, tbcclWork_t *work)
{
    return guard([&]() -> tbcclResult_t {
        if (work == nullptr) return TBCCL_INVALID_ARGUMENT;
        *work = nullptr;
        if (comm == nullptr) return TBCCL_INVALID_ARGUMENT;
        BufferView view;
        ExecutionContext exec;
        if (tbcclResult_t r = to_cpp(buffer, view)) return r;
        if (tbcclResult_t r = to_cpp(ctx, exec)) return r;
        if (root >= comm->comm->world_size()) return TBCCL_INVALID_ARGUMENT;
        WorkSlot slot;
        *work = slot.emplace(comm->comm->broadcast(view, root, exec));
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclAllGather(tbcclComm_t comm, const tbcclBuffer *send, const tbcclBuffer *recv, const tbcclExecContext *ctx, tbcclWork_t *work)
{
    return guard([&]() -> tbcclResult_t {
        if (work == nullptr) return TBCCL_INVALID_ARGUMENT;
        *work = nullptr;
        if (comm == nullptr) return TBCCL_INVALID_ARGUMENT;
        BufferView in, out;
        ExecutionContext exec;
        if (tbcclResult_t r = to_cpp(send, in)) return r;
        if (tbcclResult_t r = to_cpp(recv, out)) return r;
        if (tbcclResult_t r = to_cpp(ctx, exec)) return r;
        const std::size_t world = comm->comm->world_size();
        if (in.bytes != 0 && world > static_cast<std::size_t>(-1) / in.bytes) return TBCCL_INVALID_ARGUMENT; // send bytes * world overflows
        if (out.bytes != in.bytes * world) return TBCCL_INVALID_ARGUMENT;
        if (out.memory_kind != in.memory_kind) return TBCCL_INVALID_ARGUMENT;
        std::vector<BufferView> slots(world, out);
        for (std::size_t r = 0; r < world; ++r)
        {
            slots[r].data = static_cast<char *>(out.data) + r * in.bytes; // rank r's contribution lands at byte offset r * send bytes
            slots[r].bytes = in.bytes;
        }
        WorkSlot slot;
        *work = slot.emplace(comm->comm->all_gather(in, slots, exec));
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclAllReduce(
    tbcclComm_t comm, const tbcclBuffer *send, const tbcclBuffer *recv, uint64_t count, tbcclDataType_t dtype, tbcclReduceOp_t op,
    const tbcclExecContext *ctx, tbcclWork_t *work)
{
    return guard([&]() -> tbcclResult_t {
        if (work == nullptr) return TBCCL_INVALID_ARGUMENT;
        *work = nullptr;
        if (comm == nullptr) return TBCCL_INVALID_ARGUMENT;
        BufferView in, out;
        ExecutionContext exec;
        DataType d;
        ReduceOp o;
        if (tbcclResult_t r = to_cpp(send, in)) return r;
        if (tbcclResult_t r = to_cpp(recv, out)) return r;
        if (tbcclResult_t r = to_cpp(ctx, exec)) return r;
        if (!to_cpp(dtype, d) || !to_cpp(op, o)) return TBCCL_INVALID_ARGUMENT;
        if (count > static_cast<std::uint64_t>(static_cast<std::size_t>(-1))) return TBCCL_INVALID_ARGUMENT;
        const std::size_t elems = static_cast<std::size_t>(count), width = datatype_size(d);
        if (elems != 0 && elems > static_cast<std::size_t>(-1) / width) return TBCCL_INVALID_ARGUMENT; // count * sizeof(dtype) overflows
        if (in.bytes < elems * width || out.bytes < elems * width) return TBCCL_INVALID_ARGUMENT;    // buffers too small
        WorkSlot slot;
        *work = slot.emplace(comm->comm->all_reduce(in, out, elems, d, o, exec));
        return TBCCL_SUCCESS;
    });
}

} // extern "C"
