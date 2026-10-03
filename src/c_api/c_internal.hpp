#pragma once

// Private to the C shim (NOT installed). The shim is a thin translation layer: POD in, tbccl:: C++ calls, POD out. It owns the opaque handle wrappers, converts
// every exception to a tbcclResult_t through ONE mapping, and validates ABI-level arguments (pointers, struct_size, enum values, ranges, overflow). It never
// parses message text and never reimplements communication.

#include <tbccl/communicator.hpp>
#include <tbccl/error.hpp>
#include <tbccl/tbccl.h>

#include <cstring>
#include <memory>
#include <new>
#include <string>

struct tbcclComm_st
{
    std::unique_ptr<tbccl::Communicator> comm;
};

struct tbcclWork_st
{
    tbccl::Work work;
};

namespace tbccl::capi
{

// The one ErrorCode -> tbcclResult_t mapping.
inline tbcclResult_t to_c(ErrorCode code) noexcept
{
    switch (code)
    {
    case ErrorCode::Success: return TBCCL_SUCCESS;
    case ErrorCode::InvalidArgument: return TBCCL_INVALID_ARGUMENT;
    case ErrorCode::Unsupported: return TBCCL_UNSUPPORTED;
    case ErrorCode::ResourceExhausted: return TBCCL_RESOURCE_EXHAUSTED;
    case ErrorCode::Aborted: return TBCCL_ABORTED;
    case ErrorCode::Timeout: return TBCCL_TIMEOUT;
    case ErrorCode::ProtocolMismatch: return TBCCL_PROTOCOL_MISMATCH;
    case ErrorCode::TransportError: return TBCCL_TRANSPORT_ERROR;
    case ErrorCode::PeerFailure: return TBCCL_TRANSPORT_ERROR;
    case ErrorCode::DeviceError: return TBCCL_DEVICE_ERROR;
    case ErrorCode::InternalError: return TBCCL_INTERNAL_ERROR;
    }
    return TBCCL_INTERNAL_ERROR;
}

// Runs `f` (returning a tbcclResult_t) and converts anything it throws. No exception ever leaves an exported function.
template <typename F> tbcclResult_t guard(F &&f) noexcept
{
    try
    {
        return f();
    }
    catch (const std::exception &e)
    {
        return to_c(error_code_of(e));
    }
    catch (...)
    {
        return TBCCL_INTERNAL_ERROR;
    }
}

// struct_size validation: the caller's struct must contain at least the v1 prefix.
template <typename T> bool size_ok(const T *p, std::size_t v1_size) noexcept { return p != nullptr && p->struct_size >= v1_size; }

// Copies `text` into a caller buffer: at most capacity-1 characters plus a NUL; *required is the full length including the NUL.
inline tbcclResult_t copy_text(const std::string &text, char *buf, std::size_t capacity, std::size_t *required) noexcept
{
    if (required == nullptr) return TBCCL_INVALID_ARGUMENT;
    if (capacity > 0 && buf == nullptr) return TBCCL_INVALID_ARGUMENT;
    *required = text.size() + 1;
    if (capacity > 0)
    {
        const std::size_t n = text.size() < capacity - 1 ? text.size() : capacity - 1;
        std::memcpy(buf, text.data(), n);
        buf[n] = '\0';
    }
    return TBCCL_SUCCESS;
}

// Integer -> C++ enum conversions; false for a value outside the frozen v1 set.
inline bool to_cpp(tbcclDataType_t v, DataType &out) noexcept
{
    switch (v)
    {
    case TBCCL_INT32: out = DataType::Int32; return true;
    case TBCCL_INT64: out = DataType::Int64; return true;
    case TBCCL_FLOAT32: out = DataType::Float32; return true;
    case TBCCL_FLOAT64: out = DataType::Float64; return true;
    case TBCCL_INT8: out = DataType::Int8; return true;
    case TBCCL_UINT8: out = DataType::UInt8; return true;
    case TBCCL_FLOAT16: out = DataType::Float16; return true;
    case TBCCL_BFLOAT16: out = DataType::BFloat16; return true;
    }
    return false;
}

inline bool to_cpp(tbcclReduceOp_t v, ReduceOp &out) noexcept
{
    switch (v)
    {
    case TBCCL_SUM: out = ReduceOp::Sum; return true;
    case TBCCL_PRODUCT: out = ReduceOp::Product; return true;
    case TBCCL_MIN: out = ReduceOp::Min; return true;
    case TBCCL_MAX: out = ReduceOp::Max; return true;
    }
    return false;
}

inline bool memory_kind_to_cpp(tbcclMemoryKind_t v, MemoryKind &out) noexcept
{
    switch (v)
    {
    case TBCCL_MEMORY_HOST: out = MemoryKind::Host; return true;
    case TBCCL_MEMORY_CUDA: out = MemoryKind::Cuda; return true;
    case TBCCL_MEMORY_METAL_SHARED: out = MemoryKind::MetalShared; return true;
    }
    return false;
}

// Validates a tbcclBuffer and converts it. data may be NULL only for zero bytes.
inline tbcclResult_t to_cpp(const tbcclBuffer *b, BufferView &out) noexcept
{
    if (!size_ok(b, sizeof(tbcclBuffer))) return TBCCL_INVALID_ARGUMENT;
    if (b->reserved0 != 0 || b->reserved1[0] != 0 || b->reserved1[1] != 0) return TBCCL_INVALID_ARGUMENT;
    MemoryKind kind;
    if (!memory_kind_to_cpp(b->memory_kind, kind)) return TBCCL_INVALID_ARGUMENT;
    if (b->data == nullptr && b->bytes != 0) return TBCCL_INVALID_ARGUMENT;
    if (b->bytes > static_cast<std::uint64_t>(static_cast<std::size_t>(-1))) return TBCCL_INVALID_ARGUMENT;
    out.memory_kind = kind;
    out.data = b->data;
    out.bytes = static_cast<std::size_t>(b->bytes);
    out.device_ordinal = b->device_ordinal;
    return TBCCL_SUCCESS;
}

// NULL -> the default execution context.
inline tbcclResult_t to_cpp(const tbcclExecContext *c, ExecutionContext &out) noexcept
{
    out = ExecutionContext{};
    if (c == nullptr) return TBCCL_SUCCESS;
    if (!size_ok(c, sizeof(tbcclExecContext))) return TBCCL_INVALID_ARGUMENT;
    if (c->reserved[0] != 0) return TBCCL_INVALID_ARGUMENT;
    switch (c->kind)
    {
    case TBCCL_EXEC_DEFAULT:
        if (c->native_handle != nullptr) return TBCCL_INVALID_ARGUMENT;
        return TBCCL_SUCCESS;
    case TBCCL_EXEC_CUDA_STREAM:
        out.kind = ExecutionContextKind::CudaStream;
        out.native_handle = c->native_handle;
        return TBCCL_SUCCESS;
    }
    return TBCCL_INVALID_ARGUMENT;
}

// Wraps the C++ Work into a heap handle (an allocation failure becomes RESOURCE_EXHAUSTED through guard()).
inline tbcclWork_t make_work(Work work) { return new tbcclWork_st{std::move(work)}; }

} // namespace tbccl::capi
