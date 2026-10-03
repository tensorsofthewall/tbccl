// TBCCL C ABI v1: version, result strings, communicator lifetime and queries, CUDA registration. See docs/c_abi_v1.md.

#include "c_internal.hpp"

#include <tbccl/cuda_support.hpp>

#ifndef TBCCL_PACKAGE_VERSION_MAJOR
#define TBCCL_PACKAGE_VERSION_MAJOR 0
#define TBCCL_PACKAGE_VERSION_MINOR 0
#define TBCCL_PACKAGE_VERSION_PATCH 0
#endif

using namespace tbccl;
using namespace tbccl::capi;

extern "C"
{

tbcclResult_t TBCCL_CALL tbcclGetAbiVersion(uint32_t *abi_version)
{
    return guard([&]() -> tbcclResult_t {
        if (abi_version == nullptr) return TBCCL_INVALID_ARGUMENT;
        *abi_version = TBCCL_C_ABI_VERSION;
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclGetPackageVersion(uint32_t *major, uint32_t *minor, uint32_t *patch)
{
    return guard([&]() -> tbcclResult_t {
        if (major == nullptr || minor == nullptr || patch == nullptr) return TBCCL_INVALID_ARGUMENT;
        *major = TBCCL_PACKAGE_VERSION_MAJOR;
        *minor = TBCCL_PACKAGE_VERSION_MINOR;
        *patch = TBCCL_PACKAGE_VERSION_PATCH;
        return TBCCL_SUCCESS;
    });
}

const char *TBCCL_CALL tbcclGetResultString(tbcclResult_t result)
{
    switch (result)
    {
    case TBCCL_SUCCESS: return "success";
    case TBCCL_INVALID_ARGUMENT: return "invalid argument";
    case TBCCL_UNSUPPORTED: return "unsupported";
    case TBCCL_RESOURCE_EXHAUSTED: return "resource exhausted";
    case TBCCL_ABORTED: return "aborted";
    case TBCCL_TIMEOUT: return "timeout";
    case TBCCL_PROTOCOL_MISMATCH: return "protocol mismatch";
    case TBCCL_TRANSPORT_ERROR: return "transport error";
    case TBCCL_INTERNAL_ERROR: return "internal error";
    case TBCCL_DEVICE_ERROR: return "device error";
    }
    return "unknown result value";
}

tbcclResult_t TBCCL_CALL tbcclCommDestroy(tbcclComm_t comm)
{
    return guard([&]() -> tbcclResult_t {
        delete comm; // NULL is allowed; the Communicator's destructor aborts outstanding work (bounded)
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclCommGetRank(tbcclComm_t comm, uint32_t *rank)
{
    return guard([&]() -> tbcclResult_t {
        if (comm == nullptr || rank == nullptr) return TBCCL_INVALID_ARGUMENT;
        *rank = static_cast<uint32_t>(comm->comm->rank());
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclCommGetSize(tbcclComm_t comm, uint32_t *world_size)
{
    return guard([&]() -> tbcclResult_t {
        if (comm == nullptr || world_size == nullptr) return TBCCL_INVALID_ARGUMENT;
        *world_size = static_cast<uint32_t>(comm->comm->world_size());
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclCommAbort(tbcclComm_t comm, const char *reason)
{
    return guard([&]() -> tbcclResult_t {
        if (comm == nullptr) return TBCCL_INVALID_ARGUMENT;
        comm->comm->abort(reason != nullptr ? reason : "");
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclCommIsAborted(tbcclComm_t comm, int32_t *aborted)
{
    return guard([&]() -> tbcclResult_t {
        if (comm == nullptr || aborted == nullptr) return TBCCL_INVALID_ARGUMENT;
        *aborted = comm->comm->aborted() ? 1 : 0;
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclCommGetAbortReason(tbcclComm_t comm, char *buf, size_t capacity, size_t *required)
{
    return guard([&]() -> tbcclResult_t {
        if (comm == nullptr) return TBCCL_INVALID_ARGUMENT;
        return copy_text(comm->comm->abort_reason(), buf, capacity, required);
    });
}

tbcclResult_t TBCCL_CALL tbcclCommGetCapabilities(tbcclComm_t comm, tbcclCapabilities *capabilities)
{
    return guard([&]() -> tbcclResult_t {
        if (comm == nullptr || !size_ok(capabilities, sizeof(tbcclCapabilities))) return TBCCL_INVALID_ARGUMENT;
        const Capabilities &caps = comm->comm->capabilities();
        std::uint32_t mask = 0;
        for (tbcclMemoryKind_t k : {TBCCL_MEMORY_HOST, TBCCL_MEMORY_CUDA, TBCCL_MEMORY_METAL_SHARED})
        {
            MemoryKind kind;
            if (memory_kind_to_cpp(k, kind) && caps.supports_memory_kind(kind)) mask |= 1u << k;
        }
        const std::uint32_t size = capabilities->struct_size;
        std::memset(capabilities, 0, sizeof(tbcclCapabilities)); // v1 fields only; reserved fields are written as zero
        capabilities->struct_size = size;
        capabilities->memory_kind_mask = mask;
        capabilities->effective_max_chunk = caps.effective_max_chunk();
        capabilities->effective_alignment = caps.effective_alignment();
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclCommSupportsAllReduce(
    tbcclComm_t comm, uint32_t rank, tbcclMemoryKind_t kind, tbcclDataType_t dtype, tbcclReduceOp_t op, int32_t *supported)
{
    return guard([&]() -> tbcclResult_t {
        if (comm == nullptr || supported == nullptr) return TBCCL_INVALID_ARGUMENT;
        *supported = 0;
        MemoryKind k;
        DataType d;
        ReduceOp o;
        if (!memory_kind_to_cpp(kind, k) || !to_cpp(dtype, d) || !to_cpp(op, o)) return TBCCL_INVALID_ARGUMENT;
        const Capabilities &caps = comm->comm->capabilities();
        if (rank >= caps.world_size()) return TBCCL_INVALID_ARGUMENT;
        bool ok = caps.supports_collective_all_reduce(k, d, o);
        if (comm->comm->world_size() > 2 && (d == DataType::Float16 || d == DataType::BFloat16)) ok = false; // N>2 FP16/BF16 reduction semantics are not defined
        if (ok)
        {
            // the named rank must itself offer the memory kind
            const auto &backends = caps.for_rank(rank).memory_backends;
            const auto has = [&](MemoryBackendKind b) { for (auto x : backends) if (x == b) return true; return false; };
            switch (k)
            {
            case MemoryKind::Host: break;
            case MemoryKind::Cuda: ok = has(MemoryBackendKind::CudaPinned) || has(MemoryBackendKind::CudaPageable); break;
            case MemoryKind::MetalShared: ok = has(MemoryBackendKind::MetalShared); break;
            }
        }
        *supported = ok ? 1 : 0;
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclRegisterCudaSupport(void)
{
    return guard([]() -> tbcclResult_t {
#ifdef TBCCL_C_WITH_CUDA
        tbccl::register_cuda_support();
        return TBCCL_SUCCESS;
#else
        return TBCCL_UNSUPPORTED;
#endif
    });
}

} // extern "C"
