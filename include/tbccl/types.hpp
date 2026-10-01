#pragma once

// Phase 41: framework-independent public runtime types. This header,
// and every other header directly under include/tbccl/, has no CUDA or
// Objective-C/Metal dependency -- see docs/framework_integration_architecture.md.
// DataType/ReduceOp already exist (tbccl/reduction.hpp); this header adds
// the remaining small enums the public Communicator/BufferView API needs.

#include <string>

namespace tbccl
{

// What kind of memory a BufferView's data pointer refers to. MetalShared
// is deliberately NOT a separate code path from Host at the transport
// layer (docs/framework_integration_architecture.md Section 5): once a
// caller has taken an MTLBuffer's .contents pointer, it is ordinary
// CPU-visible memory as far as TBCCL's transport is concerned. The
// label exists for documentation/provenance and to leave room for a
// future, genuinely different MetalPrivate kind (not supported).
enum class MemoryKind
{
    Host,
    Cuda,
    MetalShared,
};

std::string memory_kind_name(MemoryKind kind);

// What kind of execution context a caller's buffer was produced under
// (Part K). Host means "already synchronized, safe to read now" --
// every prior phase's benchmark-owned-tensor convention. CudaStream
// carries an opaque native stream handle, cast to cudaStream_t only
// inside the CUDA-specific provider implementation -- this header never
// includes a CUDA header.
enum class ExecutionContextKind
{
    Host,
    CudaStream,
};

struct ExecutionContext
{
    ExecutionContextKind kind = ExecutionContextKind::Host;
    // Opaque at this layer. For CudaStream, this is a cudaStream_t
    // reinterpret_cast to void*; the caller's stream must remain valid
    // until the returned Work completes.
    void *native_handle = nullptr;
};

// Structured errors so framework adapters can branch without parsing
// exception strings (Part R). The C++ API still throws
// std::runtime_error with a human-readable message (repository style,
// matching every existing TBCCL entry point) -- ErrorCode is carried
// alongside via CommunicatorError (communicator.hpp) and is the only
// thing that crosses the C ABI.
enum class ErrorCode
{
    Success,
    InvalidArgument,
    Unsupported,
    TransportError,
    PeerFailure,
    Timeout,
    DeviceError,
    InternalError,
};

std::string error_code_name(ErrorCode code);

} // namespace tbccl
