#include <tbccl/types.hpp>

#include <stdexcept>

namespace tbccl
{

std::string memory_kind_name(MemoryKind kind)
{
    switch (kind)
    {
    case MemoryKind::Host: return "host";
    case MemoryKind::Cuda: return "cuda";
    case MemoryKind::MetalShared: return "metal-shared";
    }
    throw std::runtime_error("memory_kind_name: unrecognized MemoryKind");
}

std::string error_code_name(ErrorCode code)
{
    switch (code)
    {
    case ErrorCode::Success: return "success";
    case ErrorCode::InvalidArgument: return "invalid_argument";
    case ErrorCode::Unsupported: return "unsupported";
    case ErrorCode::TransportError: return "transport_error";
    case ErrorCode::PeerFailure: return "peer_failure";
    case ErrorCode::Timeout: return "timeout";
    case ErrorCode::DeviceError: return "device_error";
    case ErrorCode::InternalError: return "internal_error";
    case ErrorCode::Aborted: return "aborted";
    case ErrorCode::ProtocolMismatch: return "protocol_mismatch";
    case ErrorCode::ResourceExhausted: return "resource_exhausted";
    }
    throw std::runtime_error("error_code_name: unrecognized ErrorCode");
}

} // namespace tbccl
