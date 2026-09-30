#include "tensor_backend_capabilities.hpp"

namespace tbccl_bench::tensor
{

tbccl::PeerCapabilities local_tensor_capabilities()
{
    tbccl::PeerCapabilities caps = tbccl::local_capabilities();

#if defined(TBCCL_ENABLE_CUDA)
    caps.memory_backends.push_back(tbccl::MemoryBackendKind::CudaPageable);
    caps.memory_backends.push_back(tbccl::MemoryBackendKind::CudaPinned);
    caps.async_capabilities.push_back(tbccl::AsyncCapability::CudaEvents);
#endif

#if defined(TBCCL_ENABLE_METAL)
    caps.memory_backends.push_back(tbccl::MemoryBackendKind::MetalShared);
    caps.memory_backends.push_back(tbccl::MemoryBackendKind::MetalPrivateStaged);
    caps.async_capabilities.push_back(tbccl::AsyncCapability::MetalEvents);
#endif

    return caps;
}

} // namespace tbccl_bench::tensor
