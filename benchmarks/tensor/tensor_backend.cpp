#include "tensor_backend.hpp"

#include "host_backend.hpp"

#ifdef TBCCL_ENABLE_CUDA
#include "cuda_backend.hpp"
#endif

#ifdef TBCCL_ENABLE_METAL
#include "metal_backend.hpp"
#endif

#include <stdexcept>

namespace tbccl_bench::tensor
{

std::string backend_kind_name(BackendKind kind)
{
    switch (kind)
    {
    case BackendKind::Host:
        return "host";
    case BackendKind::CudaPageable:
        return "cuda-pageable";
    case BackendKind::CudaPinned:
        return "cuda-pinned";
    case BackendKind::MetalShared:
        return "metal-shared";
    case BackendKind::MetalPrivateStaged:
        return "metal-private-staged";
    }

    throw std::runtime_error("backend_kind_name: unhandled BackendKind");
}

BackendKind parse_backend_kind(const std::string &name)
{
    if (name == "host")
    {
        return BackendKind::Host;
    }
    if (name == "cuda-pageable")
    {
        return BackendKind::CudaPageable;
    }
    if (name == "cuda-pinned")
    {
        return BackendKind::CudaPinned;
    }
    if (name == "metal-shared")
    {
        return BackendKind::MetalShared;
    }
    if (name == "metal-private-staged")
    {
        return BackendKind::MetalPrivateStaged;
    }

    throw std::runtime_error("unknown tensor backend: " + name);
}

bool backend_kind_available(BackendKind kind)
{
    switch (kind)
    {
    case BackendKind::Host:
        return true;
    case BackendKind::CudaPageable:
    case BackendKind::CudaPinned:
#ifdef TBCCL_ENABLE_CUDA
        return true;
#else
        return false;
#endif
    case BackendKind::MetalShared:
    case BackendKind::MetalPrivateStaged:
#ifdef TBCCL_ENABLE_METAL
        return true;
#else
        return false;
#endif
    }

    throw std::runtime_error("backend_kind_available: unhandled BackendKind");
}

std::unique_ptr<TensorBackend> make_backend(BackendKind kind)
{
    if (!backend_kind_available(kind))
    {
        throw std::runtime_error(
            "tensor backend '" + backend_kind_name(kind) +
            "' was not compiled into this build");
    }

    switch (kind)
    {
    case BackendKind::Host:
        return make_host_backend();

    case BackendKind::CudaPageable:
    case BackendKind::CudaPinned:
#ifdef TBCCL_ENABLE_CUDA
        return make_cuda_backend(kind);
#else
        break;
#endif

    case BackendKind::MetalShared:
    case BackendKind::MetalPrivateStaged:
#ifdef TBCCL_ENABLE_METAL
        return make_metal_backend(kind);
#else
        break;
#endif
    }

    // Unreachable: backend_kind_available() already filtered out any
    // kind not compiled into this build.
    throw std::runtime_error("make_backend: unhandled BackendKind");
}

} // namespace tbccl_bench::tensor
