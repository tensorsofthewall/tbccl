#pragma once

// Declared separately from cuda_backend.cu's implementation so that
// tensor_backend.cpp (a plain .cpp, compiled by the host C++ compiler,
// never nvcc) can see this factory without including any CUDA-specific
// type. Only included when TBCCL_ENABLE_CUDA is defined.

#include "tensor_backend.hpp"

#include <memory>

namespace tbccl_bench::tensor
{

// `kind` must be BackendKind::CudaPageable or BackendKind::CudaPinned.
// Throws std::runtime_error if no CUDA device is available at
// runtime, or if any CUDA API call during construction fails.
std::unique_ptr<TensorBackend> make_cuda_backend(BackendKind kind);

} // namespace tbccl_bench::tensor
