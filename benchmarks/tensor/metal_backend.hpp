#pragma once

// Declared separately from metal_backend.mm's implementation so that
// tensor_backend.cpp (plain C++, never Objective-C++) can see this
// factory without including any Metal/Foundation type. Only included
// when TBCCL_ENABLE_METAL is defined (Apple platforms only).

#include "tensor_backend.hpp"

#include <memory>

namespace tbccl_bench::tensor
{

// `kind` must be BackendKind::MetalShared or
// BackendKind::MetalPrivateStaged. Throws std::runtime_error if no
// Metal device is available at runtime (e.g. MTLCreateSystemDefaultDevice
// returns nil), or if shader compilation/pipeline creation fails.
std::unique_ptr<TensorBackend> make_metal_backend(BackendKind kind);

} // namespace tbccl_bench::tensor
