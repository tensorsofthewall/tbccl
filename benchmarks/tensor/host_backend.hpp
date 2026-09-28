#pragma once

#include "tensor_backend.hpp"

#include <memory>

namespace tbccl_bench::tensor
{

// Plain host-memory backend: source/destination are ordinary
// heap-allocated buffers, "device generation" is a CPU loop, and
// every staging stage is a no-op (the buffer World::send()/recv()
// touch already *is* the source/destination storage). This makes
// HostBackend the network-only baseline every other backend's
// staging overhead is measured against (Part E).
std::unique_ptr<TensorBackend> make_host_backend();

} // namespace tbccl_bench::tensor
