#pragma once

// Phase 41 Part I/P: registers a CUDA tbccl::ExternalMemoryProvider
// factory for tbccl::MemoryKind::Cuda. Declared separately from its .cu
// implementation so non-CUDA translation units never see a CUDA type.
// Call this once before constructing any Communicator that will move
// MemoryKind::Cuda buffers -- an explicit, portable extension point
// (no weak symbols, no static-init-order dependency) rather than
// auto-registration, so linking tbccl_tensor_backend's CUDA component
// never silently changes Communicator's behavior without the caller's
// knowledge.

namespace tbccl_bench::tensor
{

void register_cuda_memory_provider();

} // namespace tbccl_bench::tensor
