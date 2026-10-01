#pragma once

// A staging-copy-free AsyncMemoryBackend over an existing
// TensorBackend of kind MetalShared. Unlike TensorBackendAsyncAdapter
// (which is generic over any TensorBackend and never advertises direct
// transport access), this wraps the fact that MetalShared's
// source_staging_data()/destination_staging_data() already return the
// backing MTLBuffer's own CPU-visible .contents pointer (MTLStorageModeShared
// semantics -- see benchmarks/tensor/metal_backend.mm) with nothing further
// to stage. This header touches only the portable TensorBackend interface
// -- no Metal/Objective-C types -- so it compiles on any platform;
// constructing one with a non-MetalShared backend throws.
//
// This is "staging-copy-free direct access to CPU-visible Metal-shared
// memory", NOT kernel zero-copy, GPUDirect, or RDMA.
//
// TensorBackendAsyncAdapter remains unchanged and is still the correct
// choice for MetalPrivateStaged (whose real tensor storage is not
// CPU-addressable) and remains the reference/fallback path for MetalShared.

#include <tbccl/async_transfer.hpp>

#include "tensor_backend.hpp"

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace tbccl_bench::tensor
{

class MetalSharedDirectAsyncBackend final : public tbccl::AsyncMemoryBackend
{
public:
    // `backend` must outlive this wrapper and remain valid (allocated,
    // with source already prepared for a send) for as long as any
    // TransferRequest referencing this wrapper is outstanding -- same
    // contract as TensorBackendAsyncAdapter.
    explicit MetalSharedDirectAsyncBackend(TensorBackend &backend)
        : backend_(backend)
    {
        if (backend_.kind() != BackendKind::MetalShared)
        {
            throw std::runtime_error(
                "MetalSharedDirectAsyncBackend requires BackendKind::MetalShared");
        }
    }

    // Only exercised if a caller explicitly requests chunk_hint != 0
    // (chunk_hint is otherwise ignored on the direct path -- see
    // async_transfer.hpp). Kept correct, not just present, so this
    // backend still behaves sanely if ever used with explicit chunking.
    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
    {
        const auto *src = static_cast<const std::byte *>(backend_.source_staging_data());
        std::memcpy(staging, src + chunk.offset, chunk.size);
    }

    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override
    {
        auto *dst = static_cast<std::byte *>(backend_.destination_staging_data());
        std::memcpy(dst + chunk.offset, staging, chunk.size);
    }

    bool supports_direct_transport_access() const noexcept override { return true; }

    const void *direct_source_data() const noexcept override
    {
        return backend_.source_staging_data();
    }

    void *direct_destination_data() noexcept override
    {
        return backend_.destination_staging_data();
    }

    // No-op: unlike TensorBackendAsyncAdapter, this backend has no
    // per-chunk commit bookkeeping to reset (the direct path never calls
    // stage_source_chunk()/commit_destination_chunk()). Present only so
    // call sites that unconditionally call begin_transfer() on whichever
    // concrete Metal backend is active (adapter or direct) don't need to
    // branch on that choice.
    void begin_transfer(std::size_t /*expected_chunk_count*/) noexcept {}

private:
    TensorBackend &backend_;
};

} // namespace tbccl_bench::tensor
