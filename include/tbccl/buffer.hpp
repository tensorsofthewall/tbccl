#pragma once

// Phase 41 Part E/F: the non-owning external buffer descriptor. A
// BufferView never owns the memory it points to (Part 23/24) -- the
// caller guarantees it remains valid until the Work referencing it
// completes, exactly the same contract TransferRequest's backend/
// transport pointers already carry (async_transfer.hpp).

#include <tbccl/reduction.hpp>
#include <tbccl/types.hpp>

#include <cstddef>

namespace tbccl
{

struct BufferView
{
    MemoryKind memory_kind = MemoryKind::Host;

    // Host / MetalShared: an ordinary CPU-visible pointer.
    // Cuda: a device pointer (NOT host-dereferenceable); see
    // device_ordinal below.
    void *data = nullptr;

    // Explicit, not inferred from datatype/count (Part 28) -- a
    // collective call separately supplies count+datatype and this is
    // validated against `bytes` with overflow-safe arithmetic.
    std::size_t bytes = 0;

    // Meaningful only when memory_kind == Cuda; the CUDA device this
    // pointer was allocated on. Ignored for Host/MetalShared.
    int device_ordinal = 0;
};

// count * datatype_size(datatype) <= view.bytes, computed without
// overflow (Part 30). Throws std::invalid_argument (via
// std::runtime_error family, repository style) with a descriptive
// message on failure; a zero-byte/zero-count view is valid (Part 31)
// and always passes.
void validate_buffer_view(
    const BufferView &view,
    std::size_t count,
    DataType datatype);

} // namespace tbccl
