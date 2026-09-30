#pragma once

// Phase 32 Commit 4: adapts the existing Phase 17 TensorBackend
// (host/cuda-pageable/cuda-pinned/metal-shared/metal-private-staged,
// tensor_backend.hpp) to the AsyncMemoryBackend interface
// (tbccl/async_transfer.hpp), so CUDA/Metal tensors can move through
// TensorCommWorker without the core library ever seeing a CUDA/Metal
// type -- exactly the boundary tensor_backend.hpp's own docstring
// already established.
//
// Scope note (see docs/phase32_report.md item 41-45 for the full
// rationale): this is a worker-thread-driven wrapper over
// TensorBackend's existing *synchronous* stage_device_to_host()/
// stage_host_to_device() calls, not the finer-grained per-chunk
// cudaEvent/copy-stream overlap the plan's Part S describes. The
// caller-side benefit Part K asks for -- the enqueuing thread never
// blocks on GPU work -- is still genuinely delivered, because those
// blocking calls now happen on TensorCommWorker's own staging thread,
// never on the thread that called enqueue(). What this adapter does
// NOT deliver is overlap *within* a single transfer's own device-side
// staging step with that same transfer's network traffic (the D2H copy
// for chunk 0 completes before chunk 0's network send can begin,
// because TensorBackend's own API stages the whole buffer in one call,
// not chunk-by-chunk) -- a real limitation, stated plainly rather than
// implied away.

#include <tbccl/async_transfer.hpp>

#include "tensor_backend.hpp"

#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace tbccl_bench::tensor
{

class TensorBackendAsyncAdapter final : public tbccl::AsyncMemoryBackend
{
public:
    // `backend` must outlive this adapter. For the Send direction,
    // the caller must already have called initialize_source()/
    // prepare_source() on `backend` before any TransferRequest using
    // this adapter is enqueued (this adapter only performs the
    // device-to-host staging step, not source generation -- matching
    // the existing benchmark's own stage separation, Part 27/36).
    explicit TensorBackendAsyncAdapter(TensorBackend &backend)
        : backend_(backend)
    {
    }

    // Resets the "already staged this round" / "chunks committed this
    // round" bookkeeping for a new transfer. Must be called once
    // before each TransferRequest that reuses this adapter (the
    // backend's own buffers are reused across iterations, Part 30 --
    // only this adapter's per-transfer bookkeeping needs resetting).
    void begin_transfer(std::size_t expected_chunk_count)
    {
        source_staged_ = false;
        expected_chunk_count_ = expected_chunk_count;
        committed_chunk_count_ = 0;
    }

    void stage_source_chunk(const tbccl::Chunk &chunk, void *staging) override
    {
        {
            std::lock_guard<std::mutex> lock(stage_mutex_);
            if (!source_staged_)
            {
                // Whole-buffer device-to-host staging, exactly once per
                // transfer -- TensorBackend's own API has no per-chunk
                // granularity (see this file's top-of-file scope note).
                backend_.stage_device_to_host();
                source_staged_ = true;
            }
        }

        const auto *source = static_cast<const std::uint8_t *>(backend_.source_staging_data());
        std::memcpy(staging, source + chunk.offset, chunk.size);
    }

    void commit_destination_chunk(const tbccl::Chunk &chunk, const void *staging) override
    {
        auto *destination = static_cast<std::uint8_t *>(backend_.destination_staging_data());
        std::memcpy(destination + chunk.offset, staging, chunk.size);

        const std::size_t committed = ++committed_chunk_count_;
        if (committed == expected_chunk_count_)
        {
            // Last chunk of this transfer: push the now-fully-assembled
            // host staging buffer to the device and complete any
            // device-side synchronization (Part 8's destination-ready
            // guarantee) -- exactly once per transfer, on whichever
            // thread happens to commit the final chunk (StagingPool's
            // depth bounds how many chunks can be "in flight" at once,
            // but per Part K/L this worker still commits chunks in
            // strict order on one thread, so no race is actually
            // possible here; the atomic is defensive, not load-bearing).
            backend_.stage_host_to_device();
        }
        else if (committed > expected_chunk_count_)
        {
            throw std::logic_error(
                "TensorBackendAsyncAdapter::commit_destination_chunk called "
                "more times than begin_transfer's expected_chunk_count");
        }
    }

private:
    TensorBackend &backend_;
    std::mutex stage_mutex_;
    bool source_staged_ = false;
    std::size_t expected_chunk_count_ = 0;
    std::atomic<std::size_t> committed_chunk_count_{0};
};

} // namespace tbccl_bench::tensor
