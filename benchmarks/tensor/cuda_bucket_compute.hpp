#pragma once

// A deterministic, tunable CUDA compute workload used to simulate
// a DDP-like gradient-bucket production pattern. Declared
// separately from its .cu implementation so non-CUDA translation
// units never see a CUDA type (matching
// cuda_backend.hpp/cuda_chunked_async_backend.hpp's convention).

#include <cstddef>
#include <cstdint>

namespace tbccl_bench::tensor
{

// Launches a per-byte deterministic hash-mix transform on `device_ptr`
// (device-resident, `bytes` long), with `rounds` iterations of mixing
// per byte -- more rounds means more GPU time, used for compute-time
// calibration. Writes every byte of the buffer (no "touches 4 bytes,
// transfers 16 MiB" realism gap). Asynchronous: returns once the
// kernel is launched, not once it completes -- callers synchronize
// `stream` explicitly when they need completion (host-side stream
// synchronization, not CUDA events, is the first approach
// tried).
void launch_bucket_compute(
    void *device_ptr,
    std::size_t bytes,
    std::uint32_t seed,
    int rounds,
    void *stream);

// The exact same transform, computed on the CPU, for verification (integer
// arithmetic, no floating-point determinism concerns) and for the one-time
// untimed post-loop byte-exact check (docs/development/benchmark-methodology.md's
// convention). Defined inline, here, in plain C++ (NOT in cuda_bucket_compute.cu)
// so it is available on every platform regardless of TBCCL_ENABLE_CUDA --
// critical for the CUDA compute-overlap benchmark's receiver role, which commonly
// runs on a non- CUDA machine (e.g. the Mac mini's Metal-shared destination) but
// still needs to verify CUDA-computed bytes it received. The CUDA kernel in
// cuda_bucket_compute.cu uses an intentional __device__ duplicate of this exact
// formula (matching this codebase's established convention for small CUDA-side
// pattern duplicates).
inline std::uint8_t expected_bucket_byte(std::size_t i, std::uint32_t seed, int rounds)
{
    const std::uint64_t index = static_cast<std::uint64_t>(i);
    const std::uint64_t value =
        index * 131u + (index >> 8) * 17u + static_cast<std::uint64_t>(seed);
    const std::uint8_t input = static_cast<std::uint8_t>(value & 0xffu);

    std::uint32_t v = static_cast<std::uint32_t>(input) ^ seed ^
                       static_cast<std::uint32_t>(i & 0xffffffffu);
    for (int r = 0; r < rounds; ++r)
    {
        v *= 2654435761u;
        v += 1u;
        v ^= v >> 15;
        v *= 0x85ebca6bu;
        v ^= v >> 13;
    }
    return static_cast<std::uint8_t>(v & 0xffu);
}

// Fills `device_ptr` with pattern_byte(i, seed) (same formula as
// tensor_backend.hpp's shared pattern) -- the "input" the compute
// kernel transforms, analogous to initialize_source() elsewhere in
// this codebase.
void launch_bucket_fill_input(
    void *device_ptr,
    std::size_t bytes,
    std::uint32_t seed,
    void *stream);

} // namespace tbccl_bench::tensor
