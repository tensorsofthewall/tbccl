#include "metal_backend.hpp"

#include <tbccl/world.hpp>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Compiled with -fobjc-arc (see CMakeLists.txt): id-typed members of
// the C++ classes below are ARC-managed like ordinary C++ members --
// retained on assignment, released in the (implicit) destructor. No
// manual retain/release is needed or used in this file.

namespace tbccl_bench::tensor
{

namespace
{

    // -----------------------------------------------------------------------------
    // Device-side deterministic pattern generation, as a runtime-
    // compiled MSL compute shader. Must produce exactly the same
    // bytes as pattern_byte() in tensor_backend.hpp (Part D) -- kept
    // as an intentional, independently-readable duplicate rather than
    // shared code, since that header must stay includable by a plain
    // (non-Objective-C++) C++ compiler. Mirrors cuda_backend.cu's
    // fill_pattern_kernel exactly, translated to MSL.
    // -----------------------------------------------------------------------------

    NSString *const kFillPatternSource = @"\
#include <metal_stdlib>\n\
using namespace metal;\n\
kernel void fill_pattern(device uchar *data [[buffer(0)]],\n\
                          constant uint &seed [[buffer(1)]],\n\
                          constant uint &count [[buffer(2)]],\n\
                          uint id [[thread_position_in_grid]])\n\
{\n\
    if (id >= count) { return; }\n\
    ulong index = (ulong)id;\n\
    ulong value = index * 131u + (index >> 8) * 17u + (ulong)seed;\n\
    data[id] = (uchar)(value & 0xffu);\n\
}\n";

    std::string ns_error_message(NSError *error)
    {
        if (error == nil)
        {
            return "unknown error";
        }

        NSString *description = error.localizedDescription;

        if (description == nil)
        {
            return "unknown error";
        }

        return std::string(description.UTF8String);
    }

    // -----------------------------------------------------------------------------
    // Shared GPU state (device, queue, compiled pipeline), built once
    // per backend instance and reused across every allocate()/
    // initialize_source() call -- pipeline creation happens before
    // any steady-state timing (Part 27's "shader compilation and
    // pipeline creation occur before warmup").
    // -----------------------------------------------------------------------------

    class MetalContext
    {
    public:
        MetalContext()
        {
            device_ = MTLCreateSystemDefaultDevice();

            if (device_ == nil)
            {
                throw std::runtime_error(
                    "MetalContext: MTLCreateSystemDefaultDevice() returned nil "
                    "(no Metal device available at runtime)");
            }

            queue_ = [device_ newCommandQueue];

            if (queue_ == nil)
            {
                throw std::runtime_error(
                    "MetalContext: newCommandQueue failed");
            }

            NSError *error = nil;
            id<MTLLibrary> library =
                [device_ newLibraryWithSource:kFillPatternSource
                                       options:nil
                                         error:&error];

            if (library == nil)
            {
                throw std::runtime_error(
                    "MetalContext: shader compilation failed: " +
                    ns_error_message(error));
            }

            id<MTLFunction> function =
                [library newFunctionWithName:@"fill_pattern"];

            if (function == nil)
            {
                throw std::runtime_error(
                    "MetalContext: fill_pattern function not found in "
                    "compiled library");
            }

            error = nil;
            pipeline_ =
                [device_ newComputePipelineStateWithFunction:function
                                                         error:&error];

            if (pipeline_ == nil)
            {
                throw std::runtime_error(
                    "MetalContext: pipeline creation failed: " +
                    ns_error_message(error));
            }
        }

        id<MTLDevice> device() const noexcept
        {
            return device_;
        }

        id<MTLCommandQueue> queue() const noexcept
        {
            return queue_;
        }

        id<MTLComputePipelineState> pipeline() const noexcept
        {
            return pipeline_;
        }

    private:
        id<MTLDevice> device_ = nil;
        id<MTLCommandQueue> queue_ = nil;
        id<MTLComputePipelineState> pipeline_ = nil;
    };

    id<MTLBuffer> make_buffer(
        const MetalContext &context,
        std::size_t bytes,
        MTLResourceOptions options)
    {
        if (bytes == 0)
        {
            return nil;
        }

        id<MTLBuffer> buffer =
            [context.device() newBufferWithLength:static_cast<NSUInteger>(bytes)
                                           options:options];

        if (buffer == nil)
        {
            throw std::runtime_error(
                "Metal: newBufferWithLength failed for " +
                std::to_string(bytes) + " bytes");
        }

        return buffer;
    }

    // Dispatches fill_pattern into `target` (Shared or Private, both
    // writable by a compute encoder), 1 thread per byte, and commits
    // the command buffer WITHOUT waiting -- completion is the
    // caller's responsibility (see prepare_source()'s "producer
    // completion is measured separately" contract, Part 27).
    id<MTLCommandBuffer> dispatch_fill_pattern(
        const MetalContext &context,
        id<MTLBuffer> target,
        std::size_t count,
        std::uint32_t seed)
    {
        id<MTLCommandBuffer> command_buffer = [context.queue() commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];

        [encoder setComputePipelineState:context.pipeline()];
        [encoder setBuffer:target offset:0 atIndex:0];

        const std::uint32_t seed_value = seed;
        [encoder setBytes:&seed_value length:sizeof(seed_value) atIndex:1];

        const std::uint32_t count_value = static_cast<std::uint32_t>(count);
        [encoder setBytes:&count_value length:sizeof(count_value) atIndex:2];

        NSUInteger thread_group_size =
            context.pipeline().maxTotalThreadsPerThreadgroup;

        if (thread_group_size > 256)
        {
            thread_group_size = 256;
        }

        const MTLSize grid_size = MTLSizeMake(static_cast<NSUInteger>(count), 1, 1);
        const MTLSize threadgroup_size = MTLSizeMake(thread_group_size, 1, 1);

        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];
        [command_buffer commit];

        return command_buffer;
    }

    void blit_copy(
        const MetalContext &context,
        id<MTLBuffer> source,
        id<MTLBuffer> destination,
        std::size_t bytes)
    {
        if (bytes == 0)
        {
            return;
        }

        id<MTLCommandBuffer> command_buffer = [context.queue() commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command_buffer blitCommandEncoder];

        [blit copyFromBuffer:source
                 sourceOffset:0
                     toBuffer:destination
            destinationOffset:0
                         size:static_cast<NSUInteger>(bytes)];

        [blit endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.error != nil)
        {
            throw std::runtime_error(
                "Metal: blit copy failed: " +
                ns_error_message(command_buffer.error));
        }
    }

    void wait_and_check(id<MTLCommandBuffer> command_buffer, const char *what)
    {
        if (command_buffer == nil)
        {
            return;
        }

        [command_buffer waitUntilCompleted];

        if (command_buffer.error != nil)
        {
            throw std::runtime_error(
                std::string("Metal: ") + what + " failed: " +
                ns_error_message(command_buffer.error));
        }
    }

    bool verify_buffer_contents(const void *data, std::size_t count, std::uint32_t seed)
    {
        const std::uint8_t *bytes = static_cast<const std::uint8_t *>(data);

        for (std::size_t i = 0; i < count; ++i)
        {
            const std::uint64_t index = static_cast<std::uint64_t>(i);
            const std::uint64_t value =
                index * 131u + (index >> 8) * 17u + static_cast<std::uint64_t>(seed);
            const std::uint8_t expected = static_cast<std::uint8_t>(value & 0xffu);

            if (bytes[i] != expected)
            {
                return false;
            }
        }

        return true;
    }

    // -----------------------------------------------------------------------------
    // MetalBackend: shared implementation for metal-shared and
    // metal-private-staged.
    //
    // metal-shared: source_buffer_/destination_buffer_ are
    // MTLResourceStorageModeShared -- the compute shader writes
    // directly into them, and .contents is already CPU-visible after
    // GPU completion, so stage_device_to_host()/stage_host_to_device()
    // are no-ops (Part 25: "shared storage does not eliminate the
    // need for synchronization", which prepare_source()'s
    // waitUntilCompleted provides).
    //
    // metal-private-staged: source_buffer_/destination_buffer_ are
    // MTLResourceStorageModePrivate (the "real" GPU tensor, never
    // CPU-visible); source_staging_/destination_staging_ are a
    // separate pair of Shared buffers, populated/drained via blit
    // encoder copies with explicit command-buffer-completion waits
    // (Part 26).
    // -----------------------------------------------------------------------------

    class MetalBackend final : public TensorBackend
    {
    public:
        explicit MetalBackend(BackendKind kind) : kind_(kind)
        {
            if (kind_ != BackendKind::MetalShared &&
                kind_ != BackendKind::MetalPrivateStaged)
            {
                throw std::runtime_error(
                    "MetalBackend: kind must be MetalShared or "
                    "MetalPrivateStaged");
            }

            context_ = std::make_unique<MetalContext>();
        }

        BackendKind kind() const noexcept override
        {
            return kind_;
        }

        void allocate(std::size_t bytes) override
        {
            if (bytes == capacity_ && allocated_)
            {
                ++stats_.reuse_count;
                return;
            }

            if (private_staged())
            {
                source_device_ =
                    make_buffer(*context_, bytes, MTLResourceStorageModePrivate);
                destination_device_ =
                    make_buffer(*context_, bytes, MTLResourceStorageModePrivate);
                source_staging_ =
                    make_buffer(*context_, bytes, MTLResourceStorageModeShared);
                destination_staging_ =
                    make_buffer(*context_, bytes, MTLResourceStorageModeShared);
                verify_scratch_ =
                    make_buffer(*context_, bytes, MTLResourceStorageModeShared);
            }
            else
            {
                source_device_ =
                    make_buffer(*context_, bytes, MTLResourceStorageModeShared);
                destination_device_ =
                    make_buffer(*context_, bytes, MTLResourceStorageModeShared);
            }

            capacity_ = bytes;
            allocated_ = true;
            ++stats_.allocation_count;
            stats_.capacity_bytes = bytes;
        }

        std::size_t capacity() const noexcept override
        {
            return capacity_;
        }

        AllocationStats stats() const noexcept override
        {
            return stats_;
        }

        void initialize_source(std::uint32_t seed) override
        {
            require_allocated("initialize_source");

            if (capacity_ == 0)
            {
                return;
            }

            pending_producer_ =
                dispatch_fill_pattern(*context_, source_device_, capacity_, seed);
        }

        void prepare_source() override
        {
            require_allocated("prepare_source");
            wait_and_check(pending_producer_, "initialize_source");
            pending_producer_ = nil;
        }

        void stage_device_to_host() override
        {
            require_allocated("stage_device_to_host");

            if (capacity_ == 0)
            {
                return;
            }

            if (private_staged())
            {
                blit_copy(*context_, source_device_, source_staging_, capacity_);
            }
            // metal-shared: source_device_.contents is already
            // host-visible once prepare_source() has completed.
        }

        const void *source_staging_data() const noexcept override
        {
            if (capacity_ == 0)
            {
                return nullptr;
            }

            return private_staged() ? source_staging_.contents
                                     : source_device_.contents;
        }

        void *destination_staging_data() noexcept override
        {
            if (capacity_ == 0)
            {
                return nullptr;
            }

            return private_staged() ? destination_staging_.contents
                                     : destination_device_.contents;
        }

        void host_send_data(tbccl::World &world, std::size_t peer) override
        {
            require_allocated("host_send_data");
            world.send(peer, source_staging_data(), capacity_);
        }

        void host_recv_data(tbccl::World &world, std::size_t peer) override
        {
            require_allocated("host_recv_data");
            world.recv(peer, destination_staging_data(), capacity_);
        }

        void stage_host_to_device() override
        {
            require_allocated("stage_host_to_device");

            if (capacity_ == 0)
            {
                return;
            }

            if (private_staged())
            {
                blit_copy(
                    *context_, destination_staging_, destination_device_, capacity_);
            }
            // metal-shared: World::recv() already wrote directly into
            // destination_device_.contents (Shared storage).
        }

        void synchronize() override
        {
            wait_and_check(pending_producer_, "synchronize");
            pending_producer_ = nil;
        }

        bool verify_source(std::uint32_t seed) const override
        {
            return verify_device_buffer(source_device_, seed);
        }

        bool verify_destination(std::uint32_t seed) const override
        {
            return verify_device_buffer(destination_device_, seed);
        }

    private:
        bool private_staged() const noexcept
        {
            return kind_ == BackendKind::MetalPrivateStaged;
        }

        void require_allocated(const char *who) const
        {
            if (!allocated_)
            {
                throw std::runtime_error(
                    std::string("MetalBackend::") + who +
                    ": allocate() was never called");
            }
        }

        // Genuine GPU readback (Part 26): for metal-shared, .contents
        // already IS the device buffer's contents; for
        // metal-private-staged, blits into a dedicated scratch Shared
        // buffer (kept separate from the transfer staging buffers so
        // verification never perturbs their state) before reading.
        bool verify_device_buffer(id<MTLBuffer> device_buffer, std::uint32_t seed) const
        {
            if (capacity_ == 0)
            {
                return true;
            }

            if (private_staged())
            {
                blit_copy(*context_, device_buffer, verify_scratch_, capacity_);
                return verify_buffer_contents(
                    verify_scratch_.contents, capacity_, seed);
            }

            return verify_buffer_contents(device_buffer.contents, capacity_, seed);
        }

        BackendKind kind_;
        std::unique_ptr<MetalContext> context_;

        id<MTLBuffer> source_device_ = nil;
        id<MTLBuffer> destination_device_ = nil;
        id<MTLBuffer> source_staging_ = nil;      // private-staged only
        id<MTLBuffer> destination_staging_ = nil; // private-staged only
        id<MTLBuffer> verify_scratch_ = nil;       // private-staged only

        id<MTLCommandBuffer> pending_producer_ = nil;

        std::size_t capacity_ = 0;
        bool allocated_ = false;
        AllocationStats stats_;
    };

} // namespace

std::unique_ptr<TensorBackend> make_metal_backend(BackendKind kind)
{
    return std::make_unique<MetalBackend>(kind);
}

} // namespace tbccl_bench::tensor
