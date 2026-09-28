#include "cuda_backend.hpp"

#include <tbccl/world.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tbccl_bench::tensor
{

namespace
{

    void check_cuda(cudaError_t status, const char *what)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("CUDA error in ") + what + ": " +
                cudaGetErrorString(status));
        }
    }

} // namespace

} // namespace tbccl_bench::tensor

#define TBCCL_CUDA_CHECK(expr) \
    ::tbccl_bench::tensor::check_cuda((expr), #expr)

namespace tbccl_bench::tensor
{

namespace
{

    // -----------------------------------------------------------------------------
    // Device-side deterministic pattern generation. Must produce
    // exactly the same bytes as pattern_byte() in tensor_backend.hpp
    // (Part D requires one shared cross-platform pattern) -- kept as
    // an intentional, small, independently-readable duplicate rather
    // than sharing code with the host header, since that header must
    // stay includable by a plain (non-CUDA) C++ compiler.
    // -----------------------------------------------------------------------------

    __global__ void fill_pattern_kernel(
        std::uint8_t *data,
        std::size_t count,
        std::uint32_t seed)
    {
        const std::size_t i =
            static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

        if (i >= count)
        {
            return;
        }

        const std::uint64_t index = static_cast<std::uint64_t>(i);
        const std::uint64_t value =
            index * 131u + (index >> 8) * 17u + static_cast<std::uint64_t>(seed);

        data[i] = static_cast<std::uint8_t>(value & 0xffu);
    }

    void launch_fill_pattern(
        std::uint8_t *device_data,
        std::size_t count,
        std::uint32_t seed,
        cudaStream_t stream)
    {
        if (count == 0)
        {
            return;
        }

        constexpr int kThreadsPerBlock = 256;
        const int blocks =
            static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);

        fill_pattern_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            device_data, count, seed);

        TBCCL_CUDA_CHECK(cudaGetLastError());
    }

    // -----------------------------------------------------------------------------
    // RAII wrappers. Every CUDA resource acquisition below is paired
    // with an unconditional release in the corresponding destructor,
    // and every acquisition happens after any earlier one in the same
    // object has already succeeded -- so a throwing constructor never
    // leaks a partially-constructed object's resources (its
    // destructor still runs the release for whatever *did* succeed),
    // and a throwing allocate()/reallocate() call leaves the object in
    // its prior (fully released or fully valid) state, never a leaked
    // in-between state, because free() is called before any new
    // acquisition is attempted.
    // -----------------------------------------------------------------------------

    class DeviceBuffer
    {
    public:
        DeviceBuffer() = default;

        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        ~DeviceBuffer()
        {
            release();
        }

        void allocate(std::size_t bytes)
        {
            release();

            if (bytes > 0)
            {
                TBCCL_CUDA_CHECK(cudaMalloc(&ptr_, bytes));
            }

            size_ = bytes;
        }

        void release()
        {
            if (ptr_ != nullptr)
            {
                cudaFree(ptr_);
                ptr_ = nullptr;
            }

            size_ = 0;
        }

        void *get() const noexcept
        {
            return ptr_;
        }

        std::size_t size() const noexcept
        {
            return size_;
        }

    private:
        void *ptr_ = nullptr;
        std::size_t size_ = 0;
    };

    class CudaStream
    {
    public:
        CudaStream()
        {
            TBCCL_CUDA_CHECK(cudaStreamCreate(&stream_));
        }

        CudaStream(const CudaStream &) = delete;
        CudaStream &operator=(const CudaStream &) = delete;

        ~CudaStream()
        {
            if (stream_ != nullptr)
            {
                cudaStreamDestroy(stream_);
            }
        }

        cudaStream_t get() const noexcept
        {
            return stream_;
        }

        void synchronize() const
        {
            TBCCL_CUDA_CHECK(cudaStreamSynchronize(stream_));
        }

    private:
        cudaStream_t stream_ = nullptr;
    };

    // Host-visible staging storage for the two CUDA backend variants.
    // Pageable: an ordinary heap allocation, reallocated by allocate().
    // Pinned: cudaHostAlloc()'d page-locked memory, released via
    // cudaFreeHost() -- required for cudaMemcpyAsync() to actually
    // behave asynchronously (Part F, requirement 21's caution: plain
    // pageable memory gives no such guarantee).
    class HostStagingBuffer
    {
    public:
        virtual ~HostStagingBuffer() = default;

        HostStagingBuffer(const HostStagingBuffer &) = delete;
        HostStagingBuffer &operator=(const HostStagingBuffer &) = delete;

        virtual void allocate(std::size_t bytes) = 0;
        virtual void *data() noexcept = 0;
        virtual const void *data() const noexcept = 0;

    protected:
        HostStagingBuffer() = default;
    };

    class PageableStagingBuffer final : public HostStagingBuffer
    {
    public:
        void allocate(std::size_t bytes) override
        {
            buffer_.assign(bytes, std::uint8_t{0});
        }

        void *data() noexcept override
        {
            return buffer_.data();
        }

        const void *data() const noexcept override
        {
            return buffer_.data();
        }

    private:
        std::vector<std::uint8_t> buffer_;
    };

    class PinnedStagingBuffer final : public HostStagingBuffer
    {
    public:
        ~PinnedStagingBuffer() override
        {
            release();
        }

        void allocate(std::size_t bytes) override
        {
            release();

            if (bytes > 0)
            {
                TBCCL_CUDA_CHECK(cudaHostAlloc(&ptr_, bytes, cudaHostAllocDefault));
            }

            size_ = bytes;
        }

        void *data() noexcept override
        {
            return ptr_;
        }

        const void *data() const noexcept override
        {
            return ptr_;
        }

    private:
        void release()
        {
            if (ptr_ != nullptr)
            {
                cudaFreeHost(ptr_);
                ptr_ = nullptr;
            }

            size_ = 0;
        }

        void *ptr_ = nullptr;
        std::size_t size_ = 0;
    };

    // -----------------------------------------------------------------------------
    // CudaBackend: shared implementation for both cuda-pageable and
    // cuda-pinned, differing only in which HostStagingBuffer
    // implementation backs source_staging_/destination_staging_ and
    // whether D2H/H2D copies use cudaMemcpy (pageable: synchronous,
    // matching the plan's caution not to assume meaningful async
    // behavior over pageable memory) or cudaMemcpyAsync + explicit
    // stream synchronization (pinned).
    //
    // Every stage method's completion guarantee (Part 8) is upheld by
    // synchronizing the stream (or using a synchronous cudaMemcpy)
    // before that method returns -- Phase 17 does not attempt any
    // cross-stage overlap/double-buffering (Part R).
    // -----------------------------------------------------------------------------

    class CudaBackend final : public TensorBackend
    {
    public:
        explicit CudaBackend(BackendKind kind) : kind_(kind)
        {
            if (kind_ != BackendKind::CudaPageable &&
                kind_ != BackendKind::CudaPinned)
            {
                throw std::runtime_error(
                    "CudaBackend: kind must be CudaPageable or CudaPinned");
            }

            int device_count = 0;
            const cudaError_t status = cudaGetDeviceCount(&device_count);

            if (status != cudaSuccess || device_count == 0)
            {
                throw std::runtime_error(
                    "CudaBackend: no CUDA device available at runtime "
                    "(cudaGetDeviceCount reported none, or the driver call "
                    "failed: " +
                    std::string(cudaGetErrorString(status)) + ")");
            }

            source_staging_ = make_staging_buffer();
            destination_staging_ = make_staging_buffer();
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

            source_device_.allocate(bytes);
            destination_device_.allocate(bytes);
            source_staging_->allocate(bytes);
            destination_staging_->allocate(bytes);

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

            launch_fill_pattern(
                static_cast<std::uint8_t *>(source_device_.get()),
                capacity_, seed, stream_.get());
        }

        void prepare_source() override
        {
            require_allocated("prepare_source");
            stream_.synchronize();
        }

        void stage_device_to_host() override
        {
            require_allocated("stage_device_to_host");

            if (capacity_ == 0)
            {
                return;
            }

            if (kind_ == BackendKind::CudaPinned)
            {
                TBCCL_CUDA_CHECK(cudaMemcpyAsync(
                    source_staging_->data(), source_device_.get(), capacity_,
                    cudaMemcpyDeviceToHost, stream_.get()));
                stream_.synchronize();
            }
            else
            {
                TBCCL_CUDA_CHECK(cudaMemcpy(
                    source_staging_->data(), source_device_.get(), capacity_,
                    cudaMemcpyDeviceToHost));
            }
        }

        const void *source_staging_data() const noexcept override
        {
            return source_staging_->data();
        }

        void *destination_staging_data() noexcept override
        {
            return destination_staging_->data();
        }

        void host_send_data(tbccl::World &world, std::size_t peer) override
        {
            require_allocated("host_send_data");
            world.send(peer, source_staging_->data(), capacity_);
        }

        void host_recv_data(tbccl::World &world, std::size_t peer) override
        {
            require_allocated("host_recv_data");
            world.recv(peer, destination_staging_->data(), capacity_);
        }

        void stage_host_to_device() override
        {
            require_allocated("stage_host_to_device");

            if (capacity_ == 0)
            {
                return;
            }

            if (kind_ == BackendKind::CudaPinned)
            {
                TBCCL_CUDA_CHECK(cudaMemcpyAsync(
                    destination_device_.get(), destination_staging_->data(),
                    capacity_, cudaMemcpyHostToDevice, stream_.get()));
                stream_.synchronize();
            }
            else
            {
                TBCCL_CUDA_CHECK(cudaMemcpy(
                    destination_device_.get(), destination_staging_->data(),
                    capacity_, cudaMemcpyHostToDevice));
            }
        }

        void synchronize() override
        {
            stream_.synchronize();
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
        std::unique_ptr<HostStagingBuffer> make_staging_buffer() const
        {
            if (kind_ == BackendKind::CudaPinned)
            {
                return std::make_unique<PinnedStagingBuffer>();
            }

            return std::make_unique<PageableStagingBuffer>();
        }

        void require_allocated(const char *who) const
        {
            if (!allocated_)
            {
                throw std::runtime_error(
                    std::string("CudaBackend::") + who +
                    ": allocate() was never called");
            }
        }

        // Genuine GPU readback (Part 26): copies the device buffer
        // back to a fresh host vector and compares every byte, never
        // relying on the (possibly stale, possibly never-populated)
        // staging buffers.
        bool verify_device_buffer(
            const DeviceBuffer &device_buffer,
            std::uint32_t seed) const
        {
            if (device_buffer.size() == 0)
            {
                return true;
            }

            std::vector<std::uint8_t> host_copy(device_buffer.size());

            check_cuda(
                cudaMemcpy(
                    host_copy.data(), device_buffer.get(), device_buffer.size(),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy (verify readback)");

            for (std::size_t i = 0; i < host_copy.size(); ++i)
            {
                const std::uint64_t index = static_cast<std::uint64_t>(i);
                const std::uint64_t value =
                    index * 131u + (index >> 8) * 17u +
                    static_cast<std::uint64_t>(seed);
                const std::uint8_t expected = static_cast<std::uint8_t>(value & 0xffu);

                if (host_copy[i] != expected)
                {
                    return false;
                }
            }

            return true;
        }

        BackendKind kind_;
        DeviceBuffer source_device_;
        DeviceBuffer destination_device_;
        std::unique_ptr<HostStagingBuffer> source_staging_;
        std::unique_ptr<HostStagingBuffer> destination_staging_;
        CudaStream stream_;
        std::size_t capacity_ = 0;
        bool allocated_ = false;
        AllocationStats stats_;
    };

} // namespace

std::unique_ptr<TensorBackend> make_cuda_backend(BackendKind kind)
{
    return std::make_unique<CudaBackend>(kind);
}

} // namespace tbccl_bench::tensor
