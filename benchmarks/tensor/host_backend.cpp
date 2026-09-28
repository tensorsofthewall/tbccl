#include "host_backend.hpp"

#include <tbccl/world.hpp>

#include <memory>
#include <stdexcept>
#include <vector>

namespace tbccl_bench::tensor
{

namespace
{

    class HostBackend final : public TensorBackend
    {
    public:
        BackendKind kind() const noexcept override
        {
            return BackendKind::Host;
        }

        void allocate(std::size_t bytes) override
        {
            if (bytes == capacity_ && allocated_)
            {
                ++stats_.reuse_count;
                return;
            }

            source_.assign(bytes, std::uint8_t{0});
            destination_.assign(bytes, std::uint8_t{0});
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

            for (std::size_t i = 0; i < source_.size(); ++i)
            {
                source_[i] = pattern_byte(i, seed);
            }
        }

        void prepare_source() override
        {
            // Host generation above is already synchronous; nothing
            // to wait for.
        }

        void stage_device_to_host() override
        {
            // source_ is already host-visible; no separate staging
            // copy exists for this backend.
        }

        const void *source_staging_data() const noexcept override
        {
            return source_.data();
        }

        void *destination_staging_data() noexcept override
        {
            return destination_.data();
        }

        void host_send_data(tbccl::World &world, std::size_t peer) override
        {
            require_allocated("host_send_data");
            world.send(peer, source_.data(), source_.size());
        }

        void host_recv_data(tbccl::World &world, std::size_t peer) override
        {
            require_allocated("host_recv_data");
            world.recv(peer, destination_.data(), destination_.size());
        }

        void stage_host_to_device() override
        {
            // destination_ already received the bytes directly; no
            // separate device push exists for this backend.
        }

        void synchronize() override
        {
        }

        bool verify_source(std::uint32_t seed) const override
        {
            return verify(source_, seed);
        }

        bool verify_destination(std::uint32_t seed) const override
        {
            return verify(destination_, seed);
        }

    private:
        void require_allocated(const char *who) const
        {
            if (!allocated_)
            {
                throw std::runtime_error(
                    std::string("HostBackend::") + who +
                    ": allocate() was never called");
            }
        }

        static bool verify(
            const std::vector<std::uint8_t> &buffer,
            std::uint32_t seed)
        {
            for (std::size_t i = 0; i < buffer.size(); ++i)
            {
                if (buffer[i] != pattern_byte(i, seed))
                {
                    return false;
                }
            }

            return true;
        }

        std::vector<std::uint8_t> source_;
        std::vector<std::uint8_t> destination_;
        std::size_t capacity_ = 0;
        bool allocated_ = false;
        AllocationStats stats_;
    };

} // namespace

std::unique_ptr<TensorBackend> make_host_backend()
{
    return std::make_unique<HostBackend>();
}

} // namespace tbccl_bench::tensor
