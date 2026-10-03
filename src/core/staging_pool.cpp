#include <tbccl/error.hpp>
#include <tbccl/async_transfer.hpp>

#include <stdexcept>

namespace tbccl
{

StagingPool::StagingPool(std::size_t slot_bytes, std::size_t depth)
    : slot_bytes_(slot_bytes), depth_(depth)
{
    if (depth_ == 0)
    {
        throw Error(ErrorCode::InvalidArgument, "StagingPool depth must be >= 1");
    }

    if (slot_bytes_ == 0)
    {
        throw Error(ErrorCode::InvalidArgument, "StagingPool slot_bytes must be >= 1");
    }

    buffers_.reserve(depth_);
    states_.assign(depth_, StagingSlotState::Free);

    for (std::size_t i = 0; i < depth_; ++i)
    {
        buffers_.push_back(std::make_unique<std::byte[]>(slot_bytes_));
    }
}

StagingPool::~StagingPool() = default;

std::size_t StagingPool::acquire()
{
    std::unique_lock<std::mutex> lock(mutex_);

    cv_.wait(
        lock,
        [&]()
        {
            for (std::size_t i = 0; i < depth_; ++i)
            {
                if (states_[i] == StagingSlotState::Free)
                {
                    return true;
                }
            }
            return false;
        });

    for (std::size_t i = 0; i < depth_; ++i)
    {
        if (states_[i] == StagingSlotState::Free)
        {
            states_[i] = StagingSlotState::SourcePreparing;
            return i;
        }
    }

    // Unreachable: the predicate above guarantees a Free slot exists
    // when wait() returns.
    throw Error(ErrorCode::InternalError, "StagingPool::acquire() invariant violated");
}

void *StagingPool::data(std::size_t slot_index) noexcept
{
    return buffers_[slot_index].get();
}

const void *StagingPool::data(std::size_t slot_index) const noexcept
{
    return buffers_[slot_index].get();
}

void StagingPool::release(std::size_t slot_index)
{
    {
        std::lock_guard<std::mutex> lock(mutex_);
        states_[slot_index] = StagingSlotState::Free;
    }
    cv_.notify_one();
}

void StagingPool::set_state(std::size_t slot_index, StagingSlotState state)
{
    std::lock_guard<std::mutex> lock(mutex_);
    states_[slot_index] = state;
}

StagingSlotState StagingPool::state(std::size_t slot_index) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return states_[slot_index];
}

} // namespace tbccl
