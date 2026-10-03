// TBCCL C ABI v1: Work queries. The API return value is the status of the QUERY; the operation's own terminal result is an out-parameter. See docs/c_abi_v1.md.

#include "c_internal.hpp"

#include <chrono>
#include <limits>

using namespace tbccl;
using namespace tbccl::capi;

namespace
{
tbcclResult_t terminal_result(const Work &work) noexcept { return work.has_error() ? to_c(work.error_code()) : TBCCL_SUCCESS; }
} // namespace

extern "C"
{

tbcclResult_t TBCCL_CALL tbcclWorkTest(tbcclWork_t work, int32_t *done, tbcclResult_t *operation_result)
{
    return guard([&]() -> tbcclResult_t {
        if (work == nullptr || done == nullptr || operation_result == nullptr) return TBCCL_INVALID_ARGUMENT;
        *done = 0;
        *operation_result = TBCCL_SUCCESS;
        if (work->work.is_completed())
        {
            *done = 1;
            *operation_result = terminal_result(work->work);
        }
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclWorkWait(tbcclWork_t work, tbcclResult_t *operation_result)
{
    return guard([&]() -> tbcclResult_t {
        if (work == nullptr || operation_result == nullptr) return TBCCL_INVALID_ARGUMENT;
        *operation_result = TBCCL_SUCCESS;
        work->work.wait();
        *operation_result = terminal_result(work->work);
        return TBCCL_SUCCESS;
    });
}

tbcclResult_t TBCCL_CALL tbcclWorkWaitFor(tbcclWork_t work, uint64_t timeout_ms, int32_t *done, tbcclResult_t *operation_result)
{
    return guard([&]() -> tbcclResult_t {
        if (work == nullptr || done == nullptr || operation_result == nullptr) return TBCCL_INVALID_ARGUMENT;
        *done = 0;
        *operation_result = TBCCL_SUCCESS;
        // clamp: a duration this long can overflow the clock arithmetic inside the wait
        constexpr std::uint64_t kMaxMs = 365ull * 24 * 3600 * 1000;
        const auto timeout = std::chrono::milliseconds(static_cast<std::int64_t>(timeout_ms < kMaxMs ? timeout_ms : kMaxMs));
        if (work->work.wait_for(timeout))
        {
            *done = 1;
            *operation_result = terminal_result(work->work);
        }
        return TBCCL_SUCCESS; // an expired caller timeout is not an error and does not touch the Work
    });
}

tbcclResult_t TBCCL_CALL tbcclWorkGetErrorString(tbcclWork_t work, char *buf, size_t capacity, size_t *required)
{
    return guard([&]() -> tbcclResult_t {
        if (work == nullptr) return TBCCL_INVALID_ARGUMENT;
        return copy_text(work->work.error(), buf, capacity, required);
    });
}

tbcclResult_t TBCCL_CALL tbcclWorkDestroy(tbcclWork_t work)
{
    return guard([&]() -> tbcclResult_t {
        delete work; // drops only the caller's handle; the operation continues against the shared state
        return TBCCL_SUCCESS;
    });
}

} // extern "C"
