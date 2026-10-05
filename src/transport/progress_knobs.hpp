#pragma once

// internal, environment-gated experiment switches for the cold-cadence progress investigation. Each is read
// once, defaults to off and has no public counterpart; none ships unless it is promoted by the retention rule in the plan. A knob that does
// not earn its place is removed with the commit that introduced it.
//   TBCCL_WAIT_POLL_US=<us>     a caller thread in Work::wait polls the terminal flag this long before blocking on the condition variable
//   TBCCL_EXEC_POLL_US=<us>     the CollectiveExecutor thread polls a child transfer's terminal flag this long before blocking
//   TBCCL_RX_POLL_US=<us>       a posted receive polls the socket (MSG_DONTWAIT) this long before blocking in recvmsg
//   TBCCL_TX_SPIN_US=<us>       an idle lane worker spins on its queue this long after its last request before blocking
//   TBCCL_EXEC_INLINE=<bytes>   the CollectiveExecutor thread runs its N=2 child transfers (send AND receive) of at most this size itself on an idle lane
//                                   instead of posting them to the lane worker (removes the executor->lane wake and the child->executor wake per child;
//                                   the executor, not the caller, blocks in the socket call, so submission never blocks)
//   TBCCL_DIRECT_SEND=<bytes>   DIAGNOSTIC CONTROL ONLY: an idle-lane Host send of at most this size runs on the caller thread (the earlier
//                                   result: it bypasses the lane wake but blocks the caller in sendmsg, so it is never a candidate)
// Poll loops are bounded by their budget and test the Work's terminal flag (abort and peer loss complete every Work with an error), or, for the socket
// poll, the transport's abort flag; none can outlive the operation it serves.

#include <chrono>
#include <cstdint>
#include <cstdlib>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace tbccl::detail
{

inline long progress_env(const char *name)
{
    const char *v = std::getenv(name);
    return v ? std::atol(v) : 0;
}

struct ProgressKnobs
{
    long wait_poll_us = progress_env("TBCCL_WAIT_POLL_US");
    long exec_poll_us = progress_env("TBCCL_EXEC_POLL_US");
    long rx_poll_us = progress_env("TBCCL_RX_POLL_US");
    long tx_spin_us = progress_env("TBCCL_TX_SPIN_US");
    long direct_send_max = progress_env("TBCCL_DIRECT_SEND");
    long exec_inline_max = progress_env("TBCCL_EXEC_INLINE");
};

inline const ProgressKnobs &progress()
{
    static const ProgressKnobs knobs;
    return knobs;
}

inline void cpu_relax() noexcept
{
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(__aarch64__)
    asm volatile("yield");
#endif
}

inline std::int64_t progress_now_ns() noexcept
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline thread_local bool tl_is_executor_thread = false; // set by the CollectiveExecutor thread: its waits use the executor budget
} // namespace tbccl::detail
