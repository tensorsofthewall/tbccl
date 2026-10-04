#pragma once

// internal diagnostic switches for the small-message latency investigation. Each is read once from the environment, defaults to off, and
// has no public counterpart. They exist to MEASURE, not to ship: a knob that does not pay for itself is removed.
//   TBCCL_DIAG_DIRECT_SEND=<max bytes>      N=2 Host send of at most this many bytes runs on the CALLER thread when the lane is idle (bypasses the
//                                           admission queue and the TX worker wake); everything else takes the normal path
//   TBCCL_DIAG_WORKER_SPIN_US=<us>          an idle lane worker spins this long on an atomic before blocking on its condition variable
//   TBCCL_DIAG_WAIT_SPIN_US=<us>            Work::wait spins this long on an atomic before blocking
//   TBCCL_DIAG_RECV_SPIN_US=<us>            a receive polls the socket (MSG_DONTWAIT) this long before blocking in recvmsg
//   TBCCL_DIAG_ACTIVITY_WINDOW_US=<us>      spin only if this lane/waiter saw activity within the window (0 = always spin)
//   TBCCL_DIAG_QUICKACK=1                   Linux: request TCP_QUICKACK after every receive
//   TBCCL_DIAG_SOCKOPTS=1                   print the effective options of every data socket at creation

#include <chrono>
#include <cstdint>
#include <cstdlib>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace tbccl::detail
{

inline long diag_env(const char *name)
{
    const char *v = std::getenv(name);
    return v ? std::atol(v) : 0;
}

struct DiagKnobs
{
    long direct_send_max = diag_env("TBCCL_DIAG_DIRECT_SEND");
    long worker_spin_us = diag_env("TBCCL_DIAG_WORKER_SPIN_US");
    long wait_spin_us = diag_env("TBCCL_DIAG_WAIT_SPIN_US");
    long recv_spin_us = diag_env("TBCCL_DIAG_RECV_SPIN_US");
    long activity_window_us = diag_env("TBCCL_DIAG_ACTIVITY_WINDOW_US");
    bool quickack = diag_env("TBCCL_DIAG_QUICKACK") != 0;
    bool sockopts = diag_env("TBCCL_DIAG_SOCKOPTS") != 0;
};

inline const DiagKnobs &diag()
{
    static const DiagKnobs knobs;
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

inline std::int64_t diag_now_ns() noexcept
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace tbccl::detail
