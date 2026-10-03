#pragma once

// Phase 50: opt-in communication trace (TBCCL_TRACE=1), off by default and free when off. One line per event, every line starting with
// the communicator-id prefix, rank and world size so interleaved logs of several ranks stay readable:
//
//   [tbccl 1234.567 comm=1a2b3c4d rank=2/4] work=7 all_reduce #3 post recv peer=0 bytes=4096 dtype=float32 op=sum
//
// Separate from the older per-stage timing switches (TBCCL_ASYNC_TIMING, TBCCL_ALLREDUCE_TIMING), which print raw timestamps only.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace tbccl::detail
{

class Trace
{
public:
    Trace() = default;
    Trace(const std::string &comm_prefix, std::size_t rank, std::size_t world)
        : on_(std::getenv("TBCCL_TRACE") != nullptr && std::string(std::getenv("TBCCL_TRACE")) != "0"),
          tag_("comm=" + comm_prefix + " rank=" + std::to_string(rank) + "/" + std::to_string(world))
    {
    }

    bool on() const noexcept { return on_; }

    void line(const std::string &text) const
    {
        if (!on_) return;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
        std::fprintf(stderr, "[tbccl %.3f %s] %s\n", ms, tag_.c_str(), text.c_str());
    }

private:
    bool on_ = false;
    std::string tag_;
};

} // namespace tbccl::detail
