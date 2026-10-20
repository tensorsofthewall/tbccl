// Test-only observation point for the failure transition of a staged transfer (tensor_comm_worker.cpp).
//
// The probe is a plain function pointer, null unless a test installs one, and it is read only on the failure path of a transfer: the successful path never
// touches it. It exists so that a test can stall the failing stage between the steps of the failure transition and check that the other stage still
// terminates; it is not part of the public API, is not installed, and is not controlled by the environment.
#pragma once

#include <atomic>

namespace tbccl
{
namespace detail
{

// `stage` names the role that is failing ("send.staging", "send.network", "recv.network", "recv.staging"); `step` is one of
// "begin", "recorded", "cancelled", "slot_released", in that order.
using PipelineFailureProbe = void (*)(const char *stage, const char *step);

inline std::atomic<PipelineFailureProbe> g_pipeline_failure_probe{nullptr};

inline void set_pipeline_failure_probe(PipelineFailureProbe probe) noexcept
{
    g_pipeline_failure_probe.store(probe, std::memory_order_release);
}

inline void pipeline_failure_probe(const char *stage, const char *step)
{
    if (PipelineFailureProbe probe = g_pipeline_failure_probe.load(std::memory_order_acquire)) probe(stage, step);
}

} // namespace detail
} // namespace tbccl
