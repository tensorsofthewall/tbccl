// test-only access for the pure-C tests. The C ABI deliberately has no way to hold transport progress; the tests need one to prove submission does not depend on it, so
// this file (compiled into the C tests only, never installed) reaches the private runtime hook through the shim's handle wrapper.
#include "c_internal.hpp"
#include "communicator_debug.hpp"

extern "C" void tbccl_test_pause_progress(tbcclComm_t comm, int paused)
{
    tbccl::detail::debug_set_progress_paused(*comm->comm, paused != 0);
}
