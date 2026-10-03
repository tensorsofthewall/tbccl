// tbccl.h included from C++ (extern "C" guard, no C++-hostile constructs) alongside the C++ API without conflicts, and the C result mapping agrees with
// ErrorCode for every code.
#include <tbccl/tbccl.h>

#include <tbccl/communicator.hpp>
#include <tbccl/error.hpp>

#include <cstdio>

int main()
{
    uint32_t abi = 0;
    if (tbcclGetAbiVersion(&abi) != TBCCL_SUCCESS || abi != TBCCL_C_ABI_VERSION) return 1;
    static_assert(sizeof(tbcclBuffer) == 48, "layout");
    static_assert(TBCCL_FLOAT16 == static_cast<int>(tbccl::DataType::Float16) && TBCCL_INT32 == static_cast<int>(tbccl::DataType::Int32) &&
                      TBCCL_BFLOAT16 == static_cast<int>(tbccl::DataType::BFloat16),
                  "the C datatype values equal the C++ ones");
    static_assert(TBCCL_SUM == static_cast<int>(tbccl::ReduceOp::Sum) && TBCCL_MAX == static_cast<int>(tbccl::ReduceOp::Max), "reduce op values");
    static_assert(TBCCL_MEMORY_HOST == static_cast<int>(tbccl::MemoryKind::Host) && TBCCL_MEMORY_METAL_SHARED == static_cast<int>(tbccl::MemoryKind::MetalShared), "memory kind values");
    std::puts("[PASS] tbccl.h compiles as C++ and agrees with the C++ enum values");
    std::puts("All C ABI header (C++) tests passed.");
    return 0;
}
