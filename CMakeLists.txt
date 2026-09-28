cmake_minimum_required(VERSION 3.20)

project(
    tbccl
    VERSION 0.1.0
    DESCRIPTION "Thunderbolt Collective Communication Library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

option(
    TBCCL_ENABLE_RING_TRACE
    "Enable internal ring execution tracing (diagnostic only; adds a lightweight per-thread event buffer to the ring collective path). Off by default; leave off for normal builds."
    OFF
)

add_library(tbccl STATIC
    src/transport/tcp.cpp
    src/core/world.cpp
    src/core/tcp_world.cpp
    src/collectives/barrier.cpp
    src/collectives/broadcast.cpp
    src/collectives/all_gather.cpp
    src/collectives/reduction.cpp
    src/collectives/reduce.cpp
    src/collectives/all_reduce.cpp
    src/collectives/reduce_scatter.cpp
    src/collectives/ring_trace.cpp
    src/collectives/ring_executor.cpp
    src/collectives/all_gather_ring.cpp
    src/collectives/reduce_scatter_ring.cpp
    src/collectives/all_reduce_ring.cpp
    src/collectives/all_gather_pipelined.cpp
    src/collectives/reduce_scatter_pipelined.cpp
    src/collectives/all_reduce_pipelined.cpp
    src/collectives/algorithm_selector.cpp
)

if(TBCCL_ENABLE_RING_TRACE)
    # PUBLIC: every consumer (tests, tools, benchmarks) must agree on
    # this macro's value, since ring_trace.hpp's declarations differ
    # by branch depending on it — a mismatched definition between
    # translation units would be an ODR violation.
    target_compile_definitions(tbccl
        PUBLIC
            TBCCL_ENABLE_RING_TRACE
    )
endif()

target_include_directories(tbccl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# src/core/world.cpp needs ring_executor.hpp (World::ring_executor_'s
# complete type); ring_executor.hpp/.cpp are not part of the public
# API — this path is private to the library build, not
# installed/exported.
target_include_directories(tbccl
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
)

target_link_libraries(tbccl
    PUBLIC
        Threads::Threads
)

target_compile_features(tbccl
    PUBLIC
        cxx_std_17
)

if(MSVC)
    target_compile_options(tbccl PRIVATE /W4)
else()
    target_compile_options(tbccl PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# ---------------------------------------------------------------------------
# Heterogeneous tensor-transfer backends. Benchmark-only: the tbccl library
# above has no CUDA/Metal awareness and no dependency on anything below. Both
# backends default OFF; a host-only Linux or macOS build never needs the CUDA
# Toolkit or Metal/Foundation.
# ---------------------------------------------------------------------------

option(
    TBCCL_ENABLE_CUDA
    "Build the CUDA pageable/pinned tensor-transfer backend (requires the CUDA Toolkit). Off by default."
    OFF
)

option(
    TBCCL_ENABLE_METAL
    "Build the Metal shared/private-staged tensor-transfer backend (Apple platforms only). Off by default."
    OFF
)

if(TBCCL_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

if(TBCCL_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "TBCCL_ENABLE_METAL requires an Apple platform")
    endif()
    enable_language(OBJCXX)
endif()

add_library(tbccl_tensor_backend STATIC
    benchmarks/tensor/tensor_backend.cpp
    benchmarks/tensor/host_backend.cpp
)

if(TBCCL_ENABLE_CUDA)
    target_sources(tbccl_tensor_backend PRIVATE benchmarks/tensor/cuda_backend.cu)
endif()

if(TBCCL_ENABLE_METAL)
    target_sources(tbccl_tensor_backend PRIVATE benchmarks/tensor/metal_backend.mm)
endif()

# benchmarks/ (not benchmarks/tensor/) is the public include root, so
# consumers write #include "tensor/tensor_backend.hpp" etc.
target_include_directories(tbccl_tensor_backend
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
)

target_link_libraries(tbccl_tensor_backend
    PUBLIC
        tbccl
        Threads::Threads
)

if(TBCCL_ENABLE_CUDA)
    # PUBLIC: tensor_backend.hpp's make_backend()/backend_kind_available()
    # behavior depends on this macro, so every consumer (tests, the
    # tensor benchmark) must agree on its value.
    target_compile_definitions(tbccl_tensor_backend PUBLIC TBCCL_ENABLE_CUDA)
    target_link_libraries(tbccl_tensor_backend PUBLIC CUDA::cudart)
    set_target_properties(tbccl_tensor_backend PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(tbccl_tensor_backend PROPERTIES CUDA_ARCHITECTURES native)
    endif()
endif()

if(TBCCL_ENABLE_METAL)
    target_compile_definitions(tbccl_tensor_backend PUBLIC TBCCL_ENABLE_METAL)
    target_link_libraries(tbccl_tensor_backend PUBLIC "-framework Metal" "-framework Foundation")
endif()

# -Wall/-Wextra/-Wpedantic apply to our own C++/Objective-C++ sources
# only -- CUDA sources are compiled by nvcc, whose host-side forwarding
# of these flags produces noise from NVIDIA's own headers unrelated to
# our code, so they are intentionally not applied to CUDA_LANGUAGE.
if(MSVC)
    target_compile_options(tbccl_tensor_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(tbccl_tensor_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

add_executable(tb_pingpong
    benchmarks/tcp_pingpong.cpp
)

target_link_libraries(tb_pingpong
    PRIVATE
        tbccl
)

if(MSVC)
    target_compile_options(tb_pingpong PRIVATE /W4)
else()
    target_compile_options(tb_pingpong PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_world_smoke
    tools/tbccl_world_smoke.cpp
)

target_link_libraries(tbccl_world_smoke
    PRIVATE
        tbccl
)

if(MSVC)
    target_compile_options(tbccl_world_smoke PRIVATE /W4)
else()
    target_compile_options(tbccl_world_smoke PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_barrier_smoke
    tools/tbccl_barrier_smoke.cpp
)

target_link_libraries(tbccl_barrier_smoke
    PRIVATE
        tbccl
)

if(MSVC)
    target_compile_options(tbccl_barrier_smoke PRIVATE /W4)
else()
    target_compile_options(tbccl_barrier_smoke PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_broadcast_smoke
    tools/tbccl_broadcast_smoke.cpp
)

target_link_libraries(tbccl_broadcast_smoke
    PRIVATE
        tbccl
)

if(MSVC)
    target_compile_options(tbccl_broadcast_smoke PRIVATE /W4)
else()
    target_compile_options(tbccl_broadcast_smoke PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_all_gather_smoke
    tools/tbccl_all_gather_smoke.cpp
)

target_link_libraries(tbccl_all_gather_smoke
    PRIVATE
        tbccl
)

if(MSVC)
    target_compile_options(tbccl_all_gather_smoke PRIVATE /W4)
else()
    target_compile_options(tbccl_all_gather_smoke PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_reduce_smoke
    tools/tbccl_reduce_smoke.cpp
)

target_link_libraries(tbccl_reduce_smoke
    PRIVATE
        tbccl
)

if(MSVC)
    target_compile_options(tbccl_reduce_smoke PRIVATE /W4)
else()
    target_compile_options(tbccl_reduce_smoke PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_all_reduce_smoke
    tools/tbccl_all_reduce_smoke.cpp
)

target_link_libraries(tbccl_all_reduce_smoke
    PRIVATE
        tbccl
)

if(MSVC)
    target_compile_options(tbccl_all_reduce_smoke PRIVATE /W4)
else()
    target_compile_options(tbccl_all_reduce_smoke PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_reduce_scatter_smoke
    tools/tbccl_reduce_scatter_smoke.cpp
)

target_link_libraries(tbccl_reduce_scatter_smoke
    PRIVATE
        tbccl
)

if(MSVC)
    target_compile_options(tbccl_reduce_scatter_smoke PRIVATE /W4)
else()
    target_compile_options(tbccl_reduce_scatter_smoke PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# Diagnostic only — dumps the
# internal ring trace (see ring_trace.hpp) for a handful of
# ReduceScatter invocations. Builds fine without
# TBCCL_ENABLE_RING_TRACE, but prints no events in that case (every
# ring_trace_* call is a no-op).
add_executable(tbccl_ring_trace_dump
    tools/tbccl_ring_trace_dump.cpp
)

target_link_libraries(tbccl_ring_trace_dump
    PRIVATE
        tbccl
)

# reduce_scatter_internal.hpp / ring_executor.hpp / ring_trace.hpp are
# not part of the public API — this path is private to this one
# diagnostic target, not installed/exported.
target_include_directories(tbccl_ring_trace_dump
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
)

if(MSVC)
    target_compile_options(tbccl_ring_trace_dump PRIVATE /W4)
else()
    target_compile_options(tbccl_ring_trace_dump PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_collective_bench
    benchmarks/collective_bench.cpp
)

target_link_libraries(tbccl_collective_bench
    PRIVATE
        tbccl
)

# all_gather_internal.hpp is not part of the public API — this path is
# private to this one benchmark target, not installed/exported.
target_include_directories(tbccl_collective_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
)

if(MSVC)
    target_compile_options(tbccl_collective_bench PRIVATE /W4)
else()
    target_compile_options(tbccl_collective_bench PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_tensor_transfer_bench
    benchmarks/tensor_transfer_bench.cpp
)

target_link_libraries(tbccl_tensor_transfer_bench
    PRIVATE
        tbccl_tensor_backend
        tbccl
)

if(MSVC)
    target_compile_options(tbccl_tensor_transfer_bench PRIVATE /W4)
else()
    target_compile_options(tbccl_tensor_transfer_bench PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

include(CTest)

if(BUILD_TESTING)

    find_package(Threads REQUIRED)

    add_executable(tcp_transport_test
        tests/tcp_transport_test.cpp
    )

    target_link_libraries(tcp_transport_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(tcp_transport_test PRIVATE /W4)
    else()
        target_compile_options(tcp_transport_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME tcp_transport_test
        COMMAND tcp_transport_test
    )

    add_executable(world_test
        tests/world_test.cpp
    )

    target_link_libraries(world_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(world_test PRIVATE /W4)
    else()
        target_compile_options(world_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME world_test
        COMMAND world_test
    )

    add_executable(barrier_test
        tests/barrier_test.cpp
    )

    target_link_libraries(barrier_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(barrier_test PRIVATE /W4)
    else()
        target_compile_options(barrier_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME barrier_test
        COMMAND barrier_test
    )

    add_executable(broadcast_test
        tests/broadcast_test.cpp
    )

    target_link_libraries(broadcast_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(broadcast_test PRIVATE /W4)
    else()
        target_compile_options(broadcast_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME broadcast_test
        COMMAND broadcast_test
    )

    add_executable(all_gather_test
        tests/all_gather_test.cpp
    )

    target_link_libraries(all_gather_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(all_gather_test PRIVATE /W4)
    else()
        target_compile_options(all_gather_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME all_gather_test
        COMMAND all_gather_test
    )

    add_executable(reduce_test
        tests/reduce_test.cpp
    )

    target_link_libraries(reduce_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(reduce_test PRIVATE /W4)
    else()
        target_compile_options(reduce_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME reduce_test
        COMMAND reduce_test
    )

    add_executable(all_reduce_test
        tests/all_reduce_test.cpp
    )

    target_link_libraries(all_reduce_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(all_reduce_test PRIVATE /W4)
    else()
        target_compile_options(all_reduce_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME all_reduce_test
        COMMAND all_reduce_test
    )

    add_executable(reduce_scatter_test
        tests/reduce_scatter_test.cpp
    )

    target_link_libraries(reduce_scatter_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(reduce_scatter_test PRIVATE /W4)
    else()
        target_compile_options(reduce_scatter_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME reduce_scatter_test
        COMMAND reduce_scatter_test
    )

    add_executable(all_gather_ring_test
        tests/all_gather_ring_test.cpp
    )

    target_link_libraries(all_gather_ring_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    # all_gather_internal.hpp is not part of the public API — this
    # path is private to this one test target, not installed/exported.
    target_include_directories(all_gather_ring_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
    )

    if(MSVC)
        target_compile_options(all_gather_ring_test PRIVATE /W4)
    else()
        target_compile_options(all_gather_ring_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME all_gather_ring_test
        COMMAND all_gather_ring_test
    )

    add_executable(reduce_scatter_ring_test
        tests/reduce_scatter_ring_test.cpp
    )

    target_link_libraries(reduce_scatter_ring_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    # reduce_scatter_internal.hpp / all_gather_internal.hpp are not
    # part of the public API — this path is private to this one test
    # target, not installed/exported.
    target_include_directories(reduce_scatter_ring_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
    )

    if(MSVC)
        target_compile_options(reduce_scatter_ring_test PRIVATE /W4)
    else()
        target_compile_options(reduce_scatter_ring_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME reduce_scatter_ring_test
        COMMAND reduce_scatter_ring_test
    )

    add_executable(all_reduce_ring_test
        tests/all_reduce_ring_test.cpp
    )

    target_link_libraries(all_reduce_ring_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    # all_reduce_internal.hpp / all_gather_internal.hpp /
    # reduce_scatter_internal.hpp are not part of the public API — this
    # path is private to this one test target, not installed/exported.
    target_include_directories(all_reduce_ring_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
    )

    if(MSVC)
        target_compile_options(all_reduce_ring_test PRIVATE /W4)
    else()
        target_compile_options(all_reduce_ring_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME all_reduce_ring_test
        COMMAND all_reduce_ring_test
    )

    add_executable(algorithm_selector_test
        tests/algorithm_selector_test.cpp
    )

    target_link_libraries(algorithm_selector_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    # algorithm_selector.hpp is not part of the public API — this path
    # is private to this one test target, not installed/exported.
    target_include_directories(algorithm_selector_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
    )

    if(MSVC)
        target_compile_options(algorithm_selector_test PRIVATE /W4)
    else()
        target_compile_options(algorithm_selector_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME algorithm_selector_test
        COMMAND algorithm_selector_test
    )

    add_executable(ring_executor_test
        tests/ring_executor_test.cpp
    )

    target_link_libraries(ring_executor_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    # ring_executor.hpp / all_gather_internal.hpp / all_reduce_internal
    # .hpp / reduce_scatter_internal.hpp are not part of the public API
    # — this path is private to this one test target, not
    # installed/exported.
    target_include_directories(ring_executor_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
    )

    if(MSVC)
        target_compile_options(ring_executor_test PRIVATE /W4)
    else()
        target_compile_options(ring_executor_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME ring_executor_test
        COMMAND ring_executor_test
    )

    add_executable(ring_pipeline_session_test
        tests/ring_pipeline_session_test.cpp
    )

    target_link_libraries(ring_pipeline_session_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    # ring_pipeline_session.hpp is not part of the public API — this
    # path is private to this one test target, not installed/exported.
    target_include_directories(ring_pipeline_session_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
    )

    if(MSVC)
        target_compile_options(ring_pipeline_session_test PRIVATE /W4)
    else()
        target_compile_options(ring_pipeline_session_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME ring_pipeline_session_test
        COMMAND ring_pipeline_session_test
    )

    add_executable(all_gather_pipelined_test
        tests/all_gather_pipelined_test.cpp
    )

    target_link_libraries(all_gather_pipelined_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    # all_gather_internal.hpp / ring_executor.hpp are not part of the
    # public API — this path is private to this one test target, not
    # installed/exported.
    target_include_directories(all_gather_pipelined_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
    )

    if(MSVC)
        target_compile_options(all_gather_pipelined_test PRIVATE /W4)
    else()
        target_compile_options(all_gather_pipelined_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME all_gather_pipelined_test
        COMMAND all_gather_pipelined_test
    )

    add_executable(reduce_scatter_pipelined_test
        tests/reduce_scatter_pipelined_test.cpp
    )

    target_link_libraries(reduce_scatter_pipelined_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    target_include_directories(reduce_scatter_pipelined_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
    )

    if(MSVC)
        target_compile_options(reduce_scatter_pipelined_test PRIVATE /W4)
    else()
        target_compile_options(reduce_scatter_pipelined_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME reduce_scatter_pipelined_test
        COMMAND reduce_scatter_pipelined_test
    )

    add_executable(all_reduce_pipelined_test
        tests/all_reduce_pipelined_test.cpp
    )

    target_link_libraries(all_reduce_pipelined_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    target_include_directories(all_reduce_pipelined_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
    )

    if(MSVC)
        target_compile_options(all_reduce_pipelined_test PRIVATE /W4)
    else()
        target_compile_options(all_reduce_pipelined_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME all_reduce_pipelined_test
        COMMAND all_reduce_pipelined_test
    )

    add_executable(tensor_host_test
        tests/tensor_host_test.cpp
    )

    target_link_libraries(tensor_host_test
        PRIVATE
            tbccl_tensor_backend
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(tensor_host_test PRIVATE /W4)
    else()
        target_compile_options(tensor_host_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME tensor_host_test
        COMMAND tensor_host_test
    )

endif()
