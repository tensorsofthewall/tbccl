cmake_minimum_required(VERSION 3.20)

project(
    tbccl
    VERSION 0.3.0
    DESCRIPTION "Thunderbolt Collective Communication Library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# Phase 41 Part AJ: needed early for CMAKE_INSTALL_INCLUDEDIR, used by
# tbccl's PUBLIC include directories below so the exported target works
# correctly both from the build tree and from an installed prefix.
include(GNUInstallDirs)

option(
    TBCCL_ENABLE_RING_TRACE
    "Enable internal ring execution tracing (diagnostic only; adds a lightweight per-thread event buffer to the ring collective path). Off by default; leave off for normal builds."
    OFF
)

add_library(tbccl STATIC
    src/transport/tcp.cpp
    src/core/world.cpp
    src/core/tcp_world.cpp
    src/core/peer_capabilities.cpp
    src/core/chunk_plan.cpp
    src/core/staging_pool.cpp
    src/core/tensor_comm_worker.cpp
    src/core/types.cpp
    src/core/buffer.cpp
    src/core/communicator.cpp
    src/core/rank_directory.cpp
    src/core/wire_protocol.cpp
    src/core/bootstrap_config.cpp
    src/core/connection_manager.cpp
    src/core/nrank_collectives.cpp
    src/core/collective_protocol.cpp
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
    src/collectives/hetero_allreduce.cpp
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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
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

# The installed static library may be linked into shared objects (e.g.
# out-of-tree framework adapters), so it must be position-independent.
set_target_properties(tbccl PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
# Phase 17: heterogeneous tensor-transfer backends. Benchmark-only: the
# tbccl library above has no CUDA/Metal awareness and no dependency on
# anything below. Both backends default OFF; a host-only Linux or
# macOS build never needs the CUDA Toolkit or Metal/Foundation.
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
    benchmarks/tensor/tensor_backend_capabilities.cpp
)

if(TBCCL_ENABLE_CUDA)
    target_sources(tbccl_tensor_backend PRIVATE
        benchmarks/tensor/cuda_backend.cu
        benchmarks/tensor/cuda_chunked_async_backend.cu
        benchmarks/tensor/cuda_bucket_compute.cu
        benchmarks/tensor/cuda_reduce_backend.cu
    )
endif()

if(TBCCL_ENABLE_CUDA)
    # Optional installable CUDA memory-provider component. Contains only
    # what tbccl::register_cuda_support() needs and depends solely on the
    # public tbccl headers; core `tbccl` stays CUDA-free. The private
    # header cuda_external_async_backend.hpp is NOT installed.
    add_library(tbccl_cuda STATIC
        benchmarks/tensor/cuda_external_async_backend.cu
        benchmarks/tensor/cuda_memory_provider.cu
    )
    target_include_directories(tbccl_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/tensor
    )
    target_link_libraries(tbccl_cuda
        PUBLIC tbccl
        PRIVATE CUDA::cudart
    )
    set_target_properties(tbccl_cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
    )
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(tbccl_cuda PROPERTIES CUDA_ARCHITECTURES native)
    endif()
    target_link_libraries(tbccl_tensor_backend PUBLIC tbccl_cuda)
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

    # ARC manages the id-typed C++ class members in metal_backend.mm
    # (MTLBuffer/MTLCommandBuffer/etc. handles) like ordinary C++
    # members -- retained on assignment, released on destruction --
    # so no manual retain/release appears in that file.
    set_source_files_properties(
        benchmarks/tensor/metal_backend.mm
        PROPERTIES COMPILE_FLAGS "-fobjc-arc"
    )
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

if(TBCCL_ENABLE_CUDA)
    # Phase 18 standalone CUDA diagnostic (Part E/F): no TBCCL
    # networking or TensorBackend involvement, deliberately decoupled
    # so its timing can't be perturbed by or confused with either.
    add_executable(cuda_sync_bench
        benchmarks/tensor/cuda_sync_bench.cu
    )

    target_link_libraries(cuda_sync_bench
        PRIVATE
            CUDA::cudart
    )

    set_target_properties(cuda_sync_bench PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(cuda_sync_bench PROPERTIES CUDA_ARCHITECTURES native)
    endif()
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

# Phase 14 diagnostic only — dumps the internal ring trace (see
# ring_trace.hpp) for a handful of ReduceScatter invocations. Builds
# fine without TBCCL_ENABLE_RING_TRACE, but prints no events in that
# case (every ring_trace_* call is a no-op).
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

add_executable(tbccl_async_transfer_bench
    benchmarks/async_transfer_bench.cpp
)

target_link_libraries(tbccl_async_transfer_bench
    PRIVATE
        tbccl_tensor_backend
        tbccl
)

target_include_directories(tbccl_async_transfer_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
)

if(MSVC)
    target_compile_options(tbccl_async_transfer_bench PRIVATE /W4)
else()
    target_compile_options(tbccl_async_transfer_bench PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# Phase 36: plain .cpp (not .cu) so it builds on every platform -- the
# sender role requires TBCCL_ENABLE_CUDA at runtime (checked, clear
# error otherwise), the receiver role only needs tensor_backend.hpp's
# portable backend set (host/metal-shared), exactly like
# tbccl_async_transfer_bench above.
add_executable(tbccl_cuda_overlap_bench
    benchmarks/cuda_overlap_bench.cpp
)

target_link_libraries(tbccl_cuda_overlap_bench
    PRIVATE
        tbccl_tensor_backend
        tbccl
)

target_include_directories(tbccl_cuda_overlap_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
)

if(MSVC)
    target_compile_options(tbccl_cuda_overlap_bench PRIVATE /W4)
else()
    target_compile_options(tbccl_cuda_overlap_bench PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# Phase 50 diagnostics: N=2 regression micro-benchmark (also built by hand against older libtbccl for the A/B) and the reference N-rank collectives.
add_executable(communicator_n2_bench benchmarks/communicator_n2_bench.cpp)
target_link_libraries(communicator_n2_bench PRIVATE tbccl Threads::Threads)
add_executable(communicator_nrank_bench benchmarks/communicator_nrank_bench.cpp)
target_link_libraries(communicator_nrank_bench PRIVATE tbccl Threads::Threads)

add_executable(tbccl_hetero_allreduce_bench
    benchmarks/tbccl_hetero_allreduce_bench.cpp
)

target_link_libraries(tbccl_hetero_allreduce_bench
    PRIVATE
        tbccl_tensor_backend
        tbccl
)

target_include_directories(tbccl_hetero_allreduce_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
)

if(MSVC)
    target_compile_options(tbccl_hetero_allreduce_bench PRIVATE /W4)
else()
    target_compile_options(tbccl_hetero_allreduce_bench PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_bucketed_allreduce_bench
    benchmarks/tbccl_bucketed_allreduce_bench.cpp
)

target_link_libraries(tbccl_bucketed_allreduce_bench
    PRIVATE
        tbccl_tensor_backend
        tbccl
)

target_include_directories(tbccl_bucketed_allreduce_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
)

if(MSVC)
    target_compile_options(tbccl_bucketed_allreduce_bench PRIVATE /W4)
else()
    target_compile_options(tbccl_bucketed_allreduce_bench PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(async_allreduce
    examples/async_allreduce.cpp
)

target_link_libraries(async_allreduce
    PRIVATE
        tbccl
        Threads::Threads
)

if(TBCCL_ENABLE_CUDA)
    target_link_libraries(async_allreduce PRIVATE tbccl_tensor_backend)
    target_compile_definitions(async_allreduce PRIVATE TBCCL_EXAMPLE_ENABLE_CUDA)
endif()

if(TBCCL_ENABLE_METAL)
    target_link_libraries(async_allreduce PRIVATE "-framework Metal" "-framework Foundation")
    target_compile_definitions(async_allreduce PRIVATE TBCCL_EXAMPLE_ENABLE_METAL)
    set_source_files_properties(
        examples/async_allreduce.cpp
        PROPERTIES
            LANGUAGE OBJCXX
            COMPILE_FLAGS "-fobjc-arc"
    )
endif()

if(MSVC)
    target_compile_options(async_allreduce PRIVATE /W4)
else()
    target_compile_options(async_allreduce PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_async_overlap_bench
    benchmarks/async_overlap_bench.cpp
)

target_link_libraries(tbccl_async_overlap_bench
    PRIVATE
        tbccl
)

target_include_directories(tbccl_async_overlap_bench
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
)

if(MSVC)
    target_compile_options(tbccl_async_overlap_bench PRIVATE /W4)
else()
    target_compile_options(tbccl_async_overlap_bench PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(tbccl_async_raw_transport_bench
    benchmarks/async_raw_transport_bench.cpp
)

target_link_libraries(tbccl_async_raw_transport_bench
    PRIVATE
        tbccl
)

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

    add_executable(peer_capabilities_test
        tests/peer_capabilities_test.cpp
    )

    target_link_libraries(peer_capabilities_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(peer_capabilities_test PRIVATE /W4)
    else()
        target_compile_options(peer_capabilities_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME peer_capabilities_test
        COMMAND peer_capabilities_test
    )

    add_executable(async_transfer_test
        tests/async_transfer_test.cpp
    )

    target_link_libraries(async_transfer_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(async_transfer_test PRIVATE /W4)
    else()
        target_compile_options(async_transfer_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME async_transfer_test
        COMMAND async_transfer_test
    )
    set_tests_properties(async_transfer_test PROPERTIES TIMEOUT 60)

    add_executable(hetero_allreduce_test
        tests/hetero_allreduce_test.cpp
    )

    target_link_libraries(hetero_allreduce_test
        PRIVATE
            tbccl_tensor_backend
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(hetero_allreduce_test PRIVATE /W4)
    else()
        target_compile_options(hetero_allreduce_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME hetero_allreduce_test
        COMMAND hetero_allreduce_test
    )
    set_tests_properties(hetero_allreduce_test PROPERTIES TIMEOUT 60)

    add_executable(bucket_allreduce_worker_test
        tests/bucket_allreduce_worker_test.cpp
    )

    target_link_libraries(bucket_allreduce_worker_test
        PRIVATE
            tbccl_tensor_backend
            tbccl
            Threads::Threads
    )

    target_include_directories(bucket_allreduce_worker_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
    )

    if(MSVC)
        target_compile_options(bucket_allreduce_worker_test PRIVATE /W4)
    else()
        target_compile_options(bucket_allreduce_worker_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME bucket_allreduce_worker_test
        COMMAND bucket_allreduce_worker_test
    )
    set_tests_properties(bucket_allreduce_worker_test PROPERTIES TIMEOUT 60)

    add_executable(raw_io_worker_test
        tests/raw_io_worker_test.cpp
    )

    target_link_libraries(raw_io_worker_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(raw_io_worker_test PRIVATE /W4)
    else()
        target_compile_options(raw_io_worker_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME raw_io_worker_test
        COMMAND raw_io_worker_test
    )
    set_tests_properties(raw_io_worker_test PROPERTIES TIMEOUT 60)

    add_executable(tensor_comm_worker_no_staging_test
        tests/tensor_comm_worker_no_staging_test.cpp
    )

    target_link_libraries(tensor_comm_worker_no_staging_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(tensor_comm_worker_no_staging_test PRIVATE /W4)
    else()
        target_compile_options(tensor_comm_worker_no_staging_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME tensor_comm_worker_no_staging_test
        COMMAND tensor_comm_worker_no_staging_test
    )
    set_tests_properties(tensor_comm_worker_no_staging_test PROPERTIES TIMEOUT 60)

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

    add_executable(rank_directory_test
        tests/rank_directory_test.cpp
    )
    target_link_libraries(rank_directory_test PRIVATE tbccl Threads::Threads)
    if(MSVC)
        target_compile_options(rank_directory_test PRIVATE /W4)
    else()
        target_compile_options(rank_directory_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME rank_directory_test COMMAND rank_directory_test)

    add_executable(wire_protocol_test
        tests/wire_protocol_test.cpp
    )
    target_include_directories(wire_protocol_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
    target_link_libraries(wire_protocol_test PRIVATE tbccl Threads::Threads)
    if(MSVC)
        target_compile_options(wire_protocol_test PRIVATE /W4)
    else()
        target_compile_options(wire_protocol_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME wire_protocol_test COMMAND wire_protocol_test)

    add_executable(bootstrap_config_test
        tests/bootstrap_config_test.cpp
    )
    target_include_directories(bootstrap_config_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
    target_link_libraries(bootstrap_config_test PRIVATE tbccl Threads::Threads)
    if(MSVC)
        target_compile_options(bootstrap_config_test PRIVATE /W4)
    else()
        target_compile_options(bootstrap_config_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME bootstrap_config_test COMMAND bootstrap_config_test)

    # Phase 50 multi-rank tests: dynamic ports only (kernel-assigned, pre-bound listeners), so they need no shared-port lock.
    add_executable(communicator_mesh_test
        tests/communicator_mesh_test.cpp
    )
    target_link_libraries(communicator_mesh_test PRIVATE tbccl Threads::Threads)
    if(MSVC)
        target_compile_options(communicator_mesh_test PRIVATE /W4)
    else()
        target_compile_options(communicator_mesh_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME communicator_mesh_test COMMAND communicator_mesh_test)
    list(APPEND tbccl_dynamic_port_tests communicator_mesh_test)

    add_executable(communicator_p2p_nrank_test
        tests/communicator_p2p_nrank_test.cpp
    )
    target_link_libraries(communicator_p2p_nrank_test PRIVATE tbccl Threads::Threads)
    if(MSVC)
        target_compile_options(communicator_p2p_nrank_test PRIVATE /W4)
    else()
        target_compile_options(communicator_p2p_nrank_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME communicator_p2p_nrank_test COMMAND communicator_p2p_nrank_test)
    set_tests_properties(communicator_p2p_nrank_test PROPERTIES TIMEOUT 240)
    list(APPEND tbccl_dynamic_port_tests communicator_p2p_nrank_test)

    add_executable(communicator_nrank_collectives_test
        tests/communicator_nrank_collectives_test.cpp
    )
    target_link_libraries(communicator_nrank_collectives_test PRIVATE tbccl Threads::Threads)
    if(MSVC)
        target_compile_options(communicator_nrank_collectives_test PRIVATE /W4)
    else()
        target_compile_options(communicator_nrank_collectives_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME communicator_nrank_collectives_test COMMAND communicator_nrank_collectives_test)
    set_tests_properties(communicator_nrank_collectives_test PROPERTIES TIMEOUT 300)
    list(APPEND tbccl_dynamic_port_tests communicator_nrank_collectives_test)

    add_executable(communicator_nrank_failure_test
        tests/communicator_nrank_failure_test.cpp
    )
    target_link_libraries(communicator_nrank_failure_test PRIVATE tbccl Threads::Threads)
    if(MSVC)
        target_compile_options(communicator_nrank_failure_test PRIVATE /W4)
    else()
        target_compile_options(communicator_nrank_failure_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME communicator_nrank_failure_test COMMAND communicator_nrank_failure_test)
    set_tests_properties(communicator_nrank_failure_test PROPERTIES TIMEOUT 320 ENVIRONMENT "TSAN_OPTIONS=die_after_fork=0")
    list(APPEND tbccl_dynamic_port_tests communicator_nrank_failure_test)

    add_executable(collective_protocol_test
        tests/collective_protocol_test.cpp
    )
    target_include_directories(collective_protocol_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/core)
    target_link_libraries(collective_protocol_test PRIVATE tbccl Threads::Threads)
    if(MSVC)
        target_compile_options(collective_protocol_test PRIVATE /W4)
    else()
        target_compile_options(collective_protocol_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME collective_protocol_test COMMAND collective_protocol_test)

    add_executable(datatype_metadata_test
        tests/datatype_metadata_test.cpp
    )

    target_link_libraries(datatype_metadata_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(datatype_metadata_test PRIVATE /W4)
    else()
        target_compile_options(datatype_metadata_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME datatype_metadata_test
        COMMAND datatype_metadata_test
    )

    add_executable(reduction_low_precision_test
        tests/reduction_low_precision_test.cpp
    )

    target_link_libraries(reduction_low_precision_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    # low_precision.hpp / reduction_internal.hpp / all_reduce_internal.hpp are not part of the public API; this path is private to
    # this one test target (not installed/exported).
    target_include_directories(reduction_low_precision_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    if(MSVC)
        target_compile_options(reduction_low_precision_test PRIVATE /W4)
    else()
        target_compile_options(reduction_low_precision_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME reduction_low_precision_test
        COMMAND reduction_low_precision_test
    )
    set_tests_properties(reduction_low_precision_test PROPERTIES TIMEOUT 300)

    add_executable(reduction_int8_test
        tests/reduction_int8_test.cpp
    )

    target_link_libraries(reduction_int8_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    target_include_directories(reduction_int8_test
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
            ${CMAKE_CURRENT_SOURCE_DIR}/tests
    )

    if(MSVC)
        target_compile_options(reduction_int8_test PRIVATE /W4)
    else()
        target_compile_options(reduction_int8_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME reduction_int8_test
        COMMAND reduction_int8_test
    )
    set_tests_properties(reduction_int8_test PROPERTIES TIMEOUT 300)

    add_executable(byte_transport_formats_test
        tests/byte_transport_formats_test.cpp
    )

    target_link_libraries(byte_transport_formats_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(byte_transport_formats_test PRIVATE /W4)
    else()
        target_compile_options(byte_transport_formats_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME byte_transport_formats_test
        COMMAND byte_transport_formats_test
    )
    set_tests_properties(byte_transport_formats_test PROPERTIES TIMEOUT 300)

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

    add_executable(tensor_backend_async_adapter_test
        tests/tensor_backend_async_adapter_test.cpp
    )

    target_link_libraries(tensor_backend_async_adapter_test
        PRIVATE
            tbccl_tensor_backend
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(tensor_backend_async_adapter_test PRIVATE /W4)
    else()
        target_compile_options(tensor_backend_async_adapter_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME tensor_backend_async_adapter_test
        COMMAND tensor_backend_async_adapter_test
    )
    set_tests_properties(tensor_backend_async_adapter_test PROPERTIES TIMEOUT 60)

    add_executable(communicator_abort_test tests/communicator_abort_test.cpp)
    target_link_libraries(communicator_abort_test PRIVATE tbccl Threads::Threads)
    add_test(NAME communicator_abort_test COMMAND communicator_abort_test)
    set_tests_properties(communicator_abort_test PROPERTIES TIMEOUT 120)

    add_executable(communicator_collectives_test
        tests/communicator_collectives_test.cpp
    )
    target_link_libraries(communicator_collectives_test PRIVATE tbccl Threads::Threads)
    if(MSVC)
        target_compile_options(communicator_collectives_test PRIVATE /W4)
    else()
        target_compile_options(communicator_collectives_test PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME communicator_collectives_test COMMAND communicator_collectives_test)
    set_tests_properties(communicator_collectives_test PROPERTIES TIMEOUT 120)

    add_executable(communicator_test
        tests/communicator_test.cpp
    )

    target_link_libraries(communicator_test
        PRIVATE
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(communicator_test PRIVATE /W4)
    else()
        target_compile_options(communicator_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME communicator_test
        COMMAND communicator_test
    )
    set_tests_properties(communicator_test PROPERTIES TIMEOUT 60)

    add_executable(metal_shared_direct_async_backend_test
        tests/metal_shared_direct_async_backend_test.cpp
    )

    target_link_libraries(metal_shared_direct_async_backend_test
        PRIVATE
            tbccl_tensor_backend
            tbccl
            Threads::Threads
    )

    if(MSVC)
        target_compile_options(metal_shared_direct_async_backend_test PRIVATE /W4)
    else()
        target_compile_options(metal_shared_direct_async_backend_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME metal_shared_direct_async_backend_test
        COMMAND metal_shared_direct_async_backend_test
    )
    set_tests_properties(metal_shared_direct_async_backend_test PROPERTIES TIMEOUT 60)

    # Phase 18 (Part H): exercises tbccl_tensor_transfer_bench's
    # --timing-scope ready/produce CLI behavior by spawning the actual
    # compiled binary -- its logic lives in that .cpp's anonymous
    # namespace, so this is the CLI's real testable surface. Host
    # backend only, so no CUDA/Metal dependency.
    add_executable(tensor_timing_scope_test
        tests/tensor_timing_scope_test.cpp
    )

    target_compile_definitions(tensor_timing_scope_test
        PRIVATE
            TBCCL_TENSOR_BENCH_PATH="$<TARGET_FILE:tbccl_tensor_transfer_bench>"
    )

    # $<TARGET_FILE:...> only supplies the path; it does not itself
    # make `cmake --build` build that target first, so this test could
    # otherwise run against a stale or missing binary.
    add_dependencies(tensor_timing_scope_test tbccl_tensor_transfer_bench)

    if(MSVC)
        target_compile_options(tensor_timing_scope_test PRIVATE /W4)
    else()
        target_compile_options(tensor_timing_scope_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
    endif()

    add_test(
        NAME tensor_timing_scope_test
        COMMAND tensor_timing_scope_test
    )

    if(TBCCL_ENABLE_CUDA)
        add_executable(tensor_cuda_test
            tests/tensor_cuda_test.cpp
        )

        target_link_libraries(tensor_cuda_test
            PRIVATE
                tbccl_tensor_backend
                tbccl
                Threads::Threads
        )

        if(MSVC)
            target_compile_options(tensor_cuda_test PRIVATE /W4)
        else()
            target_compile_options(tensor_cuda_test PRIVATE
                -Wall
                -Wextra
                -Wpedantic
            )
        endif()

        add_test(
            NAME tensor_cuda_test
            COMMAND tensor_cuda_test
        )

        add_executable(cuda_chunked_async_backend_test
            tests/cuda_chunked_async_backend_test.cpp
        )

        target_link_libraries(cuda_chunked_async_backend_test
            PRIVATE
                tbccl_tensor_backend
                tbccl
                Threads::Threads
        )

        if(MSVC)
            target_compile_options(cuda_chunked_async_backend_test PRIVATE /W4)
        else()
            target_compile_options(cuda_chunked_async_backend_test PRIVATE
                -Wall
                -Wextra
                -Wpedantic
            )
        endif()

        add_test(
            NAME cuda_chunked_async_backend_test
            COMMAND cuda_chunked_async_backend_test
        )

        add_executable(cuda_bucket_compute_test
            tests/cuda_bucket_compute_test.cpp
        )

        target_link_libraries(cuda_bucket_compute_test
            PRIVATE
                tbccl_tensor_backend
                tbccl
                Threads::Threads
        )

        if(MSVC)
            target_compile_options(cuda_bucket_compute_test PRIVATE /W4)
        else()
            target_compile_options(cuda_bucket_compute_test PRIVATE
                -Wall
                -Wextra
                -Wpedantic
            )
        endif()

        add_test(
            NAME cuda_bucket_compute_test
            COMMAND cuda_bucket_compute_test
        )

        add_executable(communicator_cuda_staging_test tests/communicator_cuda_staging_test.cpp)
        target_include_directories(communicator_cuda_staging_test PRIVATE ${CMAKE_SOURCE_DIR}/benchmarks/tensor)
        target_link_libraries(communicator_cuda_staging_test PRIVATE tbccl_tensor_backend CUDA::cudart Threads::Threads)
        add_test(NAME communicator_cuda_staging_test COMMAND communicator_cuda_staging_test)
        set_tests_properties(communicator_cuda_staging_test PROPERTIES TIMEOUT 300)

        add_executable(cuda_submit_bench benchmarks/tensor/cuda_submit_bench.cpp)
        target_link_libraries(cuda_submit_bench PRIVATE tbccl_tensor_backend CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
        add_library(cuda_alloc_interpose SHARED benchmarks/tensor/cuda_alloc_interpose.cpp)
        target_link_libraries(cuda_alloc_interpose PRIVATE ${CMAKE_DL_LIBS})

        add_executable(communicator_collectives_cuda_test
            tests/communicator_collectives_cuda_test.cpp
        )
        target_link_libraries(communicator_collectives_cuda_test PRIVATE tbccl_tensor_backend CUDA::cudart Threads::Threads)
        add_test(NAME communicator_collectives_cuda_test COMMAND communicator_collectives_cuda_test)
        set_tests_properties(communicator_collectives_cuda_test PROPERTIES TIMEOUT 120)

        add_executable(communicator_nrank_cuda_test
            tests/communicator_nrank_cuda_test.cpp
        )
        target_link_libraries(communicator_nrank_cuda_test PRIVATE tbccl_tensor_backend tbccl CUDA::cudart Threads::Threads)
        add_test(NAME communicator_nrank_cuda_test COMMAND communicator_nrank_cuda_test)
        set_tests_properties(communicator_nrank_cuda_test PROPERTIES TIMEOUT 300)
        list(APPEND tbccl_dynamic_port_tests communicator_nrank_cuda_test)

        add_executable(reduction_low_precision_cuda_test
            tests/reduction_low_precision_cuda_test.cpp
        )
        target_link_libraries(reduction_low_precision_cuda_test PRIVATE tbccl_tensor_backend tbccl CUDA::cudart Threads::Threads)
        target_include_directories(reduction_low_precision_cuda_test
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src/collectives
                ${CMAKE_CURRENT_SOURCE_DIR}/tests
        )
        add_test(NAME reduction_low_precision_cuda_test COMMAND reduction_low_precision_cuda_test)
        set_tests_properties(reduction_low_precision_cuda_test PROPERTIES TIMEOUT 300)

        add_executable(byte_transport_formats_cuda_test
            tests/byte_transport_formats_test.cpp
        )
        target_compile_definitions(byte_transport_formats_cuda_test PRIVATE BYTE_TEST_WITH_CUDA)
        target_link_libraries(byte_transport_formats_cuda_test PRIVATE tbccl_tensor_backend tbccl CUDA::cudart Threads::Threads)
        add_test(NAME byte_transport_formats_cuda_test COMMAND byte_transport_formats_cuda_test)
        set_tests_properties(byte_transport_formats_cuda_test PROPERTIES TIMEOUT 300)

        add_executable(communicator_cuda_test
            tests/communicator_cuda_test.cpp
        )

        target_link_libraries(communicator_cuda_test
            PRIVATE
                tbccl_tensor_backend
                tbccl
                Threads::Threads
        )

        if(MSVC)
            target_compile_options(communicator_cuda_test PRIVATE /W4)
        else()
            target_compile_options(communicator_cuda_test PRIVATE
                -Wall
                -Wextra
                -Wpedantic
            )
        endif()

        add_test(
            NAME communicator_cuda_test
            COMMAND communicator_cuda_test
        )
        set_tests_properties(communicator_cuda_test PROPERTIES TIMEOUT 180)

        add_executable(cuda_reduce_backend_test
            tests/cuda_reduce_backend_test.cpp
        )

        target_link_libraries(cuda_reduce_backend_test
            PRIVATE
                tbccl_tensor_backend
                tbccl
                Threads::Threads
        )

        if(MSVC)
            target_compile_options(cuda_reduce_backend_test PRIVATE /W4)
        else()
            target_compile_options(cuda_reduce_backend_test PRIVATE
                -Wall
                -Wextra
                -Wpedantic
            )
        endif()

        add_test(
            NAME cuda_reduce_backend_test
            COMMAND cuda_reduce_backend_test
        )

        # Phase 19 (Part J): exercises cuda_sync_bench's synchronization-
        # mode/H2D/D2H correctness and CLI validation by spawning the
        # actual compiled binary -- it is a standalone diagnostic tool
        # (a plain main(), not a library), so this is its real
        # testable surface, exactly as tensor_timing_scope_test does
        # for tbccl_tensor_transfer_bench.
        add_executable(cuda_sync_bench_test
            tests/cuda_sync_bench_test.cpp
        )

        target_compile_definitions(cuda_sync_bench_test
            PRIVATE
                TBCCL_CUDA_SYNC_BENCH_PATH="$<TARGET_FILE:cuda_sync_bench>"
        )

        add_dependencies(cuda_sync_bench_test cuda_sync_bench)

        if(MSVC)
            target_compile_options(cuda_sync_bench_test PRIVATE /W4)
        else()
            target_compile_options(cuda_sync_bench_test PRIVATE
                -Wall
                -Wextra
                -Wpedantic
            )
        endif()

        add_test(
            NAME cuda_sync_bench_test
            COMMAND cuda_sync_bench_test
        )
    endif()

    if(TBCCL_ENABLE_METAL)
        add_executable(tensor_metal_test
            tests/tensor_metal_test.mm
        )

        target_link_libraries(tensor_metal_test
            PRIVATE
                tbccl_tensor_backend
                tbccl
                Threads::Threads
        )

        target_compile_options(tensor_metal_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )

        add_test(
            NAME tensor_metal_test
            COMMAND tensor_metal_test
        )

        add_executable(communicator_metal_test
            tests/communicator_metal_test.mm
        )

        target_link_libraries(communicator_metal_test
            PRIVATE
                tbccl
                Threads::Threads
                "-framework Metal"
                "-framework Foundation"
        )

        target_compile_options(communicator_metal_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )

        add_test(
            NAME communicator_metal_test
            COMMAND communicator_metal_test
        )
        set_tests_properties(communicator_metal_test PROPERTIES TIMEOUT 60)

        add_executable(metal_reduce_backend_test
            tests/metal_reduce_backend_test.cpp
        )

        target_link_libraries(metal_reduce_backend_test
            PRIVATE
                tbccl_tensor_backend
                tbccl
                Threads::Threads
        )

        target_compile_options(metal_reduce_backend_test PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )

        add_test(
            NAME metal_reduce_backend_test
            COMMAND metal_reduce_backend_test
        )
    endif()

    # Phase 20 scripts and opt-in benchmark telemetry; no physical TB4 required.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME tb4_sweep_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tb4_sweep_test.py)
        add_test(NAME tb4_health_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tb4_health_test.py)
        add_test(NAME tensor_diagnostics_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tensor_diagnostics_test.py
                    $<TARGET_FILE:tbccl_tensor_transfer_bench>)
        add_test(NAME tb4_trace_analysis_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tb4_trace_analysis_test.py)
        add_test(NAME tb4_receive_trace_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tb4_receive_trace_test.py)
        add_test(NAME tb4_runtime_state_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/tb4_runtime_state_test.py)
        add_test(NAME analyze_training_interference_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/analyze_training_interference_test.py)
        add_test(NAME analyze_phase32_async_transport_test
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/analyze_phase32_async_transport_test.py)
        set_tests_properties(tb4_sweep_test tb4_health_test tensor_diagnostics_test tb4_trace_analysis_test tb4_receive_trace_test tb4_runtime_state_test analyze_training_interference_test analyze_phase32_async_transport_test
            PROPERTIES TIMEOUT 90)
    endif()

    # Every test binds fixed loopback ports (and the Python harness tests spawn tools on dynamic ones), and the ranges are dense: pairs of tests hard-code the
    # SAME base (28820, 28900, 29100, 29550, 32100), and Communicator-based tests also bind a data port at control port + 1000 that lands inside other
    # tests' ranges (e.g. communicator_test 29100 -> 30100 vs reduce_test 30100). Run in parallel they intermittently fail with "bind failed ... Address
    # already in use" (seen as a Mac hetero_allreduce_test failure in Phase 48, tensor_backend_async_adapter_test and, in Phase 49, reduce_test on
    # Linux). One shared RESOURCE_LOCK makes ctest run them one at a time whatever -j says, and covers tests added later automatically.
    # Phase 50 tests allocate every port dynamically (listed in tbccl_dynamic_port_tests) and are left out of the lock.
    get_property(tbccl_all_tests DIRECTORY PROPERTY TESTS)
    foreach(t IN LISTS tbccl_all_tests)
        if(NOT t IN_LIST tbccl_dynamic_port_tests)
            set_property(TEST ${t} APPEND PROPERTY RESOURCE_LOCK "tbccl_loopback_ports")
        endif()
    endforeach()

endif()

# ---------------------------------------------------------------------------
# Phase 41 Part AJ: install/export support for the core tbccl library --
# the actual public API surface (include/tbccl/*.hpp + libtbccl). The
# optional CUDA/Metal device component (tbccl_tensor_backend,
# benchmarks/tensor/), benchmarks/, examples/, and tests/ are NOT
# installed; they remain source-tree/vendored this phase (a deliberate,
# documented scope limit -- see docs/public_api.md). A host-only
# external consumer needs nothing beyond what this block installs.
# ---------------------------------------------------------------------------

include(CMakePackageConfigHelpers)

install(TARGETS tbccl
    EXPORT TBCCLTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(TBCCL_ENABLE_CUDA)
    install(TARGETS tbccl_cuda
        EXPORT TBCCLCudaTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    install(EXPORT TBCCLCudaTargets
        FILE TBCCLCudaTargets.cmake
        NAMESPACE TBCCL::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TBCCL
    )
endif()

install(DIRECTORY include/tbccl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(EXPORT TBCCLTargets
    FILE TBCCLTargets.cmake
    NAMESPACE TBCCL::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TBCCL
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/TBCCLConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TBCCLConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/TBCCLConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TBCCL
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/TBCCLConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/TBCCLConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TBCCL
)
