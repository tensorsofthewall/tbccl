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

add_library(tbccl STATIC
    src/transport/tcp.cpp
    src/core/tcp_world.cpp
    src/collectives/barrier.cpp
    src/collectives/broadcast.cpp
    src/collectives/all_gather.cpp
    src/collectives/reduction.cpp
    src/collectives/reduce.cpp
    src/collectives/all_reduce.cpp
    src/collectives/reduce_scatter.cpp
    src/collectives/all_gather_ring.cpp
)

target_include_directories(tbccl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
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

endif()
