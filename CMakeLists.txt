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

if(APPLE)
    # macOS's Local Network TCC permission is keyed to a binary's code-
    # signing identity. An ad-hoc/unsigned build gets a fresh identity
    # every rebuild, so each new binary looks like a new app and needs
    # its own manual approval. Signing every build with the same local
    # certificate keeps that identity stable across rebuilds, so one
    # approval per tool sticks. Set to the name of a "Code Signing"
    # identity in your keychain (Keychain Access > Certificate
    # Assistant > Create a Certificate...; must be trusted "Always
    # Trust" for Code Signing) to enable; leave empty to skip signing.
    set(TBCCL_CODESIGN_IDENTITY "" CACHE STRING
        "macOS code-signing identity for network-facing tbccl binaries")
endif()

function(tbccl_codesign_if_apple target)
    if(APPLE AND TBCCL_CODESIGN_IDENTITY)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND codesign --force --sign "${TBCCL_CODESIGN_IDENTITY}"
                    --identifier "com.tbccl.${target}"
                    "$<TARGET_FILE:${target}>"
            COMMENT "Code-signing ${target} with '${TBCCL_CODESIGN_IDENTITY}'"
            VERBATIM
        )
    endif()
endfunction()

add_library(tbccl STATIC
    src/transport/tcp.cpp
    src/core/tcp_world.cpp
    src/collectives/barrier.cpp
    src/collectives/broadcast.cpp
    src/collectives/all_gather.cpp
    src/collectives/reduction.cpp
    src/collectives/reduce.cpp
)

target_include_directories(tbccl
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
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

tbccl_codesign_if_apple(tb_pingpong)

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

tbccl_codesign_if_apple(tbccl_world_smoke)

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

tbccl_codesign_if_apple(tbccl_barrier_smoke)

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

tbccl_codesign_if_apple(tbccl_broadcast_smoke)

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

tbccl_codesign_if_apple(tbccl_all_gather_smoke)

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

tbccl_codesign_if_apple(tbccl_reduce_smoke)

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

endif()
