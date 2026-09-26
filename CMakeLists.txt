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

add_library(tbccl STATIC
    src/transport/tcp.cpp
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

endif()
