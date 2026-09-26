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

add_executable(tb_pingpong
    benchmarks/tcp_pingpong.cpp
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