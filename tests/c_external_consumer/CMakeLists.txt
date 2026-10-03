# Phase 52: an out-of-tree PURE-C consumer of an INSTALLED TBCCL (C language only: no C++ compiler is enabled here, no -lstdc++/-lc++ is guessed, no source-tree
# header is used). Built by scripts/check_c_external_consumer.sh.
cmake_minimum_required(VERSION 3.20)
project(tbccl_c_external_consumer C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(TBCCL_REQUIRED_VERSION 0.1 CACHE STRING "minimum TBCCL version this consumer asks for")
find_package(TBCCL ${TBCCL_REQUIRED_VERSION} CONFIG REQUIRED)
find_package(Threads REQUIRED)
if(NOT TBCCL_c_FOUND)
    message(FATAL_ERROR "this TBCCL install has no C ABI target (TBCCL::tbccl_c)")
endif()

add_executable(c_consumer main.c)
target_compile_options(c_consumer PRIVATE -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(c_consumer PRIVATE TBCCL::tbccl_c Threads::Threads)

if(TBCCL_cuda_FOUND)
    # The consumer includes the CUDA runtime itself and passes its cudaStream_t through native_handle; tbccl.h never includes CUDA.
    find_package(CUDAToolkit REQUIRED)
    add_executable(c_consumer_cuda main.c)
    target_compile_definitions(c_consumer_cuda PRIVATE CONSUMER_WITH_CUDA)
    target_compile_options(c_consumer_cuda PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_link_libraries(c_consumer_cuda PRIVATE TBCCL::tbccl_c CUDA::cudart Threads::Threads)
endif()
message(STATUS "TBCCL ${TBCCL_VERSION}; C ABI target found: ${TBCCL_c_FOUND}; CUDA component found: ${TBCCL_cuda_FOUND}")
