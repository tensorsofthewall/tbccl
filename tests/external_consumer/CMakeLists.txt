# Phase 49: an out-of-tree consumer of an INSTALLED TBCCL (no source-tree headers, no add_subdirectory). Built by scripts/check_external_consumer.sh.
cmake_minimum_required(VERSION 3.20)
project(tbccl_external_consumer CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Phase 50: the package uses SameMajorVersion, so a consumer written against an older 0.x release (default 0.1) keeps finding a newer 0.y install.
set(TBCCL_REQUIRED_VERSION 0.1 CACHE STRING "minimum TBCCL version this consumer asks for")
find_package(TBCCL ${TBCCL_REQUIRED_VERSION} CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE TBCCL::tbccl Threads::Threads)

if(TBCCL_cuda_FOUND)
    # TBCCL::tbccl_cuda only links CUDA::cudart for its own use; this consumer also calls the CUDA runtime directly (cudaMalloc/cudaMemcpy to hold a device buffer),
    # so it finds the toolkit itself. TBCCL's public headers never include CUDA headers.
    find_package(CUDAToolkit REQUIRED)
    add_executable(consumer_cuda main.cpp)
    target_compile_definitions(consumer_cuda PRIVATE CONSUMER_WITH_CUDA)
    target_link_libraries(consumer_cuda PRIVATE TBCCL::tbccl_cuda TBCCL::tbccl CUDA::cudart Threads::Threads)
endif()
message(STATUS "TBCCL ${TBCCL_VERSION}; CUDA component found: ${TBCCL_cuda_FOUND}")
