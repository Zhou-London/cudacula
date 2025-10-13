cmake_minimum_required(VERSION 3.24)

project(cudacula LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(cudacula
    src/main.cu
)

target_include_directories(cudacula PRIVATE
    src
    src/practices
)

target_sources(cudacula PRIVATE
    src/practices/add_vector.cu
)

set_target_properties(cudacula PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)