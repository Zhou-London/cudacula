cmake_minimum_required(VERSION 3.24)

project(cudacula LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RTX 5090
set(CMAKE_CUDA_ARCHITECTURES 120)

add_executable(cudacula
    src/main.cu
)

target_include_directories(cudacula PRIVATE
    src
    src/tests
    src/markets
    src/kernels
    src/pricers
)

target_sources(cudacula PRIVATE

    # Tests
    src/tests/add_vector.cu
    src/tests/mul_mat.cu
    src/tests/square_sum.cu

    # Markets
    src/markets/dummy_market_data_source.cu

    # Kernels
    src/kernels/persistent_monte_carlo.cu

    # Pricers
    src/pricers/monte_carlo_pricer.cu
)

set_target_properties(cudacula PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(cudacula PRIVATE
    NOMINMAX
)