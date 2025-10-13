
#include <iostream>

#include "add_vector.h"

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

void Practices::add_vector(int N, int threadsPerBlock, const float* h_A,
                           const float* h_B, const float* h_C) {
    size_t size = N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    std::cout << "hello\n";
}