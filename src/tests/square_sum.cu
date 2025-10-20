
#include <cuda_runtime_api.h>

#include <cstdio>

#include "square_sum.h"

constexpr int ELEMENTS_PER_THREAD = 4;

__global__ void square_sum_kernel(const float* A, const float* B, float* C,
                                  int N) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * ELEMENTS_PER_THREAD;

    float a[ELEMENTS_PER_THREAD];
    float b[ELEMENTS_PER_THREAD];

#pragma unroll
    for (int k = 0; k < ELEMENTS_PER_THREAD; ++k) {
        int idx = base + k;
        if (idx < N) {
            a[k] = A[idx];
            b[k] = B[idx];
        }
    }

#pragma unroll
    for (int k = 0; k < ELEMENTS_PER_THREAD; ++k) {
        int idx = base + k;
        if (idx < N) C[idx] = a[k] * a[k] + b[k] * b[k];
    }
}

void Practices::square_sum() {
    int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * ELEMENTS_PER_THREAD - 1) /
                        (threadsPerBlock * ELEMENTS_PER_THREAD);

    square_sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Vector A[0] = %f\n", h_A[0]);
    printf("Vector B[0] = %f\n", h_B[0]);
    printf("Vector C[0] = %f\n", h_C[0]);
    printf("Vector A[N-1] = %f\n", h_A[N - 1]);
    printf("Vector B[N-1] = %f\n", h_B[N - 1]);
    printf("Vector C[N-1] = %f\n", h_C[N - 1]);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}