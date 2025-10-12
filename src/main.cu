#include <iostream>

__global__ void helloCUDA(){
    printf("Hello CUDA\n");
}

int main(){
    helloCUDA<<<1,5>>>();
    cudaDeviceSynchronize();
    return 0;
}