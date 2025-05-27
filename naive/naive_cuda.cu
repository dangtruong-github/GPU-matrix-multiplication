#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

#define MAX_BLOCK_SIZE 512

__global__ void matmul_kernel(
    float* in1, float* in2, float* out,
    const int M, const int N, const int K
) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxA = pos / M;
    const int idxB = pos % M;

    if (pos >= M * N) return;

    float result = 0.0f;
    for (int i=0; i<K; i++) {
        result += in1[idxA * M + i] * in2[i * K + idxB];
    }

    out[pos] = result;
}

torch::Tensor matmul(torch::Tensor in1, torch::Tensor in2) {
    in1 = in1.contiguous();
    in2 = in2.contiguous();

    const int M = in1.size(0);
    const int N = in1.size(1);
    const int K = in2.size(1);

    auto result = torch::zeros({3}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    
    // Create a CPU tensor with M, N, K for debugging
    // auto debug_tensor = torch::tensor({M, N, K}, torch::dtype(torch::kInt32));
    // Copy to GPU result tensor
    // result.copy_(debug_tensor.to(torch::kCUDA));

    int max_grid_needed = 1 + ((M * N - 1) / MAX_BLOCK_SIZE);

    dim3 blockSize(MAX_BLOCK_SIZE);
    dim3 gridSize(max_grid_needed)

    matmul_kernel<<<1, 1>>>(result.data_ptr<int>());
    cudaDeviceSynchronize();  // ensure completion
    
    return result;
}
