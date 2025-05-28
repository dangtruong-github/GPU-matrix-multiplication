#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_DIM 16

__global__ void matmul_kernel(
    float* in1, float* in2, float* out,
    const int M, const int N, const int K
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float result = 0.0f;
    float c = 0.0f;
    for (int i = 0; i < K; i++) {
        // Correct indices: in1 is MxK (row idxA, column i) => idxA*K + i
        //                  in2 is KxN (row i, column idxB) => i*N + idxB
        float y = in1[row * K + i] * in2[i * N + col];
        float t = result + y;
        c = (t-result) - y;

        result = t;
    }

    out[row * N + col] = result;
}

torch::Tensor matmul(torch::Tensor in1, torch::Tensor in2) {
    in1 = in1.contiguous();
    in2 = in2.contiguous();

    const int M = in1.size(0);
    const int K = in1.size(1);
    const int N = in2.size(1);

    auto result = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize(
        (N + BLOCK_DIM - 1) / BLOCK_DIM,
        (M + BLOCK_DIM - 1) / BLOCK_DIM
    );

    matmul_kernel<<<gridSize, blockSize>>>(
        in1.data_ptr<float>(),
        in2.data_ptr<float>(),
        result.contiguous().data_ptr<float>(),
        M, N, K
    );
    cudaDeviceSynchronize();  // Ensure kernel completes
    
    return result;
}