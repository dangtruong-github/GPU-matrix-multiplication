#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

#define MAX_BLOCK_SIZE 512

__global__ void matmul_kernel(
    float* in1, float* in2, float* out,
    const int M, const int N, const int K
) {
    const int pos = blockIdx.x * blockDim.x + threadIdx.x;
    const int idxA = pos / N;  // Row index in output (0..M-1)
    const int idxB = pos % N;  // Column index in output (0..N-1)

    if (pos >= M * N) return;

    float result = 0.0f;
    for (int i = 0; i < K; i++) {
        // Correct indices: in1 is MxK (row idxA, column i) => idxA*K + i
        //                  in2 is KxN (row i, column idxB) => i*N + idxB
        result += in1[idxA * K + i] * in2[i * N + idxB];
    }

    out[pos] = result;
}

torch::Tensor matmul(torch::Tensor in1, torch::Tensor in2) {
    in1 = in1.contiguous();
    in2 = in2.contiguous();

    const int M = in1.size(0);
    const int K = in1.size(1);
    const int N = in2.size(1);

    auto result = torch::zeros({M, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    
    int max_grid_needed = (M * N + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;

    dim3 blockSize(MAX_BLOCK_SIZE);
    dim3 gridSize(max_grid_needed);

    matmul_kernel<<<gridSize, blockSize>>>(
        in1.data_ptr<float>(),
        in2.data_ptr<float>(),
        result.contiguous().data_ptr<float>(),
        M, N, K
    );
    cudaDeviceSynchronize();  // Ensure kernel completes
    
    return result;
}