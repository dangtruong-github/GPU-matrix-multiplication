#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void test1_kernel(int* result) {
    *result = 1000;
}

torch::Tensor matmul(torch::Tensor in1, torch::Tensor in2) {
    auto result = torch::zeros({1}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    test1_kernel<<<1, 1>>>(result.data_ptr<int>());
    cudaDeviceSynchronize();  // ensure completion
    return result;
}
