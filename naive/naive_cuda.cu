#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void test1_kernel(int* result) {
    *result = 1000;
}

torch::Tensor matmul(torch::Tensor in1, torch::Tensor in2) {
    torch::Tensor in1 = in1.contiguous();
    torch::Tensor in2 = in2.contiguous();

    const int M = in1.size(0);
    const int N = in1.size(1);
    const int K = in2.size(1);

    auto result = torch::zeros({3}, torch::device(torch::kCUDA).dtype(torch::kInt32));
    
    // Create a CPU tensor with M, N, K for debugging
    auto debug_tensor = torch::tensor({M, N, K}, torch::dtype(torch::kInt32));

    // Copy to GPU result tensor
    result.copy_(debug_tensor.to(torch::kCUDA));

    // test1_kernel<<<1, 1>>>(result.data_ptr<int>());
    // cudaDeviceSynchronize();  // ensure completion
    
    return result;
}
