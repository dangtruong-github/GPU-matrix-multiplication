#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

__global__ void test1_kernel(int* result) {
    *result = 1000;  // CUDA kernel sets value to 0
}

int test1() {
    int* d_result;
    int h_result;

    cudaMalloc(&d_result, sizeof(int));
    test1_kernel<<<1, 1>>>(d_result);
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return h_result;
}

PYBIND11_MODULE(naive, m) {
    m.def("test1", &test1, "A CUDA function that returns 0");
}
