#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

torch::Tensor matmul(torch::Tensor in1, torch::Tensor in2);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul", &matmul, "Naive MatMul (CUDA)");
}
