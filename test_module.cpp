#include <pybind11/pybind11.h>

int test1() {
    return 0;
}

PYBIND11_MODULE(test_module, m) {
    m.def("test1", &test1, "A function that returns 0");
}
