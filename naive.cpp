#include <pybind11/pybind11.h>

// Original function
int test1() {
    return 0;
}

// Pybind11 module
PYBIND11_MODULE(naive, m) {
    m.def("test1", &test1, "A function that returns 0");
}
