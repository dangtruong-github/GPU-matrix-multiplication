from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='naive',
    ext_modules=[
        CUDAExtension(
            name='naive',
            sources=['naive/naive.cpp', 'naive/naive_cuda.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
