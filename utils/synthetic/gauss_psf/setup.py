from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cur_path = os.getcwd()

setup(
    name='gauss_psf_cuda',
    ext_modules=[
        CUDAExtension('gauss_psf_cuda', [
            f'{cur_path}/gauss_psf_cuda.cpp',
            f'{cur_path}/gauss_psf_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })