import glob
import os
import time
from itertools import chain

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

SRC_DIR = os.path.abspath(os.path.dirname(__file__))


def get_cpp_or_cuda_sources(src_dir):
    files = glob.glob(f'{src_dir}/*.cu') + glob.glob(f'{src_dir}/*.cpp')
    print(f'\033[31mFind {len(files)} cu/cpp files in directory: {src_dir}\033[0m')
    return files


setup(
    name='C_ext',
    version='2023.09',
    description='build time {}'.format(time.strftime("%y-%m-%d %H:%M:%S", time.localtime(time.time()))),
    ext_modules=[
        CUDAExtension(
            name='C_ext',
            sources=list(get_cpp_or_cuda_sources('src')) + list(get_cpp_or_cuda_sources('src/gaussian_render')),
            extra_compile_args={
                'cxx': ["-fopenmp", "-O3"],
                'nvcc': [
                    '-O3',
                    # '-rdc=true',
                    # '--ptxas-options=-v',
                ]
            },
            # define_macros=[],
            include_dirs=[os.path.join(SRC_DIR, "include")],
            # libraries=[],
            # library_dirs=[]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
