from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Retrieve environment variables
SPUTNIK_PATH = os.getenv('SPUTNIK_PATH')
GLOG_PATH = os.getenv('GLOG_PATH')  # Default path if not set

# Configure setup
setup(
    name='DTCSpMM',
    ext_modules=[
        CUDAExtension('DTCSpMM', [
            'DTCSpMM.cpp',
            'DTCSpMM_kernel.cu',
        ],
        library_dirs=[
            os.path.join(SPUTNIK_PATH, 'build/sputnik'),
            os.path.join(GLOG_PATH, 'build'),
        ],
        libraries=['sputnik', 'glog'],
        include_dirs=[
            os.path.join(SPUTNIK_PATH, ''), 
            os.path.join(GLOG_PATH, 'build/glog'),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
