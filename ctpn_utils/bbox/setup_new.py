import numpy as np

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize


numpy_include = np.get_include()

setup(ext_modules=cythonize("bbox.pyx"),include_dirs=[numpy_include])
setup(ext_modules=cythonize("nms.pyx"),include_dirs=[numpy_include])

