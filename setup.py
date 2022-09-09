from distutils.core import setup
from distutils.extension import Extension

utils_ext = Extension(
    "c_utils",
    sources=["turbo_ninja_ga/C/c_utils.cpp"],
    libraries=["boost_python"],
)

setup(name="turbo_ninja_ga", version="0.1", ext_modules=[utils_ext])
