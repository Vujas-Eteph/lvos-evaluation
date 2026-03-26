import numpy
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

# Point towards cython code
extensions = [
    Extension(
        name="lvos_api.lvos.metric_ops._get_binary_c._get_binary_c",
        sources=["lvos_api/lvos/metric_ops/_get_binary_c/_get_binary_c.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

# Build the cython extension
setup(
    ext_modules = cythonize(extensions, compiler_directives={'language_level': "3"}),
    packages=find_packages(),
    package_data={"lvos_api": ["*.txt"]},
)