import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# Define the extension module
extensions = [
    Extension(
        "cython_image_processing.image_filters",
        ["src/cython_image_processing/image_filters.pyx"],
        include_dirs=[numpy.get_include()],
        language="c",
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={"language_level": 3}),
    zip_safe=False,
)
