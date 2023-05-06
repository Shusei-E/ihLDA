# setup.py
"""Build Cython files
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


def main():
    """Compile Cython files
    """

    # Compiler directives documentation
    # https://github.com/cython/cython/wiki/enhancements-compilerdirectives

    # Compile options: The second one ignores assertions
    compiler_args = ["-O3", "-ffast-math"]
    # compiler_args = ["-O3", "-ffast-math", "-DCYTHON_WITHOUT_ASSERTIONS"]

    ext_modules = [
                Extension("dataread", sources=["dataread.pyx"],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=compiler_args),  # read data
                Extension("htssb_cy", sources=["htssb_cy.pyx"],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=compiler_args),
                Extension("hpy_cy", sources=["hpy_cy.pyx"],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=compiler_args),
                Extension("ihLDA_cy", sources=["ihLDA_cy.pyx"],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=compiler_args),
                Extension("node_cy", sources=["node_cy.pyx"],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=compiler_args),
                Extension("tssb_cy", sources=["tssb_cy.pyx"],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=compiler_args),
                Extension("save_model", sources=["save_model.pyx"],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=compiler_args),
                Extension("tool", sources=["tool.pyx"],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=compiler_args)
            ]

    # This is the important part. By setting this compiler directive,
    # cython will embed signature information in docstrings.
    # Sphinx then knows how to extract
    # and use those signatures.
    for e in ext_modules:
        e.compiler_directives = {"embedsignature": True,
                                 "language_level": "3"}

    setup(
        name="ihLDA",
        cmdclass={'build_ext': build_ext},
        ext_modules=ext_modules
    )


if __name__ == "__main__":
    main()
