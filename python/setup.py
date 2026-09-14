import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT_DIR = Path(__file__).resolve().parents[1]

# Some Python/setuptools toolchains use CC for compiling every extension
# source, including .cc files. Select a C++ driver only when no compiler was
# explicitly configured by the user.
if os.name != "nt" and not os.environ.get("CC"):
    os.environ["CC"] = os.environ.get("CXX", "c++")

# GCC 9 names the C++20 mode c++2a; this spelling is also accepted by newer
# GCC/Clang releases. Keep the standard flag here because pybind11's helper
# does not detect C++20 on older compilers.
compile_args = ["-O3", "-std=c++2a", "-fPIC"]
link_args = []

if os.name != "nt":
    compile_args.append("-fopenmp")
    link_args.append("-fopenmp")

native_simd = os.environ.get("LSSG_ENABLE_NATIVE_SIMD", "ON").upper()
if native_simd not in {"0", "OFF", "FALSE", "NO"} and os.name != "nt":
    compile_args.extend(["-march=native", "-finline-functions"])


ext_modules = [
    Pybind11Extension(
        "pylssg._pylssg",
        ["lssg_bindings.cc"],
        include_dirs=[str(ROOT_DIR)],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        language="c++",
        cxx_std=None,
    )
]


setup(
    name="pylssg",
    version="0.1.0",
    description="Python bindings for the LSSG label-filtering ANN index",
    packages=["pylssg"],
    package_dir={"": "."},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
