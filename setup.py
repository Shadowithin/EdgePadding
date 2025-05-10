import os
import subprocess
import sys
import platform

import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import edge_padding

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        ext_name = os.path.basename(self.get_ext_fullpath(ext.name))
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        cfg = "Release" if not self.debug else "Debug"

        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)

        subprocess.check_call(["cmake", "-B", build_temp, "-A", "x64", "-DBUILD_TEST=OFF"], cwd=cur_dir)
        subprocess.check_call(["cmake", "--build", build_temp, "--config", cfg], cwd=cur_dir)
        # subprocess.check_call(["cmake", "--install", build_temp, "--config", cfg], cwd=cur_dir)

        lib_path = os.path.abspath(os.path.join(build_temp, cfg, ext_name))
        print("lib_path: ", lib_path)

        ext_path = os.path.abspath(os.path.join(ext_dir, "edge_padding", "PyEdgePadding", ext_name))
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        print("ext_path: ", ext_path)

        self.copy_file(lib_path, ext_path)

setup(
    name="edge_padding",
    version=edge_padding.__version__,
    author="Yu Liangyu",
    author_email="withindigo96@gmail.com",
    description="CUDA texture edge padding/dilation, http://wiki.polycount.com/wiki/Edge_padding.",
    packages=["edge_padding", "edge_padding.PyEdgePadding"],
    package_data={
        'edge_padding': [
            'src/*.h',
            'src/*.inl',
            'src/*.cu',
            'src/*.cpp',
            'binding/*.h',
            'binding/*.inl',
            'binding/*.cu',
            'binding/*.cpp',
        ] 
    },
    include_package_data=True,
    ext_modules=[CMakeExtension("PyEdgePadding")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Environment :: GPU :: NVIDIA CUDA :: 12",
        "Environment :: Win32 (MS Windows)",
    ],
    python_requires=">=3.7",
)
