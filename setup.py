#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import pathlib
import subprocess
import sys

import setuptools
from setuptools.command import build_ext
from distutils import spawn


class Build(build_ext.build_ext):
    def run(self):  # Necessary for pip install -e.
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        source_path = pathlib.Path(__file__).parent.resolve()
        output_path = pathlib.Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        os.makedirs(self.build_temp, exist_ok=True)

        global moodist_version

        with open(output_path / "version.py", "w") as f:
            f.write('__version__ = "%s"\n' % moodist_version)

        cmake_cmd = [
            "cmake",
            str(source_path),
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=%s" % output_path,
        ]

        if "bdist_wheel" in sys.argv:
            cmake_cmd.append("-DIS_BUILDING_WHEEL=1")

        build_cmd = ["cmake", "--build", ".", "--parallel"]

        # pip install (but not python setup.py install) runs with a modified PYTHONPATH.
        # This can prevent cmake from finding the torch libraries.
        env = os.environ.copy()
        if "PYTHONPATH" in env:
            del env["PYTHONPATH"]
        try:
            subprocess.check_call(cmake_cmd, cwd=self.build_temp, env=env)
            subprocess.check_call(build_cmd, cwd=self.build_temp, env=env)
        except subprocess.CalledProcessError:
            # Don't obscure the error with a setuptools backtrace.
            sys.exit(1)


def main():

    import torch

    torch_version = torch.__version__
    if "+" in torch_version:
        torch_version = torch_version[: torch_version.index("+")]

    extra_version = ""
    if "bdist_wheel" not in sys.argv:
        extra_version = "-dev"

    global moodist_version
    moodist_version = "%s%s+torch.%s" % (
        open("version.txt").read().strip(),
        extra_version,
        torch_version,
    )

    torch_req_version = ".".join(torch_version.split(".")[:2]) + ".*"
    
    print("Building for torch==%s" % torch_req_version)

    setuptools.setup(
        name="moodist",
        version=moodist_version,
        description=("moodist"),
        long_description="",
        long_description_content_type="text/markdown",
        author="",
        url="",
        classifiers=[
            "Programming Language :: C++",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Environment :: GPU :: NVIDIA CUDA",
        ],
        packages=["moodist"],
        package_dir={"": "py"},
        ext_modules=[setuptools.Extension("moodist._C", sources=[])],
        install_requires=["torch==%s" % torch_req_version],
        cmdclass={"build_ext": Build},
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
