#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import pathlib
import subprocess
import sys

import setuptools
from setuptools.command import build_ext
from distutils import spawn

zip_files = {}


class Build(build_ext.build_ext):
    def run(self):  # Necessary for pip install -e.
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        source_path = pathlib.Path(__file__).parent.resolve()
        output_path = pathlib.Path(self.get_ext_fullpath(ext.name)).parent.absolute()

        os.makedirs(self.build_temp, exist_ok=True)

        os.makedirs(output_path, exist_ok=True)

        global moodist_version, moodist_cversions

        with open(output_path / "version.py", "w") as f:
            f.write('__version__ = "%s"\n' % moodist_version)
            f.write("cversions = %s\n" % str(moodist_cversions))

        if ext.name.startswith("moodist.pt"):
            os.rename(output_path / "version.py", output_path.parent / "version.py")
            z, fn = zip_files[ext.name]
            open(self.get_ext_fullpath(ext.name), "wb").write(z.open(fn).read())
            # Extract libmoodist.so to package directory (only once)
            if "moodist.libmoodist" in zip_files:
                z_lib, fn_lib = zip_files.pop("moodist.libmoodist")
                libmoodist_path = output_path.parent / "libmoodist.so"
                open(libmoodist_path, "wb").write(z_lib.open(fn_lib).read())
            # Copy serialize libraries to package directory (only once)
            if "moodist.serialize_libs" in zip_files:
                serialize_libs = zip_files.pop("moodist.serialize_libs")
                for lib_path in serialize_libs:
                    import shutil
                    dest_path = output_path.parent / os.path.basename(lib_path)
                    shutil.copy(lib_path, dest_path)
                    print(f"Copied serialize lib: {dest_path}")
            return

        cmake_cmd = [
            "cmake",
            str(source_path),
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=%s" % output_path,
        ]

        # Support pre-built core library for multi-PyTorch version builds
        if "MOODIST_PREBUILT_CORE" in os.environ:
            cmake_cmd.append("-DMOODIST_PREBUILT_CORE=%s" % os.environ["MOODIST_PREBUILT_CORE"])

        # Support explicit build magic for multi-PyTorch version builds
        if "MOODIST_BUILD_MAGIC" in os.environ:
            cmake_cmd.append("-DMOODIST_BUILD_MAGIC=%s" % os.environ["MOODIST_BUILD_MAGIC"])

        # Use minimal debug info for wheel builds (smaller binaries)
        if "MOODIST_MINIMAL_DEBUG" in os.environ or "bdist_wheel" in sys.argv:
            cmake_cmd.append("-DMINIMAL_DEBUG=ON")

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

        # If using pre-built core, copy it to output directory
        if "MOODIST_PREBUILT_CORE" in os.environ:
            import shutil
            shutil.copy(os.environ["MOODIST_PREBUILT_CORE"], output_path / "libmoodist.so")


def main():

    extra_version = ""
    if "bdist_wheel" not in sys.argv:
        extra_version = "-dev"

    global moodist_version
    global moodist_cversions

    if "MOODIST_WHL_LIST" in os.environ:
        moodist_version = "%s%s" % (
            open("version.txt").read().strip(),
            extra_version,
        )

        import zipfile

        cversions = {}

        ext_modules = []

        min_version = None
        max_version = None

        # Track libmoodist.so from first wheel (same in all wheels)
        libmoodist_source = None

        for fn in os.environ["MOODIST_WHL_LIST"].split(","):
            z = zipfile.ZipFile(fn)

            g = dict()
            exec(z.open("moodist/version.py").read(), g)

            cv = g["cversions"]

            assert len(cv) == 1

            cfn = None

            for n in z.filelist:
                if n.filename.startswith("moodist/_C."):
                    assert cfn is None
                    cfn = n.filename
                # Grab libmoodist.so from first wheel
                if libmoodist_source is None and n.filename == "moodist/libmoodist.so":
                    libmoodist_source = (z, n.filename)
            assert cfn is not None

            tv: str = next(iter(cv.keys()))

            foldername = "pt%s" % tv.replace(".", "")
            assert foldername.isalnum()

            modname = "moodist.%s._C" % foldername

            ext_modules.append(setuptools.Extension(modname, sources=[], py_limited_api=True))

            zip_files[modname] = (z, cfn)

            assert tv not in cversions
            cversions[tv] = modname

            maj, min = (int(x) for x in tv.split("."))
            v = (maj, min)
            if min_version is None or v < min_version:
                min_version = v
            if max_version is None or v >= max_version:
                max_version = (maj, min + 1)

        torch_req_version = ">=%s, <%s" % (
            ".".join(str(x) for x in min_version),
            ".".join(str(x) for x in max_version),
        )
        moodist_cversions = cversions

        # Store libmoodist.so source for extraction
        if libmoodist_source is not None:
            zip_files["moodist.libmoodist"] = libmoodist_source

        # Store serialize library paths for extraction
        if "MOODIST_SERIALIZE_LIBS" in os.environ:
            serialize_libs = os.environ["MOODIST_SERIALIZE_LIBS"].split(",")
            zip_files["moodist.serialize_libs"] = serialize_libs
    else:
        import torch

        torch_version = torch.__version__
        if "+" in torch_version:
            torch_version = torch_version[: torch_version.index("+")]

        moodist_version = "%s%s+torch.%s" % (
            open("version.txt").read().strip(),
            extra_version,
            torch_version,
        )
        torch_prefix_version = ".".join(torch_version.split(".")[:2])
        torch_req_version = "==" + torch_prefix_version + ".*"

        ext_modules = [setuptools.Extension("moodist._C", sources=[])]
        moodist_cversions = {torch_prefix_version: "._C"}

    print("Building for torch%s" % torch_req_version)

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
        ext_modules=ext_modules,
        install_requires=["torch%s" % torch_req_version],
        cmdclass={"build_ext": Build},
        zip_safe=False,
    )


if __name__ == "__main__":
    main()
