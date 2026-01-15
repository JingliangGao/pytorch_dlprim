#!/usr/bin/env python3
import io
import os
from setuptools import setup, find_packages, Distribution


def read_file(fname):
    here = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(here, fname)
    with io.open(path, encoding="utf-8") as f:
        return f.read()


version = "0.0.0"
try:
    version = read_file("version.txt").strip()
except Exception:
    pass

long_description = ""
try:
    long_description = read_file("README.md")
except Exception:
    pass


setup(
    name="torch_kpu",
    version=version,
    description="PyTorch OpenCL backend (torch_kpu)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # Package from the compiled install tree so native libraries (libtorch_kpu.so)
    # under dl_install/python/torch_kpu are included in the wheel. This avoids
    # requiring the user to set PYTHONPATH to dl_install at runtime.
    packages=find_packages(where="dl_install/python"),
    package_dir={"": "dl_install/python"},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[],
    # Ensure compiled shared object is included in the package data. Add any
    # additional patterns if you have other binary assets under the package.
    package_data={
        # top-level shared lib
        "torch_kpu": ["*.so", "*.so.*", "*.json"],
        # common subfolders that may contain binary/data files
        "torch_kpu.core": ["*", "*/*"],
        "torch_kpu.testing": ["*", "*/*"],
        "torch_kpu.utils": ["*", "*/*"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    # Mark this distribution as containing extension modules / non-pure python
    # so that builders will produce a platform-specific wheel (not py3-none-any).
    # This ensures the wheel filename will include python/abi and platform tags
    # (for example: cp310-<abi>-manylinux_x86_64.whl).
    distclass=type('BinaryDistribution', (Distribution,), {
        'has_ext_modules': lambda self: True
    }),
    # Optional: set platforms metadata (informational)
    platforms=['manylinux_x86_64'],
)
