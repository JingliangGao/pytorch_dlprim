#!/usr/bin/env python3
import io
import os
from setuptools import setup, find_packages


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
    description="PyTorch OpenCL backend (torch_kpu) from pytorch_dlprim",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="torch_kpu/python"),
    package_dir={"": "torch_kpu/python"},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
