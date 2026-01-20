#!/usr/bin/env python3
"""
Custom setup.py for pytorch_dlprim (torch_kpu).

This script mirrors the project's `build-for-debug.sh` behavior and a
Huawei-style setup.py: it will (when building) invoke the local CMake-based
build to produce native libraries into `build/packages/torch_kpu`, then
package the resulting Python package and shared libraries into a wheel.

The script provides custom build_clib / build_ext / bdist_wheel commands to
coordinate CMake/Make, include generated bindings, and optionally run
`auditwheel` when building manylinux wheels.

Run examples:
  python3 setup.py build_clib build_ext bdist_wheel
  python3 setup.py install
"""

import os
import sys
import stat
import subprocess
import glob
import shutil
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.install import install
from setuptools.command.egg_info import egg_info
from wheel.bdist_wheel import bdist_wheel

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
BUILD_DIR = os.path.join(BASE_DIR, 'build')
PACKAGE_BUILD_DIR = os.path.join(BUILD_DIR, 'packages')
TORCH_PKG_NAME = os.environ.get('TORCH_KPU_PKG_NAME', 'torch_kpu')


def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


VERSION = '0.0.0'
version_file = os.path.join(BASE_DIR, 'version.txt')
if os.path.exists(version_file):
    VERSION = read_text(version_file)


def generate_version_py():
    # write version into package build dir; but when running in 'develop'
    # mode write into the source python/ package so editable installs can
    # import the version without running a full build.
    is_develop = any(a in sys.argv for a in ('develop', 'egg_info', 'bdist_egg'))
    if is_develop:
        pkg_base = Path(BASE_DIR) / 'python' / TORCH_PKG_NAME
    else:
        pkg_base = Path(PACKAGE_BUILD_DIR) / TORCH_PKG_NAME
    pkg_version_py = pkg_base / 'version.py'
    pkg_version_py.parent.mkdir(parents=True, exist_ok=True)
    sha = 'unknown'
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BASE_DIR).decode().strip()
    except Exception:
        pass
    full_version = VERSION
    if os.getenv('BUILD_WITHOUT_SHA') is None and sha:
        full_version = full_version + '+git' + sha[:7]
    pkg_version_py.write_text("__version__ = '{v}'\ngit_version = '{g}'\n".format(v=full_version, g=sha))
    return full_version


def which(exe_name):
    path = os.environ.get('PATH', os.defpath).split(os.pathsep)
    for d in path:
        candidate = os.path.join(d, exe_name)
        if os.access(candidate, os.F_OK | os.X_OK) and not os.path.isdir(candidate):
            return candidate
    return None


def get_cmake_command():
    for candidate in ('cmake3', 'cmake'):
        cmd = which(candidate)
        if cmd:
            try:
                out = subprocess.check_output([cmd, '--version']).decode()
                if 'cmake' in out.lower():
                    return cmd
            except Exception:
                continue
    raise RuntimeError('cmake not found')


def get_torch_cmake_prefix():
    try:
        import torch
        return os.path.dirname(os.path.realpath(torch.__file__)) + '/share/cmake/Torch'
    except Exception:
        raise RuntimeError('[ERROR] Failed to find torch cmake prefix')
    return None


class CPPLibBuild(build_clib):
    """Build native libs using CMake (mirrors build-for-debug.sh behaviour)."""

    def run(self):
        cmake = get_cmake_command()
        build_dir = os.path.join(BASE_DIR, 'build_debug') if os.getenv('DEBUG', '0') in ('1', 'ON', 'TRUE') else os.path.join(BASE_DIR, 'build')
        install_prefix = os.path.join(BASE_DIR, 'build', 'packages', TORCH_PKG_NAME)
        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(install_prefix, exist_ok=True)

        torch_cmake = get_torch_cmake_prefix() or ''
        print('Using torch cmake prefix:', torch_cmake)

        cmake_args = [
            '-DCMAKE_INSTALL_PREFIX=' + os.path.realpath(install_prefix),
            '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
            '-DOCL_PATH=/usr/include',
        ]

        if torch_cmake:
            cmake_args.insert(0, '-DCMAKE_PREFIX_PATH=' + torch_cmake)

        python_root = sys.prefix
        cmake_args += [
            '-DPython3_ROOT_DIR=' + python_root,
            "-DPython3_FIND_STRATEGY=LOCATION",
            "-DPython3_FIND_REGISTRY=NEVER",
        ]

        op_plugin_script = os.path.join(BASE_DIR, 'third_party', 'op-plugin', 'build-for-debug.sh')
        if os.path.exists(op_plugin_script):
            try:
                subprocess.check_call(['bash', op_plugin_script], cwd=BASE_DIR)
            except Exception:
                print('Warning: op-plugin build script failed', file=sys.stderr)

        call = [cmake, BASE_DIR] + cmake_args
        print('Running cmake:', ' '.join(call))
        subprocess.check_call(call, cwd=build_dir, env=os.environ)

        jobs = os.environ.get('MAX_JOBS', str(os.cpu_count() or 4))
        make_cmd = ['make', '-j16']
        print('Running make:', ' '.join(make_cmd))
        subprocess.check_call(make_cmd, cwd=build_dir, env=os.environ)

        subprocess.check_call(['make', 'install'], cwd=build_dir, env=os.environ)


class BuildExt(build_ext):
    def run(self):
        self.run_command('build_clib')
        super().run()


class PythonPackageBuild(build_py):
    def run(self):
        # copy python sources to package build dir so wheel includes them
        src_py = os.path.join(BASE_DIR, 'python')
        target_base = os.path.join(PACKAGE_BUILD_DIR, TORCH_PKG_NAME)
        if os.path.exists(src_py):
            for src in glob.glob(os.path.join(src_py, '**', '*'), recursive=True):
                if os.path.isfile(src):
                    rel = os.path.relpath(src, src_py)
                    dst = os.path.join(target_base, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
        # also copy dl_install/python if present
        dl_src = os.path.join(BASE_DIR, 'dl_install', 'python', TORCH_PKG_NAME)
        if os.path.exists(dl_src):
            for src in glob.glob(os.path.join(dl_src, '**', '*'), recursive=True):
                if os.path.isfile(src):
                    rel = os.path.relpath(src, dl_src)
                    dst = os.path.join(target_base, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
        return super().run()


class BdistWheel(bdist_wheel):
    def run(self):
        bdist_wheel.run(self)
        is_manylinux = os.environ.get('AUDITWHEEL_PLAT') is not None
        if is_manylinux:
            whl = glob.glob(os.path.join(self.dist_dir, '*linux*.whl'))
            if whl:
                whl = whl[0]
                audit_cmd = ['auditwheel', 'repair', '-w', self.dist_dir, whl]
                excludes = ['libgomp.so.1']
                for e in excludes:
                    audit_cmd += ['--exclude', e]
                try:
                    subprocess.check_call(audit_cmd)
                finally:
                    try:
                        os.remove(whl)
                    except Exception:
                        pass


class InstallCmd(install):
    def finalize_options(self):
        self.build_lib = os.path.relpath(PACKAGE_BUILD_DIR)
        return super().finalize_options()


generate_version_py()

# Determine packaging sources: for editable/develop installs we should point
# package_dir to the source `python/` tree to avoid setuptools complaining
# about inconsistent installation directories. For bdist_wheel / normal
# packaging we use the `build/packages` tree produced by the CMake build.
is_develop_cmd = any(a in sys.argv for a in ('develop', 'egg_info', 'bdist_egg'))
if is_develop_cmd:
    packages = find_packages(where='python')
    package_dir = {'': 'python'}
else:
    # The CMake install step places python files under
    #   build/packages/<pkg_name>/python/<pkg_name>/...
    # We must point setuptools at that 'python' subdirectory so the wheel
    # contains a top-level 'torch_kpu' package (not 'torch_kpu/python/torch_kpu').
    package_source_dir = os.path.join(PACKAGE_BUILD_DIR, TORCH_PKG_NAME, 'python')
    if not os.path.exists(package_source_dir):
        # fallback to previous layout
        package_source_dir = os.path.join(PACKAGE_BUILD_DIR, 'python')
    # find packages within that python subdir
    packages = find_packages(where=package_source_dir)
    # setuptools requires package_dir values to be relative POSIX paths
    rel_pkg_build = os.path.relpath(package_source_dir, BASE_DIR).replace(os.sep, '/')
    package_dir = {'': rel_pkg_build}

setup(
    name=TORCH_PKG_NAME,
    version=VERSION,
    description='PyTorch OpenCL backend (torch_kpu) - built from source',
    long_description=open(os.path.join(BASE_DIR, 'README.md'), 'r', encoding='utf-8').read() if os.path.exists(os.path.join(BASE_DIR, 'README.md')) else '',
    long_description_content_type='text/markdown',
    packages=packages,
    package_dir=package_dir,
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.8',
    package_data={
        TORCH_PKG_NAME: ['*.so', 'lib/*.so*', '*.py'],
    },
    cmdclass={
        'build_clib': CPPLibBuild,
        'build_ext': BuildExt,
        'build_py': PythonPackageBuild,
        'bdist_wheel': BdistWheel,
        'install': InstallCmd,
    },
)
