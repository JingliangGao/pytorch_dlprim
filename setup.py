import glob
import os
import platform
import shutil
import sys
import stat
import subprocess

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
    """run build-for-debug.sh """
    def run(self):
        subprocess.check_call(['bash', 'build-for-debug.sh'], cwd=BASE_DIR)
        pass


class BuildExt(build_ext):
    def run(self):
        self.run_command('build_clib')
        super().run()


class PythonPackageBuild(build_py):
    def run(self):
        # copy dl_install/python 
        dl_src = os.path.join(BASE_DIR, 'dl_install', 'python', TORCH_PKG_NAME)
        target_base = os.path.join(PACKAGE_BUILD_DIR, TORCH_PKG_NAME)
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
    
    def finalize_options(self):
        super().finalize_options()
        self.python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        arch = platform.machine()  
        self.plat_name = f"linux_{arch}"
        self.plat_name_supplied = True


class InstallCmd(install):
    def finalize_options(self):
        return super().finalize_options()

class EggInfoCmd(egg_info):
    def finalize_options(self):
        os.makedirs(os.path.join(BASE_DIR, 'dl_install', 'python'), exist_ok=True)
        super().finalize_options()

    def run(self):
        self.run_command('build_clib')
        super().run()

packages = find_packages(where='dl_install/python')
package_dir = {'': 'dl_install/python'}

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
    install_requires=[
        "torch==2.9.1+cpu",        
    ],
    package_data={
        TORCH_PKG_NAME: ['*.so', 'lib/*.so*', '*.py'],
    },
    cmdclass={
        'egg_info': EggInfoCmd,
        'build_clib': CPPLibBuild,
        'build_ext': BuildExt,
        'build_py': PythonPackageBuild,
        'bdist_wheel': BdistWheel,
        'install': InstallCmd,
    },
)
