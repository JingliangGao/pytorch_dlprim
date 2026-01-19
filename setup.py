

from distutils.version import LooseVersion
from setuptools import setup, distutils, Extension

import os
import platform
import subprocess
import sys
import traceback

# set necessary variables
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
BUILD_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP

# get version info
with open(os.path.join(BASE_DIR, "version.txt")) as version_f:
    VERSION = version_f.read().strip()

# get README.md content
readme = os.path.join(BASE_DIR, "README.md")
if not os.path.exists(readme):
    raise FileNotFoundError("Unable to find 'README.md'")
with open(readme, encoding="utf-8") as fdesc:
    readme_context = fdesc.read()

# PyPI classifier label
classifier_label = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: BSD License",
    "Operating System :: POSIX :: Linux",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

# function to get pytorch installation dir
def get_pytorch_dir():
    try:
        import torch
        return os.path.dirname(os.path.realpath(torch.__file__))
    except Exception:
        _, _, exc_traceback = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_traceback)[-1]
        return os.path.dirname(frame_summary.filename)

# function to get pytorch include and lib dirs
def CppExtension(name, sources, *args, **kwargs):
    pytorch_dir = get_pytorch_dir()
    temp_include_dirs = kwargs.get('include_dirs', [])
    temp_include_dirs.append(os.path.join(pytorch_dir, 'include'))
    temp_include_dirs.append(os.path.join(pytorch_dir, 'include/torch/csrc/api/include'))
    kwargs['include_dirs'] = temp_include_dirs

    temp_library_dirs = kwargs.get('library_dirs', [])
    temp_library_dirs.append(os.path.join(pytorch_dir, 'lib'))
    kwargs['library_dirs'] = temp_library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('c10')
    libraries.append('torch')
    libraries.append('torch_cpu')
    libraries.append('torch_python')
    kwargs['libraries'] = libraries
    kwargs['language'] = 'c++'
    return Extension(name, sources, *args, **kwargs)

# Setup include directories folders.
include_directories = [
    BASE_DIR,
    os.path.join(BASE_DIR, 'torch_kpu/csrc/include'),
    os.path.join(BASE_DIR, 'third_party/dlprimitives/include'),
    os.path.join(BASE_DIR, 'third_party/op-plugin/op_plugin/generate'),
    os.path.join(BASE_DIR, 'third_party/op-plugin/op_plugin/utils')
]

# extra link args
extra_compile_args = [
    '-std=c++17',
    '-Wno-sign-compare',
    '-Wno-deprecated-declarations',
    '-Wno-return-type'
]
extra_link_args = []
DEBUG = (os.getenv('DEBUG', default='').upper() in ['ON', '1', 'YES', 'TRUE', 'Y'])
if DEBUG:
    extra_compile_args += ['-O0', '-g']
    extra_link_args += ['-O0', '-g', '-Wl,-z,now']
else:
    extra_compile_args += ['-DNDEBUG']
    extra_link_args += ['-Wl,-z,now']

# change to use cxx11.abi in default since 2.7
USE_CXX11_ABI = True
if os.environ.get("_GLIBCXX_USE_CXX11_ABI") is not None and os.environ.get("_GLIBCXX_USE_CXX11_ABI") == "0":
    USE_CXX11_ABI = False

# func: get build type
def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        if sys.platform == 'win32':
            exts = os.environ.get('PATHEXT', '').split(os.pathsep)
            fnames += [fname + ext for ext in exts]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None

# func: get cmake command
def get_cmake_command():
    def _get_version(cmd):
        for line in subprocess.check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')
    "Returns cmake command."
    cmake_command = 'cmake'
    if platform.system() == 'Windows':
        return cmake_command
    cmake3 = which('cmake3')
    cmake = which('cmake')
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.18.0"):
        cmake_command = 'cmake3'
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= LooseVersion("3.18.0"):
        return cmake_command
    else:
        raise RuntimeError('no cmake or cmake3 with version >= 3.18.0 found')

# class: build cpp library
class CPPLibBuild(build_clib, object):
    def run(self):
        cmake = get_cmake_command()

        if cmake is None:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        self.cmake = cmake

        build_dir = os.path.join(BASE_DIR, "build")
        build_type_dir = os.path.join(build_dir)
        output_lib_path = os.path.join(build_type_dir, "packages/torch_npu/lib")
        os.makedirs(build_type_dir, exist_ok=True)
        os.chmod(build_type_dir, mode=BUILD_PERMISSION)
        os.makedirs(output_lib_path, exist_ok=True)
        self.build_lib = os.path.relpath(os.path.join(build_dir, "packages/torch_npu"))
        self.build_temp = os.path.relpath(build_type_dir)

        cmake_args = [
            '-DCMAKE_BUILD_TYPE=' + get_build_type(),
            '-DCMAKE_INSTALL_PREFIX=' + os.path.realpath(output_lib_path),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.realpath(output_lib_path),
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY=' + os.path.realpath(output_lib_path),
            '-DTORCHNPU_INSTALL_LIBDIR=' + os.path.realpath(output_lib_path),
            '-DPYTHON_INCLUDE_DIR=' + get_paths().get('include'),
            '-DTORCH_VERSION=' + VERSION,
            '-DPYTORCH_INSTALL_DIR=' + get_pytorch_dir()]

        if DISABLE_TORCHAIR == 'FALSE':
            if check_torchair_valid(BASE_DIR):
                cmake_args.append('-DBUILD_TORCHAIR=on')
                torchair_install_prefix = os.path.join(build_type_dir, "packages/torch_npu/dynamo/torchair")
                cmake_args.append(f'-DTORCHAIR_INSTALL_PREFIX={torchair_install_prefix}')
                cmake_args.append(f'-DTORCHAIR_TARGET_PYTHON={sys.executable}')

        if DISABLE_RPC == 'FALSE':
            if check_tensorpipe_valid(BASE_DIR):
                cmake_args.append('-DBUILD_TENSORPIPE=on')
        
        if ENABLE_LTO == "TRUE":
            cmake_args.append('-DENABLE_LTO=on')
        if PGO_MODE != 0:
            cmake_args.append('-DPGO_MODE=' + str(PGO_MODE))
        
        if USE_CXX11_ABI:
            cmake_args.append('-DGLIBCXX_USE_CXX11_ABI=1')

        if os.getenv('_ABI_VERSION') is not None:
            cmake_args.append('-DABI_VERSION=' + os.getenv('_ABI_VERSION'))

        max_jobs = os.getenv("MAX_JOBS", str(multiprocessing.cpu_count()))
        build_args = ['-j', max_jobs]

        subprocess.check_call([self.cmake, BASE_DIR] + cmake_args, cwd=build_type_dir, env=os.environ)
        for base_dir, dirs, files in os.walk(build_type_dir):
            for dir_name in dirs:
                dir_path = os.path.join(base_dir, dir_name)
                os.chmod(dir_path, mode=BUILD_PERMISSION)
            for file_name in files:
                file_path = os.path.join(base_dir, file_name)
                os.chmod(file_path, mode=BUILD_PERMISSION)

        subprocess.check_call(['make'] + build_args, cwd=build_type_dir, env=os.environ)

# setup function
setup(
    name='torch_kpu',
    version=VERSION,
    description='KPU bridge for PyTorch',
    long_description=readme_context,
    long_description_content_type="text/markdown",
    license="BSD License",
    classifiers=classifier_label,
    packages=["torch_kpu"],
    libraries=[('torch_kpu', {'sources': list()})],
    package_dir={'': os.path.relpath(os.path.join(BASE_DIR, "build/packages"))},
    ext_modules=[
            CppExtension(
                'torch_kpu._C',
                sources=["torch_kpu/csrc/Pybind.cpp"],
                libraries=["torch_kpu"],
                include_dirs=include_directories,
                extra_compile_args=extra_compile_args + ['-fstack-protector-all'] + ['-D__FILENAME__=\"Pybind.cpp\"'],
                library_dirs=["lib"],
                extra_link_args=extra_link_args + ['-Wl,-rpath,$ORIGIN/lib', '-Wl,-Bsymbolic-functions'],
                define_macros=[('_GLIBCXX_USE_CXX11_ABI', '1' if USE_CXX11_ABI else '0'), ('GLIBCXX_USE_CXX11_ABI', '1' if USE_CXX11_ABI else '0')]
            ),
    ],
    extras_require={
    },
    package_data={
        'torch_kpu': [
            '*.so', 'lib/*.so*',
        ],
    },
    cmdclass={
        'build_clib': CPPLibBuild,
        'build_ext': Build,
        'build_py': PythonPackageBuild,
        'bdist_wheel': BdistWheelBuild,
        'install': InstallCmd,
        'clean': Clean
    },
    # entry_points={
    #     'torch.backends': [
    #         'torch_kpu = torch_kpu:_autoload',
    #     ],
    # }
)