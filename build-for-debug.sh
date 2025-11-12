#!/bin/bash

# set variables
PROJECT_DIR=$(pwd)

# create folder
build_folder=build_debug
if [ -d ${build_folder} ]; then
    rm -rf ${build_folder}
fi
mkdir ${build_folder}
 
if [ -d dl_install ]; then
    rm -rf dl_install
fi
mkdir dl_install

# build project
cd ${build_folder}
TorchDir=$(pip3 show torch | awk '/Location:/ {print $2}')
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include/python${PYVER}:$CPLUS_INCLUDE_PATH
cmake -DCMAKE_PREFIX_PATH=${TorchDir}/torch/share/cmake/Torch \
      -DCMAKE_INSTALL_PREFIX=$PROJECT_DIR/dl_install  \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DOCL_PATH=/usr/include \
      -DINSTALL_PYTHON=ON \
      -DPython3_ROOT_DIR=$CONDA_PREFIX \
      -DPython3_FIND_STRATEGY=LOCATION \
      -DPython3_FIND_REGISTRY=NEVER ..
make -j6
make install

# set temporary variable
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include/python${PYVER}:$CPLUS_INCLUDE_PATH
export PYTHONPATH=${PROJECT_DIR}/dl_install/python:$PYTHONPATH
