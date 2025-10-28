#!/bin/bash

# set variables
PROJECT_DIR=$(pwd)

# create folder
if [ -d build ]; then
    rm -rf build
fi
mkdir build

if [ -d dl_install ]; then
    rm -rf dl_install
fi
mkdir dl_install

# build project
cd build
TorchDir=$(pip3 show torch | awk '/Location:/ {print $2}')
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include/python3.12:$CPLUS_INCLUDE_PATH
cmake -DCMAKE_PREFIX_PATH=${TorchDir}/torch/share/cmake/Torch \
      -DCMAKE_INSTALL_PREFIX=$PROJECT_DIR/dl_install  \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DOCL_PATH=/usr/include \
      -DPython3_ROOT_DIR=$CONDA_PREFIX \
      -DPython3_FIND_STRATEGY=LOCATION \
      -DPython3_FIND_REGISTRY=NEVER ..
make -j4
make install

# write into bashrc
grep -qxF 'export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include/python3.12:$CPLUS_INCLUDE_PATH' ~/.bashrc || echo 'export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/include/python3.12:$CPLUS_INCLUDE_PATH' >> ~/.bashrc
grep -qxF 'export PYTHONPATH='"$PROJECT_DIR"'/dl_install/python:$PYTHONPATH' ~/.bashrc || echo 'export PYTHONPATH='"$PROJECT_DIR"'/dl_install/python:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc