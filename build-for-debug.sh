#!/bin/bash

# set variables
PROJECT_DIR=$(pwd)

# clean screen
clear

# create folder
build_folder=build_debug
echo ">> [INFO]: Refresh build folder '${build_folder}' ..."
if [ -d ${build_folder} ]; then
    rm -rf ${build_folder}
fi
mkdir ${build_folder}

install_folder=dl_install
echo ">> [INFO]: Refresh install folder '${install_folder}' ..."
if [ -d ${install_folder} ]; then
    rm -rf ${install_folder}
fi
mkdir ${install_folder}

# codegen process
cd ${PROJECT_DIR}
echo ">> [INFO]: Start codegen process ..."
if [ -e ${PROJECT_DIR}/third_party/op-plugin/build-for-debug.sh ]; then
    chmod +x ${PROJECT_DIR}/third_party/op-plugin/build-for-debug.sh
    bash ${PROJECT_DIR}/third_party/op-plugin/build-for-debug.sh
else
    echo ">> [ERROR]: File 'third_party/op-plugin/build-for-debug.sh' not found, exit ..."
    exit 1
fi


# build project
cd ${PROJECT_DIR}
cd ${build_folder}
TorchDir=$(pip3 show torch | awk '/Editable project location:/ {print $2}')
PYROOT=$(python3 -c "import sys; print(sys.prefix)")
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo ">> [INFO]: Auto-detect conda environment '$CONDA_DEFAULT_ENV' for cmake ..."
    export CPLUS_INCLUDE_PATH=${PYROOT}/include/python${PYVER}
    cmake -DCMAKE_PREFIX_PATH=${TorchDir}/torch/share/cmake/Torch \
          -DCMAKE_INSTALL_PREFIX=${PROJECT_DIR}/${install_folder} \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          -DOCL_PATH=/usr/include \
          -DALLOW_PYBIND=ON \
          -DPython3_ROOT_DIR=${PYROOT} \
          -DPython3_FIND_STRATEGY=LOCATION \
          -DPython3_FIND_REGISTRY=NEVER ..
else
    echo ">> [INFO]: Auto-detect real environment '$CONDA_DEFAULT_ENV' for cmake ..."
    export CPLUS_INCLUDE_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
    echo ">> [INFO]: Set 'CPLUS_INCLUDE_PATH' path '${CPLUS_INCLUDE_PATH}' ..."
    cmake -DCMAKE_PREFIX_PATH=${TorchDir}/torch/share/cmake/Torch \
          -DCMAKE_INSTALL_PREFIX=${PROJECT_DIR}/${install_folder} \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          -DOCL_PATH=/usr/include \
          -DALLOW_PYBIND=ON \
          -DPython3_ROOT_DIR=${PYROOT} \
          -DPython3_FIND_STRATEGY=LOCATION \
          -DPython3_FIND_REGISTRY=NEVER ..
fi

echo ">> [INFO]: Start building ..."
make -j6
echo ">> [INFO]: Start installing ..."
make install

# set temporary variable
cd ${PROJECT_DIR}
echo ">> [INFO]: Set 'PYTHONPATH' path '${PROJECT_DIR}/${install_folder}/python' ..."
export PYTHONPATH=${PROJECT_DIR}/${install_folder}/python:$PYTHONPATH
echo ">> [INFO]: All finished."
