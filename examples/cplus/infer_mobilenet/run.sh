#!/bin/bash

# set variables
CURRENT_DIR="$(dirname "$(readlink -f "$0")")"

# recreate build folder
echo ">> [INFO] start to recreate build folder .. "
cd ${CURRENT_DIR}
BUILD_FOLDER="build"
if [ -d ${BUILD_FOLDER} ]; then
    rm -rf ${BUILD_FOLDER}
    mkdir ${BUILD_FOLDER}
else
    mkdir ${BUILD_FOLDER}
fi

# build
echo ">> [INFO] start to build project .. "
cd ${CURRENT_DIR}
cd ${BUILD_FOLDER}
TorchDir=$(pip3 show torch | awk '/Location:/ {print $2}')
cmake -DCMAKE_PREFIX_PATH=${TorchDir}/torch/share/cmake/Torch ..
make -j

# run example
echo ">> [INFO] start to run example .. "
cd ${CURRENT_DIR}
chmod +x ${BUILD_FOLDER}/infer_mobilenet
./${BUILD_FOLDER}/infer_mobilenet
echo ">> [INFO] succeed to run example.. "