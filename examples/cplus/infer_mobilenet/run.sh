#!/bin/bash

# set variables
CURRENT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="${CURRENT_DIR}/../../.."
DATA_DIR="${CURRENT_DIR}/../../data"

# Set environment variables for pt_ocl
export PYTHONPATH="${PROJECT_ROOT}/build:${PROJECT_ROOT}/dl_install/python:${PYTHONPATH}"
export LD_LIBRARY_PATH="${PROJECT_ROOT}/build/pytorch_ocl:${PROJECT_ROOT}/dl_install/lib:${LD_LIBRARY_PATH}"

# 验证环境
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import pytorch_ocl; print('pytorch_ocl 已加载')" || { echo "错误: pytorch_ocl 模块无法加载"; exit 1; }

# check if model exist
if [ ! -e "${DATA_DIR}/mobilenetv2_ts.pt" ]; then
    echo ">> [INFO] start to download model .. "
    python3 download_model.py
fi

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