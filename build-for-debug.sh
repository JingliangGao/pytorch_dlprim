#!/bin/bash

# set variables
PROJECT_DIR=$(pwd)

# clean screen
clear

# check sudo permission
if [ "$EUID" -eq 0 ]; then
    SUDO_ER=""
else
    SUDO_ER=sudo
fi

# check ICD vendor
if [ ! -d /etc/OpenCL/vendors ]; then
    echo ">> [ERROR]: OpenCL ICD vendor folder '/etc/OpenCL/vendors' not found, please install OpenCL runtime first, exit ..."
    exit 1
fi

# install packages
echo ">> [INFO]: Install system packages ..."
PACKAGES=(
  ocl-icd-opencl-dev
  opencl-headers
  libfmt-dev
  pybind11-dev
  python3-dev
)

for pkg in "${PACKAGES[@]}"; do
  if dpkg -s "$pkg" >/dev/null 2>&1; then
    echo ">> [INFO]: $pkg already installed"
  else
    echo ">> [INFO]: $pkg not installed, installing..."
    ${SUDO_ER} apt-get install -y "$pkg"
  fi
done
echo ">> [INFO]: Install Python packages ..."
pip3 install torch --index-url https://download.pytorch.org/whl/cpu
pip3 install torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install -r requirements.txt
echo ">> [INFO]: All packages installed."

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
TorchDir=$(pip3 show torch | awk '/Location:/ {print $2}')
PYROOT=$(python3 -c "import sys; print(sys.prefix)")
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo ">> [INFO]: Auto-detect conda environment '$CONDA_DEFAULT_ENV' for cmake ..."
    export CPLUS_INCLUDE_PATH=${PYROOT}/include/python${PYVER}
    cmake -DCMAKE_PREFIX_PATH=${TorchDir}/torch/share/cmake/Torch \
          -DCMAKE_INSTALL_PREFIX=${PROJECT_DIR}/${install_folder} \
          -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
          -DOCL_PATH=/usr/include \
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
          -DPython3_ROOT_DIR=${PYROOT} \
          -DPython3_FIND_STRATEGY=LOCATION \
          -DPython3_FIND_REGISTRY=NEVER \
          ..
fi

echo ">> [INFO]: Start building ..."
make -j16
echo ">> [INFO]: Start installing ..."
make install

# write environment variables into '~/.torch_kpu_env' file
cd ${PROJECT_DIR}
echo ">> [INFO]: Set 'PYTHONPATH' path '${PROJECT_DIR}/${install_folder}/python' ..."
LINE="export PYTHONPATH=${PROJECT_DIR}/${install_folder}/python:\$PYTHONPATH"
TORCH_RC="${HOME}/.torch_kpu_env"
## check if '${TORCH_RC}' file exists
if [ ! -f "${TORCH_RC}" ]; then
    touch "${TORCH_RC}"
fi

## write 'PYTHONPATH' into '${TORCH_RC}' file
if ! grep -Fxq "$LINE" "$TORCH_RC"; then
    echo "$LINE" >> "$TORCH_RC"
    echo ">> [INFO]: 'PYTHONPATH' added to '${TORCH_RC}'"
fi

## append sourcing line into '~/.bashrc' file
BASHRC="${HOME}/.bashrc"
LINE='[ -f ~/.torch_kpu_env ] && source ~/.torch_kpu_env'
if ! grep -Fxq "$LINE" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# torch_kpu environment" >> "$BASHRC"
    echo "$LINE" >> "$BASHRC"
fi
echo ">> [INFO]: Please run 'source ~/.bashrc' to update environment variables."

echo ">> [INFO]: All finished."
