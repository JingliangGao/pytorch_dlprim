#!/bin/bash

CURRENT_DIR=$(pwd)


clear

# refresh build folder
cd ${CURRENT_DIR}
if [ -d build ]; then
    rm -rf build
fi

# build project
cd ${CURRENT_DIR}
mkdir build && cd build
cmake ../
make -j

# run example
cd ${CURRENT_DIR}
./build/infer_demo
