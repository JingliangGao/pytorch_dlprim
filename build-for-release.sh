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

if [ -d dist ]; then
    rm -rf dist
fi

# build project
echo ">> [INFO]: Build project for release ..."
python3 setup.py build_clib build_ext bdist_wheel > build_release.log 2>&1   # build_clib build_ext bdist_wheel 
if [ $? -ne 0 ]; then
    echo ">> [ERROR]: Build failed, please check 'build_release.log' for more information."
    exit 1
fi
python3 setup.py install

echo ">> [INFO]: All finished."