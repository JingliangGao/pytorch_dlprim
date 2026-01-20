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

# build project
echo ">> [INFO]: Build project for release ..."
python3 setup.py build_clib build_ext bdist_wheel

echo ">> [INFO]: All finished."