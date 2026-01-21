#!/bin/bash

CURRENT_DIR=$(pwd)
ARCH=$(uname -m)
VERSION=0.0.1

# check sudo permission
if [ "$EUID" -eq 0 ]; then
    SUDO_ER=""
else
    SUDO_ER=sudo
fi

# refresh 'torch_kpu' folder
if [ -d ${CURRENT_DIR}/torch_kpu/usr/local/lib/torch_kpu/ ]; then
    ${SUDO_ER} rm -rf ${CURRENT_DIR}/torch_kpu/usr/local/lib/torch_kpu/
    ${SUDO_ER} mkdir -p ${CURRENT_DIR}/torch_kpu/usr/local/lib/torch_kpu/
fi

# copy 'libtorch_kpu.so'
if [ -f ${CURRENT_DIR}/../dl_install/python/torch_kpu/libtorch_kpu.so ]; then
    echo ">> [INFO]: Found 'libtorch_kpu.so' in dl_install/python/torch_kpu/, copy dl_install ..."
    ${SUDO_ER} cp -rf ${CURRENT_DIR}/../dl_install/include/ ${CURRENT_DIR}/torch_kpu/usr/local/lib/torch_kpu/
    ${SUDO_ER} cp -rf ${CURRENT_DIR}/../dl_install/lib/     ${CURRENT_DIR}/torch_kpu/usr/local/lib/torch_kpu/
    ${SUDO_ER} cp -rf ${CURRENT_DIR}/../dl_install/python/  ${CURRENT_DIR}/torch_kpu/usr/local/lib/torch_kpu/                                             
fi

# give permission
chmod 755 torch_kpu/DEBIAN/preinst
chmod 755 torch_kpu/DEBIAN/postinst
chmod 755 torch_kpu/DEBIAN/prerm

# build deb package
echo ">> [INFO]: Build deb package ..."
${SUDO_ER} dpkg-deb --build torch_kpu torch-kpu_${VERSION}_${ARCH}.deb