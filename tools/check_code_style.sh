#!/bin/bash


# set variables
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
PROJECT_DIR=${CDIR}/../

# check if 'pre-commit' exist
echo "[INFO] check if module 'pre-commit' exist ... "
if pip3 show pre-commit >/dev/null 2>&1; then
    echo "[INFO] pre-commit has been installed ."
else
    pip3 install pre-commit -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo "[INFO] pre-commit has been installed ."
fi

# code style reformat
cd ${PROJECT_DIR}
echo "[INFO] check if '.pre-commit-config.yaml' exist ... "
## 1. add .pre-commit-config.yaml
if [ ! -e .pre-commit-config.yaml ]; then
    echo "
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v2.3.0
    hooks:
    -   id: check-yaml
        exclude: ^tools/|^third_party/dlprimitives/|^examples/data/
    -   id: end-of-file-fixer
        exclude: ^tools/|^third_party/dlprimitives/|^examples/data/
    -   id: trailing-whitespace
        exclude: ^tools/|^third_party/dlprimitives/|^examples/data/
-   repo: https://github.com/psf/black
    rev: 22.10.0
    hooks:
    -   id: black
        exclude: ^tools/|^third_party/dlprimitives/|^examples/data/
-   repo: https://github.com/pre-commit/mirrors-clang-format
    rev: v17.0.6
    hooks:
    -   id: clang-format
        files: \.(c|cc|cpp|cxx|h|hpp|hxx)$
        exclude: ^tools/|^third_party/dlprimitives/|^examples/data/
" > .pre-commit-config.yaml

    echo "[INFO] Succeed to add '.pre-commit-config.yaml' file . "
fi

## 2. add hooks
echo "[INFO] check if hook '.git/hooks/pre-commit' exist ... "
if [ ! -e .git/hooks/pre-commit ]; then
    pre-commit install
    echo "[INFO] Succeed to hook '.git/hooks/pre-commit' . "
fi


## start checking code style
echo "[INFO] Succeed to check code style ... "
cd ${PROJECT_DIR}
pre-commit run --all-files

echo "[INFO] All finished . "
