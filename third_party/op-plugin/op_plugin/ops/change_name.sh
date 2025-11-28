#!/bin/bash

# 遍历当前目录下所有 *Kernel.cpp 文件
for f in *Tensor.cpp; do
    # 如果没有匹配到文件则跳过
    [ -e "$f" ] || continue

    # 去掉文件名中的 Kernel
    newname=$(echo "$f" | sed 's/Kernel//')

    echo "Renaming: $f -> $newname"
    mv "$f" "$newname"
done
