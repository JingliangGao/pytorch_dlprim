cmake_minimum_required(VERSION 3.18)
project(infer_mobilenet CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ========================
# PyTorch 配置
# ========================
# 指向你的 PyTorch libtorch 路径
# 假设你用 conda torch_cuda 环境
set(TORCH_PATH "/root/miniconda3/envs/torch_cuda/lib/python3.12/site-packages/torch")

find_package(Torch REQUIRED PATHS ${TORCH_PATH} NO_DEFAULT_PATH)

# ========================
# OpenCL 可选库
# ========================
# 如果你要链接 libpt_ocl.so
# set(PT_OCL_PATH "/home/kylin/gjl/project/pytorch_dlprim/build/debug/pytorch_ocl")
# set(PT_OCL_LIB "${PT_OCL_PATH}/libpt_ocl.so")

# CUDA 库路径
link_directories(
    /usr/local/cuda/lib64
    /usr/local/cuda-12.2/targets/x86_64-linux/lib/
)

# ========================
# 可执行文件
# ========================
add_executable(infer_mobilenet infer_mobilenet.cpp)

# 链接 PyTorch
target_link_libraries(infer_mobilenet "${TORCH_LIBRARIES}" "${PT_OCL_LIB}")

# 包含目录
target_include_directories(infer_mobilenet PRIVATE
    "${TORCH_PATH}/include"
    "${TORCH_PATH}/include/torch/csrc/api/include"
)

# 链接 CUDA / OpenCL / pthread
find_package(OpenCL REQUIRED)
target_include_directories(infer_mobilenet PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(infer_mobilenet ${OpenCL_LIBRARIES} pthread dl nccl
    cupti
    cudart) 

# 设置 RPATH 方便运行时找到动态库
set_target_properties(infer_mobilenet PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
)
