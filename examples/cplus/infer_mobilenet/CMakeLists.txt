cmake_minimum_required(VERSION 3.10)
project(infer_mobilenet)

find_package(Torch REQUIRED)

# 手动设置 OpenCV 路径 (麒麟系统特殊)
set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
set(OpenCV_LIBRARIES
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories("/usr/lib/x86_64-linux-gnu")

# DLPrim headers and libraries
set(DLPRIM_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

# Include directories
include_directories(${DLPRIM_ROOT}/dl_install/include)

# Set library paths
link_directories(${DLPRIM_ROOT}/dl_install/lib)

# Define pt_ocl library path explicitly
set(PT_OCL_LIB "${DLPRIM_ROOT}/build/pytorch_ocl/pt_ocl.so")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(infer_mobilenet infer_mobilenet.cpp)
set_property(TARGET infer_mobilenet PROPERTY CXX_STANDARD 17)

target_link_libraries(infer_mobilenet 
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    dlprim_core
    ${PT_OCL_LIB}  # Use the full path to the library
)

target_compile_definitions(infer_mobilenet PRIVATE 
        DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../data/")
