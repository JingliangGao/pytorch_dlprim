cmake_minimum_required(VERSION 3.16)
project(torch_kpu_demo LANGUAGES CXX)

# ===== C++ 标准 =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ===== Find Torch from pip =====
execute_process(
    COMMAND pip3 show torch
    OUTPUT_VARIABLE TORCH_PIP_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REGEX MATCH "Location: ([^\n\r]+)" _match "${TORCH_PIP_INFO}")
set(TORCH_LOCATION "${CMAKE_MATCH_1}")

message(STATUS "Found Torch installation at: ${TORCH_LOCATION}")

# CMake expects TorchConfig.cmake in <prefix>/torch/share/cmake/Torch
set(CMAKE_PREFIX_PATH "${TORCH_LOCATION}/torch/share/cmake/Torch")

# ===== Find Torch using CMake =====
find_package(Torch REQUIRED)

# ===== Executable =====
add_executable(infer_demo
    infer_mobilenet.cpp
)

find_library(TORCH_KPU_LIB NAMES torch_kpu PATHS /usr/local/lib/torch_kpu/python/torch_kpu)
message(STATUS "Found torch_kpu library: ${TORCH_KPU_LIB}")
message(STATUS "OCL_LIB: ${OCL_LIB}")

# Link main library and dependencies
target_link_libraries(infer_demo PRIVATE ${TORCH_KPU_LIB} ${TORCH_LIBRARIES} dl)

# Set C++17
target_compile_features(infer_demo PRIVATE cxx_std_17)

target_link_options(infer_demo PRIVATE
    -Wl,--export-dynamic
    -Wl,--no-as-needed
)
