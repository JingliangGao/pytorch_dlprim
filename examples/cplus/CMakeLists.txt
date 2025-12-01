# examples/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# 找到 OpenCV
find_package(OpenCV REQUIRED)
message("OpenCV_INCLUDE_DIRS : ${OpenCV_INCLUDE_DIRS} ")
# 递归查找所有子目录
file(GLOB CHILD_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(dir ${CHILD_DIRS})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
        # 查找该子目录下所有 .cpp 文件
        file(GLOB SRC_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.cpp)
        if(SRC_FILES)
            # 可执行文件名字与子目录名相同
            set(TARGET_NAME ${dir})

            add_executable(${TARGET_NAME} ${SRC_FILES})

            # 包含主工程头文件
            target_include_directories(${TARGET_NAME} PRIVATE
                ${PROJECT_SOURCE_DIR}/dlprimitives/include
                ${PROJECT_SOURCE_DIR}/include
                ${OpenCV_INCLUDE_DIRS}
            )

            # 链接主库和依赖
            target_link_libraries(${TARGET_NAME} PRIVATE libtorch_kpu ${TORCH_LIBRARIES} ${OCL_LIB} fmt::fmt)

            # 设置 C++17
            target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

            message(STATUS "Added example: ${TARGET_NAME} -> ${SRC_FILES}")
        endif()
    endif()
endforeach()
