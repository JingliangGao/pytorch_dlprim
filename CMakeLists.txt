cmake_minimum_required(VERSION 3.10)
project(pytorch_dlprim)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find Torch
execute_process(
    COMMAND pip3 show torch
    OUTPUT_VARIABLE TORCH_PIP_INFO
)
string(REGEX MATCH "Location: ([^\n\r]+)" _match "${TORCH_PIP_INFO}")
set(TORCH_LOCATION "${CMAKE_MATCH_1}")
set(CMAKE_PREFIX_PATH ${TORCH_LOCATION}/torch/share/cmake/Torch)   
find_package(Torch REQUIRED)

# Find fmt library
find_package(fmt REQUIRED)
if(fmt_FOUND)
    message(STATUS "fmt library found!")
    get_target_property(FMT_INCLUDE_DIRS fmt::fmt INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(FMT_LIBS fmt::fmt LOCATION)
   

    message(STATUS "fmt include dirs: ${FMT_INCLUDE_DIRS}")
    message(STATUS "fmt library location: ${FMT_LIBS}")
else()
    message(FATAL_ERROR "fmt library not found!")
endif()

# Find pybind11 
find_package(pybind11 REQUIRED)

# Start codegen
message(" Starting code generation for op-plugin ...")
execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/third_party/op-plugin/build-for-debug.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/op-plugin
)

# Set some options
option(BUILD_CORE_ONLY "Core Build" ON)
option(DLPRIM_STATIC_BUILD "Build static version only" ON)
option(ALLOW_PYBIND "Create Python bindings" OFF)
option(BUILD_EXAMPLES "Build example programs" OFF)

add_subdirectory(third_party/dlprimitives)
include_directories("${OCL_PATH}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/third_party/dlprimitives)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/dlprimitives/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/op-plugin/op_plugin/generate)
include_directories(${Python3_INCLUDE_DIRS})   # Python include
include_directories(${pybind11_INCLUDE_DIRS})  # pybind11 include

message(" Pytorch libraries: ${TORCH_LIBRARIES}")
message(" Python libraries: ${Python3_LIBRARIES}")
message(" Python include: ${Python3_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if(NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall")
endif()

# collect source files in third_party/op-plugin/
file(GLOB OP_PLUGIN_SOURCES
    third_party/op-plugin/op_plugin/ops/*.cpp
    third_party/op-plugin/op_plugin/generate/*.cpp
)

# collect source files in src/
file(GLOB PYOCL_SOURCES
    src/*.cpp
)

# Define our library target
add_library(libpt_ocl SHARED
    ${OP_PLUGIN_SOURCES}
    ${PYOCL_SOURCES}
)



target_compile_features(libpt_ocl PRIVATE cxx_std_17)

# Link against LibTorch
target_include_directories(libpt_ocl PRIVATE ${FMT_INCLUDE_DIRS})
target_compile_definitions(libpt_ocl PRIVATE FMT_HEADER_ONLY)
target_link_libraries(libpt_ocl PRIVATE ${TORCH_LIBRARIES} ${OCL_LIB} dlprim_core ${Python3_LIBRARIES})
if(WIN32)
    target_link_libraries(libpt_ocl ${Python3_LIBRARIES})
endif()
set_target_properties(libpt_ocl PROPERTIES PREFIX "" OUTPUT_NAME "libpt_ocl")
if(NOT WIN32)
	set_target_properties(libpt_ocl PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch_ocl/")
else()
	set_target_properties(libpt_ocl PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch_ocl/"
		SUFFIX ".pyd"
	) 
endif()

# Install Python bindings
foreach(PYNAME __init__.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/pytorch_ocl/${PYNAME} "${CMAKE_BINARY_DIR}/pytorch_ocl/${PYNAME}" COPYONLY)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/pytorch_ocl/${PYNAME} DESTINATION python/pytorch_ocl)
endforeach()
install(TARGETS libpt_ocl 
        LIBRARY DESTINATION python/pytorch_ocl
        RUNTIME DESTINATION python/pytorch_ocl
    )

# Add C++ examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples/cplus)
endif()
