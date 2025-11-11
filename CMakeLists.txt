cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(dlprim_backend)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch 1.13 REQUIRED)
find_package(fmt REQUIRED)

# find fmt library
if(fmt_FOUND)
    message(STATUS "fmt library found!")
    get_target_property(FMT_INCLUDE_DIRS fmt::fmt INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(FMT_LIBS fmt::fmt LOCATION)
   

    message(STATUS "fmt include dirs: ${FMT_INCLUDE_DIRS}")
    message(STATUS "fmt library location: ${FMT_LIBS}")
else()
    message(FATAL_ERROR "fmt library not found!")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
option(BUILD_CORE_ONLY "Core Build" ON)
option(DLPRIM_STATIC_BUILD "Build static version only" ON)

add_subdirectory(dlprimitives)
include_directories("${OCL_PATH}")
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dlprimitives)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dlprimitives/include)

message("  Pytorch libraries ${TORCH_LIBRARIES}")
message("  Python libraries ${PYTHON_LIBRARIES}")
message("  Python include ${Python3_INCLUDE_DIRS}")

if(NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall")
endif()

# Define our library target
add_library(pt_ocl SHARED 
        src/pointwise_ops.cpp
        src/tensor_ops.cpp
        src/vision_ops.cpp
        src/loss_ops.cpp
        src/random_ops.cpp
        src/norm_ops.cpp
        src/utils.cpp
        src/registeration.cpp
        src/CLTensor.cpp
        src/guard.cpp
        src/python_iface.cpp
        )

target_compile_features(pt_ocl PRIVATE cxx_std_17)

# Link against LibTorch
target_include_directories(pt_ocl PRIVATE ${FMT_INCLUDE_DIRS})
target_compile_definitions(pt_ocl PRIVATE FMT_HEADER_ONLY)
target_link_libraries(pt_ocl PRIVATE ${TORCH_LIBRARIES} ${OCL_LIB} dlprim_core)
if(WIN32)
    target_link_libraries(pt_ocl ${PYTHON_LIBRARIES})
endif()
set_target_properties(pt_ocl PROPERTIES PREFIX "" OUTPUT_NAME "pt_ocl")
if(NOT WIN32)
	set_target_properties(pt_ocl PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch_ocl/")
else()
	set_target_properties(pt_ocl PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch_ocl/"
		SUFFIX ".pyd"
	)
	 
endif()

	
foreach(PYNAME __init__.py)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/pytorch_ocl/${PYNAME} "${CMAKE_BINARY_DIR}/pytorch_ocl/${PYNAME}" COPYONLY)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/pytorch_ocl/${PYNAME} DESTINATION python/pytorch_ocl)
endforeach()

install(TARGETS pt_ocl 
		LIBRARY DESTINATION python/pytorch_ocl
		RUNTIME DESTINATION python/pytorch_ocl
	)

option(BUILD_EXAMPLES "Build example programs" OFF)
if(BUILD_EXAMPLES)
    add_subdirectory(examples/cplus)
endif()

