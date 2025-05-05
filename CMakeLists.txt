cmake_minimum_required(VERSION 3.20)
project(EdgePadding LANGUAGES CXX CUDA)

# 设置 unicode
add_definitions(-DUNICODE -D_UNICODE)

# 设置 CUDA 构建
find_package(CUDA REQUIRED)

# 设置 C++ 和 CUDA 标准
enable_language(CUDA)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# 查找 OpenCV
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(EdgePaddingLib STATIC
    src/EdgePaddingLib.h
    src/EdgePaddingLib.cu)

# target_include_directories(EdgePadding PUBLIC ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(CuEdgePadding PUBLIC ${OpenCV_LIBS})

# 添加可执行文件（支持 .cu）
add_executable(EdgePaddingTest
    src/EdgePadding.cpp
    src/EdgePadding.cu)

# 链接 OpenCV 库
target_link_libraries(EdgePaddingTest PUBLIC EdgePaddingLib ${OpenCV_LIBS})

option(PYBIND11_FINDPYTHON ON)
add_subdirectory(3rdParty/pybind11)
pybind11_add_module(PyEdgePadding SHARED binding/PyEdgePadding.cpp binding/PyEdgePadding.cu)

# target_include_directories(PyEdgePadding PUBLIC Src)
target_link_libraries(PyEdgePadding PUBLIC EdgePaddingLib ${OpenCV_LIBS})

function(config_target_output target_name)
    set_target_properties(${target_name} PROPERTIES 
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin
    )
endfunction()

config_target_output(EdgePaddingLib)
config_target_output(EdgePaddingTest)
config_target_output(PyEdgePadding)
