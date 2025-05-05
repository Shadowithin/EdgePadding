cmake_minimum_required(VERSION 3.20)
project(EdgePadding LANGUAGES C CXX CUDA)

# 设置 CUDA 构建
find_package(CUDA REQUIRED)

# 设置 C++ 和 CUDA 标准
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

# 添加可执行文件（支持 .cu）
add_executable(EdgePadding 
    src/EdgePadding.cu 
    src/EdgePadding.cpp)

# 链接 OpenCV 库
target_link_libraries(EdgePadding ${OpenCV_LIBS})
