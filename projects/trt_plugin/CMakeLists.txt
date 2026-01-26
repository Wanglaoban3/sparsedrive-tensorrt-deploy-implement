cmake_minimum_required(VERSION 3.14)
project(SparseDrivePlugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# =========================================================
# [修复关键点] 显式添加 CUDA 头文件路径
# g++ 编译 .cpp 文件时需要找到 cuda_runtime_api.h
# =========================================================
include_directories(/usr/local/cuda/include)

# 添加系统包含路径 (NvInfer.h)
include_directories(/usr/include/x86_64-linux-gnu)

# 添加库路径
link_directories(/usr/local/cuda-11.8/targets/x86_64-linux/lib/)

link_directories(/usr/lib/x86_64-linux-gnu)

set(SOURCE_FILES 
    deformable_aggregation_kernel.cu 
    DeformableAggregationPlugin.cpp
)

add_library(SparseDrivePlugin SHARED ${SOURCE_FILES})

# 链接 nvinfer 和 cudart
target_link_libraries(SparseDrivePlugin nvinfer cudart)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set_target_properties(SparseDrivePlugin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)