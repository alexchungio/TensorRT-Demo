cmake_minimum_required(VERSION 3.0.2)
project(cuda_sample)


file(GLOB_RECURSE SRC_LIST "${CMAKE_CURRENT_LIST_DIR}/*.cu")

add_library(cuda_sample ${SRC_LIST})