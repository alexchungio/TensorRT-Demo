cmake_minimum_required(VERSION 3.0.2)

project(src)

file(GLOB_RECURSE HEAD_LIST "${CMAKE_CURRENT_LIST_DIR}/*.hpp")
file(GLOB_RECURSE SRC_LIST "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

add_library(src ${SRC_LIST})