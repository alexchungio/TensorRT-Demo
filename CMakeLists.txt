cmake_minimum_required(VERSION 3.0.2)

project(TensorRT-Demo)

# add compile difine
add_compile_definitions(TEST_DATA_DIRS="${PROJECT_SOURCE_DIR}/test/data/")
add_compile_definitions(MODEL_DIRS="${PROJECT_SOURCE_DIR}/model/")


# set normal ariable a value
set(CMAKE_CXX_STANDARD 17)

# opencv
find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)
message(STATUS "OpenCV Libarary STATUS")
message(STATUS "    verison: ${OpenCV_VERSION}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libraries: ${OpenCV_LIBS}")

# cuda
find_package(CUDA REQUIRED)
set(CUDA_INCLUDES "/usr/local/cuda/targets/x86_64-linux/include")
set(CUDA_LIB_DIRS "/usr/local/cuda/targets/x86_64-linux/lib")
set(ALL_LIB cuda cudart cublas cudnn pthread stdc++ dl protobuf nvinfer nvinfer_plugin)
# set compile option
include_directories(${CUDA_INCLUDE_DIRS})


# enable language
message(STATUS "CUDA Libarary STATUS")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")

# eigen
# set(Eigen_INCLUDE_DIR "/usr/include/eigen3")
# include_directories(
#     ${Eigen_INCLUDE_DIR}
# )

# tensorrt
# set(TensorRT_DIR "/usr/local/TensorRT-8.5.1.7/targets/x86_64-linux-gnu")
# include_directories(
#     ${TensorRT_DIR}/include
# )
# link_directories(
#     ${TensorRT_DIR}/lib
# )

# add directories to build
# include_directories(
#     src
# )

# add subdirectory to build
set(COMMON_LIST
    test)  
    # src)

foreach(COMMON_ITER ${COMMON_LIST})
    add_subdirectory(${COMMON_ITER})
endforeach(COMMON_ITER)
