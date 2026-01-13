cmake_minimum_required(VERSION 3.0.2)

project(test)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# add executable    
macro(add_eval name sources)
    add_executable(${name} ${sources})
    target_link_libraries(${name} ${OpenCV_LIBS})
    # target_link_libraries(${name} src)
    # target_link_libraries(${name} cuda cudart nvinfer)
endmacro()

add_eval(cv_demo ../cv_sample/cv_demo.cpp)
add_eval(fs_postprocess ../cv_sample/fs_postprocess.cpp)
# add_eval(eigenDemo ../eigen_sample/eigen_demo.cpp)
# add_eval(trtDemo test_utils.cpp )
# add_eval(cudaDemo ../cuda_sample/cuda_demo.cu)