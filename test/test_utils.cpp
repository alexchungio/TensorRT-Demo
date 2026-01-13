#include <algorithm>
#include <vector>
#include <numeric>

#include "NvInfer.h"
#include "io.hpp"
#include "trt.hpp"
#include <cuda_runtime_api.h>


using namespace trtutils;

int main()
{   
    const std::string filename = TEST_DATA_DIRS "conch.jpg";
    const std::string model_name = MODEL_DIRS "resnet50_fp16.engine";       



               
    const int input_shape[2] = {224, 224};
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    auto input_dims = nvinfer1::Dims4(1, 3, input_shape[0], input_shape[1]);
    auto output_dims = nvinfer1::Dims2(1, 1000);
    int input_size = input_dims.d[0] * input_dims.d[1] * input_dims.d[2] * input_dims.d[3];
   
    auto image_read = RGBImageReader(filename, input_dims, mean, std);
    auto engine = TRTEngine(model_name, input_dims, output_dims);

    cv::Mat input_batch = image_read.read();
    
    // memcpy(input_data, input_batch.data, input_size * sizeof(float));

    engine.infer((float *)input_batch.data, nvinfer1::DataType::kHALF);
    // cv::Mat img_tensor = image_preprocess(filename, size, mean, std);
    return 0;
}     