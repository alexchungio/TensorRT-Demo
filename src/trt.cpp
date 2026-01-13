#include <algorithm>
#include <fstream>
#include <vector>
#include <numeric>
#include <chrono>

#include "io.hpp"
#include "tools.hpp"
#include "trt.hpp"


namespace trtutils
{
    size_t get_memory_size(const nvinfer1::Dims& dims, const int32_t elem_size)
    {
        return std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<int64_t>()) * elem_size;
    }

    TRTEngine::TRTEngine(const std::string & engine_name, nvinfer1::Dims input_dims, nvinfer1::Dims output_dims):
    mEnineFilename(engine_name), mInputDims(input_dims), mOutputDims(output_dims), mEngine(nullptr)
    {
        // deserialize engine
        std::ifstream engine_file(mEnineFilename, std::ios::binary);
        if(engine_file.fail())
        {
            return;
        }
        engine_file.seekg(0, std::ifstream::end);
        auto fsize = engine_file.tellg();
        engine_file.seekg(0, std::ifstream::beg);
        
        // read engine data
        std::vector<char> engine_data(fsize);
        engine_file.read(engine_data.data(), fsize);
        
        // create runtime
        trtutils::UniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(mLogger)};
        mEngine.reset(runtime->deserializeCudaEngine(engine_data.data(), fsize, nullptr));
        assert(mEngine.get() != nullptr);
    }

    bool TRTEngine::infer(const float * input_data, nvinfer1::DataType dtype)
    {
        // create context
        auto context = trtutils::UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        if(!context) return false;

        // allocate input memory
        auto input_idx = mEngine->getBindingIndex("data"); // the name of input name
        if(input_idx == -1) return false;
        assert(mEngine->getBindingDataType(input_idx) == dtype);
        // context->setBindingDimensions(input_idx, mInputDims);
        auto input_dims = context->getBindingDimensions(input_idx);
        auto input_size = get_memory_size(input_dims, sizeof(float)); // 3*224*224*4 = 602112
        if(dtype == nvinfer1::DataType::kHALF)
        {
            input_size /= 2;
        }
        void * input_mem = nullptr;
        if(cudaMalloc(&input_mem, input_size) != cudaSuccess)
        {
            gLogError << "Error: input data memory allocated falied, size = " << input_size << "bytes" << std::endl;
            return false;
        }
        // allocate output memory
        auto output_idx = mEngine->getBindingIndex("output");
        if(output_idx == -1) return false;
        assert(mEngine->getBindingDataType(output_idx) == dtype);
        auto output_dims = context->getBindingDimensions(output_idx);
        auto output_size = get_memory_size(output_dims, sizeof(float)); // 1*1000*4 = 4000
        if(dtype == nvinfer1::DataType::kHALF)
        {
            output_size /= 2;
        }
        auto output_data_half = std::unique_ptr<ushort []>(new ushort[output_size]);
        auto output_data_float = std::unique_ptr<float []>(new float[output_size]);
        void * output_mem = nullptr;
        if(cudaMalloc(&output_mem, output_size) != cudaSuccess)
        {
            gLogError << "Error: output data memory allocated falied, size = " << output_size << "bytes" << std::endl;
            return false;
        }  

        // create stream
        cudaStream_t stream;
        if(cudaStreamCreate(&stream) != cudaSuccess)
        {
            gLogError << "Error: stream create failed" << std::endl;
            return false;
        }

        // count time 
        int run_times = 1000;
        auto start = std::chrono::steady_clock::now();

        for(int i=0; i<run_times; i++)
        {
            // float v = input_data[0];
            // copy image data to input bind memory
            // input_data -> input_mem
            if(cudaMemcpyAsync(input_mem, input_data, input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
            {
                gLogError << "Error: CUDA memory copy of input falied, size = " << input_size << "bytes" << std::endl;
            }

            //run tensorRT
            // assert(mEngine->getNbBindings() == 2);
            // void * bindings[2]{nullptr}; // create buffer
            // bool status = context->enqueue(1, bindings, stream, nullptr);
            void * bindings[] = {input_mem, output_mem};
            bool status = context->enqueueV2(bindings, stream, nullptr);
            if(!status)
            {
                gLogError << "Error: TensorRT inferrence failed" << std::endl;
                return false;  
            }

            // copy prediction from output bind to memory
            // output_mem -> output_data
            if(dtype == nvinfer1::DataType::kHALF)
            {
                if(cudaMemcpyAsync(output_data_half.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess){
                    gLogError << "Error: CUDA memory copy of output failed, size = " << output_size << "bytes" << std::endl;
                    return false;
                }

            }else
            {
                if(cudaMemcpyAsync(output_data_float.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess){
                    gLogError << "Error: CUDA memory copy of output failed, size = " << output_size << "bytes" << std::endl;
                    return false;
                }

            }
            
            // synchronize threads
            cudaStreamSynchronize(stream);
        }
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Cost time: " << elapsed.count() / run_times << std::endl;

        // output prediction result
        std::vector<float> predicts(mOutputDims.d[1]);
        for(int i=0; i<mOutputDims.d[1]; i++)                                                         
        {
            if(dtype == nvinfer1::DataType::kHALF)
            {
                predicts[i] = trtutils::half_to_float(output_data_half[i]);

            }else
            {
                predicts[i] = output_data_float[i];

            }  
        }
        auto argmax = trtutils::argmax_idx(predicts);

        std::cout << "pred index: " << argmax[0] << "|prob: " << predicts[argmax[0]] << std::endl;
    
        // free space
        cudaStreamDestroy(stream);
        cudaFree(input_mem);
        cudaFree(output_mem);

        return true;     
    }
    
} // namespace trtutils
