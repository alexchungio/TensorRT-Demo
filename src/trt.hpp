#ifndef SRC_TRT_HPP
#define SRC_TRT_HPP

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>

#include "logger.hpp"


namespace trtutils
{
    /**
     * @brief create infer destroy
     * 
     */
    struct InferDeleter
    {
        template<typename T>
        void operator()(T* obj) const
        {
            if(obj)
            {
                obj->destroy();
            }
        }
    };

    // detector ptr
    template<typename T>
    using UniquePtr = std::unique_ptr<T, trtutils::InferDeleter>;

    /**
     * @brief Get the memory size object
     * 
     * @param dims 
     * @param elem_size 
     * @return size_t 
     */
    size_t get_memory_size(const nvinfer1::Dims& dims, const int32_t elem_size);

    class TRTEngine
    {
        public:
            /**
             * @brief Construct a new TRTEngine object
             * 
             * @param engine_name 
             * @param input_dims 
             * @param output_dims 
             */
            TRTEngine(const std::string & engine_name, nvinfer1::Dims input_dims, nvinfer1::Dims output_dims);
            /**
             * @brief run the tensorrt inference
             *        allocate the input and output memory, and execute the engine
             * @return
             */
            bool infer(const float * input_data, nvinfer1::DataType dtype);
        private:
            std::string mEnineFilename;
            nvinfer1::Dims mInputDims;
            nvinfer1::Dims mOutputDims;
            trtutils::UniquePtr<nvinfer1::ICudaEngine> mEngine;
            trtutils::Logger mLogger;
            // std::vector<void *> mBindings;
    };
} // trtutils


#endif