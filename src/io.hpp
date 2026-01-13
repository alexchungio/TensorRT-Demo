#ifndef SRC_IO_HPP
#define SRC_IO_HPP

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/shape.hpp>

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>


namespace trtutils
{

    /**
     * @brief image struct
     * 
     */
    struct Tensor
    {
        std::string filename;
        int b;
        int c;
        int h;
        int w;
        float * data_ptr = nullptr;
    };

    class ImageBase
    {
        public:
           ImageBase(const std::string & filename, const nvinfer1::Dims & dims);
           virtual ~ImageBase() {}
           virtual size_t volume() const;
        protected:
            nvinfer1::Dims mDims;
            Tensor mTensor; 
    };

    class RGBImageReader : public ImageBase
    {
        public:
            RGBImageReader(const std::string & filename, const nvinfer1::Dims & dims, const std::vector<float> & mean, const std::vector<float> & std);
            cv::Mat read();
        private:
            std::vector<float> mMean;
            std::vector<float> mStd;
    };

    /**
     * @brief read and preprocess image
     * 
     * @param filename 
     * @param size 
     * @param mean 
     * @param std 
     * @return cv::Mat 
     */
    cv::Mat image_preprocess(const std::string filename, 
                             const int size[2],
                             const float mean[3],
                             const float std[3]);
} // namespace trtutils

#endif