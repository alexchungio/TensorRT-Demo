#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <numeric>
#include <cassert>

#include <NvInfer.h>
#include "io.hpp"


using namespace nvinfer1;

namespace trtutils
{


    ImageBase::ImageBase(const std::string & filename, const nvinfer1::Dims & dims)
    :mDims(dims)
    {
        assert(dims.nbDims == 4);
        assert(dims.d[0] == 1);
        mTensor.filename = filename;

    }

    size_t ImageBase::volume() const
    {
        return mDims.d[2] * mDims.d[3] * 3; /* h x w x 3*/
    }

    RGBImageReader::RGBImageReader(const std::string & filename, const nvinfer1::Dims & dims, const std::vector<float> & mean, const std::vector<float> & std)
    :ImageBase(filename, dims), mMean(mean), mStd(std)
    {
        
    }


    void _blob_from_images(cv::InputArrayOfArrays images_, cv::OutputArray blob_, cv::Size size_, 
                       const cv::Scalar& mean_, const cv::Scalar& std_, bool is_bgr_ = false, int dtype_ = CV_32F){
    // image preprocess
    std::vector<cv::Mat> images;
    images_.getMatVector(images);  // Array to Mat
    CV_Assert(!images.empty());
    for (int i = 0; i < images.size(); i++)
    {
        cv::Size img_size = images[i].size();
        // resize
        if (size_ != img_size)
        {
            resize(images[i], images[i], size_, 0, 0, cv::INTER_LINEAR);
        }
        // convert bgr to rgb
        if(is_bgr_)
        {
            cv::cvtColor(images[i], images[i], cv::COLOR_BGR2RGB);
        }
        // normlize
        if (images[i].depth() == CV_8U)
            images[i].convertTo(images[i], CV_32F, 1.0 / 255.);
        cv::subtract(images[i], mean_, images[i]);
        cv::divide(images[i], std_, images[i]);
        // cv::Vec3f pixel_val_sub_mean_divide_std = images[i].at<cv::Vec3f>(0, 0);
    }

    // convert mat to tensor
    size_t _, num_imgs = images.size();
    int height = images[0].rows;
    int width = images[0].cols;
    int num_chs = images[0].channels();

    cv::Mat blob;
    CV_Assert(images[0].dims == 2);
    // CV_Assert(images[0].depth() == dtype_);
    if (num_chs == 3 || num_chs == 4)
    {
        int blob_size[] = {(int)num_imgs, num_chs, height, width};
        blob_.create(4, blob_size, dtype_);  // allocate memery
        blob = blob_.getMat();  
        cv::Mat img_chs[4]; 

        for (int i = 0; i < num_imgs; i++)
        {
            cv::Mat image = images[i];
            cv::split(image, img_chs); // split RGB channels
            for (int j = 0; j < num_chs; j++)
            {
                img_chs[j].copyTo(cv::Mat(image.rows, image.cols, dtype_, blob.ptr((int)i, j)));

            }
        }  
    }
    else
    {
        CV_Assert(num_chs == 1);
        int blob_size[] = {(int)num_imgs, 1, height, height};
        blob_.create(4, blob_size, dtype_);
        cv::Mat blob = blob_.getMat();
        for (int i = 0; i < num_imgs; i++)
        {
            cv::Mat image;
            images[i].convertTo(image, dtype_);
            image.copyTo(cv::Mat(image.rows, image.cols, dtype_, blob.ptr((int)i, 0)));
        }
    }
    // if(dtype_ == CV_16F) cv::convertFp16(blob_, blob_);
    // verification
    // get (0, 1, 112, 112) => 224 * 224 + 224 * 112 + 112 = 75376 pixel value => 2.1309526
    // float * data_ptr = (float *)blob.data;
    // float r_0_val = data_ptr[0];
    // cv::waitKey(0);
    }

    cv::Mat blob_from_images(cv::InputArrayOfArrays images, cv::Size size,
        const cv::Scalar& mean, const cv::Scalar& std_num, bool is_bgr=false, int dtype = CV_32F){

        cv::Mat blob;
        _blob_from_images(images, blob, size, mean, std_num, is_bgr, dtype);

        return blob;
    }


    cv::Mat RGBImageReader::read()
    {
        cv::Mat bgr_img, dst_img, tensor_img;
        bgr_img = cv::imread(mTensor.filename, cv::IMREAD_COLOR);
        cv::resize(bgr_img, bgr_img, cv::Size(mDims.d[2], mDims.d[3]));
        cv::cvtColor(bgr_img, dst_img, cv::COLOR_BGR2RGB);


        // normallize
        // dst_img.convertTo(dst_img, CV_32FC3, 1.0 / 255.);
        // cv::subtract(dst_img, cv::Scalar(mMean[0], mMean[1], mMean[2]), dst_img, cv::noArray(), -1);
        // cv::divide(dst_img, cv::Scalar(mStd[0], mStd[1], mStd[2]), dst_img, 1, -1);

        // // mat to tensor
        // cv::dnn::blobFromImage(dst_img, tensor_img, 1.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false, CV_32F);
        std::vector<cv::Mat> blob_images = {dst_img};
        tensor_img = blob_from_images(blob_images, cv::Size(224, 224), cv::Scalar(mMean[0], mMean[1], mMean[2]),
                                      cv::Scalar(mStd[0], mStd[1], mStd[2]), false, CV_16F);
        return tensor_img;
 
        // mTensor.data_ptr = (float *){new float[volume()]};
        // mTensor.data_ptr = (float *) tensor_img.data;
        // input_data = (float *){new float[volume()]};
        // input_data = (float *) tensor_img.data;
    }
} // namespace trtutils
