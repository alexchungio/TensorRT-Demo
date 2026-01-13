#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "cv_demo.hpp"


int A::b = 3;


void fill_poly(cv::Mat& mask, const std::vector<std::vector<uint8_t>> val)
{   
    std::vector<std::vector<cv::Point>> polygons;
    std::vector<cv::Point> polygon;
    
    cv::Scalar color(1);
    for(int i=0; i<3; i++)
    {
        // polygon[0][i] = cv::Point(val[i][0], val[i][1]);
        polygon.push_back(cv::Point(val[i][0], val[i][1]));
    }
    polygons.push_back(polygon);
    cv::fillPoly(mask, polygons, color);
}




int main()
{
    // cv::Mat bgr_img = cv::imread(TEST_DATA_DIRS "conch.jpg");
    // cv::Mat resize_bgr_img;

    // cv::resize(bgr_img, resize_bgr_img, cv::Size(0, 0), 0.5, 0.5);
    // cv::imshow("image", resize_bgr_img);
    // cv::waitKey(0);
    // return 0;

    cv::Mat mask = cv::Mat::zeros(320, 320, CV_8UC1);
    
    std::vector<std::vector<uint8_t>> val = {
        {4, 4},
        {80, 10},
        {10, 30}
    };

    A a;
    
    a.b = 4;

    
    


    fill_poly(mask, val);

    std::map<std::string, std::vector<std::vector<uint8_t>>> demo_map;
    // // std::pair<std::string, uint8_t[3][2]> p = std::make_pair("fs", val);
    demo_map.insert(std::make_pair("fs", val));

    std::vector<std::vector<uint8_t>> tmp = demo_map.at("fs");


    cv::imshow("mask", mask*255);
    cv::waitKey();
    return 0;
    
}