#include <iostream>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>


namespace fs = std::filesystem;  // c++17

// convert label freespace -> 0 & obstacle -> 1
void convetMaskToBinary(const cv::Mat & mask, cv::Mat & binary_mask){
    for (int i=0; i<320; i++) {
        for(int j=0; j<320; j++) {
            uint8_t val = mask.at<uchar>(i, j);
            if (val == 255) { // for debug
            // if (val == 1) {
                binary_mask.at<uchar>(i, j) = 0;
            } else {
                binary_mask.at<uchar>(i, j) = 1;
            }
        }
    }
    
}

void convetMaskToBinary(uint8_t * mask_data, cv::Mat & binary_mask){
    for (int i=0; i<320; i++) {
        for(int j=0; j<320; j++) {
            // uint8_t val = mask.at<int>(i, j);
            uint8_t val = *(mask_data + i*320 +j);
            
            if (val == 255) { // for debug
            // if (val == 1) {
                binary_mask.at<uchar>(i, j) = 0;
            } else {
                binary_mask.at<uchar>(i, j) = 255;
            }
        }
    }
    
}

// detect small region
void detectSamllRegions(const cv::Mat& binary_mask, std::vector<std::vector<int>> & small_regions, cv::Mat& labels, cv::Mat& stats, cv::Mat& centroids, const int area_thres) {
    int num_regions = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids);
    
    for(int i=1; i<num_regions; i++) {
        std::vector<int> region;
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area <= area_thres) {
            region.push_back(x);
            region.push_back(y);
            region.push_back(w);
            region.push_back(h);
            region.push_back(area);
            small_regions.push_back(region);
        }

    }
   
}

float calAreaMinWidth(const int& w, const int& h, const float& area) {
    float length = std::sqrt(w*w + h*h);
    float width = area / length;
    
    return width;
}

void removeRegion(cv::Mat& mask, const int& x, const int& y, const int& w, const int& h) {
    for(int i=y; i<y+h; i++){
        for(int j=x; j<x+w; j++) {
            mask.at<uchar>(i, j) = 255; // for debug
            // mask.at<uchar>(i, j) = 1;
        }
    }
   
}

void removeRegion(uchar * mask_data, const int& x, const int& y, const int& w, const int& h) {
    for(int i=y; i<y+h; i++){
        for(int j=x; j<x+w; j++) {
            *(mask_data + i * 320 + j)= 255; // for debug
            // mask.at<uchar>(i, j) = 1;
        }
    }
}


void filterSmallRegion(uchar * mask_data, std::vector<std::vector<int>>& small_regions, const int& min_area, const int& second_min_area, const float& second_min_width) {
    for(const auto& region: small_regions) {
        int x = region[0];
        int y = region[1];
        int w = region[2];
        int h = region[3];
        float area = (float) region[4];
        if (area <= min_area) {
            removeRegion(mask_data, x, y, w, h);
        } else if(area < second_min_area) {
            float min_width = calAreaMinWidth(w, h, area);
            if (min_width < second_min_width) {
                removeRegion(mask_data, x, y, w, h);
            }
        }
    }
    small_regions.clear();
}


int main()
{
    int small_area_thres = 60;
    int min_area = 10;
    int second_min_area = 40;
    float second_min_width = 2.0;
    std::string data_path = "";
    
    cv::Mat origin_mask(320, 320, CV_8UC1);
    cv::Mat binary_mask {0};
    binary_mask = cv::Mat::ones(320, 320, CV_8UC1);
    uint8_t * origin_mask_data;
    // cv::Mat filter_mask(320, 320, CV_8UC1);
    std::vector<std::vector<int>> small_regions;
    
    int MAX_COMPONENT = 150;
    // cv::Mat labels, stats, centroids;
    cv::Mat labels(binary_mask.size(), CV_32S);
    cv::Mat stats(MAX_COMPONENT, 5, CV_32S);
    cv::Mat centroids(MAX_COMPONENT, 2, CV_64F);

    // load filename
    std::vector<std::string> filenames;
    for (const auto& entry: fs::directory_iterator(data_path)) {
        if(entry.is_regular_file()) {
            filenames.push_back(entry.path().filename().string());
        }
    }
    std::sort(filenames.begin(), filenames.end());

    
    for(auto& filename: filenames) {
        std::string img_path = data_path + "/" + filename;
        origin_mask = cv::imread(img_path, cv::IMREAD_UNCHANGED);
        origin_mask_data = (uint8_t *)origin_mask.data;
        // cv::imshow("origin_mask", origin_mask);
        // convetMaskToBinary(origin_mask, binary_mask);
        convetMaskToBinary(origin_mask_data, binary_mask);
        detectSamllRegions(binary_mask, small_regions, labels, stats, centroids, small_area_thres);
        filterSmallRegion(origin_mask_data, small_regions, min_area, second_min_area, second_min_width);
        
        cv::imshow("binary_mask", binary_mask*255);
        cv::imshow("filter_mask", origin_mask);
        // cv::imshow("binary_mask", binary_mask);
        cv::waitKey();
        
    }
    

    return 0;
}