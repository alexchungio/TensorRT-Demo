# include "fs_postprocess_new.hpp"


class FreespacePostprocess {
public:          
   
    bool FreespacePostprocess::GenerateFilterArea( cv::Mat& img_mask, const std::vector<int> veh_coord, const float piece_det) 
    {
        float frontAlpha = 30;
        float backAlpha = 75;
        float frontAlpha_min = frontAlpha - piece_det;
        float backAlpha_min = backAlpha - piece_det;
        float frontAlpha_max = frontAlpha + piece_det;
        float backAlpha_max = backAlpha + piece_det;
        frontAlpha_min = frontAlpha_min * CV_PI / 180;
        backAlpha_min = backAlpha_min * CV_PI / 180;
        frontAlpha_max = frontAlpha_max * CV_PI / 180;
        backAlpha_max = backAlpha_max * CV_PI / 180;
        float theta = 0;
        for (int i = 0; i < img_mask.rows; i++) {
            for (int j = 0; j < img_mask.cols; j++) {
                if ( i < veh_coord[1] && j < veh_coord[0] ){
                    theta = atan2(std::abs(int(j - veh_coord[0])), std::abs(int(i - veh_coord[1])));
                    if (theta > frontAlpha_min && theta < frontAlpha_max) {
                        img_mask.at<uchar>(i, j) = 0;
                    }
                }
                else if (i < veh_coord[1] && j > veh_coord[2]) {
                    theta = atan2(std::abs(int(j - veh_coord[2])), std::abs(int(i - veh_coord[1])));
                    if (theta > frontAlpha_min && theta < frontAlpha_max) {
                        img_mask.at<uchar>(i, j) = 0;
                    }
                }
                else if (i > veh_coord[3] && j < veh_coord[0]) {
                    theta = atan2(std::abs(int(j - veh_coord[0])), std::abs(int(i - veh_coord[3])));
                    if (theta > backAlpha_min && theta < backAlpha_max) {
                        img_mask.at<uchar>(i, j) = 0;
                    }
                }
                else if (i > veh_coord[3] && j > veh_coord[2]) {
                    theta = atan2(std::abs(int(j - veh_coord[2])), std::abs(int(i - veh_coord[3])));
                    if (theta > backAlpha_min && theta < backAlpha_max) {
                        img_mask.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
        return true;
    }
    
    bool convetMaskToBinary(const cv::Mat & mask, cv::Mat & binary_mask)
    {
        for (int i=0; i<320; i++) {
            for(int j=0; j<320; j++) {
                uint8_t val = mask.at<uchar>(i, j);
                // if (val == 255) { // for debug
                if (val == 1) {
                    binary_mask.at<uchar>(i, j) = 0;
                } else {
                    binary_mask.at<uchar>(i, j) = 1;
                }
            }
       }
        return true;

    }
    bool convetMaskToBinary(uchar * mask_data, cv::Mat & binary_mask)
    {   
        for (int i=0; i<320; i++) {
            for(int j=0; j<320; j++) {
                uint8_t val = *(mask_data + i * 320 + j);
                // if (val == 255) { // for debug
                if (val == 1) {
                    binary_mask.at<uchar>(i, j) = 0;
                } else {
                    binary_mask.at<uchar>(i, j) = 1;
                }
            }
        }
        return true;
    }

    bool detectSamllRegions(const cv::Mat& binary_mask, 
                                   std::vector<std::vector<int>>& small_regions,
                                   cv::Mat& labels,
                                   cv::Mat& stats, 
                                   cv::Mat& centroids,
                                   const int area_thres)
    
    {
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
        return true;                               
    }
    
    bool calAreaMinWidth(const int& w, const int& h, const float& area, float& width)
    {
        float length = std::sqrt(w*w + h*h);
        width = area / length;
        
        return true;
    }


    
    bool removeRegion(cv::Mat& mask, const cv::Mat& filter_area_mask,  const int& x, const int& y, const int& w, const int& h)
    {
        for(int i=y; i<y+h; i++){
            for(int j=x; j<x+w; j++) {
                if(filter_area_mask.at<int>(i, j) == 1) {
                    // mask.at<uchar>(i, j) = 255; // for debug
                    mask.at<uchar>(i, j) = 1;
                }
                
            }
        }
        return true;
    }   
    bool removeRegion(uchar* mask_data, const cv::Mat& filter_area_mask, const int& x, const int& y, const int& w, const int& h)
    {
        for(int i=y; i<y+h; i++){
            for(int j=x; j<x+w; j++) {
                if(filter_area_mask.at<int>(i, j) == 1) {
                    // mask.at<uchar>(i, j) = 255; // for debug
                    *(mask_data + i * 320 + j) = 1;
                }
            }
        }
        return true;
    }
              
    bool FilterSmallObstacelRegion(uchar * fs_outs,
                                          const cv::Mat& filter_area_mask,
                                          cv::Mat& filter_binary_mask,
                                          std::vector<std::vector<int>>& small_regions,
                                          cv::Mat& labels,
                                          cv::Mat& stats, 
                                          cv::Mat& centroids,
                                          const float small_area_thres, 
                                          const float min_area_thres,
                                          const float second_min_area_thres,
                                          const float second_min_width_thres)
    {
    
        if(!convetMaskToBinary(fs_outs, filter_binary_mask)) {
            
        }

        if(!detectSamllRegions(filter_binary_mask, small_regions, labels, stats, centroids, small_area_thres)) {
        }
        
        for(const auto& region: small_regions) {
            int x = region[0];
            int y = region[1];
            int w = region[2];
            int h = region[3];
            float area = (float) region[4];
            float min_width;
            if (area <= min_area_thres) {
                removeRegion(fs_outs, filter_area_mask, x, y, w, h);
            } else if(area < second_min_area_thres) {
                calAreaMinWidth(w, h, area, min_width);
                if (min_width < second_min_width_thres) {
                    removeRegion(fs_outs, filter_area_mask, x, y, w, h);
                }
            }
        }


        small_regions.clear();
        
        return true;
    }

}; 