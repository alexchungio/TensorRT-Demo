#include <opencv2/opencv.hpp>



class FreespacePostprocess {
public:          
    static bool get_bottom_gray_level_mean( const cv::Mat& hist, 
                                            const int size, 
                                            float& mean_gray);
    /**
     * @brief generate filter area
     *
     * @return bool
     */           
    static bool GenerateFilterArea( cv::Mat& img_mask, 
                                    const std::vector<int> veh_coord, 
                                    const float piece_det);
    
    /**
     * @brief convet gray mask to binary mask
     *
     * @return bool
     */
    static bool convetMaskToBinary(const cv::Mat & mask, cv::Mat & binary_mask);
    static bool convetMaskToBinary(uchar * mask_data, cv::Mat & binary_mask);

    /**
     * @brief detect small region of obstacle
     *
     * @return bool
     */
    static bool detectSamllRegions(const cv::Mat& binary_mask, 
                                   std::vector<std::vector<int>>& small_regions,
                                   cv::Mat& labels,
                                   cv::Mat& stats, 
                                   cv::Mat& centroids,
                                   const int area_thres);
    
    /**
     * @brief calculate min width of area
     *
     * @return bool
     */
    static bool calAreaMinWidth(const int& w, const int& h, const float& area, float& width);

    /**
     * @brief remove area
     *
     * @return bool
     */
    static bool removeRegion(cv::Mat& mask, const cv::Mat& filter_area_mask,  const int& x, const int& y, const int& w, const int& h);
    static bool removeRegion(uchar* mask_data, const cv::Mat& filter_area_mask, const int& x, const int& y, const int& w, const int& h);
    
    /**
     * @brief Filter small obstacle area with small region
     *
     * @return bool
     */           
    static bool FilterSmallObstacelRegion(uchar * fs_outs,
                                          const cv::Mat& filter_area_mask,
                                          cv::Mat& filter_binary_mask,
                                          std::vector<std::vector<int>>& small_regions,
                                          cv::Mat& labels,
                                          cv::Mat& stats, 
                                          cv::Mat& centroids,
                                          const float small_area_thres, 
                                          const float min_area_thres,
                                          const float second_min_area_thres,
                                          const float second_min_width_thres);

private:
    // config file param
  
    int m_bufferIndex = 0;
    std::vector<cv::Mat> m_vecMatFs   = {};
    std::vector<cv::Mat> m_vecMatSign = {};
    std::vector<uchar> imgVec         = {};
}; 