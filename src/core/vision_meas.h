/**
 * @file vision_meas.h
 * @author Jae Hyung Jung (lastflowers@snu.ac.kr)
 * @brief Vision measurerment preprocessor
 * @date 2021-05-04
 *
 * @copyright Copyright (c) 2021 Jae Hyung Jung
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */


#include <string>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>


#include "../utils/define.h"
#include "../utils/utils.h"

namespace nesl{

class vision_meas {
 public:
    vision_meas(const std_msgs::Header& header);
    
    vision_meas() {} // default constructor
    
    static void LoadParameters(ros::NodeHandle& n);

    void processStereoImage(const sensor_msgs::ImageConstPtr& img_l,
      const sensor_msgs::ImageConstPtr& img_r, const cv::Mat& mask);

    void processStereoORBImage(const sensor_msgs::ImageConstPtr& img_l,
      const sensor_msgs::ImageConstPtr& img_r, const cv::Mat& mask);

    // Getter
    inline const std::vector<cv::Point2d>& u_l() const { return u_l_; }

    inline const std::vector<cv::Point2d>& u_r() const { return u_r_; }

    inline const std::vector<double>& d_l() const { return depth_l_; }

    inline const std::vector<double>& d_r() const { return depth_r_; }

    inline double timestamp() { return timestamp_; }

    inline const cv::Mat& uimg_l() const { return uimg_l_; }

    inline const cv::Mat& uimg_r() const { return uimg_r_; }

    inline const M3d& Kl_inv() const { return Kl_inv_; }

    inline const M3d& Kl() const { return Kl_; }

    inline const M3d& Kr() const { return Kr_; }

    inline const std_msgs::Header& header() const { return header_; }

    inline const M4d& T_rl() const { return T_rl_; }


 private:
    // ROS header
    std_msgs::Header header_;

    // Number of grids
    static int nx_;
    static int ny_;

    // Triangulation threshold
    static double min_depth_;
    static double max_depth_;
    static double min_parallax_;
    static double ransac_thr_;

    // Raw image resolution 
    static int raw_wimg_, raw_himg_;

    // In/extrinsic matrix for left and right
    static M3d Kl_, Kr_, Kl_inv_, Kr_inv_;
    static M4d T_rl_, T_lr_;

    // Distortion coefficients for left and right
    static std::string distortion_model_;
    static V4d dcl_, dcr_;

    // Undistortion maps
    static cv::Mat undistort_map1_left_, undistort_map2_left_;
    static cv::Mat undistort_map1_right_, undistort_map2_right_;

    // Feature extraction margin size [pix]
    int marg_size_ = 10;

    // Feature extraction threshold (8bit)
    double thr_feature_ = 0.9;
    double thr_gradmin_ = 1e-2;
    double thr_gradmax_ = 400;

    // Current timestamp
    double timestamp_;

    // Undistorted left and right images
    cv::Mat uimg_l_, uimg_r_;

    // Feature locations and depths at left and right
    std::vector<cv::Point2d> u_l_, u_r_;
    std::vector<double> depth_l_, depth_r_;

    // Undistorted feature mask
    // 0: occupied, 1: free
    cv::Mat mask_margin_;

    void TwoViewTriangulation(const cv::Point2f& pt0,
      const cv::Point2f& pt1, double& depth0, double& depth1);

    bool IsFov(const cv::Point2f& pt);

    void epipolarLineSearch(const std::vector<cv::Point2f>& u0,
      std::vector<cv::Point2f>& u1, std::vector<double>& d0,
      std::vector<double>& d1, std::vector<uchar>& is_valid);

 };
}