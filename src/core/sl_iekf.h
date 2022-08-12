/**
 * @file sl_iekf.h
 * @author Jae Hyung Jung (lastflowers@snu.ac.kr)
 * @brief Stochastic linearization iterated EKF on matrix Lie group
 * @date 2021-05-12
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

#include <random>
#include <chrono>
#include <list>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>

#include "../utils/define.h"
#include "../utils/Attitude.h"
#include "../utils/utils.h"
#include "vision_meas.h"

namespace nesl {

class sl_iekf {
 public:
    sl_iekf();

    void InitializeState(std::vector<sensor_msgs::ImuConstPtr>& imu_buf);

    void LoadParameters(ros::NodeHandle &n);

    void registerPub(ros::NodeHandle &n);

    void publishPoseAndMap(const std_msgs::Header& header,
        const cv::Mat& uimg_l, const cv::Mat& uimg_r);

    void propagate(const sensor_msgs::ImuConstPtr& imu_msg,
        const double& dt);

    void processMeasurement(const nesl::vision_meas& vis_meas);

    // Getters
    inline const double& getTimeOffset() const { return td_; }

    inline const cv::Mat& mask() const { return mask_; }

    inline V4d q_gb() { 
        M3d R = Xi_.block<3,3>(0,0);
        V4d q = nesl::dcm2quat(R);
        return q; }

    inline V3d p_gb() { return Xi_.block<3,1>(0,3); }

    inline V3d v_gb() { return Xi_.block<3,1>(0,4); }

    inline V3d ba() { return ba_; }

    inline V3d bg() { return bg_; }

    inline V3d std_q() {
        V3d std;
        std << sqrt(cov_(0,0)), sqrt(cov_(1,1)), sqrt(cov_(2,2));
        return std; }

    inline V3d std_p() {
        V3d std;
        std << sqrt(cov_(3,3)), sqrt(cov_(4,4)), sqrt(cov_(5,5));
        return std; }

    inline V3d std_v() {
        V3d std;
        std << sqrt(cov_(6,6)), sqrt(cov_(7,7)), sqrt(cov_(8,8));
        return std; }

    inline V3d std_ba() {
        V3d std;
        std << sqrt(cov_(9,9)), sqrt(cov_(10,10)), sqrt(cov_(11,11));
        return std; }

    inline V3d std_bg() {
        V3d std;
        std << sqrt(cov_(12,12)), sqrt(cov_(13,13)), sqrt(cov_(14,14));
        return std; }

    int NUM_INIT_SAMPLES_;

    M3d Rvi_;

 private:
    // Filter state
    M5d Xi_;
    M4d T0_;
    V3d ba_;
    V3d bg_;

    struct photometric_map {
        int life_time_;  // num of tracking
        V2d uv_l_;  // Pixel coordinate on left
        V2d uv_r_;  // Pixel coordinate on right
        double depth_l_;  // depth on left
        double depth_r_;  // depth on right
        std::vector<std::vector<double>> intensity_; // pyramidal intensities

        V3d pg_f_;  // global feature position
    };
    // Contained in the filter state
    std::vector<photometric_map> photo_map_;

    // For drawing: xyz global position and intensity
    std::vector<V4d> draw_marg_map_;

    // Filter matrices
    Eigen::MatrixXd cov_;
    M12d Q_;
    double R_;

    // Parameters
    V3d grav_;

    M4d Tlb_;

    M4d Tbl_;

    double td_;

    double rho_var0_;  // [m^2] or [m^-2]

    int thr_num_;  // max num of tracking features

    int max_iter_;  // max iter num of iterated ekf

    double max_itime_;  // max time for early break

    double thr_stop_;  // stop threshold for iterated ekf

    int max_lifetime_;

    int uniform_dist_;

    double max_diff_;

    double draw_max_depth_;

    int max_lvl_ = 3;  // maximum number of pyramid level

    // Pattern padding
    Eigen::MatrixXi pad_;
    int idx_center_;

    cv::Mat mask_;

    // Previous left image for tracking quality check
    cv::Mat uimg_l_prev_;

    double marg_size_ = 10;  // margin size [px]

    int Nen_;  // Number of ensembles
    bool use_sl_;  // use stochastic gradient?
    double thr_weak_;  // sl threshold

    // Dimensions
    int Dx = 15;
    int Ds = 6;

    bool is_idepth_;

    // Feature candidate manager
    std::vector<cv::Point2f> ul_prev_;
    std::vector<double> dl_prev_;
    std::vector<int> ul_ntrack_;
    int max_candidate_ = 100;

    // Ensemble generator
    std::default_random_engine generator_;

    // INS implementation
    void propagateIMU_Euler(const V3d& fib_b, const V3d& wib_b,
        const double& dt);

    void propagateCovariance(const double& dt);

    void marginalizeFromIndex(const std::vector<int>& marg_idx);

    void marginalizeEnsemble(const std::vector<int>& marg_idx,
        Eigen::MatrixXd& Z_en);

    void trackFeatures(std::vector<cv::Mat> pyr_uimg,
        const M3d& Kl, const M3d& Kl_inv, const M3d& Kr, const M4d& T_rl);

    void stateReplacement();

    void initializeFeatures(std::vector<cv::Mat> pyr_uimg,
        const std::vector<cv::Point2d>& u_l,
        const std::vector<double>& d_l,
        const std::vector<cv::Point2d>& u_r,
        const std::vector<double>& d_r,
        const M3d& Kl_inv);

    void initializeDelayFeatures(const cv::Mat& uimg_l,
        const std::vector<cv::Point2d>& u_l,
        const std::vector<double>& d_l,
        const std::vector<cv::Point2d>& u_r,
        const std::vector<double>& d_r,
        const M3d& Kl_inv,
        const M3d& Kr,
        const M4d& Trl);

    void updateMask(const cv::Mat& uimg_l);

    void filterUpdate(std::vector<cv::Mat> pyr_uimg,
        const M3d& Kl);

    void Exp_sed3(const V9d& xi, M5d& X);

    void Exp_se3(const V6d& xi, M4d& X);

    void validateCovMtx(const Eigen::MatrixXd& P_mtx,
        Eigen::MatrixXd& P_pd);

    void sampleDeltaEnsembles(std::vector<V6d>& delta_T0,
        std::vector<V6d>& delta_T1, std::vector<double>& delta_Z);

    void sampleEnsembles(const std::vector<V6d>& delta_T0,
        const std::vector<V6d>& delta_T1, const std::vector<double>& delta_Z,
        std::vector<M4d>& T0_en, std::vector<M4d>& T1_en,
        std::vector<Eigen::VectorXd>& Z_en);

    void computeStochasticGradient(const cv::Mat& uimg_l, const M3d& Kl,
        const M3d& Kl_inv, const V3d& uv0_h, const V3d& uv1_h,
        const std::vector<M4d>& T0_en, const std::vector<M4d>& T1_en,
        const std::vector<Eigen::VectorXd>& Z_en, const int& jth,
        double& dI_x, double& dI_y);

    ros::Publisher pub_state_;
    ros::Publisher pub_path_;
    ros::Publisher pub_tracking_points_;
    ros::Publisher pub_marg_points_;
    ros::Publisher pub_match0_, pub_match1_;
    nav_msgs::Path path_;

    void TwoViewTriangulation(const cv::Point2f& pt0,
        const cv::Point2f& pt1, const M3d& K0_inv, const M3d& K1_inv,
        const M4d& T_01, double& depth0, double& depth1);


    inline void reduceVector(std::vector<cv::Point2f> &v,
        std::vector<uchar> status) {
        // This function was copied from T. Qin, P. Li, Z. Yang, and S. Shen in VINS-Mono (feature_tracker.cpp)
        // https://github.com/HKUST-Aerial-Robotics/VINS-Mono
        int j = 0;
        for (int i = 0; i < int(v.size()); i++)
            if (status[i])
                v[j++] = v[i];
        v.resize(j);
    }

    inline void reduceVector(std::vector<int> &v,
        std::vector<uchar> status) {
        // This function was copied from T. Qin, P. Li, Z. Yang, and S. Shen in VINS-Mono (feature_tracker.cpp)
        // https://github.com/HKUST-Aerial-Robotics/VINS-Mono
        int j = 0;
        for (int i = 0; i < int(v.size()); i++)
            if (status[i])
                v[j++] = v[i];
        v.resize(j);
    }

    inline void reduceVector(std::vector<double> &v,
        std::vector<uchar> status) {
        // This function was copied from T. Qin, P. Li, Z. Yang, and S. Shen in VINS-Mono (feature_tracker.cpp)
        // https://github.com/HKUST-Aerial-Robotics/VINS-Mono
        int j = 0;
        for (int i = 0; i < int(v.size()); i++)
            if (status[i])
                v[j++] = v[i];
        v.resize(j);
    }


};
}