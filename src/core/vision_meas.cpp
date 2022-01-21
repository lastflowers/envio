/**
 * @file vision_meas.cpp
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

#include "vision_meas.h"

namespace nesl {

// Static variable definition
int vision_meas::nx_ = 0;
int vision_meas::ny_ = 0;
int vision_meas::raw_wimg_ = 0;
int vision_meas::raw_himg_ = 0;

double vision_meas::min_depth_ = 0;
double vision_meas::max_depth_ = 0;
double vision_meas::min_parallax_ = 0;
double vision_meas::ransac_thr_ = 0;

M3d vision_meas::Kl_ = M3d::Zero();
M3d vision_meas::Kr_ = M3d::Zero();
M3d vision_meas::Kl_inv_ = M3d::Zero();
M3d vision_meas::Kr_inv_ = M3d::Zero();

M4d vision_meas::T_rl_ = M4d::Zero();
M4d vision_meas::T_lr_ = M4d::Zero();

std::string vision_meas::distortion_model_ = "radtan";
V4d vision_meas::dcl_ = V4d::Zero();
V4d vision_meas::dcr_ = V4d::Zero();

cv::Mat vision_meas::undistort_map1_left_ = cv::Mat::zeros(1,1,CV_16SC2);
cv::Mat vision_meas::undistort_map2_left_ = cv::Mat::zeros(1,1,CV_16SC2);
cv::Mat vision_meas::undistort_map1_right_ = cv::Mat::zeros(1,1,CV_16SC2);
cv::Mat vision_meas::undistort_map2_right_ = cv::Mat::zeros(1,1,CV_16SC2);


vision_meas::vision_meas(const std_msgs::Header& header) {
    // Set current timestamp
    header_ = header;
    timestamp_ = header.stamp.toSec();
    
    // Set margin mask
    mask_margin_ = cv::Mat::zeros(raw_himg_, raw_wimg_, CV_8U);
    cv::Rect roi_margin(marg_size_, marg_size_,
        raw_wimg_-2*marg_size_, raw_himg_-2*marg_size_);
    mask_margin_(roi_margin) = 255;
}

void vision_meas::LoadParameters(ros::NodeHandle& n) {
    // read yaml file
    std::vector<double> cam0_intrinsics, cam1_intrinsics;
    n.getParam("/envio_node/cam0/intrinsics", cam0_intrinsics);
    n.getParam("/envio_node/cam1/intrinsics", cam1_intrinsics);

    std::vector<double> cam0_dcf, cam1_dcf;
    n.getParam("/envio_node/cam0/distortion_model", distortion_model_);
    n.getParam("/envio_node/cam0/distortion_coeffs", cam0_dcf);
    n.getParam("/envio_node/cam1/distortion_coeffs", cam1_dcf);

    std::vector<int> resolution;
    n.getParam("/envio_node/cam0/resolution", resolution);

    Eigen::Isometry3d T_rl = utils::
        getSE3matrix(n, "/envio_node/cam1/T_cn_cnm1");

    n.param<int>("/envio_node/nx", nx_, 25);
    n.param<int>("/envio_node/ny", ny_, 15);
    n.param<double>("/envio_node/min_depth", min_depth_, 0.3);
    n.param<double>("/envio_node/max_depth", max_depth_, 20);
    n.param<double>("/envio_node/min_parallax", min_parallax_, 1);
    n.param<double>("/envio_node/ransac_thr", ransac_thr_, 1);


    // Print
    ROS_INFO("Cam0 intrinsics: %f, %f, %f, %f", cam0_intrinsics[0],
        cam0_intrinsics[1], cam0_intrinsics[2], cam0_intrinsics[3]);

    ROS_INFO("Cam1 intrinsics: %f, %f, %f, %f", cam1_intrinsics[0],
        cam1_intrinsics[1], cam1_intrinsics[2], cam1_intrinsics[3]);

    ROS_INFO("Camera distortion model: %s", distortion_model_.c_str());
    ROS_INFO("Cam0 distortion coeffs: %f, %f, %f, %f",
        cam0_dcf[0], cam0_dcf[1], cam0_dcf[2], cam0_dcf[3]);

    ROS_INFO("Cam1 distortion coeffs: %f, %f, %f, %f",
        cam1_dcf[0], cam1_dcf[1], cam1_dcf[2], cam1_dcf[3]);

    ROS_INFO("Image resolution: %d, %d", resolution[0], resolution[1]);

    ROS_INFO("Image grid: %d x %d", nx_, ny_);
    ROS_INFO("Min/Max depths: %f, %f", min_depth_, max_depth_);
    ROS_INFO("Min parallax: %f", min_parallax_);


    // Assign
    M3d Kl_orig;
    Kl_orig << cam0_intrinsics[0], 0, cam0_intrinsics[2],
                0, cam0_intrinsics[1], cam0_intrinsics[3],
                0, 0, 1;

    M3d Kr_orig;
    Kr_orig << cam1_intrinsics[0], 0, cam1_intrinsics[2],
                0, cam1_intrinsics[1], cam1_intrinsics[3],
                0, 0, 1;

    dcl_ << cam0_dcf[0], cam0_dcf[1], cam0_dcf[2], cam0_dcf[3];
    dcr_ << cam1_dcf[0], cam1_dcf[1], cam1_dcf[2], cam1_dcf[3];

    raw_wimg_ = resolution[0];
    raw_himg_ = resolution[1];

    T_rl_ = M4d::Identity();
    T_rl_.block<3,3>(0,0) = T_rl.linear();
    T_rl_.block<3,1>(0,3) = T_rl.translation();
    T_lr_ = T_rl_.inverse();

    // Convert eigen to opencv matrix
    cv::Mat cv_Kl, cv_dcl, cv_Kr, cv_dcr;

    cv::eigen2cv(Kl_orig, cv_Kl);
    cv::eigen2cv(dcl_, cv_dcl);

    cv::eigen2cv(Kr_orig, cv_Kr);
    cv::eigen2cv(dcr_, cv_dcr);

    cv::Size img_size(raw_wimg_, raw_himg_);

    // Initialize distortion map
    if (distortion_model_.compare("radtan") == 0) {
        cv::Mat cv_Kl_new, cv_Kr_new;

        // Get optimal K matrix for rescaling
        cv_Kl_new = cv::getOptimalNewCameraMatrix(cv_Kl, cv_dcl,
            img_size, 0, img_size);

        cv_Kr_new = cv::getOptimalNewCameraMatrix(cv_Kr, cv_dcr,
            img_size, 0, img_size);
        
        // convert to eigen K matrix
        Kl_ << cv_Kl_new.at<double>(0,0), 0.0, cv_Kl_new.at<double>(0,2),
            0.0, cv_Kl_new.at<double>(1,1), cv_Kl_new.at<double>(1,2),
            0.0, 0.0, 1.0;
        Kl_inv_ = Kl_.inverse();

        Kr_ << cv_Kr_new.at<double>(0,0), 0.0, cv_Kr_new.at<double>(0,2),
            0.0, cv_Kr_new.at<double>(1,1), cv_Kr_new.at<double>(1,2),
            0.0, 0.0, 1.0;
        Kr_inv_ = Kr_.inverse();

        // initialize undistortion map
        cv::initUndistortRectifyMap(cv_Kl, cv_dcl, cv::Mat(),
            cv_Kl_new, img_size, CV_16SC2,
            undistort_map1_left_, undistort_map2_left_);

        cv::initUndistortRectifyMap(cv_Kr, cv_dcr, cv::Mat(),
            cv_Kr_new, img_size, CV_16SC2,
            undistort_map1_right_, undistort_map2_right_);
    }
    else if (distortion_model_.compare("equidistant") == 0) {
        cv::Mat cv_Kl_new, cv_Kr_new;

        // Get optimal K matrix for rescaling
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
            cv_Kl, cv_dcl, img_size, cv::Mat(), cv_Kl_new, 0, img_size);
        
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
            cv_Kr, cv_dcr, img_size, cv::Mat(), cv_Kr_new, 0, img_size);
        
        // convert to eigen K matrix
        Kl_ << cv_Kl_new.at<double>(0,0), 0.0, cv_Kl_new.at<double>(0,2),
            0.0, cv_Kl_new.at<double>(1,1), cv_Kl_new.at<double>(1,2),
            0.0, 0.0, 1.0;
        Kl_inv_ = Kl_.inverse();

        Kr_ << cv_Kr_new.at<double>(0,0), 0.0, cv_Kr_new.at<double>(0,2),
            0.0, cv_Kr_new.at<double>(1,1), cv_Kr_new.at<double>(1,2),
            0.0, 0.0, 1.0;
        Kr_inv_ = Kr_.inverse();

        // initialize undistortion map
        cv::fisheye::initUndistortRectifyMap(cv_Kl, cv_dcl, cv::Mat(),
            cv_Kl_new, img_size, CV_16SC2,
            undistort_map1_left_, undistort_map2_left_);

        cv::fisheye::initUndistortRectifyMap(cv_Kr, cv_dcr, cv::Mat(),
            cv_Kr_new, img_size, CV_16SC2,
            undistort_map1_right_, undistort_map2_right_);
    }
    else {
        ROS_ERROR("Invalid camera distortion model !!!");
    }

}

void vision_meas::processStereoImage(
    const sensor_msgs::ImageConstPtr& img_l,
    const sensor_msgs::ImageConstPtr& img_r,
    const cv::Mat& mask) {

    // Undistort images
    cv::Mat dimg_l = cv_bridge::toCvCopy(
        img_l, sensor_msgs::image_encodings::MONO8)->image;
    cv::Mat dimg_r = cv_bridge::toCvCopy(
        img_r, sensor_msgs::image_encodings::MONO8)->image;

    cv::remap(dimg_l, uimg_l_, undistort_map1_left_,
        undistort_map2_left_, CV_INTER_LINEAR);
    cv::remap(dimg_r, uimg_r_, undistort_map1_right_,
        undistort_map2_right_, CV_INTER_LINEAR);

    // Compute edge map
    cv::Mat dx_uimg_l, dy_uimg_l;
    cv::Mat dx_abs, dy_abs, d_uimg_m;
    cv::Sobel(uimg_l_, dx_uimg_l, CV_8U, 1, 0, 3);
    cv::Sobel(uimg_l_, dy_uimg_l, CV_8U, 0, 1, 3);
    cv::convertScaleAbs(dx_uimg_l, dx_abs);
    cv::convertScaleAbs(dy_uimg_l, dy_abs);
    cv::addWeighted(dx_abs, 0.5, dy_abs, 0.5, 0, d_uimg_m);

    // Extract locally high gradient pixels
    int roix_size = ceil(d_uimg_m.cols/nx_);
    int roiy_size = ceil(d_uimg_m.rows/ny_);
    cv::Mat mask_edge = cv::Mat::zeros(
        d_uimg_m.rows, d_uimg_m.cols, CV_8U);
    for (int i = 0; i < nx_; i++) {
        for (int j = 0; j < ny_; j++) {
            // Get a grid
            cv::Rect roi_ij(roix_size*i, roiy_size*j, roix_size, roiy_size);
            cv::Mat mag_ij = d_uimg_m(roi_ij);

            // Find the largest intensity
            double max_val;
            cv::Point max_loc;
            cv::minMaxLoc(mag_ij, 0, &max_val, 0, &max_loc);

            // select the strongest
            cv::Mat mask_ij = cv::Mat::zeros(roiy_size, roix_size, CV_8U);
            if (max_val > thr_gradmin_) mask_ij.at<uchar>(max_loc) = 255;
            if (max_val != 0) {
                mask_edge(roi_ij).setTo(255, mask_ij);
            }
        }
    }
    mask_edge = mask_edge & mask_margin_ & mask;

    // Stereo triangulation
    std::vector<cv::Point2f> u_l, u_r;
    std::vector<cv::Point> ui_l;
    cv::findNonZero(mask_edge, ui_l);
    if (ui_l.size() == 0) return;

    // Convert interger to double
    cv::Mat(ui_l).convertTo(u_l, cv::Mat(u_l).type());

    // Stereo matching
    // Opencv LK tracker is more robust to feature-rich environments
    bool use_lk = true;
    std::vector<uchar> status_stereo;
    std::vector<double> d_l, d_r;
    if (use_lk) {

        // LK tracker
        std::vector<float> err_static;
        cv::calcOpticalFlowPyrLK(
            uimg_l_, uimg_r_, u_l, u_r, status_stereo, err_static,
            cv::Size(21,21), 3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
            20, 0.02));

        // Do ransac
        std::vector<uchar> status_ransac;
        if (u_r.size() > 20) {
            cv::findFundamentalMat(u_l, u_r, cv::FM_RANSAC,
                ransac_thr_, 0.99, status_ransac);
        }
        else {
            for (int i = 0; i < u_l.size(); i++) status_ransac.push_back(1);
        }

        for (int j = 0; j < u_l.size(); j ++) {
            status_stereo[j] = status_stereo[j] && status_ransac[j] &&
                (err_static[j] < 40.0);

            // Two-view triangulation
            double dl_j, dr_j;
            TwoViewTriangulation(u_l[j], u_r[j], dl_j, dr_j);
            d_l.push_back(dl_j);
            d_r.push_back(dr_j);
        }
    }
    else {
        epipolarLineSearch(u_l, u_r, d_l, d_r, status_stereo);
    }

    // Pack features
    for (int i = 0; i < status_stereo.size(); i++) {
        if (status_stereo[i] && IsFov(u_r[i])) {

            double parallax_x = u_l[i].x - u_r[i].x;
            double parallax_y = u_l[i].y - u_r[i].y;
            double parallax = sqrt(parallax_x*parallax_x
                + parallax_y*parallax_y);

            if (min_depth_ <= d_l[i] && d_l[i] <= max_depth_ &&
                min_depth_ <= d_r[i] && d_r[i] <= max_depth_ &&
                parallax > min_parallax_) {
                // Save
                u_l_.push_back(u_l[i]);
                u_r_.push_back(u_r[i]);
                depth_l_.push_back(d_l[i]);
                depth_r_.push_back(d_r[i]);
            }
        }
    }
}


void vision_meas::TwoViewTriangulation(const cv::Point2f& pt0,
    const cv::Point2f& pt1, double& depth0, double& depth1) {
    V3d xl(pt0.x, pt0.y, 1.0);
    V3d xr(pt1.x, pt1.y, 1.0);
    xl = Kl_inv_*xl;
    xr = Kr_inv_*xr;
    Eigen::Matrix<double, 3, 2> A;
    A << xl, -T_lr_.block<3,3>(0,0)*xr;
    V2d lambda = (A.transpose()*A).ldlt().
        solve(A.transpose()*T_lr_.block<3,1>(0,3));
    depth0 = lambda(0);
    depth1 = lambda(1);
}

bool vision_meas::IsFov(const cv::Point2f& pt) {
    if (0 < pt.x && pt.x < raw_wimg_ && 0 < pt.y && pt.y < raw_himg_) {
        return true;
    }
    else {
        return false;
    }
}


void vision_meas::epipolarLineSearch(const std::vector<cv::Point2f>& u0,
    std::vector<cv::Point2f>& u1, std::vector<double>& d0,
    std::vector<double>& d1, std::vector<uchar>& is_valid) {

    int patch_size = 6;
    int grid_size = (2*patch_size + 1) * (2*patch_size + 1);

    // Decompose SE(3) matrix
    M3d R_10 = T_rl_.block<3,3>(0,0);
    V3d p_10 = T_rl_.block<3,1>(0,3);

    for (int j = 0; j < u0.size(); j ++) {

        V3d uv0_h(u0[j].x, u0[j].y, 1.0);
        V3d uv0_n = Kl_inv_ * uv0_h;
        double d_min = min_depth_;
        double d_max = max_depth_;

        // Compute reference patch
        Eigen::VectorXd I0_j = Eigen::VectorXd::Zero(grid_size);
        int cnt_0 = 0;
        for (int dx = -patch_size; dx < patch_size+1; dx ++) {
            for (int dy = -patch_size; dy < patch_size+1; dy ++) {
                double u = round(uv0_h(0,0) + dx);
                double v = round(uv0_h(1,0) + dy);

                I0_j(cnt_0, 0) = static_cast<double>(
                    uimg_l_.at<uchar>((int)v, (int)u));
                cnt_0 ++;
            }
        }

        // Find minimum depth projection
        V3d pc0_f_min = d_min * uv0_n;
        V3d pc1_f_min = R_10 * pc0_f_min + p_10;
        double d1_min = pc1_f_min(2,0);
        V3d uv1_n_min = pc1_f_min / d1_min;
        V3d uv1_h_min = Kr_ * uv1_n_min;

        // Find maximum depth projection
        V3d pc0_f_max = d_max * uv0_n;
        V3d pc1_f_max = R_10 * pc0_f_max + p_10;
        double d1_max = pc1_f_max(2,0);
        V3d uv1_n_max = pc1_f_max / d1_max;
        V3d uv1_h_max = Kr_ * uv1_n_max;

        // Obtain epipolar line direction
        double ld_u = uv1_h_max(0,0) - uv1_h_min(0,0);
        double ld_v = uv1_h_max(1,0) - uv1_h_min(1,0);
        double mag_ld = sqrt(ld_u*ld_u + ld_v*ld_v);
        int N_can = static_cast<int>(mag_ld);
        ld_u = ld_u / mag_ld;
        ld_v = ld_v / mag_ld;

        // Initialize epipolar cost
        Eigen::VectorXd epipolar_cost = Eigen::VectorXd::Zero(N_can);
        std::vector<V3d> uv1_draw;
        int min_index = -1;
        double min_value = 1e+10;  // the smallest cost
        double min2_value = 1e+10;  // the 2nd smallest cost
        cv::Point2f opt_uv1;

        // Search along epipolar line per pixel
        for (int i = 0; i < N_can; i ++) {
            
            double u_i = uv1_h_min(0,0) + ld_u*i;
            double v_i = uv1_h_min(1,0) + ld_v*i;
            int cnt_1 = 0;
            double ssd_i = 0;

            V3d uv_i(u_i, v_i, 1.0);
            uv1_draw.push_back(uv_i);

            for (int dx = -patch_size; dx < patch_size+1; dx ++) {
                for (int dy = -patch_size; dy < patch_size+1; dy ++) {
                    double u = round(u_i + dx);
                    double v = round(v_i + dy);

                    double diff = I0_j(cnt_1, 0) 
                        - static_cast<double>(uimg_r_.at<uchar>((int)v, (int)u));
                    cnt_1 ++;

                    ssd_i += diff*diff;
                }
            }

            epipolar_cost(i,0) = ssd_i;

            if (min_value > ssd_i) {
                min_index = i;
                min_value = ssd_i;
                opt_uv1.x = u_i;
                opt_uv1.y = v_i;
            }
        }

        // Compute depth
        double d0_est, d1_est;
        TwoViewTriangulation(u0[j], opt_uv1, d0_est, d1_est);

        // Find the 2nd smallest
        for (int i = 0; i < epipolar_cost.size(); i ++) {
            if ((min2_value > epipolar_cost(i,0)) &&
                (epipolar_cost(i,0) > min_value)) {
                min2_value = epipolar_cost(i,0);
            }
        }

        // Compute image gradient at the optimal point
        double d_Ix = 0.5 *
                (utils::GetPixelValue(uimg_r_, opt_uv1.x+1, opt_uv1.y) -
                    utils::GetPixelValue(uimg_r_, opt_uv1.x-1, opt_uv1.y));

        double d_Iy = 0.5 *
                (utils::GetPixelValue(uimg_r_, opt_uv1.x, opt_uv1.y+1) -
                    utils::GetPixelValue(uimg_r_, opt_uv1.x, opt_uv1.y-1));

        double m_dI = sqrt(d_Ix*d_Ix + d_Iy*d_Iy);
        d_Ix = d_Ix / m_dI;
        d_Iy = d_Iy / m_dI;
                
        double dot_grad = ld_u*d_Ix + ld_v*d_Iy;

        u1.push_back(opt_uv1);
        d0.push_back(d0_est);
        d1.push_back(d1_est);

        if (((min_value/min2_value) < 0.8) && fabs(dot_grad) > 0.2 &&
            d0_est > min_depth_ && d0_est < max_depth_) {
            is_valid.push_back(true);
            // std::cout << "Pass ! " << std::endl;
        }
        else {
            is_valid.push_back(false);
        }
    }

}

}