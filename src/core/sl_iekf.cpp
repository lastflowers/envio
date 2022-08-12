/**
 * @file sl_iekf.cpp
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

#include "sl_iekf.h"

namespace nesl {

sl_iekf::sl_iekf() {
    unsigned int seed =
        std::chrono::steady_clock::now().time_since_epoch().count();
    generator_.seed(seed);

    // Must be odd size
    // 1) No pattern
    // pad_ = MatrixXi::Zero(1,2); 
    // pad_ << 0, 0;

    // 2) 
    pad_ = MatrixXi::Zero(5,2); 
    pad_ << -1, -1,
             1, -1,
             0,  0,
            -1,  1,
             1,  1;

    // 3)
    // pad_ = MatrixXi::Zero(9,2); 
    // pad_ <<  0, -2,
    //         -1, -1,
    //          1, -1,
    //         -2,  0,
    //          0,  0,
    //          2,  0,
    //         -1,  1,
    //          1,  1,
    //          0,  2;
    idx_center_ = static_cast<int>(floor(0.5*pad_.rows()));
}

void sl_iekf::processMeasurement(const nesl::vision_meas& vis_meas) {
    // Build pyramidal images
    std::vector<cv::Mat> pyr_uimg;
    int w_img = vis_meas.uimg_l().cols;
    int h_img = vis_meas.uimg_l().rows;
    for (int yy = 0; yy < max_lvl_; yy++) {
        // resize in the order of 0.5
        double dfactor = pow(2,-yy);
        int wimg_y = static_cast<int>(w_img * dfactor);
        int himg_y = static_cast<int>(h_img * dfactor);

        if (yy == 0) {
            pyr_uimg.push_back(vis_meas.uimg_l());
        }
        else {
            cv::Mat img_y;
            cv::resize(vis_meas.uimg_l(), img_y, cv::Size(wimg_y, himg_y),
                0, 0, CV_INTER_LINEAR);
            pyr_uimg.push_back(img_y);
        }
    }

    // implement filter update
    ros::Time start_time = ros::Time::now();
    if (photo_map_.size() > 0) {
        filterUpdate(pyr_uimg, vis_meas.Kl());
    }
    double filter_update_time = (ros::Time::now()-start_time).toSec();


    // Track features with posterior
    start_time = ros::Time::now();
    trackFeatures(pyr_uimg, vis_meas.Kl(), vis_meas.Kl_inv(),
        vis_meas.Kr(), vis_meas.T_rl());
    double tracking_time = (ros::Time::now()-start_time).toSec();
    

    // Replace old pose
    start_time = ros::Time::now();
    stateReplacement();
    double state_replace_time = (ros::Time::now()-start_time).toSec();


    // Feature initialization
    start_time = ros::Time::now();
    initializeFeatures(pyr_uimg, vis_meas.u_l(), vis_meas.d_l(),
        vis_meas.u_r(), vis_meas.d_r(), vis_meas.Kl_inv());
    double initialize_time = (ros::Time::now()-start_time).toSec();


    // visualize in rviz
    start_time = ros::Time::now();
    updateMask(pyr_uimg[0]);
    double update_mask_time = (ros::Time::now()-start_time).toSec();

    uimg_l_prev_ = pyr_uimg[0];
    T0_ = Xi_.block<4,4>(0,0);  // state replacement

    std::cout << "update: " << filter_update_time << "   " <<
        "tracking: " << tracking_time << "   " <<
        "replace: " << state_replace_time << "   " <<
        "init: " << initialize_time << "   " <<
        "mask: " << update_mask_time << std::endl;
}


void sl_iekf::filterUpdate(std::vector<cv::Mat> pyr_uimg,
    const M3d& Kl) {
    // compute inverse of cov
    Eigen::MatrixXd Pinv = cov_.llt().solve(
        Eigen::MatrixXd::Identity(cov_.cols(), cov_.cols()));

    // Set priors
    M5d X_p = Xi_;
    V3d ba_p = ba_;
    V3d bg_p = bg_;
    M4d T_p = T0_;
    double cost_prev = 1e+4;
    Eigen::VectorXd delx;

    // Sample delta ensembles
    std::vector<V6d> delta_T0, delta_T1;
    std::vector<double> delta_Z;

    if (use_sl_) {
        // ros::Time start_time = ros::Time::now();
        sampleDeltaEnsembles(delta_T0, delta_T1, delta_Z);

        // std::cout << "Delta ensemble sampling time: " << 
        //     (ros::Time::now() - start_time).toSec() << std::endl;
    }

    bool is_break = false;
    ros::Time loop_start_time = ros::Time::now();
    for (int yy = max_lvl_-1; yy >= 0; yy--) {
        // Down scale images
        double dfactor = pow(2,-yy);
        double dmarg_size = dfactor * marg_size_;
        cv::Mat uimg_l = pyr_uimg[yy];
        M3d Ky = dfactor * Kl;
        Ky(2,2) = 1;
        M3d Ky_inv = Ky.inverse();

        for (int iter = 0; iter < max_iter_; iter++) {
            // initialize big matrix
            int N_d = photo_map_.size();
            int N = cov_.cols();
            M6d HpT_Hp = M6d::Zero();
            Eigen::MatrixXd HpT_h = Eigen::MatrixXd::Zero(6,N_d);
            Eigen::VectorXd hT_h = Eigen::VectorXd::Zero(N_d,1);
            Eigen::VectorXd HT_Rinv_r = Eigen::VectorXd::Zero(N,1);

            M4d T_gc0 = T0_ * Tbl_;
            M3d R_gc0 = T_gc0.block<3,3>(0,0);
            V3d p_gc0 = T_gc0.block<3,1>(0,3);
            M4d T_gc1 = Xi_.block<4,4>(0,0) * Tbl_;
            M4d T_c1g = M4d::Identity();
            T_c1g.block<3,3>(0,0) = T_gc1.block<3,3>(0,0).transpose();
            T_c1g.block<3,1>(0,3) = -T_gc1.block<3,3>(0,0).transpose()
                *T_gc1.block<3,1>(0,3);
            M3d R_c1g = T_c1g.block<3,3>(0,0);
            M4d T_c10 = T_c1g * T_gc0;
            M3d R_c10 = T_c10.block<3,3>(0,0);
            V3d p_c10 = T_c10.block<3,1>(0,3);

            // sample ensembles
            std::vector<M4d> T0_en, T1_en;
            std::vector<Eigen::VectorXd> Z_en;

            if (use_sl_) {
                // ros::Time start_time = ros::Time::now();

                sampleEnsembles(delta_T0, delta_T1, delta_Z,
                    T0_en, T1_en, Z_en);

                // std::cout << "Ensemble sampling time: " << 
                //     (ros::Time::now() - start_time).toSec() << std::endl;
            }

            std::vector<int> marg_idx;
            double cost_accum = 0;
            int cnt_stack = 0;
            int cnt_pixel = 0;
            int cnt_sl = 0;

            // Compute Jacobian per tracking features
            for (int j = 0; j < N_d; j++) {
                // Initialize accumulators
                M6d HpT_Hp_j = M6d::Zero();
                V6d HpT_h_j = V6d::Zero();
                double hT_h_j = 0;
                V6d HT_Rinv_rp_j = V6d::Zero();
                V6d HT_Rinv_rs_j = V6d::Zero();
                double HT_Rinv_rd_j = 0;
                bool is_fov = true;

                // iter per pattern
                for (int l = 0; l < pad_.rows(); l++) {
                    V3d uv0_h(dfactor*photo_map_[j].uv_l_(0,0)+pad_(l,0),
                        dfactor*photo_map_[j].uv_l_(1,0)+pad_(l,1), 1);
                    
                    // warp features
                    V3d uv0_n = Ky_inv*uv0_h;
                    V3d pc0_f = photo_map_[j].depth_l_ * uv0_n;
                    V3d pg_f = R_gc0 * pc0_f + p_gc0;
                    V3d pc1_f = R_c10 * pc0_f + p_c10;
                    double X1 = pc1_f(0,0);
                    double Y1 = pc1_f(1,0);
                    double Z1 = pc1_f(2,0);
                    double inv_Z1 = 1/Z1;
                    V3d uv1_n = pc1_f / Z1;
                    V3d uv1_h = Ky * uv1_n;

                    if ( (uv1_h(0,0) > dmarg_size) &&
                        (uv1_h(0,0) < uimg_l.cols - dmarg_size) &&
                        (uv1_h(1,0) > dmarg_size) &&
                        (uv1_h(1,0) < uimg_l.rows - dmarg_size) && 
                        (Z1 > 0) ) {

                        // Compute measurement Jacobian
                        double dIx = 0.5 * Ky(0,0) * 
                        (utils::GetPixelValue(uimg_l, uv1_h(0,0)+1, uv1_h(1,0)) -
                            utils::GetPixelValue(uimg_l, uv1_h(0,0)-1, uv1_h(1,0)));

                        double dIy = 0.5 * Ky(1,1) * 
                        (utils::GetPixelValue(uimg_l, uv1_h(0,0), uv1_h(1,0)+1) -
                            utils::GetPixelValue(uimg_l, uv1_h(0,0), uv1_h(1,0)-1));

                        // Compute stochastic gradient
                        if (use_sl_) {
                            double sl_x = 0.0;
                            double sl_y = 0.0;
                            bool is_sl = false;
                            bool x_weak = fabs(dIx) < Ky(0,0)*thr_weak_;
                            bool y_weak = fabs(dIy) < Ky(1,1)*thr_weak_;

                            if (x_weak || y_weak) {
                                computeStochasticGradient(uimg_l, Ky, Ky_inv,
                                    uv0_h, uv1_h, T0_en, T1_en, Z_en, j, sl_x, sl_y);
            
                                is_sl = true;
                            }

                            // Replace small gradients
                            if (x_weak) dIx = Ky(0,0) * sl_x;

                            if (y_weak) dIy = Ky(1,1) * sl_y;

                            if (is_sl) cnt_sl ++;
                        }

                        Eigen::Matrix<double, 1, 3> dI_dp;
                        dI_dp << dIx*inv_Z1, dIy*inv_Z1,
                            -(dIx*X1)*(inv_Z1*inv_Z1) - (dIy*Y1)*(inv_Z1*inv_Z1);
                        
                        Eigen::Matrix<double, 1, 6> Hp_jl;
                        Hp_jl << -dI_dp*R_c1g*nesl::Vec2SkewMat(pg_f),
                                dI_dp*R_c1g;

                        V3d dp_drho;
                        if (is_idepth_) {
                            dp_drho = -pc0_f(2,0) * (R_c10*pc0_f);
                        }
                        else {
                            dp_drho = R_c10 * uv0_n;
                        }
                        double h_jl = dI_dp * dp_drho;

                        // Compute innovation
                        double r_jl = photo_map_[j].intensity_[yy][l] -
                            utils::GetPixelValue(uimg_l, uv1_h(0,0), uv1_h(1,0));
                        cost_accum += r_jl * r_jl;
                        r_jl /= R_;

                        // Accumlate
                        HpT_Hp_j += (Hp_jl.transpose() * Hp_jl) / R_;
                        HpT_h_j += (Hp_jl.transpose() * h_jl) / R_;
                        hT_h_j += (h_jl * h_jl) / R_;

                        HT_Rinv_rp_j += Hp_jl.transpose() * r_jl;
                        HT_Rinv_rd_j += h_jl * r_jl;

                        cnt_pixel ++;
                    }
                    else {
                        is_fov = false;
                        break;
                    }
                } // end for pattern

                if (is_fov) {
                    HpT_Hp += HpT_Hp_j;
                    HpT_h.block<6,1>(0,cnt_stack) = HpT_h_j;
                    hT_h(cnt_stack,0) = hT_h_j;

                    V6d tmp = HT_Rinv_r.block<6,1>(0,0);
                    HT_Rinv_r.block<6,1>(0,0) = tmp + HT_Rinv_rp_j;

                    tmp = HT_Rinv_r.block<6,1>(Dx,0);
                    HT_Rinv_r.block<6,1>(Dx,0) = tmp - HT_Rinv_rp_j;

                    HT_Rinv_r(Ds+Dx+cnt_stack,0) = HT_Rinv_rd_j;

                    cnt_stack ++;
                } 
                else {
                    marg_idx.push_back(j);
                }
            } // end for feature

            // Resize Jacobian matrices
            HpT_h.conservativeResize(Ds, cnt_stack);
            hT_h.conservativeResize(cnt_stack);
            HT_Rinv_r.conservativeResize(Dx+Ds+cnt_stack);

            // Marginalize invalid features
            if (marg_idx.size() > 0) {
                std::cout << "In estimator, throw " << marg_idx.size() 
                    << " features" << std::endl;

                marginalizeFromIndex(marg_idx);

                // reassign inverse of cov
                Pinv = cov_.llt().solve(
                    Eigen::MatrixXd::Identity(cov_.cols(), cov_.cols()));
            }

            // Compute updated covariance
            Eigen::MatrixXd cov_inv = Pinv;
            cov_inv.block<6,6>(0,0) = Pinv.block<6,6>(0,0) + HpT_Hp;
            cov_inv.block<6,6>(Dx,Dx) = Pinv.block<6,6>(Dx,Dx) + HpT_Hp;
            cov_inv.block<6,6>(0,Dx) = Pinv.block<6,6>(0,Dx) - HpT_Hp;
            cov_inv.block<6,6>(Dx,0) = Pinv.block<6,6>(Dx,0) - HpT_Hp;

            cov_inv.block(0,Ds+Dx,Ds,cnt_stack) = Pinv.block(0,Ds+Dx,Ds,cnt_stack) + HpT_h;
            cov_inv.block(Dx,Ds+Dx,Ds,cnt_stack) = Pinv.block(Dx,Ds+Dx,Ds,cnt_stack) - HpT_h;
            cov_inv.block(Ds+Dx,0,cnt_stack,Ds) = Pinv.block(Ds+Dx,0,cnt_stack,Ds) + HpT_h.transpose();
            cov_inv.block(Dx+Ds,Dx,cnt_stack,Ds) = Pinv.block(Ds+Dx,Dx,cnt_stack,Ds) - HpT_h.transpose();

            for (int i = 0; i < cnt_stack; i++) cov_inv(Ds+Dx+i,Ds+Dx+i) = Pinv(Ds+Dx+i,Ds+Dx+i) + hT_h(i,0);
            delx = cov_inv.llt().solve(HT_Rinv_r);

            // State update
            M5d del_X;
            Exp_sed3(-delx.block<9,1>(0,0), del_X);
            M5d Xi_new = del_X * X_p;
            Xi_ = Xi_new;

            M4d del_T;
            Exp_se3(-delx.block<6,1>(15,0), del_T);
            M4d T0_new = del_T * T_p;
            T0_ = T0_new;

            // Report iteration
            // Check processing time
            double loop_iter_time =
                (ros::Time::now() - loop_start_time).toSec();

            double cost_i = 1.5379e-5*cost_accum;
            std::cout << "[" << yy+1 << "-" << iter+1 << " iter]"
                << " # features: " << cnt_stack << " / " << N_d 
                << "  SL: " << cnt_sl << " / " << cnt_pixel 
                << "  cost: " << cost_i 
                << "  loop time: " << loop_iter_time << std::endl;

            if ((fabs(cost_i - cost_prev)/cost_i < thr_stop_) ||
                (yy == 0 && iter == max_iter_-1) || 
                loop_iter_time > max_itime_) {
                // bias update
                ba_ = ba_p - delx.block<3,1>(9,0);
                bg_ = bg_p - delx.block<3,1>(12,0);

                // map update (only in the last iter)
                for (int j = 0; j < cnt_stack; j ++) {
                    if (is_idepth_) {
                        double idepth_new = 1/photo_map_[j].depth_l_ + delx(Dx+Ds+j, 0);
                        photo_map_[j].depth_l_ = 1/idepth_new;
                    }
                    else {
                        photo_map_[j].depth_l_ += delx(Dx+Ds+j, 0);
                    }
                }

                // Covariance assignment (only in the last iter)
                Eigen::MatrixXd cov_new = cov_inv.llt().solve(
                    Eigen::MatrixXd::Identity(cov_inv.cols(), cov_inv.cols()));
                validateCovMtx(cov_new, cov_);

                is_break = true;
                break;
            }
            cost_prev = cost_i;
        } // iter loop
        if (is_break) break;
    } // pyramid level loop
}


void sl_iekf::Exp_sed3(const V9d& xi, M5d& X) {
    X = M5d::Identity();
    V3d phi = xi.block<3,1>(0,0);
    double m = phi.norm();
    if (m < 1e-20) {
        return;
    }
    else {
        M3d R = nesl::rvec2dcm(phi);
        M3d phi_hat = nesl::Vec2SkewMat(phi);
        M3d J = M3d::Identity() + ((1-cos(m))/(m*m))*phi_hat + 
            ((m-sin(m))/(m*m*m))*(phi_hat*phi_hat);
        X.block<3,3>(0,0) = R;
        X.block<3,1>(0,3) = J*xi.block<3,1>(3,0);
        X.block<3,1>(0,4) = J*xi.block<3,1>(6,0);
    }
}

void sl_iekf::Exp_se3(const V6d& xi, M4d& X) {
    X = M4d::Identity();
    V3d phi = xi.block<3,1>(0,0);
    double m = phi.norm();
    if (m < 1e-20) {
        return;
    }
    else {
        M3d R = nesl::rvec2dcm(phi);
        M3d phi_hat = nesl::Vec2SkewMat(phi);
        M3d J = M3d::Identity() + ((1-cos(m))/(m*m))*phi_hat + 
            ((m-sin(m))/(m*m*m))*(phi_hat*phi_hat);
        X.block<3,3>(0,0) = R;
        X.block<3,1>(0,3) = J*xi.block<3,1>(3,0);
    }
}


void sl_iekf::updateMask(const cv::Mat& uimg_l) {
    mask_ = cv::Mat(uimg_l.size(), CV_8UC1, cv::Scalar(255));
    for (int j = 0; j < photo_map_.size(); j++) {
        cv::Point2f draw_u;
        draw_u.x = photo_map_[j].uv_l_(0,0);
        draw_u.y = photo_map_[j].uv_l_(1,0);
        cv::circle(mask_, draw_u, uniform_dist_, 0, -1);
    }
}



void sl_iekf::initializeFeatures(std::vector<cv::Mat> pyr_uimg,
    const std::vector<cv::Point2d>& u_l,
    const std::vector<double>& d_l,
    const std::vector<cv::Point2d>& u_r,
    const std::vector<double>& d_r,
    const M3d& Kl_inv) {

    if (photo_map_.size() < thr_num_) {
        int M_new = u_l.size();

        M4d T_gc = Xi_.block<4,4>(0,0) * Tbl_;

        for (int i = 0; i < M_new; i++) {

            V3d uv_h(u_l[i].x, u_l[i].y, 1);
            V3d uv_n = Kl_inv*uv_h;
            V3d pc_f = d_l[i] * uv_n;
            V3d pg_f = T_gc.block<3,3>(0,0) * pc_f
                + T_gc.block<3,1>(0,3);

            photometric_map feature_i;

            feature_i.life_time_ = 1;
            feature_i.uv_l_ << u_l[i].x, u_l[i].y;
            feature_i.uv_r_ << u_r[i].x, u_r[i].y;
            feature_i.depth_l_ = d_l[i];
            feature_i.depth_r_ = d_r[i];
            for (int yy = 0; yy < max_lvl_; yy++) {
                double dfactor = pow(2,-yy);
                
                std::vector<double> intensity_patch;
                for (int l = 0; l < pad_.rows(); l++) {
                    intensity_patch.push_back(utils::GetPixelValue(pyr_uimg[yy],
                        dfactor*u_l[i].x+pad_(l,0), dfactor*u_l[i].y+pad_(l,1)));
                }
                feature_i.intensity_.push_back(intensity_patch);
            }
            feature_i.pg_f_ = pg_f;
            photo_map_.push_back(feature_i);
        }
        int csize = cov_.cols();
        // Augment covariance matrix 
        // untouching pre-existed value and filled with new zeros
        cov_.conservativeResizeLike(
            Eigen::MatrixXd::Zero(csize+M_new, csize+M_new));
        cov_.block(csize, csize, M_new, M_new) =
            rho_var0_*Eigen::MatrixXd::Identity(M_new, M_new);
    }
    std::cout << "Add " << u_l.size() << " features" 
        << "   Covariance size: " << cov_.cols() << "x" << cov_.rows()
        << "   Current num features: " << photo_map_.size()
        << std::endl;

}


void sl_iekf::stateReplacement() {
    // Pose covariance replacement
    if (cov_.cols() < Dx + Ds) {
        cov_.conservativeResizeLike(Eigen::MatrixXd::Zero(Dx+Ds, Dx+Ds));
    }
    cov_.block<15,6>(0,Dx) = cov_.block<15,6>(0,0);
    cov_.block<6,15>(Dx,0) = cov_.block<15,6>(0,Dx).transpose();
    cov_.block<6,6>(Dx,Dx) = cov_.block<6,6>(0,0);

    // Feature covariance replacement
    if (cov_.cols() > Dx + Ds) {
        int M = cov_.cols() - (Dx + Ds);
        cov_.block(Dx, Dx+Ds, Ds, M) = cov_.block(0, Dx+Ds, 6, M);
        cov_.block(Dx+Ds, Dx, M, Ds) = cov_.block(Dx, Dx+Ds, Ds, M).transpose();
    }
    Eigen::MatrixXd cov_new = 0.5*(cov_ + cov_.transpose());
    cov_ = cov_new;
}


void sl_iekf::trackFeatures(std::vector<cv::Mat> pyr_uimg,
    const M3d& Kl, const M3d& Kl_inv,
    const M3d& Kr, const M4d& T_rl) {

    std::vector<int> marg_idx;
    int N_d = photo_map_.size();
    M4d T_gc0 = T0_ * Tbl_;
    M4d T_gc1 = Xi_.block<4,4>(0,0) * Tbl_;
    M4d T_c1g = M4d::Identity();
    T_c1g.block<3,3>(0,0) = T_gc1.block<3,3>(0,0).transpose();
    T_c1g.block<3,1>(0,3) = -T_gc1.block<3,3>(0,0).transpose()
        *T_gc1.block<3,1>(0,3);
    M4d T_c10 = T_c1g * T_gc0;
    M3d R_c10 = T_c10.block<3,3>(0,0);
    V3d p_c10 = T_c10.block<3,1>(0,3);

    for (int j = 0; j < N_d; j ++) {
        // warp features
        V3d uv0_h(photo_map_[j].uv_l_(0,0), photo_map_[j].uv_l_(1,0), 1);
        V3d uv0_n = Kl_inv*uv0_h;
        V3d pc0_f = photo_map_[j].depth_l_ * uv0_n;
        V3d pc1_f = R_c10 * pc0_f + p_c10;
        double Z1 = pc1_f(2,0);
        V3d uv1_n = pc1_f / Z1;
        V3d uv1_h = Kl * uv1_n;
        double intensity_new =
            utils::GetPixelValue(pyr_uimg[0], uv1_h(0,0), uv1_h(1,0));

        // Check patch neiborgood to erase occluded features
        // Note that NCC quality check is very important in practical sense
        int patch_size = 6;
        int n_pixel = (2*patch_size+1)*(2*patch_size+1);
        Eigen::VectorXd I0 = Eigen::VectorXd::Zero(n_pixel);
        Eigen::VectorXd I1 = Eigen::VectorXd::Zero(n_pixel);
        double I0_mean = 0;
        double I1_mean = 0;
        int cnt = 0;
        for (int x = -patch_size; x <= patch_size; x ++) {
            for (int y = -patch_size; y<= patch_size; y++) {

                I0(cnt, 0) = utils::GetPixelValue(uimg_l_prev_, uv0_h(0,0)+x, uv0_h(1,0)+y);;
                I1(cnt, 0) = utils::GetPixelValue(pyr_uimg[0], uv1_h(0,0)+x, uv1_h(1,0)+y);
                I0_mean += I0(cnt,0);
                I1_mean += I1(cnt,0);

                cnt ++;
            }
        }

        I0_mean = I0_mean / n_pixel;
        I1_mean = I1_mean / n_pixel;
        double N0 = 0;
        double D0 = 0;
        double D1 = 0;
        for (int i = 0; i < n_pixel; i ++) {
            N0 += (I0(i,0) - I0_mean) * (I1(i,0) - I1_mean);
            D0 += (I0(i,0) - I0_mean) * (I0(i,0) - I0_mean);
            D1 += (I1(i,0) - I1_mean) * (I1(i,0) - I1_mean);
        }
        double ncc_j = N0/sqrt(D0*D1);  // compute NCC

        // Lift to the global frame
        V3d pg_f = T_gc1.block<3,3>(0,0) * pc1_f
            + T_gc1.block<3,1>(0,3);

        if ( (uv1_h(0,0) > marg_size_) &&
            (uv1_h(0,0) < pyr_uimg[0].cols - marg_size_) &&
            (uv1_h(1,0) > marg_size_) &&
            (uv1_h(1,0) < pyr_uimg[0].rows - marg_size_) && 
            (Z1 > 0) && (photo_map_[j].life_time_ < max_lifetime_) &&
            (abs(intensity_new - photo_map_[j].intensity_[0][idx_center_]) < max_diff_) &&
            (ncc_j > 0.80)) {

            // project on right camera
            V3d pr1_f = T_rl.block<3,3>(0,0)*pc1_f
                + T_rl.block<3,1>(0,3);
            double Zr1 = pr1_f(2,0);
            V3d uv_r1_n = pr1_f / Zr1;
            V3d uv_r1_h = Kr * uv_r1_n;

            photo_map_[j].depth_l_ = Z1;
            photo_map_[j].depth_r_ = Zr1;

            photo_map_[j].intensity_.clear();
            for (int yy = 0; yy < max_lvl_; yy++) {
                double dfactor = pow(2,-yy);

                std::vector<double> intensity_patch;
                for (int l = 0; l < pad_.rows(); l++) {
                    intensity_patch.push_back(utils::GetPixelValue(pyr_uimg[yy],
                        dfactor*uv1_h(0,0)+pad_(l,0), dfactor*uv1_h(1,0)+pad_(l,1)));
                }
                photo_map_[j].intensity_.push_back(intensity_patch); 
            }
            photo_map_[j].life_time_ += 1;
            photo_map_[j].uv_l_ = uv1_h.head<2>();
            photo_map_[j].uv_r_ = uv_r1_h.head<2>();
            photo_map_[j].pg_f_ = pg_f;
        }
        else {
            marg_idx.push_back(j);
        }

        if (photo_map_[j].life_time_ >= max_lifetime_) {
            V4d marg_map_j;
            marg_map_j << pg_f, intensity_new;
            draw_marg_map_.push_back(marg_map_j);
        }
    }

    // Marginalize invalid features
    if (marg_idx.size() > 0) {
        std::cout << "In tracking, throw " << marg_idx.size() 
                << " features" << std::endl;
        marginalizeFromIndex(marg_idx);
    }
}

void sl_iekf::marginalizeEnsemble(const std::vector<int>& marg_idx,
    Eigen::MatrixXd& delta_Z) {
    
    int cnt_stack = delta_Z.rows() - marg_idx.size();
    Eigen::MatrixXd dZ_en_marg(cnt_stack, Nen_);
    int cnt_r = 0;
    for (int r = 0; r < delta_Z.rows(); r ++) {
        bool is_valid = true;
        for (int j = 0; j < marg_idx.size(); j ++) {
            if (r == marg_idx[j]) {
                is_valid = false;
                break;
            }
        }
        if (is_valid) {
            dZ_en_marg.block(cnt_r,0,1,Nen_) =
                delta_Z.block(r,0,1,Nen_);
            cnt_r ++;
        }
    }
    delta_Z = dZ_en_marg;
}

void sl_iekf::marginalizeFromIndex(
    const std::vector<int>& marg_idx) {

    std::vector<photometric_map> photo_map_marg;

    Eigen::MatrixXd cov_r_marg =
            Eigen::MatrixXd::Zero(cov_.rows()-marg_idx.size(), cov_.cols());

    // Erase invalid rows in photo_map_ and cov_
    int cnt_r = 0;
    for (int r = 0; r < cov_.rows(); r ++) {
        bool is_valid = true;
        for (int j = 0; j < marg_idx.size(); j ++) {
            if (r == Dx+Ds+marg_idx[j]) {
                is_valid = false;
                break;
            }
        }
        if (is_valid) {
            cov_r_marg.block(cnt_r, 0, 1, cov_.cols()) =
                cov_.block(r, 0, 1, cov_.cols());
            cnt_r ++;
            if (r > Dx+Ds-1) {
                photo_map_marg.push_back(photo_map_[r-Dx-Ds]);
            }
        }
    }
    photo_map_ = photo_map_marg;

    // Erase invalid cols in cov_
    Eigen::MatrixXd cov_marg =
        Eigen::MatrixXd::Zero(cov_.rows()-marg_idx.size(),
                              cov_.cols()-marg_idx.size());

    int cnt_c = 0;
    for (int c = 0; c < cov_.cols(); c ++) {
        bool is_valid = true;
        for (int j = 0; j < marg_idx.size(); j ++) {
            if (c == Dx+Ds+marg_idx[j]) {
                is_valid = false;
                break;
            }
        }
        if (is_valid) {
            cov_marg.block(0, cnt_c, cov_marg.rows(), 1) =
                cov_r_marg.block(0, c, cov_r_marg.cols(), 1);
            cnt_c ++;
        }
    }
    cov_ = cov_marg;

    if (cov_.cols() != cov_.rows()) {
        ROS_ERROR("Covariance matrix is not square !!!");
    }
    if (cov_.cols()-Dx-Ds != photo_map_.size()) {
        ROS_ERROR("Covariance matrix has no the proper size !!!");
    }
}

void sl_iekf::propagateIMU_Euler(const V3d& fib_b, const V3d& wib_b,
        const double& dt) {
    // State update
    V3d theta = wib_b*dt;
    M3d del_R = nesl::rvec2dcm(theta);
    M3d R0 = Xi_.block<3,3>(0,0);
    V4d q0 = nesl::dcm2quat(R0);
    V3d v0 = Xi_.block<3,1>(0,4);

    M3d R1 = R0 * del_R;
    V3d v1 = v0 + dt*(R0*fib_b + grav_);
    V3d p1 = Xi_.block<3,1>(0,3) + dt*v0 + 0.5*dt*dt*(R0*fib_b + grav_);
    Xi_ << R1, p1, v1,
        0, 0, 0, 1,0,
        0, 0, 0, 0, 1;
}

void sl_iekf::propagateCovariance(const double& dt) {
    M15d F = M15d::Zero();
    M3d R0 = Xi_.block<3,3>(0,0);
    V3d p0 = Xi_.block<3,1>(0,3);
    V3d v0 = Xi_.block<3,1>(0,4);
    F.block<3,3>(0,12) = -R0;
    F.block<3,3>(3,6) = M3d::Identity();
    F.block<3,3>(3,12) = -nesl::Vec2SkewMat(p0)*R0;
    F.block<3,3>(6,0) = nesl::Vec2SkewMat(grav_);
    F.block<3,3>(6,9) = -R0;
    F.block<3,3>(6,12) = -nesl::Vec2SkewMat(v0)*R0;

    Eigen::Matrix<double, 15, 12> G =
        Eigen::Matrix<double, 15, 12>::Zero();
    G.block<3,3>(0,3) = -R0;
    G.block<3,3>(3,3) = -nesl::Vec2SkewMat(p0)*R0;
    G.block<3,3>(6,0) = -R0;
    G.block<3,3>(6,3) = -nesl::Vec2SkewMat(v0)*R0;
    G.block<3,3>(9,6) = M3d::Identity();
    G.block<3,3>(12,9) = M3d::Identity();

    // State-transition matrix
    M15d Phi = M15d::Identity() + F*dt + 0.5*dt*dt*F*F;
    M15d cov_i = cov_.block<15,15>(0,0);
    cov_.block<15,15>(0,0) = Phi*cov_i*Phi.transpose()
        + G*Q_*G.transpose()*dt;

    // Propagate cross-correlations
    if (cov_.cols() > 15) {
        int N_d = photo_map_.size();
        Eigen::MatrixXd cov_cross = cov_.block(0,Dx,Dx,Ds+N_d);
        cov_.block(0,Dx,Dx,Ds+N_d) = Phi*cov_cross;
        cov_.block(Dx,0,Ds+N_d,Dx) = cov_.block(0,Dx,Dx,Ds+N_d).transpose();
    }
}

void sl_iekf::propagate(const sensor_msgs::ImuConstPtr& imu_msg,
        const double& dt) {
    // Nominal propagation - Euler integration
    // retrieve IMU measurements
    double ax = imu_msg->linear_acceleration.x;
    double ay = imu_msg->linear_acceleration.y;
    double az = imu_msg->linear_acceleration.z;
    V3d fib_b;
    fib_b << ax, ay, az;
    fib_b -= ba_;

    double wx = imu_msg->angular_velocity.x;
    double wy = imu_msg->angular_velocity.y;
    double wz = imu_msg->angular_velocity.z;
    V3d wib_b;
    wib_b << wx, wy, wz;
    wib_b -= bg_;

    // IMU-state propagation
    propagateCovariance(dt);
    propagateIMU_Euler(fib_b, wib_b, dt);
}

void sl_iekf::InitializeState(
    std::vector<sensor_msgs::ImuConstPtr>& imu_buf) {

    // Coarse alignment with IMU window
    V3d f_mean = V3d::Zero();
    V3d w_mean = V3d::Zero();
    for (unsigned int i = 0; i < imu_buf.size(); i++) {
        V3d f_i(imu_buf[i]->linear_acceleration.x,
        imu_buf[i]->linear_acceleration.y,
        imu_buf[i]->linear_acceleration.z);
        V3d w_i(imu_buf[i]->angular_velocity.x,
        imu_buf[i]->angular_velocity.y,
        imu_buf[i]->angular_velocity.z);
        f_mean += f_i;
        w_mean += w_i;
    }
    f_mean /= imu_buf.size();
    w_mean /= imu_buf.size();

    double roll0 = std::atan2(-f_mean(1), -f_mean(2));
    double pitch0 = std::atan(f_mean(0) /
        std::sqrt(f_mean(1)*f_mean(1) + f_mean(2)*f_mean(2)));
    V3d euler0(roll0, pitch0, 0.0);
    
    Xi_ << nesl::euler2dcm(euler0), V3d::Zero(), V3d::Zero(),
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;
    T0_ = Xi_.block<4,4>(0,0);
    ba_ << 0.0, 0.0, 0.0;
    bg_ = w_mean;  // when stationary at begining
    grav_ << 0.0, 0.0, 9.81;  // referenced at ned frame

    std::cout << "IMU buffer size: " << imu_buf.size() << std::endl;
    std::cout << "Mean fm: " << f_mean(0) << ", " << f_mean(1) << ", " << f_mean(2) << std::endl;
    std::cout << "Initial roll, pitch [deg]: " << R2D*roll0 << ", " << R2D*pitch0 << std::endl;
    std::cout << "Initial gyro bias [deg/s]: " << R2D*w_mean(0) << ", "
        << R2D*w_mean(1) << ", " << R2D*w_mean(2) << std::endl;

    ROS_INFO("State initialization completed !");

}

void sl_iekf::LoadParameters(ros::NodeHandle &n) {
    // read from yaml
    double std0_r, std0_p, std0_v, std0_ba, std0_bg, std0_depth, std0_idepth;
    n.param<double>("/envio_node/P0/attitude", std0_r, 1.7e-3);
    n.param<double>("/envio_node/P0/position", std0_p, 1e-10);
    n.param<double>("/envio_node/P0/velocity", std0_v, 1e-3);
    n.param<double>("/envio_node/P0/ba", std0_ba, 0.1962);
    n.param<double>("/envio_node/P0/bg", std0_bg, 8.7e-3);
    n.param<double>("/envio_node/P0/depth", std0_depth, 0.5);
    n.param<double>("/envio_node/P0/idepth", std0_idepth, 0.1);

    double std_a, std_g, std_ba, std_bg;
    n.param<double>("/envio_node/Q/velocity", std_a, 2.3e-3);
    n.param<double>("/envio_node/Q/attitude", std_g, 2.3562e-4);
    n.param<double>("/envio_node/Q/ba", std_ba, 2.4525e-4);
    n.param<double>("/envio_node/Q/bg", std_bg, 7.0298e-6);

    n.param<double>("/envio_node/cam0/timeshift_cam_imu", td_, 0.0);

    double std_meas;
    n.param<double>("/envio_node/R_std", std_meas, 16);

    Eigen::Isometry3d T_cam_imu = utils::
        getSE3matrix(n, "/envio_node/cam0/T_cam_imu");

    double max_depth;

    n.param<bool>("/envio_node/inverse_depth", is_idepth_, true);
    n.param<int>("/envio_node/thr_num", thr_num_, 300);
    n.param<int>("/envio_node/max_lifetime", max_lifetime_, 60);
    n.param<int>("/envio_node/uniform_dist", uniform_dist_, 15);
    n.param<int>("/envio_node/max_iter", max_iter_, 1);
    n.param<double>("/envio_node/max_itime", max_itime_, 0.02);
    n.param<double>("/envio_node/thr_stop", thr_stop_, 1e-3);
    n.param<double>("/envio_node/max_diff", max_diff_, 100);
    n.param<double>("/envio_node/max_depth", max_depth, 20);
    n.param<int>("/envio_node/num_init_samples", NUM_INIT_SAMPLES_, 600);

    n.param<int>("/envio_node/N_en", Nen_, 100);
    n.param<bool>("/envio_node/use_stochastic_gradient", use_sl_, true);
    n.param<double>("/envio_node/thr_weak", thr_weak_, 3.0);

    // R^{vehicle}_{IMU}
    double roll_vi, pitch_vi, yaw_vi;
    n.param<double>("/envio_node/roll_imu_vehicle", roll_vi, 0.0);
    n.param<double>("/envio_node/pitch_imu_vehicle", pitch_vi, 0.0);
    n.param<double>("/envio_node/yaw_imu_vehicle", yaw_vi, 0.0);
    V3d euler_vi(D2R*roll_vi, D2R*pitch_vi, D2R*yaw_vi);
    Rvi_ = nesl::euler2dcm(euler_vi);

    // Print
    ROS_INFO("IMU parameters.P0.attitude (std_dev): %f", std0_r);
    ROS_INFO("IMU parameters.P0.position (std_dev): %f", std0_p);
    ROS_INFO("IMU parameters.P0.velocity (std_dev): %f", std0_v);
    ROS_INFO("IMU parameters.P0.ba (std_dev): %f", std0_ba);
    ROS_INFO("IMU parameters.P0.bg (std_dev): %f", std0_bg);
    if (is_idepth_) {
        ROS_INFO("Mapping parameter.P0.idepth (std_dev): %f", std0_idepth);
    }
    else {
        ROS_INFO("Mapping parameter.P0.depth (std_dev): %f", std0_depth);
    }

    ROS_INFO("IMU parameters.Q.velocity (std_dev): %f", std_a);
    ROS_INFO("IMU parameters.Q.attitude (std_dev): %f", std_g);
    ROS_INFO("IMU parameters.Q.ba (std_dev): %f", std_ba);
    ROS_INFO("IMU parameters.Q.bg (std_dev): %f", std_bg);

    ROS_INFO("Measurement parameters.td: %f [ms]", 1000*td_);
    ROS_INFO("Measurement parameters.R (std_dev): %f", std_meas);

    ROS_INFO("Estimator parameters.thr_num: %d", thr_num_);
    ROS_INFO("Estimator parameters.max_lifetime: %d", max_lifetime_);
    ROS_INFO("Estimator parameters.uniform_dist: %d", uniform_dist_);
    ROS_INFO("Estimator parameters.max_iter: %d", max_iter_);
    ROS_INFO("Estimator parameters.max_itime: %f", max_itime_);
    ROS_INFO("Estimator parameters.thr_stop: %f", thr_stop_);
    ROS_INFO("Estimator parameters.max_diff: %f", max_diff_);
    ROS_INFO("Estimator parameters.max_depth: %f", max_depth);

    if (use_sl_) {
        ROS_INFO("Use stochastic linearization: ");
        ROS_INFO("%d ensembles", Nen_);
        ROS_INFO("%f weak gradient threshold", thr_weak_);
    }
    else {
        ROS_INFO("No stochastic linearization: ");
    }


    // Assign
    cov_ = Eigen::MatrixXd::Zero(15, 15);
    cov_.block<3,3>(0,0) = pow(std0_r,2)*M3d::Identity();
    cov_(2,2) = 1e-20;  // heading
    cov_.block<3,3>(3,3) = pow(std0_p,2)*M3d::Identity();
    cov_.block<3,3>(6,6) = pow(std0_v,2)*M3d::Identity();
    cov_.block<3,3>(9,9) = pow(std0_ba,2)*M3d::Identity();
    cov_.block<3,3>(12,12) = pow(std0_bg,2)*M3d::Identity();
    if (is_idepth_) {
        rho_var0_ = std0_idepth*std0_idepth;
    }
    else {
        rho_var0_ = std0_depth*std0_depth;
    }

    Q_ = M12d::Zero();
    Q_.block<3,3>(0,0) = pow(std_a,2)*M3d::Identity();
    Q_.block<3,3>(3,3) = pow(std_g,2)*M3d::Identity();
    Q_.block<3,3>(6,6) = pow(std_ba,2)*M3d::Identity();
    Q_.block<3,3>(9,9) = pow(std_bg,2)*M3d::Identity();

    R_ = std_meas*std_meas;

    Tlb_ = M4d::Identity();
    Tlb_.block<3,3>(0,0) = T_cam_imu.linear() * Rvi_.transpose();
    Tlb_.block<3,1>(0,3) = T_cam_imu.translation();
    Tbl_ = Tlb_.inverse();

    draw_max_depth_ = 0.8 * max_depth;

    // Set initial mask
    std::vector<int> resolution;
    n.getParam("/envio_node/cam0/resolution", resolution);
    mask_ = cv::Mat(resolution[1], resolution[0], CV_8UC1, cv::Scalar(255));
}

void sl_iekf::validateCovMtx(const Eigen::MatrixXd& P_mtx,
    Eigen::MatrixXd& P_pd){

  // force to make symmetric
  P_pd = 0.5 * (P_mtx + P_mtx.transpose());
  Eigen::LLT<Eigen::MatrixXd> lltOfP(P_pd);

  if (lltOfP.info() == Eigen::NumericalIssue){
    ROS_WARN("System Cov Mtx is not Positive Definite !");
    Eigen::SelfAdjointEigenSolver<MatrixXd> sv(P_pd);
    Eigen::MatrixXd D = sv.eigenvalues().asDiagonal();
    Eigen::MatrixXd V = sv.eigenvectors();

    for (int i = 0; i < D.rows(); i++){
      if (D(i,i) < 2.2204e-16) D(i,i) = 1e-12;
    }
    P_pd = V*D*V.transpose();
  }
}

void sl_iekf::registerPub(ros::NodeHandle &n) {
    pub_state_ = n.advertise<nav_msgs::Odometry>
        ("/envio_nesl/odom", 1000);

    pub_path_ = n.advertise<nav_msgs::Path>
        ("/envio_nesl/path", 1000);
    
    pub_tracking_points_ = n.advertise<sensor_msgs::PointCloud2>
        ("/envio_nesl/tracking_points", 1000);

    pub_marg_points_ = n.advertise<sensor_msgs::PointCloud2>
        ("/envio_nesl/marg_points", 1000);

    pub_match0_ = n.advertise<sensor_msgs::Image>
        ("/envio_nesl/feature_img0", 1000);

    pub_match1_ = n.advertise<sensor_msgs::Image>
        ("/envio_nesl/feature_img1", 1000);
}

void sl_iekf::publishPoseAndMap(const std_msgs::Header& header,
    const cv::Mat& uimg_l, const cv::Mat& uimg_r) {
    // Odometry and image publish was copied from the implementation 
    // by T. Qin, P. Li, Z. Yang, and S. Shen in VINS-Mono (estimator_node.cpp)
    // https://github.com/HKUST-Aerial-Robotics/VINS-Mono

    // Map publish was copied from the implementation 
    // by P. Geneva, K. Eckenhoff, W. Lee, Y. Yang and G. Huang 
    // in OpenVINS (ov_msckf/src/core/RosVisualizer.cpp)
    // https://github.com/rpng/open_vins

    // publish tf (imu pose)
    static tf::TransformBroadcaster tf_pub;
    geometry_msgs::TransformStamped odom_trans;
    odom_trans.header = header;
    odom_trans.header.frame_id = "world";
    odom_trans.child_frame_id = "IMU";

    M3d R_gb = Xi_.block<3,3>(0,0);
    V4d q_gb = nesl::dcm2quat(R_gb);

    odom_trans.transform.translation.x = Xi_(0,3);
    odom_trans.transform.translation.y = Xi_(1,3);
    odom_trans.transform.translation.z = Xi_(2,3);
    odom_trans.transform.rotation.w = q_gb(0,0);
    odom_trans.transform.rotation.x = q_gb(1,0);
    odom_trans.transform.rotation.y = q_gb(2,0);
    odom_trans.transform.rotation.z = q_gb(3,0);

    tf_pub.sendTransform(odom_trans);

    geometry_msgs::TransformStamped cam0_trans;
    cam0_trans.header = header;
    cam0_trans.header.frame_id = "IMU";
    cam0_trans.child_frame_id = "cam0";

    M3d R_bc = Tbl_.block<3,3>(0,0);
    V4d q_bc = nesl::dcm2quat(R_bc);

    // T^{header.frame_id}_{child_frame_id}
    cam0_trans.transform.translation.x = Tbl_(0,3);
    cam0_trans.transform.translation.y = Tbl_(1,3);
    cam0_trans.transform.translation.z = Tbl_(2,3);
    cam0_trans.transform.rotation.w = q_bc(0,0);
    cam0_trans.transform.rotation.x = q_bc(1,0);
    cam0_trans.transform.rotation.y = q_bc(2,0);
    cam0_trans.transform.rotation.z = q_bc(3,0);
    tf_pub.sendTransform(cam0_trans);

    // publish odometry
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "IMU";
    odometry.pose.pose.position.x = Xi_(0,3);
    odometry.pose.pose.position.y = Xi_(1,3);
    odometry.pose.pose.position.z = Xi_(2,3);
    odometry.pose.pose.orientation.w = q_gb(0,0);
    odometry.pose.pose.orientation.x = q_gb(1,0);
    odometry.pose.pose.orientation.y = q_gb(2,0);
    odometry.pose.pose.orientation.z = q_gb(3,0);

    V3d v_body = R_gb.transpose()*Xi_.block<3,1>(0,4);
    odometry.twist.twist.linear.x = v_body(0,0);
    odometry.twist.twist.linear.y = v_body(1,0);
    odometry.twist.twist.linear.z = v_body(2,0);

    pub_state_.publish(odometry);

    // publish path
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    path_.header = header;
    path_.header.frame_id = "world";
    path_.poses.push_back(pose_stamped);
    pub_path_.publish(path_);

    if (photo_map_.size() > 0) {
        sensor_msgs::PointCloud2 cloud;
        cloud.header = header;
        cloud.header.frame_id = "world";
        cloud.width = 3*photo_map_.size();
        cloud.height = 1;
        cloud.is_bigendian = false;
        cloud.is_dense = false;

        // Setup pointcloud fields
        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(3*photo_map_.size());

        // Iterators
        sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");

        // Fill our iterators
        for (int i = 0; i < photo_map_.size(); i ++) {
            *out_x = photo_map_[i].pg_f_(0,0); ++out_x;
            *out_y = photo_map_[i].pg_f_(1,0); ++out_y;
            *out_z = photo_map_[i].pg_f_(2,0); ++out_z;
        }

        pub_tracking_points_.publish(cloud);
    }

    // publish marginalized map
    if (draw_marg_map_.size() > 0) {
        sensor_msgs::PointCloud2 cloud;
        cloud.header = header;
        cloud.header.frame_id = "world";
        cloud.width = 6*draw_marg_map_.size();
        cloud.height = 1;
        cloud.is_bigendian = false;
        cloud.is_dense = false;

        // Setup pointcloud fields
        // "xyz": global map
        // "rgb": global map intensity
        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        modifier.resize(6*draw_marg_map_.size());

        // Iterators
        sensor_msgs::PointCloud2Iterator<float> out_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> out_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> out_z(cloud, "z");
        sensor_msgs::PointCloud2Iterator<uchar> out_r(cloud, "r");
        sensor_msgs::PointCloud2Iterator<uchar> out_g(cloud, "g");
        sensor_msgs::PointCloud2Iterator<uchar> out_b(cloud, "b");

        // Fill our iterators
        for (int i = 0; i < draw_marg_map_.size(); i ++) {
            *out_x = draw_marg_map_[i](0,0); ++out_x;
            *out_y = draw_marg_map_[i](1,0); ++out_y;
            *out_z = draw_marg_map_[i](2,0); ++out_z;

            *out_r = static_cast<unsigned char>(draw_marg_map_[i](3,0));
            ++out_r;
            *out_g = static_cast<unsigned char>(draw_marg_map_[i](3,0));
            ++out_g;
            *out_b = static_cast<unsigned char>(draw_marg_map_[i](3,0));
            ++out_b; 
        }

        pub_marg_points_.publish(cloud);

        draw_marg_map_.clear();
    }


    // publish images with features
    cv::Mat bgr_l(uimg_l.size(), CV_8UC3);
    cv::Mat bgr_r(uimg_r.size(), CV_8UC3);

    cv::cvtColor(uimg_l, bgr_l, CV_GRAY2RGB);
    cv::cvtColor(uimg_r, bgr_r, CV_GRAY2RGB);

    for (int j = 0; j < photo_map_.size(); j++) {
        cv::Point2f draw_u, draw_ur;
        draw_u.x = photo_map_[j].uv_l_(0,0);
        draw_u.y = photo_map_[j].uv_l_(1,0);

        draw_ur.x = photo_map_[j].uv_r_(0,0);
        draw_ur.y = photo_map_[j].uv_r_(1,0);

        if (photo_map_[j].life_time_ > 2) {
            cv::circle(bgr_l, draw_u, 4, cv::Scalar(0,
                255.0-255.0*photo_map_[j].depth_l_/draw_max_depth_,
                255.0*photo_map_[j].depth_l_/draw_max_depth_), -1);

            cv::circle(bgr_r, draw_ur, 4, cv::Scalar(0,
                255.0-255.0*photo_map_[j].depth_r_/draw_max_depth_,
                255.0*photo_map_[j].depth_r_/draw_max_depth_), -1);
        }
    }
    sensor_msgs::ImagePtr img_l_msg =
        cv_bridge::CvImage(header, "bgr8", bgr_l).toImageMsg();

    sensor_msgs::ImagePtr img_r_msg =
        cv_bridge::CvImage(header, "bgr8", bgr_r).toImageMsg();

    pub_match0_.publish(img_l_msg);
    pub_match1_.publish(img_r_msg);

}


void sl_iekf::sampleDeltaEnsembles(std::vector<V6d>& delta_T0,
    std::vector<V6d>& delta_T1, std::vector<double>& delta_Z) {
    
    std::normal_distribution<double> sampler(0., 1.0);

    double std_depth = sqrt(rho_var0_);

    for (int i = 0; i < Nen_; i ++) {
        // ith delta ensembles
        V6d dT0_i, dT1_i;

        for (int j = 0; j < Ds; j ++) {
            // ignore cross-correlation, down weight by a constant factor
            dT1_i(j,0) = 0.2 * sqrt(cov_(j,j)) * sampler(generator_);
            dT0_i(j,0) = 0.2 * sqrt(cov_(Dx+j, Dx+j)) * sampler(generator_);
        }

        // Fix std_dev for simple implementation
        double delta_Zi = std_depth * sampler(generator_);

        delta_T1.push_back(dT1_i);
        delta_T0.push_back(dT0_i);
        delta_Z.push_back(delta_Zi);
    }
}

void sl_iekf::sampleEnsembles(const std::vector<V6d>& delta_T0,
        const std::vector<V6d>& delta_T1, const std::vector<double>& delta_Z,
        std::vector<M4d>& T0_en, std::vector<M4d>& T1_en,
        std::vector<Eigen::VectorXd>& Z_en) {

    if (Nen_ != delta_T0.size()) {
        ROS_ERROR("Invalid ensemble size: Nen_ != delta_T0.size() !!!");
    }

    for (int i = 0; i < Nen_; i ++) {
        M4d T0_i, T1_i, dT0_i, dT1_i;

        Exp_se3(-delta_T1[i], dT1_i);
        Exp_se3(-delta_T0[i], dT0_i);

        T1_i = dT1_i * Xi_.block<4,4>(0,0);
        T0_i = dT0_i * T0_;

        Eigen::VectorXd Z_i(photo_map_.size());
        for (int j = 0; j < photo_map_.size(); j ++) {
            Z_i(j) = photo_map_[j].depth_l_ + delta_Z[i];
        }

        T1_en.push_back(T1_i);
        T0_en.push_back(T0_i);
        Z_en.push_back(Z_i);
    }
}


void sl_iekf::computeStochasticGradient(const cv::Mat& uimg_l,
    const M3d& Kl, const M3d& Kl_inv,
    const V3d& uv0_h, const V3d& uv1_h,
    const std::vector<M4d>& T0_en, const std::vector<M4d>& T1_en,
    const std::vector<Eigen::VectorXd>& Z_en, const int& jth,
    double& dI_x, double& dI_y) {

    // Predicted intensity
    double h_hat = utils::GetPixelValue(uimg_l, uv1_h(0,0), uv1_h(1,0));

    M2d cov_x_accum = M2d::Zero();
    V2d cov_yx_accum = V2d::Zero();
    V2d mean_dx_accum = V2d::Zero();
    int cnt_valid = 0;

    for (int i = 0; i < Nen_; i ++) {
        // warp features
        M4d T_gc0 = T0_en[i] * Tbl_;
        M4d T_gc1 = T1_en[i] * Tbl_;
        M4d T_c1g = M4d::Identity();
        T_c1g.block<3,3>(0,0) = T_gc1.block<3,3>(0,0).transpose();
        T_c1g.block<3,1>(0,3) = -T_gc1.block<3,3>(0,0).transpose()
            *T_gc1.block<3,1>(0,3);
        M4d T_c10 = T_c1g * T_gc0;
        M3d R_c10 = T_c10.block<3,3>(0,0);
        V3d p_c10 = T_c10.block<3,1>(0,3);

        V3d uv0_n = Kl_inv*uv0_h;
        V3d pc0_f = Z_en[i](jth,0) * uv0_n;
        V3d pc1_f = R_c10 * pc0_f + p_c10;
        double Z1 = pc1_f(2,0);
        double inv_Z1 = 1/Z1;
        V3d uv1_n = pc1_f *inv_Z1;
        V3d uv1_en = Kl * uv1_n;

        if ( (uv1_en(0,0) > 0) && (uv1_en(0,0) < uimg_l.cols) &&
            (uv1_en(1,0) > 0) && (uv1_en(1,0) < uimg_l.rows) && (Z1 > 0) ) {
            
            cnt_valid ++;

            V2d dx_i = uv1_en.head<2>() - uv1_h.head<2>();
            double y_i = utils::GetPixelValue(uimg_l, uv1_en(0,0), uv1_en(1,0));

            // mean of delta uv
            mean_dx_accum = mean_dx_accum + dx_i;

            // Covariance of delta uv
            cov_x_accum = cov_x_accum + (dx_i * dx_i.transpose());

            // Covariance of yx
            cov_yx_accum = cov_yx_accum + (y_i * dx_i);
        }
    }
    if (cnt_valid < 0.1*Nen_) {
        ROS_WARN("Not enough valid ensembels !!!");
    }

    M2d cov_x = cov_x_accum/(cnt_valid - 1);
    V2d cov_yx = cov_yx_accum/(cnt_valid - 1);
    V2d mean_dx = mean_dx_accum/cnt_valid;

    double det_covx = cov_x(0,0)*cov_x(1,1) - cov_x(0,1)*cov_x(1,0);

    if (det_covx < 1e-10) {
        ROS_WARN("Stochastic gradient is close to singular !!!");
    }
    M2d icov_x;
    icov_x << cov_x(1,1), -cov_x(0,1), -cov_x(1,0), cov_x(0,0);
    icov_x = (1/det_covx)*icov_x;

    dI_x = (icov_x*(cov_yx - h_hat*mean_dx))(0,0);
    dI_y = (icov_x*(cov_yx - h_hat*mean_dx))(1,0);
}


void sl_iekf::TwoViewTriangulation(const cv::Point2f& pt0,
    const cv::Point2f& pt1, const M3d& K0_inv, const M3d& K1_inv,
    const M4d& T_01, double& depth0, double& depth1) {

    V3d x0(pt0.x, pt0.y, 1.0);
    V3d x1(pt1.x, pt1.y, 1.0);
    x0 = K0_inv*x0;
    x1 = K1_inv*x1;
    Eigen::Matrix<double, 3, 2> A;
    A << x0, -T_01.block<3,3>(0,0)*x1;
    V2d lambda = (A.transpose()*A).ldlt().
        solve(A.transpose()*T_01.block<3,1>(0,3));
    depth0 = lambda(0);
    depth1 = lambda(1);
}

}