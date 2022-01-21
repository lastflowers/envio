/**
 * @file define.h
 * @author Sejong Heo
 * @author Jae Hyung Jung (lastflowers@snu.ac.kr)
 * @brief Type definitions
 * @date 2021-03-30
 *
 * @copyright Copyright (c) 2021 Sejong Heo, Jae Hyung Jung
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

#ifndef MSCKF_TEST_DEFINE_H
#define MSCKF_TEST_DEFINE_H

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>

/* DEFINE ================================================================== */
const double  PI_F =3.14159265358979f;
#define D2R		(double)(PI_F/180.)			//deg to rad
#define R2D		(double)(180./PI_F)			//rad to deg

/* Matrix & Vector TYPEDEF ================================================= */
typedef Eigen::Matrix<double, 2, 2> M2d;
typedef Eigen::Matrix<double, 3, 3> M3d;
typedef Eigen::Matrix<double, 4, 4> M4d;
typedef Eigen::Matrix<double, 5, 5> M5d;
typedef Eigen::Matrix<double, 6, 6> M6d;
typedef Eigen::Matrix<double, 9, 9> M9d;
typedef Eigen::Matrix<double, 12, 12> M12d;
typedef Eigen::Matrix<double, 15, 15> M15d;
typedef Eigen::Matrix<double, 16, 16> M16d;
typedef Eigen::Matrix<double, 2, 1> V2d;
typedef Eigen::Matrix<double, 3, 1> V3d;
typedef Eigen::Matrix<double, 4, 1> V4d;
typedef Eigen::Matrix<double, 5, 1> V5d;
typedef Eigen::Matrix<double, 6, 1> V6d;
typedef Eigen::Matrix<double, 9, 1> V9d;
typedef Eigen::Matrix<double, 15, 1> V15d;
/* STURCTURE =============================================================== */
typedef struct {
    V3d accel;          // accelerometer measurements
    V3d gyro;           // gyroscope measurements
} ImuMeas;	//structure for storing sensor related data

typedef struct {
    size_t  feautre_id;
    std::vector<V2d> pts;
    std::vector<size_t> frames;
} VisMeas;	//structure for storing sensor related data

typedef struct {
    V3d p;              // postion
    V4d q;              // orientation expressed in quaternion
    M3d R;              // orientation expressed in SO3
    V3d v;              // velocity (a.k.a extended pose)
} Pose;

typedef struct{
    M3d K;              // camera calibration matrix
    Pose Tbc;           // Pose from body to camera
    V3d rad_dist;       // radial distortion
    V2d tan_dist;       // tangential distortion
} CalibInfo;

typedef struct {
    Pose T;             // pose from global to body
    V3d v;              // velocity in global frame
    V3d ba;             // accelerometer bias
    V3d bg;             // gyro bias
    double time_stamp;   // time stamp
} NavState;

typedef struct {
    V3d llh;            // latitude, longitude, height
    V3d v;              // velocity in {n}
    V4d q;              // orientation in quaternion from {n} to {b}
    V3d ba;             // accelerometer bias
    V3d bg;             // gyro bias
    double time_stamp;   // time stamp
} NavState2;
typedef struct {
    Pose T;             // pose from global to body
    size_t  frame;      // frame number of related image
    double time_stamp;   // time stamp
} CamState;

typedef struct {
    NavState IMU;                   // IMU state
    std::vector<CamState> SLW;      // Camera Sliding Window
} MsckfState;

typedef struct{
//    bool mono;    // 0/1 : mono/stereo
    double gait_trust;   // trust probability for Mahalanobis gaiting test
    double max_slw;      // maximum number of sliding window
    double min_rcond;    // minimum rank condition for good triangulation
    double max_cost;     // maximum error cost for good triangulation
    double gravity;      // gravity vector in global frame
} MsckfParams;

typedef struct{
    double vel_rw;       // velocity random walk [m/s^{\frac{3}{2}}]
    double ang_rw;       // angle random walk    [rad/s^{\frac{1}{2}}]
    double acc_bi;       // accelerometer bias (in)stability  [m/s^2]
    double gyr_bi;       // gyro bias (in)stability [rad/s^2]
} ProcessNoise;

typedef struct{
    double p;            // initial position uncertainty(1\sigma) [m]
    double v;            // initial velocity uncertainty(1\sigma) [m/s]
    double q;            // initial orientation uncertainty(1\sigma) [rad]
    double ba;           // initial accelerometer bias uncertainty(1\sigma) [m/s^2]
    double init_bg;      // initial gyro bias  uncertainty(1\sigma) [rad/s]
} InitUncertainty;

typedef struct{
    ProcessNoise pn;
    InitUncertainty iu;
    double px;           // pixelwise image feature uncertainty(1\sigma) [px]
} NoiseParams;



#endif //MSCKF_TEST_DEFINE_H
