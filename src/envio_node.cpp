/**
 * @file envio_node.cpp
 * @author Jae Hyung Jung (lastflowers@snu.ac.kr)
 * @brief vio frontend using photometric innovation
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

#include <iostream>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <fstream>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include "core/sl_iekf.h"

std::condition_variable con;
std::queue<sensor_msgs::ImuConstPtr> imu_buf;
std::queue<nesl::vision_meas> vis_buf;
std::vector<sensor_msgs::ImuConstPtr> imu_init_buf;
std::mutex m_buf;  // imu and vision buffer mutex
std::mutex m_estimator;  // estimator mutex

nesl::sl_iekf estimator;
double last_imu_t = 0;
double last_img_t = 0;
cv::Mat mask_cur;
bool init_filter = false;
bool flag_update = false;

int image_callback_counter = 0;
int image_process_counter = 0;
int image_throw_counter = 0;

// R^{Vehicle}_{IMU}
M3d R_vi;

std::ofstream est_fout;

void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    // ROS_INFO("imu data received.");
    if (imu_msg->header.stamp.toSec() <= last_imu_t) {
        ROS_WARN("imu messages in distorder!");
        return;
    }

    // Rotate from imu to vehicle frame
    V3d f_raw(imu_msg->linear_acceleration.x,
        imu_msg->linear_acceleration.y,
        imu_msg->linear_acceleration.z);
    
    V3d w_raw(imu_msg->angular_velocity.x,
        imu_msg->angular_velocity.y,
        imu_msg->angular_velocity.z);

    V3d f_rot = R_vi * f_raw;
    V3d w_rot = R_vi * w_raw;

    sensor_msgs::Imu *imu_rot = new sensor_msgs::Imu();
    imu_rot->header = imu_msg->header;
    imu_rot->angular_velocity.x = w_rot(0);
    imu_rot->angular_velocity.y = w_rot(1);
    imu_rot->angular_velocity.z = w_rot(2);

    imu_rot->linear_acceleration.x = f_rot(0);
    imu_rot->linear_acceleration.y = f_rot(1);
    imu_rot->linear_acceleration.z = f_rot(2);

    sensor_msgs::ImuConstPtr imu_ptr(imu_rot);

    // protect imu_buf from other thread
    m_buf.lock();
    imu_buf.push(imu_ptr);
    m_buf.unlock();
    con.notify_one();
}

void stereo_callback(const sensor_msgs::ImageConstPtr& caml_img,
    const sensor_msgs::ImageConstPtr& camr_img) {
    // ROS_INFO("Stereo image received.");
    m_estimator.lock();
    cv::Mat mask = estimator.mask();
    m_estimator.unlock();
    
    nesl::vision_meas vismeas_cur(caml_img->header);
        
    ros::Time start_time = ros::Time::now();
    vismeas_cur.processStereoImage(caml_img, camr_img, mask);

    double vis_processing_time = (ros::Time::now()-start_time).toSec();
    std::cout << "Vision precessing time: " <<
        vis_processing_time << std::endl;

    m_buf.lock();
    vis_buf.push(vismeas_cur);
    m_buf.unlock();
    con.notify_one();

    image_callback_counter++;
}


std::vector<std::pair
    <std::vector<sensor_msgs::ImuConstPtr>,
    nesl::vision_meas>> getMeasurements() {

    std::vector<std::pair
        <std::vector<sensor_msgs::ImuConstPtr>,
        nesl::vision_meas>> measurements;

    while(true) {
        if (imu_buf.empty() || vis_buf.empty()) break;

        if (!(imu_buf.back()->header.stamp.toSec() >
            vis_buf.front().timestamp() + estimator.getTimeOffset())) break;
        
        if (!(imu_buf.front()->header.stamp.toSec() <
            vis_buf.front().timestamp() + estimator.getTimeOffset())) {
            ROS_WARN("throw img, only happen at the beginning");
            image_throw_counter++;
            vis_buf.pop();
            continue;
        }
        nesl::vision_meas vis_meas = vis_buf.front();
        vis_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() <=
            vis_meas.timestamp() + estimator.getTimeOffset()) {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        if (IMUs.empty()) ROS_WARN("no imu btw two images");
        std::pair<std::vector<sensor_msgs::ImuConstPtr>, nesl::vision_meas> temp_pair; 
        temp_pair.first = IMUs;
        temp_pair.second = vis_meas;
        measurements.emplace_back(temp_pair);
    }
    return measurements;
}


void process(){
    // This process framework was implemented by T. Qin, P. Li, Z. Yang, and S. Shen in VINS-Mono (estimator_node.cpp)
    // https://github.com/HKUST-Aerial-Robotics/VINS-Mono
    // We have modified it to adopt our filtering-based estimator
    while(true) {
        std::vector<std::pair
            <std::vector<sensor_msgs::ImuConstPtr>,
            nesl::vision_meas>> measurements;

        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
            { return (measurements = getMeasurements()).size() != 0; });
        lk.unlock();

        // Filter main loop
        m_estimator.lock();
        for (auto& measurement : measurements) {
            double processing_time_frarme = 0;

            // Filter initialization
            if (!init_filter) {
                imu_init_buf.insert(std::end(imu_init_buf),
                    std::begin(measurement.first), std::end(measurement.first));

                if (imu_init_buf.size() > estimator.NUM_INIT_SAMPLES_) {
                    estimator.InitializeState(imu_init_buf);
                    imu_init_buf.clear();
                    last_imu_t = measurement.first.back()->header.stamp.toSec();
                    last_img_t = measurement.second.timestamp()
                        + estimator.getTimeOffset();
                    init_filter = true;
                }
            }
            else {
                ros::Time start_time = ros::Time::now();
                // INS prediction
                for (auto &imu_msg : measurement.first) {
                    double now_imu_t = imu_msg->header.stamp.toSec();
                    double imu_dt;
                    if (flag_update) {
                        flag_update = false;
                        imu_dt = now_imu_t - last_img_t;
                    }
                    else {
                        imu_dt = now_imu_t - last_imu_t;
                    }
                    last_imu_t = now_imu_t;
                    estimator.propagate(imu_msg, imu_dt);
                }
                double imu_processing_time =
                    (ros::Time::now()-start_time).toSec();

                // Process vision measurement
                double img_t = measurement.second.timestamp()
                    + estimator.getTimeOffset();

                start_time = ros::Time::now();

                double gap = img_t - measurement.first.back()
                    ->header.stamp.toSec();
                estimator.propagate(measurement.first.back(), gap);
                
                estimator.processMeasurement(measurement.second);
                flag_update = true;

                double vis_processing_time =
                    (ros::Time::now()-start_time).toSec();
                
                processing_time_frarme = vis_processing_time
                    + imu_processing_time;


                start_time = ros::Time::now();

                std_msgs::Header header = measurement.second.header();
                estimator.publishPoseAndMap(header,
                    measurement.second.uimg_l(),
                    measurement.second.uimg_r());

                double pub_processing_time =
                    (ros::Time::now()-start_time).toSec();
                    
                std::cout << "Filter propagation time: " << imu_processing_time
                    << "   Filter update time: " << vis_processing_time 
                    << "   Publish time: " << pub_processing_time 
                    << "\n" << std::endl;

                last_img_t = img_t;

                // Save after initialization
                est_fout.precision(19);
                est_fout << last_img_t << " ";
                est_fout.precision(19);
                for (int i = 0; i < 4; i ++) est_fout << estimator.q_gb()(i) << " ";
                for (int i = 0; i < 3; i ++) est_fout << estimator.p_gb()(i) << " ";
                for (int i = 0; i < 3; i ++) est_fout << estimator.v_gb()(i) << " ";
                for (int i = 0; i < 3; i ++) est_fout << estimator.ba()(i) << " ";
                for (int i = 0; i < 3; i ++) est_fout << estimator.bg()(i) << " ";
                for (int i = 0; i < 3; i ++) est_fout << estimator.std_q()(i) << " ";
                for (int i = 0; i < 3; i ++) est_fout << estimator.std_p()(i) << " ";
                for (int i = 0; i < 3; i ++) est_fout << estimator.std_v()(i) << " ";
                for (int i = 0; i < 3; i ++) est_fout << estimator.std_ba()(i) << " ";
                for (int i = 0; i < 3; i ++) est_fout << estimator.std_bg()(i) << " ";
                est_fout << imu_processing_time << " ";
                est_fout << vis_processing_time << std::endl;

            }
            image_process_counter++;
        }
        m_estimator.unlock();
    }
}

int main(int argc, char **argv) {

    est_fout.open("/home/nesl/Desktop/est_out.txt");
    ros::init(argc, argv, "envio_node");
    ros::NodeHandle n;

    // load from yaml file
    estimator.LoadParameters(n);
    R_vi = estimator.Rvi_;
    nesl::vision_meas::LoadParameters(n);

    // Register publisher
    estimator.registerPub(n);

    // IMU subscriber
    ros::Subscriber sub_imu = n.subscribe("/imu", 2000, imu_callback,
        ros::TransportHints().tcpNoDelay());

    // Stereo camera subscriber
    message_filters::Subscriber<sensor_msgs::Image>
        caml_sub(n, "/left_image", 200);
    message_filters::Subscriber<sensor_msgs::Image>
        camr_sub(n, "/right_image", 200);
    message_filters::TimeSynchronizer
        <sensor_msgs::Image, sensor_msgs::Image>
            stereo_sub(caml_sub, camr_sub, 200);
    stereo_sub.registerCallback(
            boost::bind(&stereo_callback, _1, _2));

    std::thread VIO_thread{process};
    ros::spin();

    std::cout << "Image callback counter: " << image_callback_counter
        << "   Image processing counter: " << image_process_counter
        << "   #throws: " << image_throw_counter << std::endl;

    est_fout.close();

    return 0; 
}