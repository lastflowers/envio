/**
 * @file Attitude.cpp
 * @author Jae Hyung Jung (lastflowers@snu.ac.kr)
 * @brief Attitude transform functions
 * @date 2021-03-30
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
#include <Eigen/Eigen>
#include <math.h>
#include <unsupported/Eigen/MatrixFunctions>

#include "define.h"
#include "Attitude.h"

namespace nesl {

M3d euler2dcm(V3d &euler){
    double phi_ = euler(0,0);
    double theta_ = euler(1,0);
    double psi_ = euler(2,0);

    M3d Rz_;
    Rz_ << cos(psi_), -sin(psi_), 0.0,
           sin(psi_),  cos(psi_), 0.0,
           0.0,        0.0,       1.0;

    M3d Ry_;
    Ry_ << cos(theta_),  0.0, sin(theta_),
           0.0,          1.0, 0.0,
          -sin(theta_),  0.0, cos(theta_);

    M3d Rx_;
    Rx_ << 1.0, 0.0,        0.0,
           0.0, cos(phi_), -sin(phi_),
           0.0, sin(phi_),  cos(phi_);

    M3d dcm_ = Rz_ * Ry_ * Rx_;

    return dcm_;
}

V4d euler2quat(V3d &euler){
    double phi_ = euler(0,0);
    double theta_ = euler(1,0);
    double psi_ = euler(2,0);

    V4d quat_;
    quat_(0,0) = cos(phi_/2.0)*cos(theta_/2.0)*cos(psi_/2.0)
        + sin(phi_/2.0)*sin(theta_/2.0)*sin(psi_/2.0);
    quat_(1,0) = sin(phi_/2.0)*cos(theta_/2.0)*cos(psi_/2.0)
        - cos(phi_/2.0)*sin(theta_/2.0)*sin(psi_/2.0);
    quat_(2,0) = cos(phi_/2.0)*sin(theta_/2.0)*cos(psi_/2.0)
        + sin(phi_/2.0)*cos(theta_/2.0)*sin(psi_/2.0);
    quat_(3,0) = cos(phi_/2.0)*cos(theta_/2.0)*sin(psi_/2.0)
        - sin(phi_/2.0)*sin(theta_/2.0)*cos(psi_/2.0);

    return quat_;
}

M3d quat2dcm(V4d &quat){
//  from quaternion([scalar;vec] form) to direction cosine matrix
//  if you have quat({1}->{2}) that rotates frame {1} to frame {2},
//  this function calculates dcm(C^{1}_{2})
//  Input
//  quat: [q0(scalar); q_vec([q1;q2;q3])] [4x1] vector, from {1} to {2}
//  Output
//  dcm: C^{1}_{2} [3x3] matrix
    double qs = quat(0,0);
    V3d q_vec;
    q_vec << quat(1,0), quat(2,0), quat(3,0);
    M3d R = (std::pow(qs,2) - std::pow(q_vec.norm(),2))*M3d::Identity() +
        2*qs*Vec2SkewMat(q_vec) + 2*q_vec*q_vec.transpose();

    return R;

}

V3d quat2euler(V4d &quat){
    M3d dcm_ = quat2dcm(quat);
    V3d euler_ = dcm2euler(dcm_);

    return euler_;
}

V3d quat2rvec(V4d &quat){
    double qs_ = quat(0,0);
    V3d qv_;
    qv_ << quat(1,0), quat(2,0), quat(3,0);

    V3d rvec_;
    if (qv_.norm() == 0){
        rvec_ << 0.0, 0.0, 0.0;
    }
    else {
        rvec_ = (2.0*atan2(qv_.norm(), qs_))/qv_.norm() * qv_;
    }

    return rvec_;
}

M4d quatLeftComp(V4d &quat){
    double qs_ = quat(0,0);
    V3d qv_;
    qv_ << quat(1,0), quat(2,0), quat(3,0);

    M4d Q_;
    Q_(0,0) = qs_;
    Q_.block(0,1,1,3) = -qv_.transpose();
    Q_.block(1,0,3,1) = qv_;
    Q_.block(1,1,3,3) = qs_*M3d::Identity()
        + Vec2SkewMat(qv_);

    return Q_;
}

M4d quatRightComp(V4d &quat){
    double qs_ = quat(0,0);
    V3d qv_;
    qv_ << quat(1,0), quat(2,0), quat(3,0);

    M4d Q_;
    Q_(0,0) = qs_;
    Q_.block(0,1,1,3) = -qv_.transpose();
    Q_.block(1,0,3,1) = qv_;
    Q_.block(1,1,3,3) = qs_*M3d::Identity()
        - Vec2SkewMat(qv_);

    return Q_;
}

V4d quatInverse(V4d &quat){
    V4d q_;
    q_(0,0) =  quat(0,0);
    q_(1,0) = -quat(1,0);
    q_(2,0) = -quat(2,0);
    q_(3,0) = -quat(3,0);

    return q_;
}

V4d quatNormalize(V4d &quat){
// This function normalizes a given quaternion.
// Input :
// q: quaternion [q0(scalar); q_vec([q1;q2;q3]) [4x1] vector
// Output :
// q_n: quaternion [q0(scalar); q_vec([q1;q2;q3]) [4x1] vector

    V4d q_n;
    q_n = quat/quat.norm();
    return q_n;
} // inconsistent at ATTITUDE::m and ATTITUDE::ccp


V4d quatMultiply(V4d &q, V4d &p){
//  This function calculates quaternion multiplication (q*p)
//  of given two quaternions q and p.
//  Input :
//  q: quaternion [qs(scalar); q_vec([q1;q2;q3])] [4x1] vector
//  p: quaternion [ps(scalar); p_vec([p1;p2;p3])] [4x1] vector
//  Output :
//  q_result : quaternion [q0(scalar); q_vec([q1;q2;q3])] [4x1] vector
    double qs = q(0,0);
    double ps = p(0,0);
    V3d q_vec, p_vec;
    q_vec << q(1,0), q(2,0), q(3,0);
    p_vec << p(1,0), p(2,0), p(3,0);

    V4d q_result;
    q_result(0,0) = qs*ps - q_vec.transpose()*p_vec;
    q_result.block<3,1>(1,0) = qs*p_vec + ps*q_vec + q_vec.cross(p_vec);

    if(q_result(0,0) < 0.0){
      q_result = -q_result;
    }

    q_result = quatNormalize(q_result);

    return q_result;
}

V3d dcm2euler(M3d &dcm){
    double phi_, psi_;
    double theta_ = atan(-dcm(2,0)/sqrt(dcm(2,1)*dcm(2,1)
                                        + dcm(2,2)*dcm(2,2)));

    if (dcm(2,0) <= -0.999){
        phi_ = 1.0/0.0; // inducing runtime error (NaN)
        psi_ = atan2(dcm(1,2)-dcm(0,1), dcm(0,2)+dcm(1,1));
    }
    else if (dcm(2,0) >= 0.999){
        phi_ = 1.0/0.0; // inducing runtime error (NaN)
        psi_ = PI_F
            + atan2(dcm(1,2)+dcm(0,1), dcm(0,2)-dcm(1,1));
    }
    else {
        phi_ = atan2(dcm(2,1), dcm(2,2));
        psi_ = atan2(dcm(1,0), dcm(0,0));
    }

    V3d euler_;
    euler_ << phi_, theta_, psi_;

    return euler_;
}

V4d dcm2quat(M3d& dcm){
    V4d quat_;
    quat_(0,0) = sqrt(1.0/4.0*(1.0 + dcm(0,0) + dcm(1,1) + dcm(2,2)));
    quat_(1,0) = sqrt(1.0/4.0*(1.0 + dcm(0,0) - dcm(1,1) - dcm(2,2)));
    quat_(2,0) = sqrt(1.0/4.0*(1.0 - dcm(0,0) + dcm(1,1) - dcm(2,2)));
    quat_(3,0) = sqrt(1.0/4.0*(1.0 - dcm(0,0) - dcm(1,1) + dcm(2,2)));

    int idx_ = 0;

    for (int i=1;i<=3;i++){
        if (quat_(i,0)>quat_(idx_,0)){
            idx_ = i;
        }
    }

    if (idx_ == 0){
        //quat(0,0) = quat(0,0);
        quat_(1,0) = (dcm(2,1) - dcm(1,2))/(4.0*quat_(0,0));
        quat_(2,0) = (dcm(0,2) - dcm(2,0))/(4.0*quat_(0,0));
        quat_(3,0) = (dcm(1,0) - dcm(0,1))/(4.0*quat_(0,0));
    }
    else if (idx_ == 1){
        quat_(0,0) = (dcm(2,1) - dcm(1,2))/(4.0*quat_(1,0));
        //quat(1,0) = quat(1,0);
        quat_(2,0) = (dcm(1,0) + dcm(0,1))/(4.0*quat_(1,0));
        quat_(3,0) = (dcm(0,2) + dcm(2,0))/(4.0*quat_(1,0));
    }
    else if (idx_ == 2){
        quat_(0,0) = (dcm(0,2) - dcm(2,0))/(4.0*quat_(2,0));
        quat_(1,0) = (dcm(1,0) + dcm(0,1))/(4.0*quat_(2,0));
        //quat(2,0) = quat(2,0);
        quat_(3,0) = (dcm(2,1) + dcm(1,2))/(4.0*quat_(2,0));
    }
    else if (idx_ == 3){
        quat_(0,0) = (dcm(1,0) - dcm(0,1))/(4.0*quat_(3,0));
        quat_(1,0) = (dcm(0,2) + dcm(2,0))/(4.0*quat_(3,0));
        quat_(2,0) = (dcm(2,1) + dcm(1,2))/(4.0*quat_(3,0));
        //quat(3,0) = quat(3,0);
    }

    // Keep q0(scalar) be positive
    if (quat_(0,0) < 0){
        quat_ = -quat_;
    }

    return quat_;
}

V3d dcm2rvec(M3d &dcm){
    double theta_ = acos((dcm(0,0)+dcm(1,1)+dcm(2,2)-1.0)/2.0);
    V3d vect_;
    vect_ << dcm(2,1)-dcm(1,2), dcm(0,2)-dcm(2,0), dcm(1,0)-dcm(0,1);
    vect_ = 1.0/(2.0*sin(theta_))*vect_;
    V3d rvec_;
    rvec_ = theta_*vect_;

    return rvec_;
}

M3d Vec2SkewMat(V3d &vec){
    M3d Mat;
    Mat << 0.0, -vec(2,0), vec(1,0),
           vec(2,0), 0.0, -vec(0,0),
          -vec(1,0), vec(0,0), 0.0;

    return Mat;
}

M3d rvec2dcm(V3d &vec){
    double a_ = vec.norm();
    M3d dcm_ = M3d::Identity();
    if (a_ != 0) {
        V3d vect_ = (1.0/a_) * vec;
        M3d xMat_ = Vec2SkewMat(vect_);
        dcm_ = M3d::Identity() + sin(a_)*xMat_
            + (1.0 - cos(a_))*xMat_*xMat_;
    }
    // dcm = (ATTITUDE::Vec2SkewMat(vec)).exp();
    return dcm_;
} // unnecessary lines ?

V4d rvec2quat(V3d &rvec){
  double rot_ang = rvec.norm();
  V4d quat;

  if (rot_ang == 0.0){
    quat << 1.0, 0.0, 0.0, 0.0;
  }
  else {
    double cR = cos(rot_ang/2.0);
    double sR = sin(rot_ang/2.0);
    V3d sRk = sR * (rvec/rot_ang);
    quat(0,0) = cR;
    quat(1,0) = sRk(0,0);
    quat(2,0) = sRk(1,0);
    quat(3,0) = sRk(2,0);
  }

  return quat;

}
}