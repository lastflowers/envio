/**
 * @file Attitude.h
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

#include <Eigen/Eigen>
#include "define.h"

#ifndef ATTITUDE_H
#define ATTITUDE_H

using namespace Eigen;

namespace nesl {

    M3d euler2dcm(V3d &euler);
    V4d euler2quat(V3d &euler);

    M3d quat2dcm(V4d &quat);
    V3d quat2euler(V4d &quat);
    V3d quat2rvec(V4d &quat);
    M4d quatLeftComp(V4d &quat);
    M4d quatRightComp(V4d &quat);
    V4d quatInverse(V4d &quat);
    V4d quatNormalize(V4d &quat);
    V4d quatMultiply(V4d &q, V4d &p);

    V3d dcm2euler(M3d &dcm);
    V4d dcm2quat(M3d &dcm);
    V3d dcm2rvec(M3d &dcm);

    M3d Vec2SkewMat(V3d &vec);
    M3d rvec2dcm(V3d &vec);
    V4d rvec2quat(V3d &rvec);
}

#endif // ATTITUDE_H
