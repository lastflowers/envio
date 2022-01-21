/**
 * @file utils.cpp
 * @author Jae Hyung Jung (lastflowers@snu.ac.kr)
 * @brief util functions
 * @date 2022-01-03
 *
 * @copyright Copyright (c) 2022 Jae Hyung Jung
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

#include "utils.h"
#include <vector>

namespace utils {


Eigen::Isometry3d getSE3matrix(const ros::NodeHandle& n,
                               const std::string& input_str) {
  Eigen::Isometry3d T;

  std::vector<double> tmp_v;
  // Row-major order
  n.getParam(input_str, tmp_v);

  T.linear()(0, 0) = tmp_v[0];
  T.linear()(0, 1) = tmp_v[1];
  T.linear()(0, 2) = tmp_v[2];
  T.linear()(1, 0) = tmp_v[4];
  T.linear()(1, 1) = tmp_v[5];
  T.linear()(1, 2) = tmp_v[6];
  T.linear()(2, 0) = tmp_v[8];
  T.linear()(2, 1) = tmp_v[9];
  T.linear()(2, 2) = tmp_v[10];

  T.translation()(0) = tmp_v[3];
  T.translation()(1) = tmp_v[7];
  T.translation()(2) = tmp_v[11];

  return T;
}


double GetPixelValue(const cv::Mat& img, double x, double y) {
  // This function was copied from code examples by X. Gao in slambook2 (ch8/direct_method.cpp)
  // https://github.com/gaoxiang12/slambook2
  
  // boundary check
  if (x < 0) x = 0;
  if (y < 0) y = 0;
  if (x >= img.cols) x = img.cols - 1;
  if (y >= img.rows) y = img.rows - 1;
  uchar *data = &img.data[int(y) * img.step + int(x)];
  double xx = x - floor(x);
  double yy = y - floor(y);
  return double(
      (1 - xx) * (1 - yy) * data[0] +
      xx * (1 - yy) * data[1] +
      (1 - xx) * yy * data[img.step] +
      xx * yy * data[img.step + 1]);
}

} // namespace utils
