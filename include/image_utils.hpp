//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef IMAGE_UTILS_HPP
#define IMAGE_UTILS_HPP

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

float fill_factor(const cv::Mat& mat);

std::vector<double> mat_to_image(const cv::Mat& mat, bool gray = true);

cv::Mat open_image(const std::string& path, bool resize = true);

#endif
