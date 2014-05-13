//=======================================================================
// Copyright Baptiste Wicht 2013-2014.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SUDOKU_DETECTOR_HPP
#define SUDOKU_DETECTOR_HPP

#include<vector>

#include <opencv2/opencv.hpp>

constexpr const size_t CELL_SIZE = 32;

typedef std::pair<cv::Point2f, cv::Point2f> line_t;
typedef std::pair<cv::Point2f, cv::Point2f> grid_cell;

std::vector<line_t> detect_lines(const cv::Mat& source_image, cv::Mat& dest_image);
std::vector<cv::Rect> detect_grid(const cv::Mat& source_image, cv::Mat& dest_image, std::vector<line_t>& lines);
std::vector<cv::Mat> split(const cv::Mat& source_image, cv::Mat& dest_image, const std::vector<cv::Rect>& cells, std::vector<line_t>& lines);

std::vector<cv::Mat> detect(const cv::Mat& source_image, cv::Mat& dest_image);

#endif
