#ifndef SUDOKU_DETECTOR_HPP
#define SUDOKU_DETECTOR_HPP

#include<vector>

#include <opencv2/opencv.hpp>

constexpr const size_t CELL_SIZE = 32;

typedef std::pair<cv::Point2f, cv::Point2f> line_t;
typedef std::pair<cv::Point2f, cv::Point2f> grid_cell;

std::vector<line_t> detect_lines(const cv::Mat& source_image, cv::Mat& dest_image);

std::vector<cv::Mat> detect(const cv::Mat& source_image, cv::Mat& dest_image);

#endif
