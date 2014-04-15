#ifndef SUDOKU_DETECTOR_HPP
#define SUDOKU_DETECTOR_HPP

#include<vector>

#include <opencv2/opencv.hpp>

constexpr const size_t CELL_SIZE = 64;

std::vector<cv::RotatedRect> detect_grid(const cv::Mat& source_image, cv::Mat& dest_image);
std::vector<cv::Mat> split(const cv::Mat& source_image, const std::vector<cv::RotatedRect>& cells);

#endif
