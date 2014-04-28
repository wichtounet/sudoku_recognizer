#ifndef SUDOKU_DETECTOR_HPP
#define SUDOKU_DETECTOR_HPP

#include<vector>

#include <opencv2/opencv.hpp>

constexpr const size_t CELL_SIZE = 32;

std::vector<cv::Mat> detect(const cv::Mat& source_image, cv::Mat& dest_image);

#endif
