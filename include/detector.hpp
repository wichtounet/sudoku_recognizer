//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SUDOKU_SUDOKU_HPP
#define SUDOKU_SUDOKU_HPP

#include<vector>

#include <opencv2/opencv.hpp>

#include "sudoku.hpp"

constexpr const size_t CELL_SIZE = 32;
constexpr const size_t BIG_CELL_SIZE = 48;

typedef std::pair<cv::Point2f, cv::Point2f> line_t;
typedef std::pair<cv::Point2f, cv::Point2f> grid_cell;

void sudoku_binarize(const cv::Mat& source_image, cv::Mat& dest_image);
void cell_binarize(const cv::Mat& gray_image, cv::Mat& dest_image, bool mixed);

std::vector<line_t> detect_lines(const cv::Mat& source_image, cv::Mat& dest_image, bool mixed = false);
std::vector<line_t> detect_lines_binary(const cv::Mat& source_image, cv::Mat& dest_image, bool mixed = false);
std::vector<cv::Rect> detect_grid(const cv::Mat& source_image, cv::Mat& dest_image, std::vector<line_t>& lines, bool mixed = false);
sudoku_grid split(const cv::Mat& source_image, cv::Mat& dest_image, const std::vector<cv::Rect>& cells, std::vector<line_t>& lines, bool mixed = false);

sudoku_grid detect(const cv::Mat& source_image, cv::Mat& dest_image, bool mixed = false);
sudoku_grid detect_binary(const cv::Mat& source_image, cv::Mat& dest_image, bool mixed = false);

void show_regrid(sudoku_grid& grid, int mode);

#endif
