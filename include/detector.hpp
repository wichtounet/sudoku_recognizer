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

struct sudoku_cell {
    bool m_empty = true;
    cv::Mat final_mat;
    cv::Rect bounding;

    bool empty(){
        return m_empty;
    }
};

struct sudoku_grid {
    std::vector<sudoku_cell> cells;
    cv::Mat source_image;

    sudoku_cell& operator()(std::size_t x, std::size_t y){
        return cells[y * 9 + x];
    }

    const sudoku_cell& operator()(std::size_t x, std::size_t y) const {
        return cells[y * 9 + x];
    }

    bool valid() const {
        return cells.size() == 9 * 9;
    }
};

std::vector<line_t> detect_lines(const cv::Mat& source_image, cv::Mat& dest_image);
std::vector<line_t> detect_lines_binary(const cv::Mat& source_image, cv::Mat& dest_image);
std::vector<cv::Rect> detect_grid(const cv::Mat& source_image, cv::Mat& dest_image, std::vector<line_t>& lines);
sudoku_grid split(const cv::Mat& source_image, cv::Mat& dest_image, const std::vector<cv::Rect>& cells, std::vector<line_t>& lines);

sudoku_grid detect(const cv::Mat& source_image, cv::Mat& dest_image);
sudoku_grid detect_binary(const cv::Mat& source_image, cv::Mat& dest_image);

#endif
