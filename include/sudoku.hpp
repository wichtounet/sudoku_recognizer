//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SUDOKU_DETECTOR_HPP
#define SUDOKU_DETECTOR_HPP

#include<vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "image_utils.hpp"

struct sudoku_cell {
    bool m_empty = true;

    cv::Mat binary_mat;  //Binary final cv::Mat
    cv::Mat gray_mat;    //Gray final cv::Mat
    cv::Mat color_mat;   //RGB final cv::Mat

    cv::Rect bounding;
    cv::Rect digit_bounding;
    uint8_t m_value = 0;
    uint8_t m_correct = 0;      //This is the value coming from the ground truth in [0,9]

    bool empty() const {
        return m_empty;
    }

    uint8_t value() const {
        return m_value;
    }

    uint8_t& value(){
        return m_value;
    }

    uint8_t correct() const {
        return m_correct;
    }

    uint8_t& correct(){
        return m_correct;
    }

    const cv::Mat& mat(const config& conf) const {
        if(conf.gray){
            return gray_mat;
        } else {
            return binary_mat;
        }
    }

    std::vector<double> image(const config& conf) const {
        return mat_to_image(mat(conf), conf.gray);
    }
};

struct sudoku_grid {
    std::vector<sudoku_cell> cells;
    std::string source_image_path;
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

std::ostream& operator<<(std::ostream& os, const sudoku_grid& grid);

#endif
