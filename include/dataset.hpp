//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SUDOKU_DATASET_HPP
#define SUDOKU_DATASET_HPP

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include "data.hpp"
#include "config.hpp"

struct dataset {
    std::vector<std::vector<double>> all_images;
    std::vector<uint8_t> all_labels;

    std::vector<std::vector<double>> training_images;
    std::vector<uint8_t> training_labels;

    std::vector<std::vector<double>> test_images;
    std::vector<uint8_t> test_labels;

    std::vector<std::string> source_files;
    std::vector<std::vector<cv::Mat>> source_images;
    std::vector<gt_data> source_data;
};

dataset get_dataset(const config& conf, bool gray = false);

#endif
