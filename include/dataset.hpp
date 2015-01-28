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
#include "sudoku.hpp"

struct dataset {
    std::vector<std::vector<double>> all_images;
    std::vector<uint8_t> all_labels;

    std::vector<std::vector<double>> training_images;
    std::vector<uint8_t> training_labels;

    std::vector<std::vector<double>> test_images;
    std::vector<uint8_t> test_labels;

    //All the grids
    std::vector<sudoku_grid> source_grids;
};

void preprocess(std::vector<double>& image, const config& conf);

dataset get_dataset(const config& conf);

#endif
