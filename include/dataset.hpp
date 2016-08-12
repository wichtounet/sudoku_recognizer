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

    std::vector<etl::dyn_matrix<double, 1>> etl_all_images_1d;
    std::vector<etl::dyn_matrix<double, 1>> etl_training_images_1d;
    std::vector<etl::dyn_matrix<double, 1>> etl_test_images_1d;

    //All the grids
    std::vector<sudoku_grid> source_grids;

    std::vector<etl::dyn_matrix<double, 1>>& training_images_1d(){
        if(etl_training_images_1d.empty()){
            etl_training_images_1d.reserve(training_images.size());
            for(auto& image : training_images){
                etl_training_images_1d.emplace_back(image);
            }
        }

        return etl_training_images_1d;
    }

    std::vector<etl::dyn_matrix<double, 1>>& test_images_1d(){
        if(etl_test_images_1d.empty()){
            etl_test_images_1d.reserve(test_images.size());
            for(auto& image : test_images){
                etl_test_images_1d.emplace_back(image);
            }
        }

        return etl_test_images_1d;
    }

    std::vector<etl::dyn_matrix<double, 1>>& all_images_1d(){
        if(etl_all_images_1d.empty()){
            etl_all_images_1d.reserve(all_images.size());
            for(auto& image : all_images){
                etl_all_images_1d.emplace_back(image);
            }
        }

        return etl_all_images_1d;
    }
};

void preprocess(std::vector<double>& image, const config& conf);

dataset get_dataset(const config& conf);

#endif
