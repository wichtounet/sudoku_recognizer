//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "cpp_utils/data.hpp"

#include "dataset.hpp"
#include "detector.hpp"
#include "image_utils.hpp"

//Real constants used to divide the dataset if necessary
constexpr const std::size_t test_divide = 5;
constexpr const std::size_t subset_divide = 10;

void preprocess(std::vector<double>& image, const config& conf){
    if(conf.gray){
        for(auto& pixel : image){
            pixel = 255 - pixel;
        }

        cpp::normalize(image);
    }
}

dataset get_dataset(const config& conf){
    if(conf.oracle){
        std::cout << "Start loading oracle dataset..." << std::endl;
    } else {
        std::cout << "Start loading dataset..." << std::endl;
    }

    dataset ds;

    for(auto& image_source_path : conf.files){
        if(!conf.quiet){
            std::cout << "Load and detect "<< image_source_path << std::endl;
        }

        auto source_image = open_image(image_source_path);

        if (!source_image.data){
            std::cout << "Invalid source_image" << std::endl;
            continue;
        }

        cv::Mat source;
        if(conf.oracle){
            sudoku_binarize(source_image, source);
        }

        //Read metadata

        auto data = read_data(image_source_path);
        auto pure_data = read_data_pure(image_source_path);

        cv::Mat dest_image;
        auto grid = detect(source_image, dest_image, conf.mixed);

        grid.source_image_path = image_source_path;

        for(size_t i = 0; i < 9; ++i){
            for(size_t j = 0; j < 9; ++j){
                auto& cell = grid(j, i);

                //This is a handwritten digit
                if(conf.oracle && !pure_data.results[i][j]){
                    const auto& bounding_rect = cell.bounding;

                    static_assert(CELL_SIZE == 32, "Oracle algorith is out of sync with CELL_SIZE");

                    auto width = 28 + 4;
                    auto height = 28 + 4;
                    auto x_start = -2 + bounding_rect.x + (bounding_rect.width - 28) / 2;
                    auto y_start = -2 + bounding_rect.y + (bounding_rect.height - 28) / 2;

                    cell.digit_bounding.x = x_start;
                    cell.digit_bounding.y = y_start;
                    cell.digit_bounding.width = width;
                    cell.digit_bounding.height = height;

                    cell.color_mat = cv::Mat(source_image, cell.digit_bounding);
                    cv::cvtColor(cell.color_mat, cell.gray_mat, CV_RGB2GRAY);

                    cell_binarize(cell.gray_mat, cell.binary_mat, conf.mixed);
                }

                cell.correct() = data.results[i][j];

                if(data.results[i][j]){
                    ds.all_labels.push_back(data.results[i][j]-1);
                    ds.all_images.emplace_back(cell.image(conf));
                }
            }
        }

        ds.source_grids.push_back(grid);
    }

    for(auto& image : ds.all_images){
        preprocess(image, conf);
    }

    if(conf.subset){
        std::size_t count = 0;
        auto filter_lambda = [&count](auto&){ return count++ % subset_divide > 0; };
        ds.all_labels.erase(std::remove_if(ds.all_labels.begin(), ds.all_labels.end(), filter_lambda), ds.all_labels.end());
        count = 0;
        ds.all_images.erase(std::remove_if(ds.all_images.begin(), ds.all_images.end(), filter_lambda), ds.all_images.end());
    }

    if(conf.test){
        for(std::size_t i = 0; i < ds.all_images.size(); ++i){
            if(i % test_divide == 0){
                ds.test_labels.push_back(ds.all_labels[i]);
                ds.test_images.push_back(ds.all_images[i]);
            } else {
                ds.training_labels.push_back(ds.all_labels[i]);
                ds.training_images.push_back(ds.all_images[i]);
            }
        }
    } else {
        ds.training_labels = ds.all_labels;
        ds.training_images = ds.all_images;
    }

    assert(ds.all_images.size() == ds.all_labels.size());
    assert(ds.training_images.size() == ds.training_labels.size());
    assert(ds.test_images.size() == ds.test_labels.size());

    std::cout << "...dataset loaded" << std::endl;

    return ds;
}
