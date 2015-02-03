//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include <random>

#include <opencv2/opencv.hpp>

#include "fill.hpp"
#include "detector.hpp"
#include "data.hpp"
#include "solver.hpp"

cv::Mat fill_image(const std::string& source, mnist_dataset_t& mnist_dataset, const std::vector<cv::Vec3b>& colors, bool write){
    static std::random_device rd{};
    static std::default_random_engine rand_engine{rd()};

    static std::uniform_int_distribution<std::size_t> digit_distribution(0, mnist_size_1 + mnist_size_2);
    static std::uniform_int_distribution<int> offset_distribution(-3, 3);
    static std::uniform_int_distribution<std::size_t> color_distribution(0, n_colors - 1);

    static auto digit_generator = std::bind(digit_distribution, rand_engine);
    static auto offset_generator = std::bind(offset_distribution, rand_engine);
    static auto color_generator = std::bind(color_distribution, rand_engine);

    std::cout << "Process image " << source << std::endl;

    auto source_image = open_image(source);
    auto original_image = open_image(source, false);

    if (!source_image.data || !original_image.data){
        std::cout << "Invalid source_image" << std::endl;
        return original_image;
    }

    cv::Mat dest_image = original_image.clone();

    //Detect if image was resized

    bool resized = source_image.size() != original_image.size();
    auto w_ratio = static_cast<double>(source_image.size().width) / original_image.size().width;
    auto h_ratio = static_cast<double>(source_image.size().height) / original_image.size().height;

    //Detect the grid/cells

    cv::Mat detect_dest_image;
    auto grid = detect(source_image, detect_dest_image);

    if(!grid.valid()){
        std::cout << "Invalid grid" << std::endl;
    }

    //We use the ground truth to complete/fix the detection pass

    auto data = read_data(source);

    if(!data.valid){
        std::cout << "The ground truth data is not valid" << std::endl;
    }

    for(size_t i = 0; i < 9; ++i){
        for(size_t j = 0; j < 9; ++j){
            auto& cell = grid(j, i);

            cell.value() = data.results[i][j];
            cell.m_empty = !cell.value();
        }
    }

    if(!is_valid(grid)){
        std::cout << "The grid is not valid" << std::endl;
    }

    //Solve the grid (if it fails (bad detection/ground truth), random fill)

    if(!solve(grid)){
        std::cout << "The grid is not solvable" << std::endl;
        solve_random(grid);
    }

    //Update the ground truth

    for(size_t i = 0; i < 9; ++i){
        for(size_t j = 0; j < 9; ++j){
            data.results[i][j] = grid(j, i).value();
        }
    }

    //Pick a random color for the whole sudoku
    const auto& fill_color = colors[color_generator()];

    for(auto& cell : grid.cells){
        if(cell.empty()){
            auto& bounding_rect = cell.bounding;

            //Note to self: This is pretty stupid (theoretical possible infinite loop)
            auto r = digit_generator();
            while(true){
                auto label = r < mnist_size_1 ? mnist_dataset.training_labels[r] : mnist_dataset.test_labels[r - mnist_size_1];

                if(label == cell.value()){
                    break;
                }

                r = digit_generator();
            }

            //Get the digit from MNIST

            auto image = r < mnist_size_1 ? mnist_dataset.training_images[r] : mnist_dataset.test_images[r - mnist_size_1];

            cv::Mat image_mat(28, 28, CV_8U);
            for(std::size_t xx = 0; xx < 28; ++xx){
                for(std::size_t yy = 0; yy < 28; ++yy){
                    image_mat.at<uchar>(cv::Point(xx, yy)) = image[yy * 28 + xx];
                }
            }

            //Center the digit inside the cell (plus some random offsets)

            auto x_start = offset_generator() + bounding_rect.x + (bounding_rect.width - 28) / 2;
            auto y_start = offset_generator() + bounding_rect.y + (bounding_rect.height - 28) / 2;

            //Apply reverse ratio
            if(resized){
                cv::Mat resized;
                cv::resize(image_mat, resized, cv::Size(), 1.0 / w_ratio, 1.0 / h_ratio, CV_INTER_CUBIC);
                image_mat = resized;

                x_start *= (1.0 / w_ratio);
                y_start *= (1.0 / h_ratio);
            }

            //Draw the digit

            for(int xx = 0; xx < image_mat.size().width; ++xx){
                for(int yy = 0; yy < image_mat.size().height; ++yy){
                    auto mnist_color = image_mat.at<uchar>(cv::Point(xx, yy));

                    if(mnist_color > 40){
                        auto& color = dest_image.at<cv::Vec3b>(cv::Point(xx + x_start, yy + y_start));

                        color[0] = fill_color[0];
                        color[1] = fill_color[1];
                        color[2] = fill_color[2];
                    }
                }
            }

            //Apply a light blur on the drawed digit

            cv::Rect mnist_rect(x_start, y_start, image_mat.size().width, image_mat.size().height);
            cv::GaussianBlur(dest_image(mnist_rect), dest_image(mnist_rect), cv::Size(0,0), 1);
        }
    }

    if(write){
        std::string dest(source);
        dest.insert(dest.rfind('.'), ".mixed");
        imwrite(dest.c_str(), dest_image);

        write_data(dest, data);
    }

    return dest_image;
}

