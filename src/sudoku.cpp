#include <opencv2/opencv.hpp>

#include <iostream>

#include "dbn/dbn.hpp"
#include "dbn/layer.hpp"
#include "dbn/conf.hpp"
#include "dbn/labels.hpp"
#include "dbn/test.hpp"

#include "detector.hpp"
#include "data.hpp"

int main(int argc, char** argv ){
    if(argc < 2){
        std::cout << "Usage: sudoku <command> <options>" << std::endl;
        return -1;
    }

    std::string command(argv[1]);

    if(command == "detect"){
        if(argc < 3){
            std::cout << "Usage: sudoku detect <image>..." << std::endl;
            return -1;
        }

        if(argc == 3){
            std::string image_source_path(argv[2]);

            auto source_image = cv::imread(image_source_path.c_str(), 1);

            if (!source_image.data){
                std::cout << "Invalid source_image" << std::endl;
                return -1;
            }

            cv::Mat dest_image;
            auto cells = detect_grid(source_image, dest_image);
            split(source_image, cells);

            cv::namedWindow("Sudoku Grid", cv::WINDOW_AUTOSIZE);
            cv::imshow("Sudoku Grid", dest_image);

            cv::waitKey(0);
        } else {
            for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
                std::string image_source_path(argv[i]);

                std::cout << image_source_path << std::endl;

                auto source_image = cv::imread(image_source_path.c_str(), 1);

                if (!source_image.data){
                    std::cout << "Invalid source_image" << std::endl;
                    continue;
                }

                cv::Mat dest_image;
                auto cells = detect_grid(source_image, dest_image);
                split(source_image, cells);

                image_source_path.insert(image_source_path.rfind('.'), ".lines");
                imwrite(image_source_path.c_str(), dest_image);
            }
        }
    } else if(command == "train"){
        typedef dbn::dbn<
            dbn::layer<dbn::conf<true, 100, true, true>, 64 * 64, 30>,
            //dbn::layer<dbn::conf<true, 100, false, true>, 300, 300>,
            dbn::layer<dbn::conf<true, 100, false, true>, 30, 30>,
            dbn::layer<dbn::conf<true, 100, false, true, true, dbn::Type::EXP>, 30, 10>> dbn_t;

        auto dbn = std::make_unique<dbn_t>();

        dbn->display();

        std::vector<vector<double>> training_images;
        std::vector<uint8_t> training_labels;

        for(size_t i = 2; i < static_cast<size_t>(argc); ++i){
            std::string image_source_path(argv[i]);

            std::cout << image_source_path << std::endl;

            auto source_image = cv::imread(image_source_path.c_str(), 1);

            if (!source_image.data){
                std::cout << "Invalid source_image" << std::endl;
                continue;
            }

            auto data = read_data(image_source_path);

            for(size_t i = 0; i < 9; ++i){
                for(size_t j = 0; j < 9; ++j){
                    training_labels.push_back(data.results[i][j]);
                }
            }

            cv::Mat dest_image;
            auto cells = detect_grid(source_image, dest_image);
            auto mats = split(source_image, cells);

            for(auto& mat : mats){
                vector<double> image(64 * 64);

                for(size_t i = 0; i < static_cast<size_t>(mat.rows); ++i){
                    for(size_t j = 0; j < static_cast<size_t>(mat.cols); ++j){
                        auto value_c = mat.at<unsigned char>(i, j);

                        double value_d;
                        if(value_c == 255){
                            value_d = 0.0;
                        } else {
                            assert(value_c == 0);
                            value_d = 1.0;
                        }

                        image[i * mat.cols + j] = value_d;
                    }
                }

                training_images.emplace_back(std::move(image));
            }
        }

        auto labels = dbn::make_fake(training_labels);

        std::cout << "Start pretraining" << std::endl;
        dbn->pretrain(training_images, 5);

        std::cout << "Start fine-tuning" << std::endl;
        dbn->fine_tune(training_images, labels, 1, 1000);

        std::ofstream os("dbn.dat", std::ofstream::binary);
        dbn->store(os);

        auto error_rate = dbn::test_set(dbn, training_images, training_labels, dbn::predictor());
        std::cout << "\tError rate (normal): " << 100.0 * error_rate << std::endl;
    } else {
        std::cout << "Invalid command \"" << command << "\"" << std::endl;
        return -1;
    }

    return 0;
}