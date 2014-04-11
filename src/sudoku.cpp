#include <opencv2/opencv.hpp>

#include <iostream>

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

            cv::Mat source_image;
            source_image = cv::imread(image_source_path.c_str(), 1);

            if (!source_image.data){
                std::cout << "Invalid source_image" << std::endl;
                return -1;
            }

            auto data = read_data(image_source_path);

            cv::Mat dest_image;
            auto cells = detect_grid(source_image, dest_image);
            split(source_image, cells);

            cv::namedWindow("Sudoku Grid", cv::WINDOW_AUTOSIZE);
            cv::imshow("Sudoku Grid", dest_image);

            cv::waitKey(0);
        } else {
            for(size_t i = 1; i < static_cast<size_t>(argc); ++i){
                std::string image_source_path(argv[i]);

                std::cout << image_source_path << std::endl;

                cv::Mat source_image;
                source_image = cv::imread(image_source_path.c_str(), 1);

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
    } else {
        std::cout << "Invalid command \"" << command << "\"" << std::endl;
        return -1;
    }

    return 0;
}