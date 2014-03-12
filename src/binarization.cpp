#include <opencv2/opencv.hpp>

#include <iostream>

int main(int argc, char** argv ){
    if(argc != 2){
        std::cout << "Usage: binarize <image>" << std::endl;
        return -1;
    }

    cv::Mat image;
    image = cv::imread(argv[1], 1);

    if (!image.data){
        std::cout << "Invalid image" << std::endl;
        return -1;
    }

    cv::namedWindow("Binarized Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Binarized Image", image);

    cv::waitKey(0);

    return 0;
}