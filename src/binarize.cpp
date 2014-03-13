#include <opencv2/opencv.hpp>

#include <iostream>

namespace {

void method_1(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    cv::threshold(gray_image, dest_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
}

void method_2(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    cv::Mat blurred_image;
    cv::GaussianBlur(gray_image, blurred_image, cv::Size(5,5), 0, 0);

    cv::threshold(blurred_image, dest_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
}

void method_3(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    cv::adaptiveThreshold(gray_image, dest_image, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
}

void method_4(const cv::Mat& source_image, cv::Mat& dest_image){
    cv::Mat gray_image;
    cv::cvtColor(source_image, gray_image, CV_RGB2GRAY);

    cv::Mat blurred_image;
    cv::medianBlur(gray_image, blurred_image, 5);

    cv::adaptiveThreshold(blurred_image, dest_image, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 11, 2);
}

}

int main(int argc, char** argv ){
    if(argc != 2){
        std::cout << "Usage: binarize <image>" << std::endl;
        return -1;
    }

    cv::Mat source_image;
    source_image = cv::imread(argv[1], 1);

    if (!source_image.data){
        std::cout << "Invalid source_image" << std::endl;
        return -1;
    }

    cv::Mat dest_image;

    cv::namedWindow("Source", cv::WINDOW_AUTOSIZE);
    cv::imshow("Source", source_image);

    method_1(source_image, dest_image);
    cv::namedWindow("Otsu", cv::WINDOW_AUTOSIZE);
    cv::imshow("Otsu", dest_image);

    method_2(source_image, dest_image);
    cv::namedWindow("Gaussian Otsu", cv::WINDOW_AUTOSIZE);
    cv::imshow("Gaussian Otsu", dest_image);

    method_3(source_image, dest_image);
    cv::namedWindow("Adaptive", cv::WINDOW_AUTOSIZE);
    cv::imshow("Adaptive", dest_image);

    method_4(source_image, dest_image);
    cv::namedWindow("Blurred Adaptive", cv::WINDOW_AUTOSIZE);
    cv::imshow("Blurred Adaptive", dest_image);

    cv::waitKey(0);

    return 0;
}