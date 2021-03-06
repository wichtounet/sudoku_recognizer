//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "image_utils.hpp"
#include "detector.hpp" //For CELL_SIZE

float fill_factor(const cv::Mat& mat){
    auto non_zero = cv::countNonZero(mat);
    auto area = mat.cols * mat.rows;
    return (static_cast<float>(non_zero) / area);
}

std::vector<double> mat_to_image(const cv::Mat& mat, bool gray){
    std::vector<double> image(mat.rows * mat.cols);

    for(std::size_t i = 0; i < static_cast<std::size_t>(mat.rows); ++i){
        for(std::size_t j = 0; j < static_cast<std::size_t>(mat.cols); ++j){
            auto value_c = static_cast<std::size_t>(mat.at<uint8_t>(i, j));

            if(gray){
                image[i * mat.cols + j] = static_cast<double>(value_c);
            } else {
                assert(value_c == 0 || value_c == 255);

                image[i * mat.cols + j] = value_c == 0 ? 1.0 : 0.0;
            }
        }
    }

    return image;
}

cv::Mat open_image(const std::string& path, bool resize){
    auto source_image = cv::imread(path.c_str(), 1);

    if (!source_image.data){
        return source_image;
    }

    if(resize && (source_image.rows > 800 || source_image.cols > 800)){
        auto factor = 800.0f / std::max(source_image.rows, source_image.cols);

        cv::Mat resized_image;

        cv::resize(source_image, resized_image, cv::Size(), factor, factor, cv::INTER_AREA);

        return resized_image;
    }

    return source_image;
}
