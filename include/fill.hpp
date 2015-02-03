//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SUDOKU_FILL_HPP
#define SUDOKU_FILL_HPP

#include <vector>

#include "mnist/mnist_reader.hpp"

//These constants need to be sync when changing MNIST dataset and the fill colors
constexpr const std::size_t mnist_size_1 = 60000;
constexpr const std::size_t mnist_size_2 = 10000;
constexpr const std::size_t n_colors = 6;

using mnist_dataset_t = decltype(mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>());

cv::Mat fill_image(const std::string& source, mnist_dataset_t& mnist_dataset, const std::vector<cv::Vec3b>& colors, bool write);

#endif
