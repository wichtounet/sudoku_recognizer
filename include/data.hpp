//=======================================================================
// Copyright Baptiste Wicht 2013-2014.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SUDOKU_DATA_HPP
#define SUDOKU_DATA_HPP

#include<string>

struct gt_data {
    std::string phone_type;
    std::string image_type;
    uint8_t results[9][9];
    bool valid;
};

gt_data read_data(const std::string& path);
void write_data(const std::string& path, const gt_data& data);

#endif
