#ifndef SUDOKU_DATA_HPP
#define SUDOKU_DATA_HPP

#include<string>

struct gt_data {
    std::string phone_type;
    std::string image_type;
    uint8_t results[9][9];
};

gt_data read_data(const std::string& path);

#endif
