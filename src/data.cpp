#include "data.hpp"
#include <fstream>

gt_data read_data(const std::string& path){
    gt_data data;

    std::ifstream is(path);

    std::getline(is, data.phone_type);
    std::getline(is, data.image_type);

    for(size_t i = 0; i < 9; ++i){
        for(size_t j = 0; j < 9; ++j){
            is >> data.results[i][j];
        }
    }

    return data;
}
