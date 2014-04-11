#include "data.hpp"
#include <fstream>

gt_data read_data(const std::string& image_source_path){
    std::string data_source_path(image_source_path);
    data_source_path.replace(data_source_path.end() - 3, data_source_path.end(), "dat");

    gt_data data;

    std::ifstream is(data_source_path);

    std::getline(is, data.phone_type);
    std::getline(is, data.image_type);

    for(size_t i = 0; i < 9; ++i){
        for(size_t j = 0; j < 9; ++j){
            is >> data.results[i][j];
        }
    }

    return data;
}
