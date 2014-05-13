//=======================================================================
// Copyright Baptiste Wicht 2013-2014.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

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
            size_t value;
            is >> value;
            data.results[i][j] = static_cast<uint8_t>(value);
        }
    }

    return data;
}
