//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include "data.hpp"
#include <fstream>

namespace {

gt_data read_metadata(const std::string& data_source_path){
    gt_data data;
    data.valid = true;

    std::ifstream is(data_source_path);

    if(!is.good()){
        data.valid = false;
        return data;
    }

    std::getline(is, data.phone_type);

    if(!is.good()){
        data.valid = false;
        return data;
    }

    std::getline(is, data.image_type);

    if(!is.good()){
        data.valid = false;
        return data;
    }

    for(size_t i = 0; i < 9; ++i){
        for(size_t j = 0; j < 9; ++j){
            size_t value;
            is >> value;
            data.results[i][j] = static_cast<uint8_t>(value);

            if(!is.good()){
                data.valid = false;
                return data;
            }
        }
    }

    return data;
}

} //end of anonymous namespace

gt_data read_data(const std::string& image_source_path){
    std::string data_source_path(image_source_path);
    data_source_path.replace(data_source_path.end() - 3, data_source_path.end(), "dat");

    return read_metadata(data_source_path);
}

gt_data read_data_pure(const std::string& image_source_path){
    std::string image_name(image_source_path.begin() + image_source_path.rfind('/'), image_source_path.end());

    std::string data_source_path = "/home/wichtounet/dev/sudoku_dataset/images/" + image_name;
    data_source_path.replace(data_source_path.end() - 3, data_source_path.end(), "dat");

    return read_metadata(data_source_path);
}

void write_data(const std::string& image_source_path, const gt_data& data){
    std::string data_source_path(image_source_path);
    data_source_path.replace(data_source_path.end() - 3, data_source_path.end(), "dat");

    std::ofstream os(data_source_path);

    os << data.phone_type << "\n";
    os << data.image_type << "\n";

    for(size_t i = 0; i < 9; ++i){
        for(size_t j = 0; j < 9; ++j){
            os << static_cast<std::size_t>(data.results[i][j]);

            if(j < 8){
                os << " ";
            }
        }
        os << "\n";
    }
}
