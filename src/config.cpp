//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>

#include "config.hpp"

void print_usage(){
    std::cout << "Usage: sudoku [options] <command> file [file...]" << std::endl;
    std::cout << "Supported commands: " << std::endl;
    std::cout << " * detect/detect_save" << std::endl;
    std::cout << " * fill/fill_save" << std::endl;
    std::cout << " * train" << std::endl;
    std::cout << " * recog" << std::endl;
    std::cout << " * recog_binary" << std::endl;
    std::cout << " * time" << std::endl;
    std::cout << "Supported options: " << std::endl;
    std::cout << " -c : Convolutional DBN" << std::endl;
    std::cout << " -m : Mixed mode" << std::endl;
    std::cout << " -s : Take only subsets" << std::endl;
    std::cout << " -q : Quiet mode" << std::endl;
    std::cout << " -o : Oracle mode" << std::endl;
    std::cout << " -r : Shuffle input files" << std::endl;
    std::cout << " -g : Grid search during training" << std::endl;
}

config parse_args(int argc, char** argv){
    config conf;

    for(std::size_t i = 1; i < static_cast<size_t>(argc); ++i){
        conf.args.emplace_back(argv[i]);
    }

    std::size_t i = 0;
    for(; i < conf.args.size(); ++i){
        if(conf.args[i] == "-s"){
            conf.subset = true;
        } else if(conf.args[i] == "-m"){
            conf.mixed = true;
        } else if(conf.args[i] == "-c"){
            conf.conv = true;
        } else if(conf.args[i] == "-q"){
            conf.quiet = true;
        } else if(conf.args[i] == "-t"){
            conf.test = true;
        } else if(conf.args[i] == "-g"){
            conf.grid = true;
        } else if(conf.args[i] == "-o"){
            conf.oracle = true;
        } else if(conf.args[i] == "-r"){
            conf.shuffle = true;
        } else {
            break;
        }
    }

    conf.command = conf.args[i++];

    for(; i < conf.args.size(); ++i){
        conf.files.push_back(conf.args[i]);
    }

    return conf;
}
