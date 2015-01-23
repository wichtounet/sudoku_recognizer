//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SUDOKU_CONFIG_HPP
#define SUDOKU_CONFIG_HPP

#include <vector>
#include <string>

struct config {
    std::vector<std::string> args;
    std::vector<std::string> files;
    std::string command;
    bool subset = false;
    bool mixed = false;
    bool quiet = false;
    bool test = false;
};

void print_usage();

config parse_args(int argc, char** argv);

#endif
