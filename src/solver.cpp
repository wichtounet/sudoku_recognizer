//=======================================================================
// Copyright Baptiste Wicht 2013-2014.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <random>

#include "solver.hpp"

namespace {

bool valid_move(const sudoku_grid& grid, std::size_t next_x, std::size_t next_y, std::size_t new_value){
    //1. Ensure that there is not already this value on the row
    for (std::size_t x = 0; x < 9; ++x){
        if(x != next_x && grid(next_y, x).value() == new_value){
            return false;
        }
    }

    //2. Ensure that there is not already this value on the column
    for (std::size_t y = 0; y < 9; ++y){
        if(y != next_y && grid(y, next_x).value() == new_value){
            return false;
        }
    }

    //2. Ensure that there is not already this value in the box
    for(std::size_t x = 0; x < 3; ++x){
        for(std::size_t y = 0; y < 3; ++y){
            std::size_t global_x = (next_x - next_x % 3) + x;
            std::size_t global_y = (next_y - next_y % 3) + y;

            if(!(global_x == next_x && global_y == next_y) && grid(global_y, global_x).value() == new_value){
                return false;
            }
        }
    }

    return true;
}

bool next_empty_cell(const sudoku_grid& grid, std::size_t& next_x, std::size_t& next_y){
    for(std::size_t x = 0; x < 9; ++x){
        for(std::size_t y = 0; y < 9; ++y){
            if(!grid(y, x).value()){
                next_x = x;
                next_y = y;
                return true;
            }
        }
    }

    return false;
}


} //end of anonymous namespace

bool solve(sudoku_grid& grid){
    std::size_t x = 0;
    std::size_t y = 0;

    if(!next_empty_cell(grid, x, y)){
        return true;
    }

    for(std::size_t new_value = 1; new_value < 10; ++new_value){
        if(valid_move(grid, x, y, new_value)){
            grid(y, x).value() = new_value;
            if(solve(grid)){
                return true;
            }
            grid(y, x).value() = 0;
        }
    }

    return false;
}

void solve_random(sudoku_grid& grid){
    static std::random_device rd;
    static std::default_random_engine rand_engine(rd());
    static std::uniform_int_distribution<std::size_t> digit_distribution(1, 9);
    static auto digit_generator = std::bind(digit_distribution, rand_engine);

    for(std::size_t x = 0; x < 9; ++x){
        for(std::size_t y = 0; y < 9; ++y){
            if(!grid(y, x).value()){
                grid(y, x).value() = digit_generator();
            }
        }
    }
}
