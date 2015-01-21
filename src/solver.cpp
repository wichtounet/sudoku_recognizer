//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
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

bool is_valid(sudoku_grid& grid){
    //Check all columns
    for(std::size_t x = 0; x < 9; ++x){
        for(std::size_t y = 0; y < 9; ++y){
            //Don't test zero
            if(grid(y, x).value()){
                for(std::size_t yy = 0; yy < 9; ++yy){
                    if(y != yy && grid(yy, x).value() == grid(y, x).value()){
                        return false;
                    }
                }
            }
        }
    }

    //Check all rows
    for(std::size_t y = 0; y < 9; ++y){
        for(std::size_t x = 0; x < 9; ++x){
            //Don't test zero
            if(grid(y, x).value()){
                for(std::size_t xx = 0; xx < 9; ++xx){
                    if(x != xx && grid(y, xx).value() == grid(y, x).value()){
                        return false;
                    }
                }
            }
        }
    }

    //Check all squares
    for(std::size_t x = 0; x < 3; ++x){
        for(std::size_t y = 0; y < 3; ++y){
            auto start_x = x * 3;
            auto start_y = y * 3;

            for(std::size_t xx = 0; xx < 3; ++xx){
                for(std::size_t yy = 0; yy < 3; ++yy){
                    if(grid(start_y + yy, start_x + xx).value()){
                        for(std::size_t xxx = 0; xxx < 3; ++xxx){
                            for(std::size_t yyy = 0; yyy < 3; ++yyy){
                                if(!(xxx == xx && yyy == yy) && grid(start_y + yy, start_x + xx).value() == grid(start_y + yyy, start_x + xxx).value()){
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return true;
}

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
