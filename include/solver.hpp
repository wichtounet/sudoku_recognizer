//=======================================================================
// Copyright Baptiste Wicht 2013-2014.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SUDOKU_SOLVER_HPP
#define SUDOKU_SOLVER_HPP

#include "detector.hpp"

bool solve(sudoku_grid& grid);
bool is_valid(sudoku_grid& grid);
void solve_random(sudoku_grid& grid);

#endif
