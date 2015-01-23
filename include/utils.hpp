//=======================================================================
// Copyright Baptiste Wicht 2013-2015.
// Distributed under the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#ifndef SUDOKU_UTILS_HPP
#define SUDOKU_UTILS_HPP

template<typename T>
T min(const std::vector<T>& vec){
    return *std::min_element(vec.begin(), vec.end());
}

template<typename T>
T max(const std::vector<T>& vec){
    return *std::max_element(vec.begin(), vec.end());
}

template<typename T>
T mean(const std::vector<T>& vec){
    return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}

template<typename T>
T median(std::vector<T>& vec){
    std::sort(vec.begin(), vec.end());

    if(vec.size() % 2 == 0){
        return vec[vec.size() / 2 + 1];
    } else {
        return (vec[vec.size() / 2] + vec[vec.size() / 2 + 1]) / 2.0;
    }
}


#endif
