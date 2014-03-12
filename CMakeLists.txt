cmake_minimum_required(VERSION 2.8)

project(sudoku_ml)

# Activate warnings and C++11 support
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS           "-stdlib=libc++ -std=c++1y -Qunused-arguments")
    set(CMAKE_CXX_FLAGS_DEBUG     "-g -pedantic -Wuninitialized -Wsometimes-uninitialized -Wall -Wextra -Wno-long-long -Winit-self -Wdocumentation")
    set(CMAKE_CXX_FLAGS_RELEASE   "-march=native -g -DNDEBUG -Ofast -fvectorize -fslp-vectorize-aggressive -fomit-frame-pointer")
else()
    MESSAGE(FATAL_ERROR "ERROR: this project can only be compiled clang for now")
endif()

include_directories(include/)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(binarize binarization.cpp)
target_link_libraries(binarize ${OpenCV_LIBS})