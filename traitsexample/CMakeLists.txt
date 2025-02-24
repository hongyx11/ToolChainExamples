cmake_minimum_required(VERSION 3.26)
project(traits VERSION 2.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

add_executable(main traitsexample.cpp traitsexample.hpp another.cpp another.hpp main.cpp)
