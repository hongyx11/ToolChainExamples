cmake_minimum_required(VERSION 3.3)
project(
  CombBLAS
  VERSION 2.0.1
  LANGUAGES C CXX)

# combblas requies c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
include(cmake/CPM.cmake)

# ADD CCCL
CPMAddPackage(
    NAME CCCL
    GITHUB_REPOSITORY NVIDIA/cccl
    GIT_TAG v2.5.0
)
# Add CPM
CPMAddPackage(
    NAME RMM
    GITHUB_REPOSITORY rapidsai/rmm
    GIT_TAG branch-24.10
    SYSTEM Off
)
add_executable(main main.cpp)
target_link_libraries(main PUBLIC rmm::rmm)
