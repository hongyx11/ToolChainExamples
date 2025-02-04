cmake_minimum_required(VERSION 3.10)
project(MPI_CUDA_Project LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Enable CUDA
find_package(CUDA REQUIRED)

# Find NCCL using NCCL_HOME environment variable
# if(DEFINED ENV{NCCL_HOME})
#     set(NCCL_INCLUDE_DIR "$ENV{NCCL_HOME}/include")
#     set(NCCL_LIBRARY "$ENV{NCCL_HOME}/lib/libnccl.so")
#     message(STATUS "NCCL INCLUDE DIR: ${NCCL_INCLUDE_DIR}")
#     message(STATUS "NCCL LIBRARY: ${NCCL_LIBRARY}")
#     include_directories(${NCCL_INCLUDE_DIR})
#     link_libraries(${NCCL_LIBRARY})
# else()
#     message(FATAL_ERROR "NCCL_HOME environment variable is not set. Please set NCCL_HOME to your NCCL installation path.")
# endif()

# Add the executable
cuda_add_executable(mpi_cuda main.cu)
add_executable(cudaaware cuda-aware.cpp)
# cuda_add_executable(nccltest nccltest.cu)
cuda_add_executable(nocudaaware nocuda-aware.cu)
# Link libraries
target_link_libraries(mpi_cuda ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(cudaaware ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(nocudaaware ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
# target_link_libraries(nccltest ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
# Include CUDA headers
include_directories(${CUDA_INCLUDE_DIRS})

# Ensure MPI is correctly linked
if(MPI_COMPILE_FLAGS)
    set_target_properties(mpi_cuda PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(cudaaware PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    set_target_properties(nocudaaware PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    # set_target_properties(nccltest PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(mpi_cuda PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
    set_target_properties(cudaaware PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
    set_target_properties(nocudaaware PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
    # set_target_properties(nccltest PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()