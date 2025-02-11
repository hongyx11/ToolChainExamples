cmake_minimum_required(VERSION 3.25)
project(MPI_CUDA_Project LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Find MPI
find_package(MPI REQUIRED)
if(CMAKE_CXX_COMPILER MATCHES ".*cray.*" OR CMAKE_C_COMPILER MATCHES ".*cray.*")
    message(STATUS "Detected Cray compiler: ${CMAKE_CXX_COMPILER}")
    execute_process(
        COMMAND ${MPI_C_COMPILER} --cray-print-opts=all
        OUTPUT_VARIABLE MPI_FLAGS
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    string(REGEX MATCHALL "-I([^ ]+)" MPI_INCLUDE_FLAGS "${MPI_FLAGS}")
    list(GET MPI_INCLUDE_FLAGS 0 MPI_INCLUDE_DIR_RAW)
    string(REPLACE "-I" "" MPI_INCLUDE_PATH "${MPI_INCLUDE_DIR_RAW}")

    message(STATUS "Detected MPI include path: ${MPI_INCLUDE_PATH}")
    
    # Set MPI include path globally
    set(MPI_INCLUDE_PATH ${MPI_INCLUDE_PATH} CACHE PATH "MPI include path")
endif()

# Check if MPI_INCLUDE_PATH is defined
if(DEFINED MPI_INCLUDE_PATH)
    # Output the value of MPI_INCLUDE_PATH
    message(STATUS "MPI_INCLUDE_PATH is set to: ${MPI_INCLUDE_PATH}")
else()
    # Output a message if MPI_INCLUDE_PATH is not defined
    message(STATUS "MPI_INCLUDE_PATH is not set.")
endif()
# Find CUDAToolkit
find_package(CUDAToolkit REQUIRED)
add_executable(mpi_cuda main.cu)
target_include_directories(mpi_cuda PUBLIC ${MPI_INCLUDE_PATH})
target_link_libraries(mpi_cuda MPI::MPI_CXX CUDA::cudart)
# important!! otherwise cray system will use g++ as liner exec ... 
set_target_properties(mpi_cuda PROPERTIES LINKER_LANGUAGE CXX)

# # Add the executable

# add_executable(cudaaware cuda-aware.cpp)
# # cuda_add_executable(nccltest nccltest.cu)
# cuda_add_executable(nocudaaware nocuda-aware.cu)
# # Link libraries
# target_link_libraries(mpi_cuda ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
# target_link_libraries(cudaaware ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
# target_link_libraries(nocudaaware ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
# # target_link_libraries(nccltest ${MPI_LIBRARIES} ${CUDA_LIBRARIES})
# # Include CUDA headers
# include_directories(${CUDA_INCLUDE_DIRS})

# # Ensure MPI is correctly linked
# if(MPI_COMPILE_FLAGS)
#     set_target_properties(mpi_cuda PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#     set_target_properties(cudaaware PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#     set_target_properties(nocudaaware PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#     # set_target_properties(nccltest PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
# endif()

# if(MPI_LINK_FLAGS)
#     set_target_properties(mpi_cuda PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
#     set_target_properties(cudaaware PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
#     set_target_properties(nocudaaware PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
#     # set_target_properties(nccltest PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
# endif()
