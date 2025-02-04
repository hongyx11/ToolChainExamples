cmake_minimum_required(VERSION 3.3)
project(CombBLAS VERSION 2.0.1 LANGUAGES C CXX)
# require c++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)
# set(CMAKE_CXX_EXTENSIONS OFF)
# set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA)

if(CUDA_FOUND)
    #target_compile_definitions(CombBLAS PUBLIC GPU_ENABLED)
    enable_language(CUDA)
    #set(CUDA_HOST_COMPILER "nvc++") # NVHPC requires this...sorry if this causes any issues for anyone
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --ftemplate-backtrace-limit 1 --expt-relaxed-constexpr --disable-warnings")
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DUSE_CUDA")
    if(ENABLE_ASAN)
    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DUSE_CUDA -DENABLE_ASAN")
    endif()
    # target_compile_definitions(CombBLAS PUBLIC -DUSE_CUDA)
    set(THREADS_PREFER_PTHREAD_FLAG OFF)
    set_property(TARGET Threads::Threads
    PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -pthread>
    "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>")
else()
  message(STATUS "CUDA features disabled")
endif()

include(cmake/CPM.cmake)
# CPMAddPackage("gh:fmtlib/fmt#11.0.1")
CPMAddPackage("gh:gabime/spdlog#v1.15.1")

find_package(spdlog REQUIRED)

cuda_add_executable(main main.cu)
target_link_libraries(main spdlog::spdlog)
