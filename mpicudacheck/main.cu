#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <unistd.h> // For getting hostname

// CUDA Kernel
__global__ void simpleKernel() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from CUDA thread %d \n", idx);
}

void checkCudaDevices() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    std::cout << "Number of CUDA devices available: " << deviceCount << std::endl;
}

void launchSimpleKernel() {
    simpleKernel<<<1, 10>>>();
    cudaDeviceSynchronize();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    std::cout << "MPI Rank: " << rank << " of " << size << ", Hostname: " << hostname << std::endl;

    if (rank == 0) {
        std::cout << "Rank 0 checking CUDA devices..." << std::endl;
        checkCudaDevices();

        std::cout << "Launching a simple CUDA kernel..." << std::endl;
        launchSimpleKernel();
    }

    MPI_Finalize();
    return 0;
}