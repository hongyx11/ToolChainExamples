#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define N 1024  // Number of elements

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate GPU memory
    float* d_data;
    checkCuda(cudaMalloc((void**)&d_data, N * sizeof(float)), "Allocating GPU memory");

    // Initialize data on Rank 0
    if (rank == 0) {
        std::vector<float> h_data(N, 42.0f);  // Host buffer
        checkCuda(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice), "Copying data to GPU");
    }

    // MPI Broadcast directly using GPU memory
    std::cout << "Rank " << rank << " performing MPI_Bcast on GPU memory..." << std::endl;
    MPI_Bcast(d_data, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Copy data back to Host for verification
    std::vector<float> h_result(N);
    checkCuda(cudaMemcpy(h_result.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost), "Copying data back to Host");

    // Verify first element
    std::cout << "Rank " << rank << " received first element: " << h_result[0] << std::endl;

    // Cleanup
    cudaFree(d_data);
    MPI_Finalize();
    return 0;
}