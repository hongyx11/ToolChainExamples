#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define N 1024  // Number of elements

__global__ void initializeData(int* d_data, int value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        d_data[idx] = value;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate GPU memory
    int* d_data;
    cudaMalloc((void**)&d_data, N * sizeof(int));

    // Initialize data on GPU (Rank 0 sets to 100, Rank 1 sets to 200)
    int init_value = (rank == 0) ? 100 : 200;
    initializeData<<<(N + 255) / 256, 256>>>(d_data, init_value);
    cudaDeviceSynchronize();

    // Allocate host memory and copy data back from GPU
    std::vector<int> h_data(N);
    cudaMemcpy(h_data.data(), d_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    // MPI Communication: Send data from Rank 0 → Rank 1
    if (rank == 0) {
        std::cout << "Rank 0 sending data to Rank 1..." << std::endl;
        MPI_Send(h_data.data(), N, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } 
    else if (rank == 1) {
        std::vector<int> received_data(N);
        MPI_Recv(received_data.data(), N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Rank 1 received data. First element: " << received_data[0] << std::endl;
    }

    // Free GPU memory
    cudaFree(d_data);

    MPI_Finalize();
    return 0;
}