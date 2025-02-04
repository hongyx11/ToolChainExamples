#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define N 1024  // Number of elements

void checkNccl(ncclResult_t result, const char* message) {
    if (result != ncclSuccess) {
        std::cerr << "NCCL Error: " << message << " : " << ncclGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Assign a GPU to each rank
    int device = mpi_rank % 2;  // Assign GPUs in a round-robin way
    cudaSetDevice(device);

    // Initialize NCCL
    ncclComm_t comm;
    ncclUniqueId id;

    if (mpi_rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    checkNccl(ncclCommInitRank(&comm, mpi_size, id, mpi_rank), "Initializing NCCL communicator");

    // Allocate GPU memory
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // Initialize data (only Rank 0 sets values)
    if (mpi_rank == 0) {
        std::vector<float> h_data(N, 42.0f);  // Initialize with 42.0
        cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Perform NCCL Broadcast (Rank 0 → all ranks)
    checkNccl(ncclBroadcast(d_data, d_data, N, ncclFloat, 0, comm, cudaStreamDefault),"ncclbcast");

    // Copy back to host and verify results
    std::vector<float> h_data(N);
    cudaMemcpy(h_data.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Rank " << mpi_rank << " NCCL Broadcast result (first element): " << h_data[0] << std::endl;

    // Cleanup
    // cudaFree(d_data);
    // ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}