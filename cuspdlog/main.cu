#include <stdio.h>
#include <spdlog/cfg/env.h>  // Enable SPDLOG_FORCE_COLOR
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>  // Required for stdout_color_mt
#include <spdlog/spdlog.h>
#include <spdlog/spdlog.h>
#include <mpi.h>
#include <sstream>
#include <iostream>



std::string get_filename(const std::string &filepath)
{
    size_t pos = filepath.find_last_of("/\\");  // Handles both Windows (`\`) and Unix (`/`)
    return (pos == std::string::npos) ? filepath : filepath.substr(pos + 1);
}

class CombBLASLogger
{
   public:
    CombBLASLogger(const std::string &filename = "")
    {
        setenv("SPDLOG_FORCE_COLOR", "1", 1);
    }
    void loggerinit(int rank, int size, bool log2file, const std::string &filename = "")
    {
        log_to_file_ = log2file;
        filenameprefix_ = filename;
        rank_ = rank;
        size_ = size;
        if (log_to_file_) {
            // Append MPI rank and size to the filename
            std::string final_filename = filenameprefix_ + "_rank" + std::to_string(rank_) +
                                         "_of_" + std::to_string(size_) + ".txt";
            logger_ =
                spdlog::basic_logger_mt("file_logger_" + std::to_string(rank_), final_filename);
        } else {
            logger_ = spdlog::stderr_color_mt("console_logger_" + std::to_string(rank_));
        }

        // Set log pattern with timestamp, rank, file, and line info
        std::string pattern = fmt::format("[%Y-%m-%d %H:%M:%S] [Rank {}/{}] %v", rank, size);
        logger_->set_pattern(pattern);
    }
    // INFO Logs
    void info_rank0(const std::string &msg, const char *file, int line)
    {
        std::string tmp = get_filename(file);
        std::cerr << "debug, I'm rank " << rank_ << std::endl;
        if (rank_ == 0) {
            logger_->info("[{}:{}] {}", tmp, line, msg);
        }
    }

    void info_all(const std::string &msg, const char *file, int line)
    {
        std::string tmp = get_filename(file);
        logger_->info("[{}:{}] {}", tmp, line, msg);
    }

    void info_ordered(const std::string &msg, const char *file, int line)
    {
        std::string tmp = get_filename(file);
        log_ordered(spdlog::level::info, msg, tmp, line);
    }

    // WARNING Logs
    void warn_rank0(const std::string &msg, const char *file, int line)
    {
        if (rank_ == 0) {
            logger_->warn("[{}:{}] {}", file, line, msg);
        }
    }

    void warn_all(const std::string &msg, const char *file, int line)
    {
        logger_->warn("[{}:{}] {}", file, line, msg);
    }

    void warn_ordered(const std::string &msg, const char *file, int line)
    {
        log_ordered(spdlog::level::warn, msg, file, line);
    }

    // ERROR Logs
    void error_rank0(const std::string &msg, const char *file, int line)
    {
        if (rank_ == 0) {
            logger_->error("[{}:{}] {}", file, line, msg);
        }
    }

    void error_all(const std::string &msg, const char *file, int line)
    {
        logger_->error("[{}:{}] {}", file, line, msg);
    }

    void error_ordered(const std::string &msg, const char *file, int line)
    {
        log_ordered(spdlog::level::err, msg, file, line);
    }

    // CRITICAL Logs
    void critical_rank0(const std::string &msg, const char *file, int line)
    {
        if (rank_ == 0) {
            logger_->critical("[{}:{}] {}", file, line, msg);
        }
    }

    void critical_all(const std::string &msg, const char *file, int line)
    {
        logger_->critical("[{}:{}] {}", file, line, msg);
    }

    void critical_ordered(const std::string &msg, const char *file, int line)
    {
        log_ordered(spdlog::level::critical, msg, file, line);
    }

    // DEBUG Logs
    void debug_rank0(const std::string &msg, const char *file, int line)
    {
        if (rank_ == 0) {
            logger_->debug("[{}:{}] {}", file, line, msg);
        }
    }

    void debug_all(const std::string &msg, const char *file, int line)
    {
        logger_->debug("[{}:{}] {}", file, line, msg);
    }

    void debug_ordered(const std::string &msg, const char *file, int line)
    {
        log_ordered(spdlog::level::debug, msg, file, line);
    }

    // Set logging level
    void set_level(spdlog::level::level_enum level)
    {
        logger_->set_level(level);
    }

   private:
    int rank_;
    int size_;
    bool log_to_file_;
    std::string filenameprefix_;
    std::shared_ptr<spdlog::logger> logger_;
    // Logs in order from rank 0 to rank N-1
    void log_ordered(spdlog::level::level_enum lvl, const std::string &msg, const std::string file,
                     int line)
    {
        std::ostringstream oss;
        oss << fmt::format("[{}:{}] {}", file, line, msg);
        std::string message = oss.str();

        for (int r = 0; r < size_; ++r) {
            if (rank_ == r) {
                logger_->log(lvl, "{}", message);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

std::shared_ptr<CombBLASLogger> cblogger;

// Macros for logging
#define INFO_RANK0(logger, msg) logger->info_rank0(msg, __FILE__, __LINE__);
#define INFO_ALL(logger, msg) logger->info_all(msg, __FILE__, __LINE__);
#define INFO_ORDERED(logger, msg) logger->info_ordered(msg, __FILE__, __LINE__);

#define WARN_RANK0(logger, msg) logger->warn_rank0(msg, __FILE__, __LINE__);
#define WARN_ALL(logger, msg) logger->warn_all(msg, __FILE__, __LINE__);
#define WARN_ORDERED(logger, msg) logger->warn_ordered(msg, __FILE__, __LINE__);

#define ERROR_RANK0(logger, msg) logger->error_rank0(msg, __FILE__, __LINE__);
#define ERROR_ALL(logger, msg) logger->error_all(msg, __FILE__, __LINE__);
#define ERROR_ORDERED(logger, msg) logger->error_ordered(msg, __FILE__, __LINE__);

#define CRITICAL_RANK0(logger, msg) logger->critical_rank0(msg, __FILE__, __LINE__);
#define CRITICAL_ALL(logger, msg) logger->critical_all(msg, __FILE__, __LINE__);
#define CRITICAL_ORDERED(logger, msg) logger->critical_ordered(msg, __FILE__, __LINE__);

#define DEBUG_RANK0(logger, msg) logger->debug_rank0(msg, __FILE__, __LINE__);
#define DEBUG_ALL(logger, msg) logger->debug_all(msg, __FILE__, __LINE__);
#define DEBUG_ORDERED(logger, msg) logger->debug_ordered(msg, __FILE__, __LINE__);




// Size of array
#define N 1048576


// Kernel
__global__ void add_vectors(double *a, double *b, double *c)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N) c[id] = a[id] + b[id];
}

// Main program
int main()
{
    MPI_Init(NULL, NULL);
    // Number of bytes to allocate for N doubles
    size_t bytes = N * sizeof(double);
    // cblogger.loggerinit
    cblogger.reset(new CombBLASLogger());
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    cblogger->loggerinit(myrank, nprocs, false);
    INFO_RANK0(cblogger, "hello");
    // INFO_RANK0(cblogger, "hello")
    // Allocate memory for arrays A, B, and C on host
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C = (double *)malloc(bytes);

    // Allocate memory for arrays d_A, d_B, and d_C on device
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Fill host arrays A and B
    for (int i = 0; i < N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
    }

    // Copy data from host arrays A and B to device arrays d_A and d_B
    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    // Set execution configuration parameters
    //		thr_per_blk: number of CUDA threads per grid block
    //		blk_in_grid: number of blocks in grid
    int thr_per_blk = 256;
    int blk_in_grid = ceil(float(N) / thr_per_blk);

    // Launch kernel
    add_vectors<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C);

    // Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    double tolerance = 1.0e-14;
    for (int i = 0; i < N; i++) {
        if (fabs(C[i] - 3.0) > tolerance) {
            printf("\nError: value of C[%d] = %d instead of 3.0\n\n", i, C[i]);
            exit(1);
        }
    }

    // Free CPU memory
    free(A);
    free(B);
    free(C);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\n---------------------------\n");
    printf("__SUCCESS__\n");
    printf("---------------------------\n");
    printf("N                 = %d\n", N);
    printf("Threads Per Block = %d\n", thr_per_blk);
    printf("Blocks In Grid    = %d\n", blk_in_grid);
    printf("---------------------------\n\n");
    MPI_Finalize();
    return 0;
}