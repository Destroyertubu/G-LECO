#include <cuda_runtime.h>       // for cudaError_t
#include <cstdio>               // for fprintf in CUDA_CHECK
#include <cstdlib>              // for exit in CUDA_CHECK

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

    // Configuration constants
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024
#define TILE_SIZE 4096          // Default partition size for fixed-length
#define MAX_DELTA_BITS 64      // Max bits for a single delta value
#define MIN_PARTITION_SIZE 128  // Minimum partition size for variable-length partitioning
#define SPLIT_THRESHOLD 0.1     // Split threshold for variable-length partitioning