// gpu_compression_framework.cu
// Complete GPU Compression Framework with full 64-bit support and GPU serialization/deserialization
// Compile command: nvcc -O3 -arch=sm_70 -std=c++14 gpu_compression_framework.cu -o gpu_compression

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iostream>
#include <random>
#include <numeric>
#include <functional>
#include <iomanip>
#include <cstring>
#include <memory>
#include <cmath>
#include <limits>
#include <type_traits>
#include <set>

// =============================================================================
//                       General Configuration Parameters
// =============================================================================
#define BLOCK_SIZE 128      // Number of integers per logical data block
#define MINIBLOCK_SIZE 32   // Number of integers per miniblock
#define MINIBLOCKS_PER_BLOCK (BLOCK_SIZE / MINIBLOCK_SIZE) // Number of miniblocks per block (4)
#define THREADS_PER_BLOCK 128 // Number of threads per CUDA thread block
#define WARP_SIZE 32        // CUDA warp size

// Random access test parameters
#define NUM_RANDOM_QUERIES 100000 // Number of random queries to perform
#define CACHE_SIZE 64           // Number of cache blocks (unused)

// Special bitwidth flag - now we need to support up to 64 bits
#define RLE_BITWIDTH_FLAG 255  // Indicates this miniblock is RLE encoded
                              // Max actual bitwidth can be 64 for 64-bit types

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// =============================================================================
//                           Data Type Definitions
// =============================================================================

enum DataTypeId {
    TYPE_INT = 0, TYPE_UINT = 1, TYPE_LONG = 2, TYPE_ULONG = 3,
    TYPE_LONGLONG = 4, TYPE_ULONGLONG = 5, TYPE_UNKNOWN = 255
};

enum CompressionAlgorithm {
    ALGO_FOR = 0, ALGO_DFOR = 1, ALGO_RFOR = 2, ALGO_NONE = 3
};

template<typename T>
struct BlockMetadata {
    T reference; 
    uint32_t bitwidth_word;  // For 64-bit support, we'll need to handle this differently
    uint32_t compressed_offset; 
    uint32_t compressed_size_bytes;
    uint32_t original_offset;
    uint32_t element_count;  // Add actual element count in this block
};

template<typename T>
struct DFORMetadata { 
    T first_value; 
    T reference_delta; 
};

struct RFORMetadata {
    uint32_t rle_miniblock_count; 
    uint32_t for_packed_data_offset_bytes;
    uint32_t num_rle_values_in_block;
};

struct CompressionStats {
    size_t original_size_bytes; 
    size_t compressed_total_size_bytes;
    double compression_ratio; 
    double compression_time_ms;
    double decompression_time_ms; 
    int blocks_processed;
};

struct SerializedDataHeader {
    uint32_t magic; 
    uint32_t version; 
    uint32_t algorithm_type; 
    uint32_t data_type_id;
    uint64_t total_original_values; 
    uint32_t num_blocks;
    uint32_t block_metadata_entry_size; 
    uint64_t block_metadata_offset;
    uint64_t block_metadata_size_bytes; 
    uint32_t algo_specific_metadata_entry_size;
    uint64_t algo_specific_metadata_offset; 
    uint64_t algo_specific_metadata_size_bytes;
    uint64_t compressed_data_stream_offset; 
    uint64_t compressed_data_stream_size_bytes;
    uint32_t reserved[4];
};
const uint32_t GPCC_MAGIC = 0x43435047; 
const uint32_t GPCC_VERSION = 3;  // Incremented version for 64-bit support

// =============================================================================
//                       Device-side Helper Functions
// =============================================================================

// Enhanced bit extraction function that supports up to 64 bits
__device__ __forceinline__ uint64_t extract_bits_from_byte_stream_64(
    const uint8_t* data, uint64_t bit_offset, uint32_t bit_width
) {
    if (bit_width == 0) return 0;
    if (bit_width > 64) bit_width = 64;  // Clamp to 64 bits
    
    uint64_t byte_idx = bit_offset >> 3;
    uint32_t bit_in_first_byte = bit_offset & 7;
    
    // For 64-bit values, we might need up to 9 bytes
    uint64_t temp_val = 0;
    int bytes_needed = (bit_in_first_byte + bit_width + 7) / 8;
    
    // Read up to 9 bytes to handle worst case of 64 bits starting at bit offset 7
    for (int i = 0; i < bytes_needed && i < 9; i++) {
        temp_val |= ((uint64_t)data[byte_idx + i] << (i * 8));
    }
    
    temp_val >>= bit_in_first_byte;
    
    // Create mask for the required number of bits
    uint64_t mask = (bit_width == 64) ? ~0ULL : ((1ULL << bit_width) - 1);
    return temp_val & mask;
}

// Legacy 32-bit version for backward compatibility
__device__ __forceinline__ uint32_t extract_bits_from_byte_stream(
    const uint8_t* data, uint32_t bit_offset, uint32_t bit_width
) {
    return (uint32_t)extract_bits_from_byte_stream_64(data, bit_offset, bit_width);
}

// =============================================================================
//                       GPU Serialization Implementation
// =============================================================================

// Helper kernel to calculate offsets for serialization
__global__ void calculate_serialization_offsets_kernel(
    SerializedDataHeader* header,
    uint32_t num_blocks,
    uint32_t block_metadata_entry_size,
    uint32_t algo_specific_metadata_entry_size,
    uint64_t compressed_data_stream_size_bytes,
    CompressionAlgorithm algo
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        header->block_metadata_size_bytes = num_blocks * block_metadata_entry_size;
        
        if (algo == ALGO_DFOR || algo == ALGO_RFOR) {
            header->algo_specific_metadata_size_bytes = num_blocks * algo_specific_metadata_entry_size;
        } else {
            header->algo_specific_metadata_size_bytes = 0;
        }
        
        header->compressed_data_stream_size_bytes = compressed_data_stream_size_bytes;
        
        size_t current_offset = sizeof(SerializedDataHeader);
        header->block_metadata_offset = current_offset;
        current_offset += header->block_metadata_size_bytes;
        header->algo_specific_metadata_offset = current_offset;
        current_offset += header->algo_specific_metadata_size_bytes;
        header->compressed_data_stream_offset = current_offset;
    }
}

// Main GPU serialization kernel
template<typename T>
__global__ void serialize_data_gpu_kernel(
    uint8_t* serialized_buffer,
    const SerializedDataHeader* header,
    const uint8_t* compressed_data_stream,
    const BlockMetadata<T>* block_metadata,
    const DFORMetadata<T>* dfor_metadata,
    const RFORMetadata* rfor_metadata,
    CompressionAlgorithm algo
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Copy header (only thread 0)
    if (tid == 0) {
        memcpy(serialized_buffer, header, sizeof(SerializedDataHeader));
    }
    
    // Copy block metadata in parallel
    if (tid < header->num_blocks) {
        memcpy(serialized_buffer + header->block_metadata_offset + tid * sizeof(BlockMetadata<T>),
               &block_metadata[tid], sizeof(BlockMetadata<T>));
    }
    
    // Copy algorithm-specific metadata in parallel
    if (algo == ALGO_DFOR && tid < header->num_blocks && dfor_metadata != nullptr) {
        memcpy(serialized_buffer + header->algo_specific_metadata_offset + tid * sizeof(DFORMetadata<T>),
               &dfor_metadata[tid], sizeof(DFORMetadata<T>));
    } else if (algo == ALGO_RFOR && tid < header->num_blocks && rfor_metadata != nullptr) {
        memcpy(serialized_buffer + header->algo_specific_metadata_offset + tid * sizeof(RFORMetadata),
               &rfor_metadata[tid], sizeof(RFORMetadata));
    }
    
    // Copy compressed data stream (coalesced memory access)
    uint64_t compressed_size = header->compressed_data_stream_size_bytes;
    uint64_t total_threads = gridDim.x * blockDim.x;
    uint64_t chunk_size = (compressed_size + total_threads - 1) / total_threads;
    uint64_t start_idx = tid * chunk_size;
    uint64_t end_idx = min(start_idx + chunk_size, compressed_size);
    
    if (start_idx < compressed_size) {
        for (uint64_t i = start_idx; i < end_idx; i++) {
            serialized_buffer[header->compressed_data_stream_offset + i] = compressed_data_stream[i];
        }
    }
}

// GPU serialization wrapper function
template<typename T>
std::vector<uint8_t> serialize_data_gpu(
    CompressionAlgorithm algo, DataTypeId dtype_id, uint64_t total_original_values,
    const std::vector<uint8_t>& h_compressed_byte_stream,
    const std::vector<BlockMetadata<T>>& h_block_metadata,
    const std::vector<DFORMetadata<T>>& h_dfor_metadata,
    const std::vector<RFORMetadata>& h_rfor_metadata
) {
    // Prepare header on host
    SerializedDataHeader h_header = {};
    h_header.magic = GPCC_MAGIC;
    h_header.version = GPCC_VERSION;
    h_header.algorithm_type = static_cast<uint32_t>(algo);
    h_header.data_type_id = static_cast<uint32_t>(dtype_id);
    h_header.total_original_values = total_original_values;
    h_header.num_blocks = h_block_metadata.size();
    h_header.block_metadata_entry_size = sizeof(BlockMetadata<T>);
    
    if (algo == ALGO_DFOR) {
        h_header.algo_specific_metadata_entry_size = sizeof(DFORMetadata<T>);
    } else if (algo == ALGO_RFOR) {
        h_header.algo_specific_metadata_entry_size = sizeof(RFORMetadata);
    }
    
    // Allocate device memory
    SerializedDataHeader* d_header;
    uint8_t *d_compressed_stream = nullptr;
    BlockMetadata<T> *d_block_metadata = nullptr;
    DFORMetadata<T> *d_dfor_metadata = nullptr;
    RFORMetadata *d_rfor_metadata = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_header, sizeof(SerializedDataHeader)));
    CUDA_CHECK(cudaMemcpy(d_header, &h_header, sizeof(SerializedDataHeader), cudaMemcpyHostToDevice));
    
    if (!h_compressed_byte_stream.empty()) {
        CUDA_CHECK(cudaMalloc(&d_compressed_stream, h_compressed_byte_stream.size()));
        CUDA_CHECK(cudaMemcpy(d_compressed_stream, h_compressed_byte_stream.data(),
                            h_compressed_byte_stream.size(), cudaMemcpyHostToDevice));
    }
    
    if (!h_block_metadata.empty()) {
        CUDA_CHECK(cudaMalloc(&d_block_metadata, h_block_metadata.size() * sizeof(BlockMetadata<T>)));
        CUDA_CHECK(cudaMemcpy(d_block_metadata, h_block_metadata.data(),
                            h_block_metadata.size() * sizeof(BlockMetadata<T>), cudaMemcpyHostToDevice));
    }
    
    if (algo == ALGO_DFOR && !h_dfor_metadata.empty()) {
        CUDA_CHECK(cudaMalloc(&d_dfor_metadata, h_dfor_metadata.size() * sizeof(DFORMetadata<T>)));
        CUDA_CHECK(cudaMemcpy(d_dfor_metadata, h_dfor_metadata.data(),
                            h_dfor_metadata.size() * sizeof(DFORMetadata<T>), cudaMemcpyHostToDevice));
    } else if (algo == ALGO_RFOR && !h_rfor_metadata.empty()) {
        CUDA_CHECK(cudaMalloc(&d_rfor_metadata, h_rfor_metadata.size() * sizeof(RFORMetadata)));
        CUDA_CHECK(cudaMemcpy(d_rfor_metadata, h_rfor_metadata.data(),
                            h_rfor_metadata.size() * sizeof(RFORMetadata), cudaMemcpyHostToDevice));
    }
    
    // Calculate offsets on GPU
    calculate_serialization_offsets_kernel<<<1, 1>>>(
        d_header, h_header.num_blocks, h_header.block_metadata_entry_size,
        h_header.algo_specific_metadata_entry_size, h_compressed_byte_stream.size(), algo
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get updated header back
    CUDA_CHECK(cudaMemcpy(&h_header, d_header, sizeof(SerializedDataHeader), cudaMemcpyDeviceToHost));
    
    // Calculate total size
    size_t total_size = sizeof(SerializedDataHeader) + h_header.block_metadata_size_bytes +
                       h_header.algo_specific_metadata_size_bytes + h_header.compressed_data_stream_size_bytes;
    
    // Allocate serialized buffer on device
    uint8_t* d_serialized_buffer;
    CUDA_CHECK(cudaMalloc(&d_serialized_buffer, total_size));
    
    // Launch serialization kernel
    int threads = 256;
    int blocks = (h_header.num_blocks + threads - 1) / threads;
    blocks = max(blocks, (int)((h_compressed_byte_stream.size() + threads - 1) / threads));
    
    serialize_data_gpu_kernel<T><<<blocks, threads>>>(
        d_serialized_buffer, d_header, d_compressed_stream, d_block_metadata,
        d_dfor_metadata, d_rfor_metadata, algo
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    std::vector<uint8_t> serialized_buffer(total_size);
    CUDA_CHECK(cudaMemcpy(serialized_buffer.data(), d_serialized_buffer, total_size, cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_header));
    if (d_compressed_stream) CUDA_CHECK(cudaFree(d_compressed_stream));
    if (d_block_metadata) CUDA_CHECK(cudaFree(d_block_metadata));
    if (d_dfor_metadata) CUDA_CHECK(cudaFree(d_dfor_metadata));
    if (d_rfor_metadata) CUDA_CHECK(cudaFree(d_rfor_metadata));
    CUDA_CHECK(cudaFree(d_serialized_buffer));
    
    return serialized_buffer;
}

// =============================================================================
//                       GPU Deserialization Implementation
// =============================================================================

// GPU deserialization validation kernel
__global__ void validate_deserialization_kernel(
    const SerializedDataHeader* header,
    bool* is_valid,
    CompressionAlgorithm* algo_out,
    DataTypeId* dtype_out,
    uint64_t* total_values_out
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *is_valid = (header->magic == GPCC_MAGIC) && (header->version <= GPCC_VERSION);
        if (*is_valid) {
            *algo_out = static_cast<CompressionAlgorithm>(header->algorithm_type);
            *dtype_out = static_cast<DataTypeId>(header->data_type_id);
            *total_values_out = header->total_original_values;
        }
    }
}

// Main GPU deserialization kernel
template<typename T>
__global__ void deserialize_data_gpu_kernel(
    const uint8_t* serialized_buffer,
    const SerializedDataHeader* header,
    uint8_t* compressed_data_stream,
    BlockMetadata<T>* block_metadata,
    DFORMetadata<T>* dfor_metadata,
    RFORMetadata* rfor_metadata,
    CompressionAlgorithm algo
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Copy block metadata in parallel
    if (tid < header->num_blocks) {
        memcpy(&block_metadata[tid],
               serialized_buffer + header->block_metadata_offset + tid * sizeof(BlockMetadata<T>),
               sizeof(BlockMetadata<T>));
    }
    
    // Copy algorithm-specific metadata in parallel
    if (algo == ALGO_DFOR && tid < header->num_blocks && dfor_metadata != nullptr) {
        memcpy(&dfor_metadata[tid],
               serialized_buffer + header->algo_specific_metadata_offset + tid * sizeof(DFORMetadata<T>),
               sizeof(DFORMetadata<T>));
    } else if (algo == ALGO_RFOR && tid < header->num_blocks && rfor_metadata != nullptr) {
        memcpy(&rfor_metadata[tid],
               serialized_buffer + header->algo_specific_metadata_offset + tid * sizeof(RFORMetadata),
               sizeof(RFORMetadata));
    }
    
    // Copy compressed data stream (coalesced memory access)
    uint64_t compressed_size = header->compressed_data_stream_size_bytes;
    uint64_t total_threads = gridDim.x * blockDim.x;
    uint64_t chunk_size = (compressed_size + total_threads - 1) / total_threads;
    uint64_t start_idx = tid * chunk_size;
    uint64_t end_idx = min(start_idx + chunk_size, compressed_size);
    
    if (start_idx < compressed_size) {
        for (uint64_t i = start_idx; i < end_idx; i++) {
            compressed_data_stream[i] = serialized_buffer[header->compressed_data_stream_offset + i];
        }
    }
}

// GPU deserialization wrapper function
template<typename T>
bool deserialize_data_gpu(
    const std::vector<uint8_t>& serialized_buffer, SerializedDataHeader& header,
    CompressionAlgorithm& algo, DataTypeId& dtype_id, uint64_t& total_original_values,
    std::vector<uint8_t>& h_compressed_byte_stream, std::vector<BlockMetadata<T>>& h_block_metadata,
    std::vector<DFORMetadata<T>>& h_dfor_metadata, std::vector<RFORMetadata>& h_rfor_metadata
) {
    if (serialized_buffer.size() < sizeof(SerializedDataHeader)) return false;
    
    // Copy header to verify
    memcpy(&header, serialized_buffer.data(), sizeof(SerializedDataHeader));
    
    // Quick host-side validation
    if (header.magic != GPCC_MAGIC || header.version > GPCC_VERSION) return false;
    if (header.block_metadata_entry_size != sizeof(BlockMetadata<T>)) return false;
    
    algo = static_cast<CompressionAlgorithm>(header.algorithm_type);
    dtype_id = static_cast<DataTypeId>(header.data_type_id);
    total_original_values = header.total_original_values;
    
    // Allocate device memory for entire serialized buffer
    uint8_t* d_serialized_buffer;
    CUDA_CHECK(cudaMalloc(&d_serialized_buffer, serialized_buffer.size()));
    CUDA_CHECK(cudaMemcpy(d_serialized_buffer, serialized_buffer.data(),
                        serialized_buffer.size(), cudaMemcpyHostToDevice));
    
    // Allocate device memory for outputs
    SerializedDataHeader* d_header = (SerializedDataHeader*)d_serialized_buffer;
    uint8_t* d_compressed_stream = nullptr;
    BlockMetadata<T>* d_block_metadata = nullptr;
    DFORMetadata<T>* d_dfor_metadata = nullptr;
    RFORMetadata* d_rfor_metadata = nullptr;
    
    if (header.compressed_data_stream_size_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_compressed_stream, header.compressed_data_stream_size_bytes));
    }
    
    if (header.num_blocks > 0) {
        CUDA_CHECK(cudaMalloc(&d_block_metadata, header.num_blocks * sizeof(BlockMetadata<T>)));
    }
    
    if (algo == ALGO_DFOR && header.num_blocks > 0) {
        CUDA_CHECK(cudaMalloc(&d_dfor_metadata, header.num_blocks * sizeof(DFORMetadata<T>)));
    } else if (algo == ALGO_RFOR && header.num_blocks > 0) {
        CUDA_CHECK(cudaMalloc(&d_rfor_metadata, header.num_blocks * sizeof(RFORMetadata)));
    }
    
    // Launch deserialization kernel
    int threads = 256;
    int blocks = (header.num_blocks + threads - 1) / threads;
    blocks = max(blocks, (int)((header.compressed_data_stream_size_bytes + threads - 1) / threads));
    
    deserialize_data_gpu_kernel<T><<<blocks, threads>>>(
        d_serialized_buffer, d_header, d_compressed_stream, d_block_metadata,
        d_dfor_metadata, d_rfor_metadata, algo
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    h_compressed_byte_stream.resize(header.compressed_data_stream_size_bytes);
    if (header.compressed_data_stream_size_bytes > 0) {
        CUDA_CHECK(cudaMemcpy(h_compressed_byte_stream.data(), d_compressed_stream,
                            header.compressed_data_stream_size_bytes, cudaMemcpyDeviceToHost));
    }
    
    h_block_metadata.resize(header.num_blocks);
    if (header.num_blocks > 0) {
        CUDA_CHECK(cudaMemcpy(h_block_metadata.data(), d_block_metadata,
                            header.num_blocks * sizeof(BlockMetadata<T>), cudaMemcpyDeviceToHost));
    }
    
    if (algo == ALGO_DFOR) {
        h_dfor_metadata.resize(header.num_blocks);
        if (header.num_blocks > 0) {
            CUDA_CHECK(cudaMemcpy(h_dfor_metadata.data(), d_dfor_metadata,
                                header.num_blocks * sizeof(DFORMetadata<T>), cudaMemcpyDeviceToHost));
        }
    } else {
        h_dfor_metadata.clear();
    }
    
    if (algo == ALGO_RFOR) {
        h_rfor_metadata.resize(header.num_blocks);
        if (header.num_blocks > 0) {
            CUDA_CHECK(cudaMemcpy(h_rfor_metadata.data(), d_rfor_metadata,
                                header.num_blocks * sizeof(RFORMetadata), cudaMemcpyDeviceToHost));
        }
    } else {
        h_rfor_metadata.clear();
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_serialized_buffer));
    if (d_compressed_stream) CUDA_CHECK(cudaFree(d_compressed_stream));
    if (d_block_metadata) CUDA_CHECK(cudaFree(d_block_metadata));
    if (d_dfor_metadata) CUDA_CHECK(cudaFree(d_dfor_metadata));
    if (d_rfor_metadata) CUDA_CHECK(cudaFree(d_rfor_metadata));
    
    return true;
}

// =============================================================================
//                       GPU-FOR Decompression Implementation
// =============================================================================
template<typename T, int BLOCKS_PER_TB>
__global__ void decompress_gpu_for_kernel(
    const uint8_t* __restrict__ compressed_data_stream, 
    const BlockMetadata<T>* __restrict__ block_metadata,
    T* __restrict__ output, int num_total_blocks 
) {
    const int tid = threadIdx.x; 
    const int logical_block_in_tb = tid / BLOCK_SIZE; 
    const int thread_in_logical_block = tid % BLOCK_SIZE; 
    const int global_logical_block_id = blockIdx.x * BLOCKS_PER_TB + logical_block_in_tb;
    if (global_logical_block_id >= num_total_blocks) return;

    const BlockMetadata<T>& meta = block_metadata[global_logical_block_id];
    const T reference = meta.reference;
    const uint32_t bitwidth_word = meta.bitwidth_word;
    const uint8_t* this_block_packed_data_start = compressed_data_stream + meta.compressed_offset;
    
    const int miniblock_id_in_logical_block = thread_in_logical_block / MINIBLOCK_SIZE;
    const int index_in_miniblock = thread_in_logical_block % MINIBLOCK_SIZE;
    const uint32_t bitwidth_code = (bitwidth_word >> (miniblock_id_in_logical_block * 8)) & 0xFF;
    T final_value;

    if (bitwidth_code == 0) { 
        final_value = reference; 
    }
    else if (bitwidth_code != RLE_BITWIDTH_FLAG) {
        uint64_t miniblock_bit_offset_in_block = 0;
        for (int i = 0; i < miniblock_id_in_logical_block; i++) {
            uint32_t mb_bw_code = (bitwidth_word >> (i * 8)) & 0xFF;
            if (mb_bw_code != RLE_BITWIDTH_FLAG) { 
                miniblock_bit_offset_in_block += mb_bw_code * MINIBLOCK_SIZE;
            }
        }
        const uint64_t bit_pos_in_block_stream = miniblock_bit_offset_in_block + index_in_miniblock * bitwidth_code;
        
        // Use 64-bit extraction for types larger than 32 bits
        if (sizeof(T) > 4 && bitwidth_code > 32) {
            uint64_t extracted_val = extract_bits_from_byte_stream_64(this_block_packed_data_start, bit_pos_in_block_stream, bitwidth_code);
            final_value = reference + static_cast<T>(extracted_val);
        } else {
            uint32_t extracted_val = extract_bits_from_byte_stream(this_block_packed_data_start, bit_pos_in_block_stream, bitwidth_code);
            final_value = reference + static_cast<T>(extracted_val);
        }
    }
    output[global_logical_block_id * BLOCK_SIZE + thread_in_logical_block] = final_value;
}

// =============================================================================
//                       GPU-DFOR Decompression Implementation
// =============================================================================
template<typename T>
__device__ void block_prefix_sum_inclusive(T* data_array, int N, T* temp_array) {
    int tid = threadIdx.x;
    if (tid < N) { 
        temp_array[tid] = data_array[tid];
    }
    __syncthreads();

    for (int stride = 1; stride < N; stride <<= 1) {
        T val_read_from_left = 0;
        if (tid >= stride && tid < N) {
            val_read_from_left = temp_array[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < N) {
            temp_array[tid] += val_read_from_left;
        }
        __syncthreads();
    }

    if (tid < N) {
        data_array[tid] = temp_array[tid];
    }
    __syncthreads();
}

template<typename T>
__device__ void block_prefix_sum_inclusive_with_limit(T* data_array, int N, T* temp_array, int valid_count) {
    int tid = threadIdx.x;
    if (tid < N) { 
        temp_array[tid] = (tid < valid_count) ? data_array[tid] : 0;
    }
    __syncthreads();

    for (int stride = 1; stride < N; stride <<= 1) {
        T val_read_from_left = 0;
        if (tid >= stride && tid < N && tid < valid_count) {
            val_read_from_left = temp_array[tid - stride];
        }
        __syncthreads();
        if (tid >= stride && tid < N && tid < valid_count) {
            temp_array[tid] += val_read_from_left;
        }
        __syncthreads();
    }

    if (tid < N && tid < valid_count) {
        data_array[tid] = temp_array[tid];
    }
    __syncthreads();
}

template<typename T, int BLOCKS_PER_TB>
__global__ void decompress_gpu_dfor_kernel(
    const uint8_t* __restrict__ compressed_data_stream,
    const BlockMetadata<T>* __restrict__ block_metadata,
    const DFORMetadata<T>* __restrict__ dfor_metadata,
    T* __restrict__ output, int num_total_blocks,
    uint64_t total_element_count  // Add total element count parameter
) {
    // Use T type for shared memory to avoid type conversion issues
    __shared__ T s_block_values[BLOCK_SIZE];
    __shared__ T s_temp_scan_buffer[BLOCK_SIZE];

    const int tid_in_cuda_block = threadIdx.x;
    const int logical_block_in_tb_offset = tid_in_cuda_block / BLOCK_SIZE;
    const int thread_in_logical_block = tid_in_cuda_block % BLOCK_SIZE;
    const int global_logical_block_id = blockIdx.x * BLOCKS_PER_TB + logical_block_in_tb_offset;

    if (global_logical_block_id >= num_total_blocks) return;

    const BlockMetadata<T>& meta = block_metadata[global_logical_block_id];
    const DFORMetadata<T>& dfor_meta_ref = dfor_metadata[global_logical_block_id];
    
    // Get the actual number of elements in this block
    const int elements_in_block = meta.element_count;
    const uint64_t block_start_idx = (uint64_t)global_logical_block_id * BLOCK_SIZE;
    
    T l_first_value = dfor_meta_ref.first_value;
    T l_ref_delta = dfor_meta_ref.reference_delta;
    uint32_t l_bitwidth_word = meta.bitwidth_word;

    const uint8_t* this_block_packed_deltas_start = compressed_data_stream + meta.compressed_offset;
    
    T current_value_for_scan = static_cast<T>(0);

    if (thread_in_logical_block == 0) {
        current_value_for_scan = l_first_value;
    } else if (thread_in_logical_block < elements_in_block) {
        // This thread represents a valid element
        const int delta_idx_in_block = thread_in_logical_block - 1;
        const int miniblock_id = delta_idx_in_block / MINIBLOCK_SIZE;
        const int index_in_miniblock = delta_idx_in_block % MINIBLOCK_SIZE;
        const uint32_t bitwidth_code = (l_bitwidth_word >> (miniblock_id * 8)) & 0xFF;

        T actual_delta_T = static_cast<T>(0);
        if (bitwidth_code == 0) {
            actual_delta_T = l_ref_delta;
        } else {
            uint64_t miniblock_bit_offset_in_block = 0;
            for (int i = 0; i < miniblock_id; i++) {
                uint32_t prev_mb_bw_code = (l_bitwidth_word >> (i * 8)) & 0xFF;
                miniblock_bit_offset_in_block += prev_mb_bw_code * MINIBLOCK_SIZE;
            }
            const uint64_t bit_pos_in_delta_stream = miniblock_bit_offset_in_block + index_in_miniblock * bitwidth_code;
            
            // Use 64-bit extraction for larger types
            if (sizeof(T) > 4 && bitwidth_code > 32) {
                uint64_t extracted_offset_val = extract_bits_from_byte_stream_64(
                    this_block_packed_deltas_start,
                    bit_pos_in_delta_stream,
                    bitwidth_code
                );
                actual_delta_T = l_ref_delta + static_cast<T>(extracted_offset_val);
            } else {
                uint32_t extracted_offset_val = extract_bits_from_byte_stream(
                    this_block_packed_deltas_start,
                    bit_pos_in_delta_stream,
                    bitwidth_code
                );
                actual_delta_T = l_ref_delta + static_cast<T>(extracted_offset_val);
            }
        }
        current_value_for_scan = actual_delta_T;
    } else {
        // This thread is beyond valid elements (padding)
        current_value_for_scan = static_cast<T>(0);
    }
    
    if (threadIdx.x < BLOCK_SIZE) { 
        s_block_values[threadIdx.x] = current_value_for_scan;
    }
    __syncthreads();

    // Perform prefix sum
    if (threadIdx.x < BLOCK_SIZE) {
        // Use the limited version only for blocks that aren't full
        if (elements_in_block < BLOCK_SIZE) {
            block_prefix_sum_inclusive_with_limit<T>(s_block_values, BLOCK_SIZE, s_temp_scan_buffer, elements_in_block);
        } else {
            block_prefix_sum_inclusive<T>(s_block_values, BLOCK_SIZE, s_temp_scan_buffer);
        }
    }
    __syncthreads();

    // Write output only for valid elements
    if (threadIdx.x < BLOCK_SIZE && thread_in_logical_block < elements_in_block) {
        const uint64_t global_output_idx = block_start_idx + thread_in_logical_block;
        if (global_output_idx < total_element_count) {
            output[global_output_idx] = s_block_values[threadIdx.x];
        }
    }
}

// =============================================================================
//                       GPU-RFOR Decompression Implementation
// =============================================================================
template<typename T, int BLOCKS_PER_TB>
__global__ void decompress_gpu_rfor_kernel(
    const uint8_t* __restrict__ compressed_data_stream,
    const BlockMetadata<T>* __restrict__ block_metadata,
    const RFORMetadata* __restrict__ rfor_metadata_device, 
    T* __restrict__ output, int num_total_blocks
) {
    const int tid = threadIdx.x;
    const int logical_block_in_tb = tid / BLOCK_SIZE;
    const int thread_in_logical_block = tid % BLOCK_SIZE;
    const int global_logical_block_id = blockIdx.x * BLOCKS_PER_TB + logical_block_in_tb;
    if (global_logical_block_id >= num_total_blocks) return;

    const BlockMetadata<T>& meta = block_metadata[global_logical_block_id];
    const RFORMetadata& rfor_meta = rfor_metadata_device[global_logical_block_id];
    const T block_reference = meta.reference; 
    const uint32_t miniblock_info_word = meta.bitwidth_word; 
    const uint8_t* this_block_data_start = compressed_data_stream + meta.compressed_offset;
    const int miniblock_id_in_logical_block = thread_in_logical_block / MINIBLOCK_SIZE;
    const int index_in_miniblock = thread_in_logical_block % MINIBLOCK_SIZE;
    const uint32_t miniblock_code = (miniblock_info_word >> (miniblock_id_in_logical_block * 8)) & 0xFF;
    T final_value;

    if (miniblock_code == RLE_BITWIDTH_FLAG) {
        int rle_value_idx = 0;
        for (int i = 0; i < miniblock_id_in_logical_block; i++) {
            if (((miniblock_info_word >> (i * 8)) & 0xFF) == RLE_BITWIDTH_FLAG) rle_value_idx++;
        }
        memcpy(&final_value, this_block_data_start + (rle_value_idx * sizeof(T)), sizeof(T));
    } else { 
        if (miniblock_code == 0) {
            final_value = block_reference;
        } else {
            const uint8_t* for_data_sec_start = this_block_data_start + rfor_meta.for_packed_data_offset_bytes;
            uint64_t for_mb_bit_offset = 0;
            for (int i = 0; i < miniblock_id_in_logical_block; i++) {
                uint32_t prev_mb_code = (miniblock_info_word >> (i * 8)) & 0xFF;
                if (prev_mb_code != RLE_BITWIDTH_FLAG) for_mb_bit_offset += prev_mb_code * MINIBLOCK_SIZE;
            }
            
            // Use 64-bit extraction for larger types
            if (sizeof(T) > 4 && miniblock_code > 32) {
                uint64_t extracted_val = extract_bits_from_byte_stream_64(for_data_sec_start, 
                    for_mb_bit_offset + index_in_miniblock * miniblock_code, miniblock_code);
                final_value = block_reference + static_cast<T>(extracted_val);
            } else {
                uint32_t extracted_val = extract_bits_from_byte_stream(for_data_sec_start, 
                    for_mb_bit_offset + index_in_miniblock * miniblock_code, miniblock_code);
                final_value = block_reference + static_cast<T>(extracted_val);
            }
        }
    }
    output[global_logical_block_id * BLOCK_SIZE + thread_in_logical_block] = final_value;
}

// =============================================================================
//                       Random Access Decompression
// =============================================================================
template<typename T>
__global__ void decompress_random_access_kernel(
    const uint8_t* __restrict__ compressed_data_stream,
    const BlockMetadata<T>* __restrict__ block_metadata,
    const DFORMetadata<T>* __restrict__ dfor_metadata, 
    const RFORMetadata* __restrict__ rfor_metadata_device, 
    const uint32_t* __restrict__ query_indices, 
    T* __restrict__ output_values, int num_queries,
    CompressionAlgorithm algo, uint64_t total_original_values_count 
) {
    const int query_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_id >= num_queries) return;
    const uint32_t global_original_idx = query_indices[query_id];
    if (global_original_idx >= total_original_values_count) return;

    const int logical_block_id = global_original_idx / BLOCK_SIZE;
    const int index_in_logical_block = global_original_idx % BLOCK_SIZE;
    const BlockMetadata<T>& meta = block_metadata[logical_block_id];
    const uint8_t* this_block_data_start = compressed_data_stream + meta.compressed_offset;
    T final_value = static_cast<T>(0);

    if (algo == ALGO_FOR) {
        const T reference = meta.reference;
        const uint32_t bitwidth_word = meta.bitwidth_word;
        const int miniblock_id = index_in_logical_block / MINIBLOCK_SIZE;
        const int idx_in_miniblock = index_in_logical_block % MINIBLOCK_SIZE;
        const uint32_t bitwidth_code = (bitwidth_word >> (miniblock_id * 8)) & 0xFF;
        if (bitwidth_code == 0) {
            final_value = reference;
        } else {
            uint64_t mb_bit_offset = 0;
            for (int i = 0; i < miniblock_id; i++) {
                uint32_t prev_mb_bw_code = (bitwidth_word >> (i * 8)) & 0xFF;
                if (prev_mb_bw_code != RLE_BITWIDTH_FLAG) mb_bit_offset += prev_mb_bw_code * MINIBLOCK_SIZE;
            }
            
            if (sizeof(T) > 4 && bitwidth_code > 32) {
                uint64_t extracted_val = extract_bits_from_byte_stream_64(this_block_data_start, 
                    mb_bit_offset + idx_in_miniblock * bitwidth_code, bitwidth_code);
                final_value = reference + static_cast<T>(extracted_val);
            } else {
                uint32_t extracted_val = extract_bits_from_byte_stream(this_block_data_start, 
                    mb_bit_offset + idx_in_miniblock * bitwidth_code, bitwidth_code);
                final_value = reference + static_cast<T>(extracted_val);
            }
        }
    } else if (algo == ALGO_DFOR) {
        const DFORMetadata<T>& dfor_meta = dfor_metadata[logical_block_id];
        T current_cumulative_value = dfor_meta.first_value;
        
        // Check how many valid elements are in this block
        const uint64_t block_start_idx = (uint64_t)logical_block_id * BLOCK_SIZE;
        const int valid_elements_in_block = (block_start_idx + BLOCK_SIZE > total_original_values_count) ?
                                           (total_original_values_count - block_start_idx) : BLOCK_SIZE;
        
        if (index_in_logical_block > 0 && index_in_logical_block < valid_elements_in_block) {
            const uint8_t* current_block_packed_deltas_start = this_block_data_start;
            for (int i = 0; i < index_in_logical_block; ++i) {
                const int delta_idx_in_block = i;
                const int miniblock_id = delta_idx_in_block / MINIBLOCK_SIZE;
                const int idx_in_miniblock = delta_idx_in_block % MINIBLOCK_SIZE;
                const uint32_t bitwidth_code = (meta.bitwidth_word >> (miniblock_id * 8)) & 0xFF;
                T current_delta_val;
                if (bitwidth_code == 0) {
                    current_delta_val = dfor_meta.reference_delta;
                } else {
                    uint64_t mb_bit_offset = 0;
                    for (int k=0; k<miniblock_id; ++k) 
                        mb_bit_offset += ((meta.bitwidth_word >> (k*8)) & 0xFF) * MINIBLOCK_SIZE;
                    
                    if (sizeof(T) > 4 && bitwidth_code > 32) {
                        uint64_t extracted_val = extract_bits_from_byte_stream_64(current_block_packed_deltas_start, 
                            mb_bit_offset + idx_in_miniblock * bitwidth_code, bitwidth_code);
                        current_delta_val = dfor_meta.reference_delta + static_cast<T>(extracted_val);
                    } else {
                        uint32_t extracted_val = extract_bits_from_byte_stream(current_block_packed_deltas_start, 
                            mb_bit_offset + idx_in_miniblock * bitwidth_code, bitwidth_code);
                        current_delta_val = dfor_meta.reference_delta + static_cast<T>(extracted_val);
                    }
                }
                current_cumulative_value += current_delta_val;
            }
        }
        final_value = current_cumulative_value;
    } else if (algo == ALGO_RFOR) {
        const RFORMetadata& rfor_meta = rfor_metadata_device[logical_block_id];
        const T block_ref = meta.reference;
        const uint32_t mb_info_word = meta.bitwidth_word;
        const int mb_id = index_in_logical_block / MINIBLOCK_SIZE;
        const int idx_in_mb = index_in_logical_block % MINIBLOCK_SIZE;
        const uint32_t mb_code = (mb_info_word >> (mb_id * 8)) & 0xFF;
        if (mb_code == RLE_BITWIDTH_FLAG) {
            int rle_val_idx = 0;
            for (int i = 0; i < mb_id; i++) 
                if (((mb_info_word >> (i * 8)) & 0xFF) == RLE_BITWIDTH_FLAG) rle_val_idx++;
            memcpy(&final_value, this_block_data_start + (rle_val_idx * sizeof(T)), sizeof(T));
        } else {
            if (mb_code == 0) {
                final_value = block_ref;
            } else {
                const uint8_t* for_data_sec_start = this_block_data_start + rfor_meta.for_packed_data_offset_bytes;
                uint64_t for_mb_bit_offset = 0;
                for (int i = 0; i < mb_id; i++) {
                    uint32_t prev_mb_code = (mb_info_word >> (i * 8)) & 0xFF;
                    if (prev_mb_code != RLE_BITWIDTH_FLAG) for_mb_bit_offset += prev_mb_code * MINIBLOCK_SIZE;
                }
                
                if (sizeof(T) > 4 && mb_code > 32) {
                    uint64_t extracted_val = extract_bits_from_byte_stream_64(for_data_sec_start, 
                        for_mb_bit_offset + idx_in_mb * mb_code, mb_code);
                    final_value = block_ref + static_cast<T>(extracted_val);
                } else {
                    uint32_t extracted_val = extract_bits_from_byte_stream(for_data_sec_start, 
                        for_mb_bit_offset + idx_in_mb * mb_code, mb_code);
                    final_value = block_ref + static_cast<T>(extracted_val);
                }
            }
        }
    }
    output_values[query_id] = final_value;
}

// 按块解压的随机访问实现 - 每个查询都解压整个块
template<typename T>
__global__ void decompress_random_access_block_based_kernel(
    const uint8_t* __restrict__ compressed_data_stream,
    const BlockMetadata<T>* __restrict__ block_metadata,
    const DFORMetadata<T>* __restrict__ dfor_metadata, 
    const RFORMetadata* __restrict__ rfor_metadata_device, 
    const uint32_t* __restrict__ query_indices, 
    T* __restrict__ output_values, int num_queries,
    CompressionAlgorithm algo, uint64_t total_original_values_count 
) {
    const int query_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_id >= num_queries) return;
    
    const uint32_t global_original_idx = query_indices[query_id];
    if (global_original_idx >= total_original_values_count) return;
    
    const int logical_block_id = global_original_idx / BLOCK_SIZE;
    const int index_in_logical_block = global_original_idx % BLOCK_SIZE;
    
    // 对于随机访问，我们必须解压整个块来获取一个值
    // 这模拟了真实成本 - 我们不能高效地只提取一个值
    
    const BlockMetadata<T>& meta = block_metadata[logical_block_id];
    const uint8_t* this_block_data_start = compressed_data_stream + meta.compressed_offset;
    
    // 在寄存器中解压整个块
    T block_values[BLOCK_SIZE];
    
    if (algo == ALGO_FOR) {
        const T reference = meta.reference;
        const uint32_t bitwidth_word = meta.bitwidth_word;
        
        // 解压块中的所有128个值
        for (int i = 0; i < BLOCK_SIZE; i++) {
            const int miniblock_id = i / MINIBLOCK_SIZE;
            const int idx_in_miniblock = i % MINIBLOCK_SIZE;
            const uint32_t bitwidth_code = (bitwidth_word >> (miniblock_id * 8)) & 0xFF;
            
            if (bitwidth_code == 0) {
                block_values[i] = reference;
            } else if (bitwidth_code != RLE_BITWIDTH_FLAG) {
                uint64_t mb_bit_offset = 0;
                for (int j = 0; j < miniblock_id; j++) {
                    uint32_t prev_mb_bw_code = (bitwidth_word >> (j * 8)) & 0xFF;
                    if (prev_mb_bw_code != RLE_BITWIDTH_FLAG) 
                        mb_bit_offset += prev_mb_bw_code * MINIBLOCK_SIZE;
                }
                
                if (sizeof(T) > 4 && bitwidth_code > 32) {
                    uint64_t extracted_val = extract_bits_from_byte_stream_64(
                        this_block_data_start, 
                        mb_bit_offset + idx_in_miniblock * bitwidth_code, 
                        bitwidth_code
                    );
                    block_values[i] = reference + static_cast<T>(extracted_val);
                } else {
                    uint32_t extracted_val = extract_bits_from_byte_stream(
                        this_block_data_start, 
                        mb_bit_offset + idx_in_miniblock * bitwidth_code, 
                        bitwidth_code
                    );
                    block_values[i] = reference + static_cast<T>(extracted_val);
                }
            } else {
                // RLE miniblock in FOR - should not happen in pure FOR
                block_values[i] = reference;
            }
        }
    } else if (algo == ALGO_DFOR) {
        // DFOR需要解压整个块并进行前缀和计算
        const DFORMetadata<T>& dfor_meta = dfor_metadata[logical_block_id];
        const uint64_t block_start_idx = (uint64_t)logical_block_id * BLOCK_SIZE;
        const int valid_elements_in_block = (block_start_idx + BLOCK_SIZE > total_original_values_count) ?
                                           (total_original_values_count - block_start_idx) : BLOCK_SIZE;
        
        T l_first_value = dfor_meta.first_value;
        T l_ref_delta = dfor_meta.reference_delta;
        uint32_t l_bitwidth_word = meta.bitwidth_word;
        
        // 首先解压所有的delta值
        T deltas[BLOCK_SIZE - 1];
        for (int i = 0; i < BLOCK_SIZE - 1; i++) {
            if (i < valid_elements_in_block - 1) {
                const int miniblock_id = i / MINIBLOCK_SIZE;
                const int idx_in_miniblock = i % MINIBLOCK_SIZE;
                const uint32_t bitwidth_code = (l_bitwidth_word >> (miniblock_id * 8)) & 0xFF;
                
                if (bitwidth_code == 0) {
                    deltas[i] = l_ref_delta;
                } else {
                    uint64_t mb_bit_offset = 0;
                    for (int j = 0; j < miniblock_id; j++) {
                        uint32_t prev_mb_bw_code = (l_bitwidth_word >> (j * 8)) & 0xFF;
                        mb_bit_offset += prev_mb_bw_code * MINIBLOCK_SIZE;
                    }
                    
                    if (sizeof(T) > 4 && bitwidth_code > 32) {
                        uint64_t extracted_val = extract_bits_from_byte_stream_64(
                            this_block_data_start, 
                            mb_bit_offset + idx_in_miniblock * bitwidth_code, 
                            bitwidth_code
                        );
                        deltas[i] = l_ref_delta + static_cast<T>(extracted_val);
                    } else {
                        uint32_t extracted_val = extract_bits_from_byte_stream(
                            this_block_data_start, 
                            mb_bit_offset + idx_in_miniblock * bitwidth_code, 
                            bitwidth_code
                        );
                        deltas[i] = l_ref_delta + static_cast<T>(extracted_val);
                    }
                }
            } else {
                deltas[i] = static_cast<T>(0);
            }
        }
        
        // 执行前缀和来重建原始值
        block_values[0] = l_first_value;
        for (int i = 1; i < BLOCK_SIZE; i++) {
            if (i < valid_elements_in_block) {
                block_values[i] = block_values[i-1] + deltas[i-1];
            } else {
                block_values[i] = static_cast<T>(0);
            }
        }
    } else if (algo == ALGO_RFOR) {
        // RFOR解压整个块
        const RFORMetadata& rfor_meta = rfor_metadata_device[logical_block_id];
        const T block_ref = meta.reference;
        const uint32_t miniblock_info_word = meta.bitwidth_word;
        
        // 首先统计RLE值的数量
        int rle_value_count = 0;
        for (int m = 0; m < MINIBLOCKS_PER_BLOCK; m++) {
            uint32_t mb_code = (miniblock_info_word >> (m * 8)) & 0xFF;
            if (mb_code == RLE_BITWIDTH_FLAG) {
                rle_value_count++;
            }
        }
        
        // 读取所有RLE值
        T rle_values[MINIBLOCKS_PER_BLOCK];  // 最多4个RLE miniblock
        for (int i = 0; i < rle_value_count; i++) {
            memcpy(&rle_values[i], this_block_data_start + (i * sizeof(T)), sizeof(T));
        }
        
        const uint8_t* for_data_start = this_block_data_start + rfor_meta.for_packed_data_offset_bytes;
        
        // 解压每个元素
        int current_rle_idx = 0;
        for (int i = 0; i < BLOCK_SIZE; i++) {
            const int mb_id = i / MINIBLOCK_SIZE;
            const int idx_in_mb = i % MINIBLOCK_SIZE;
            const uint32_t mb_code = (miniblock_info_word >> (mb_id * 8)) & 0xFF;
            
            if (mb_code == RLE_BITWIDTH_FLAG) {
                // 这是RLE miniblock，使用对应的RLE值
                if (idx_in_mb == 0) {
                    // 找到这是第几个RLE miniblock
                    int rle_idx = 0;
                    for (int j = 0; j < mb_id; j++) {
                        if (((miniblock_info_word >> (j * 8)) & 0xFF) == RLE_BITWIDTH_FLAG) {
                            rle_idx++;
                        }
                    }
                    current_rle_idx = rle_idx;
                }
                block_values[i] = rle_values[current_rle_idx];
            } else {
                // FOR编码的miniblock
                if (mb_code == 0) {
                    block_values[i] = block_ref;
                } else {
                    // 计算这个miniblock在FOR数据中的位偏移
                    uint64_t for_mb_bit_offset = 0;
                    for (int j = 0; j < mb_id; j++) {
                        uint32_t prev_mb_code = (miniblock_info_word >> (j * 8)) & 0xFF;
                        if (prev_mb_code != RLE_BITWIDTH_FLAG) {
                            for_mb_bit_offset += prev_mb_code * MINIBLOCK_SIZE;
                        }
                    }
                    
                    if (sizeof(T) > 4 && mb_code > 32) {
                        uint64_t extracted_val = extract_bits_from_byte_stream_64(
                            for_data_start, 
                            for_mb_bit_offset + idx_in_mb * mb_code, 
                            mb_code
                        );
                        block_values[i] = block_ref + static_cast<T>(extracted_val);
                    } else {
                        uint32_t extracted_val = extract_bits_from_byte_stream(
                            for_data_start, 
                            for_mb_bit_offset + idx_in_mb * mb_code, 
                            mb_code
                        );
                        block_values[i] = block_ref + static_cast<T>(extracted_val);
                    }
                }
            }
        }
    } else {
        // ALGO_NONE or unsupported algorithm
        for (int i = 0; i < BLOCK_SIZE; i++) {
            block_values[i] = static_cast<T>(0);
        }
    }
    
    // 提取请求的值
    output_values[query_id] = block_values[index_in_logical_block];
}

// =============================================================================
//                       CPU Compression Implementation
// =============================================================================
template<typename T_VAL>
inline int get_bits_needed_cpu(T_VAL val) { 
    if (val == 0) return 0;
    if (std::is_unsigned<T_VAL>::value || val > 0) {
        if (sizeof(T_VAL) <= 4) {
            unsigned long v_ul = static_cast<unsigned long>(val); 
            if (v_ul == 0) return 0;
            return (sizeof(unsigned long) * 8) - __builtin_clzl(v_ul);
        } else { 
            unsigned long long v_ull = static_cast<unsigned long long>(val);
            if (v_ull == 0) return 0;
            return (sizeof(unsigned long long) * 8) - __builtin_clzll(v_ull);
        }
    } else { 
        unsigned long long v_ull = static_cast<unsigned long long>(-val); 
        if (v_ull == 0) return 0; 
        return ((sizeof(unsigned long long) * 8) - __builtin_clzll(v_ull)) + 1; 
    }
}

void append_bits_to_byte_stream(
    std::vector<uint8_t>& byte_stream, uint64_t& current_bit_offset_in_stream, 
    uint64_t value_to_pack, int num_bits  // Changed from uint32_t to uint64_t
) {
    if (num_bits == 0) return;
    if (num_bits > 64) num_bits = 64;  // Clamp to 64 bits
    
    uint64_t current_byte_idx = current_bit_offset_in_stream / 8;
    uint8_t bit_in_target_byte = current_bit_offset_in_stream % 8;
    size_t required_size = current_byte_idx + 9;  // Up to 9 bytes for 64 bits
    if (byte_stream.size() < required_size) byte_stream.resize(required_size, 0); 
    
    uint64_t value_shifted = value_to_pack << bit_in_target_byte;
    int bytes_to_write = (bit_in_target_byte + num_bits + 7) / 8;
    
    for (int i = 0; i < bytes_to_write && i < 9; ++i) { 
        byte_stream[current_byte_idx + i] |= static_cast<uint8_t>((value_shifted >> (i * 8)) & 0xFF);
    }
    current_bit_offset_in_stream += num_bits;
}

template<typename T>
void compress_for_cpu(
    const T* data, int count,
    std::vector<uint8_t>& compressed_byte_stream_out, 
    std::vector<BlockMetadata<T>>& metadata_out
) {
    int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    metadata_out.resize(num_blocks);
    compressed_byte_stream_out.clear();
    uint64_t current_total_bit_offset_in_stream = 0; 

    for (int b = 0; b < num_blocks; b++) {
        int block_start_idx = b * BLOCK_SIZE;
        int block_end_idx = std::min(block_start_idx + BLOCK_SIZE, count);
        int current_block_element_count = block_end_idx - block_start_idx;
        metadata_out[b].original_offset = block_start_idx;
        metadata_out[b].element_count = current_block_element_count;  // Set element count
        metadata_out[b].compressed_offset = (current_total_bit_offset_in_stream + 7) / 8; 

        if (current_block_element_count == 0) {
            metadata_out[b].reference = static_cast<T>(0); 
            metadata_out[b].bitwidth_word = 0;
            metadata_out[b].compressed_size_bytes = 0; 
            continue;
        }
        
        T min_val = data[block_start_idx];
        for (int i = block_start_idx + 1; i < block_end_idx; i++) {
            if (data[i] < min_val) min_val = data[i];
        }
        metadata_out[b].reference = min_val;
        uint32_t bitwidth_word = 0;

        for (int m = 0; m < MINIBLOCKS_PER_BLOCK; m++) {
            int mb_start_idx = block_start_idx + m * MINIBLOCK_SIZE;
            int mb_end_idx = std::min(mb_start_idx + MINIBLOCK_SIZE, block_end_idx);
            if (mb_start_idx >= block_end_idx) { 
                bitwidth_word |= (0U << (m * 8)); 
                continue; 
            }
            
            using UnsignedEquivalentT = typename std::make_unsigned<T>::type;
            UnsignedEquivalentT mb_max_delta_unsigned = 0; 
            for (int i = mb_start_idx; i < mb_end_idx; i++) {
                UnsignedEquivalentT current_delta_unsigned = static_cast<UnsignedEquivalentT>(data[i] - min_val);
                if (current_delta_unsigned > mb_max_delta_unsigned) {
                    mb_max_delta_unsigned = current_delta_unsigned;
                }
            }
            
            int bits_needed_for_mb = get_bits_needed_cpu(mb_max_delta_unsigned);
            if (bits_needed_for_mb < 0) bits_needed_for_mb = 0; 
            if (bits_needed_for_mb > 64) bits_needed_for_mb = 64;  // Increased from 32 to 64
            
            // For 64-bit support, we need to handle cases where bitwidth > 255
            // For now, we'll clamp to 64 and store in the 8-bit field
            if (bits_needed_for_mb > 64) bits_needed_for_mb = 64;
            bitwidth_word |= (static_cast<uint32_t>(bits_needed_for_mb) << (m * 8));

            if (bits_needed_for_mb > 0) {
                for (int i = mb_start_idx; i < mb_end_idx; i++) {
                    UnsignedEquivalentT delta_val_unsigned = static_cast<UnsignedEquivalentT>(data[i] - min_val);
                    append_bits_to_byte_stream(compressed_byte_stream_out, current_total_bit_offset_in_stream, 
                                             static_cast<uint64_t>(delta_val_unsigned), bits_needed_for_mb);
                }
                for (int i = mb_end_idx; i < mb_start_idx + MINIBLOCK_SIZE; ++i) {
                    append_bits_to_byte_stream(compressed_byte_stream_out, current_total_bit_offset_in_stream, 0, bits_needed_for_mb);
                }
            }
        }
        metadata_out[b].bitwidth_word = bitwidth_word;
        uint64_t end_byte_offset_for_block_data = (current_total_bit_offset_in_stream + 7) / 8;
        metadata_out[b].compressed_size_bytes = end_byte_offset_for_block_data - metadata_out[b].compressed_offset;
    }
}

template<typename T>
void compress_dfor_cpu(
    const T* data, int count,
    std::vector<uint8_t>& compressed_byte_stream_out,
    std::vector<BlockMetadata<T>>& metadata_out,
    std::vector<DFORMetadata<T>>& dfor_metadata_out
) {
    int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    metadata_out.resize(num_blocks); 
    dfor_metadata_out.resize(num_blocks);
    compressed_byte_stream_out.clear(); 
    uint64_t current_total_bit_offset_in_stream = 0;

    for (int b = 0; b < num_blocks; b++) {
        int block_start_idx = b * BLOCK_SIZE;
        int block_end_idx = std::min(block_start_idx + BLOCK_SIZE, count);
        int current_block_element_count = block_end_idx - block_start_idx;
        metadata_out[b].original_offset = block_start_idx;
        metadata_out[b].element_count = current_block_element_count;  // Set element count
        metadata_out[b].compressed_offset = (current_total_bit_offset_in_stream + 7) / 8;

        if (current_block_element_count == 0) {
            dfor_metadata_out[b].first_value = static_cast<T>(0);
            dfor_metadata_out[b].reference_delta = static_cast<T>(0);
            metadata_out[b].bitwidth_word = 0; 
            metadata_out[b].compressed_size_bytes = 0; 
            continue;
        }
        
        // Set first value from actual data
        dfor_metadata_out[b].first_value = data[block_start_idx];
        
        // Initialize delta array with zeros
        std::vector<T> deltas_T_type(BLOCK_SIZE - 1, static_cast<T>(0));
        
        // Compute deltas only for valid elements
        for (int i = 1; i < current_block_element_count; i++) {
            deltas_T_type[i-1] = data[block_start_idx + i] - data[block_start_idx + i - 1];
        }
        
        T min_delta_val_T_type = static_cast<T>(0);
        if (current_block_element_count > 1) {
            min_delta_val_T_type = deltas_T_type[0];
            for (int i = 1; i < current_block_element_count - 1; i++) {
                if (deltas_T_type[i] < min_delta_val_T_type) {
                    min_delta_val_T_type = deltas_T_type[i];
                }
            }
        }
        dfor_metadata_out[b].reference_delta = min_delta_val_T_type;
        
        // Remove debug output to reduce clutter
        /*
        if (b >= num_blocks - 2) {
            std::cout << "    Compress Block " << b << ": start_idx=" << block_start_idx
                      << ", end_idx=" << block_end_idx << ", elements=" << current_block_element_count 
                      << ", first_value=" << dfor_metadata_out[b].first_value 
                      << ", ref_delta=" << dfor_metadata_out[b].reference_delta << std::endl;
            if (current_block_element_count > 0 && block_start_idx < count) {
                std::cout << "      First element at [" << block_start_idx << "]: " << data[block_start_idx] << std::endl;
                if (current_block_element_count > 1 && block_end_idx - 1 < count) {
                    std::cout << "      Last element at [" << (block_end_idx - 1) << "]: " << data[block_end_idx - 1] << std::endl;
                }
            }
        }
        */
        
        uint32_t bitwidth_word = 0;
        using UnsignedEquivalentT = typename std::make_unsigned<T>::type;

        for (int m = 0; m < MINIBLOCKS_PER_BLOCK; m++) { 
            int mb_start_delta_idx = m * MINIBLOCK_SIZE; 
            int mb_end_delta_idx = mb_start_delta_idx + MINIBLOCK_SIZE;
            
            UnsignedEquivalentT mb_max_offset_from_ref_delta_unsigned = 0;
            // Only check valid deltas, not padding
            int valid_end = std::min(mb_end_delta_idx, current_block_element_count - 1);
            if (mb_start_delta_idx < valid_end) {
                for (int i = mb_start_delta_idx; i < valid_end; i++) {
                    UnsignedEquivalentT current_offset_unsigned = static_cast<UnsignedEquivalentT>(deltas_T_type[i] - min_delta_val_T_type);
                    if (current_offset_unsigned > mb_max_offset_from_ref_delta_unsigned) {
                        mb_max_offset_from_ref_delta_unsigned = current_offset_unsigned;
                    }
                }
            }
            
            int bits_needed_for_mb = 0;
            if (mb_start_delta_idx < current_block_element_count - 1) {
                bits_needed_for_mb = get_bits_needed_cpu(mb_max_offset_from_ref_delta_unsigned);
                if (bits_needed_for_mb < 0) bits_needed_for_mb = 0; 
                if (bits_needed_for_mb > 64) bits_needed_for_mb = 64;
            }
            bitwidth_word |= (static_cast<uint32_t>(bits_needed_for_mb) << (m * 8));

            if (bits_needed_for_mb > 0) {
                // Always pack full miniblocks (MINIBLOCK_SIZE values)
                for (int i = mb_start_delta_idx; i < mb_end_delta_idx; i++) {
                    UnsignedEquivalentT offset_val_unsigned = 0; // Initialize to 0 for padding
                    // Only calculate and pack actual delta offsets for valid delta indices
                    if (i < current_block_element_count - 1) {
                        offset_val_unsigned = static_cast<UnsignedEquivalentT>(deltas_T_type[i] - min_delta_val_T_type);
                    }
                    append_bits_to_byte_stream(compressed_byte_stream_out, current_total_bit_offset_in_stream,
                                             static_cast<uint64_t>(offset_val_unsigned), bits_needed_for_mb);
                }
            }
        }
        metadata_out[b].bitwidth_word = bitwidth_word;
        uint64_t end_byte_offset_for_block_data = (current_total_bit_offset_in_stream + 7) / 8;
        metadata_out[b].compressed_size_bytes = end_byte_offset_for_block_data - metadata_out[b].compressed_offset;
    }
}

template<typename T>
void compress_rfor_cpu(
    const T* data, int count,
    std::vector<uint8_t>& compressed_byte_stream_out,
    std::vector<BlockMetadata<T>>& metadata_out,
    std::vector<RFORMetadata>& rfor_metadata_out 
) {
    int num_blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    metadata_out.resize(num_blocks); 
    rfor_metadata_out.resize(num_blocks);
    compressed_byte_stream_out.clear();

    for (int b = 0; b < num_blocks; b++) {
        int block_start_idx = b * BLOCK_SIZE;
        int block_end_idx = std::min(block_start_idx + BLOCK_SIZE, count);
        int current_block_element_count = block_end_idx - block_start_idx;
        metadata_out[b].original_offset = block_start_idx;
        metadata_out[b].element_count = current_block_element_count;  // Set element count
        size_t byte_offset_start_of_this_block_data = compressed_byte_stream_out.size(); 
        metadata_out[b].compressed_offset = byte_offset_start_of_this_block_data;

        if (current_block_element_count == 0) {
            metadata_out[b].reference = static_cast<T>(0); 
            metadata_out[b].bitwidth_word = 0; 
            rfor_metadata_out[b].rle_miniblock_count = 0; 
            rfor_metadata_out[b].num_rle_values_in_block = 0;
            rfor_metadata_out[b].for_packed_data_offset_bytes = 0; 
            metadata_out[b].compressed_size_bytes = 0; 
            continue;
        }
        
        T block_min_val = data[block_start_idx]; 
        for (int i = block_start_idx + 1; i < block_end_idx; i++) {
            if (data[i] < block_min_val) block_min_val = data[i];
        }
        metadata_out[b].reference = block_min_val;
        
        uint32_t miniblock_info_word = 0;
        std::vector<T> rle_values_for_this_block_storage; 
        std::vector<uint8_t> for_packed_data_for_this_block_stream; 
        uint64_t current_for_bit_offset_in_block_for_stream = 0; 
        uint32_t rle_miniblock_counter_for_block = 0;
        using UnsignedEquivalentT = typename std::make_unsigned<T>::type;

        for (int m = 0; m < MINIBLOCKS_PER_BLOCK; m++) {
            int mb_start_idx = block_start_idx + m * MINIBLOCK_SIZE;
            int mb_end_idx = std::min(mb_start_idx + MINIBLOCK_SIZE, block_end_idx);
            if (mb_start_idx >= block_end_idx) { 
                miniblock_info_word |= (0U << (m * 8)); 
                continue; 
            }
            
            bool is_rle_miniblock = true;
            T first_val_in_mb = data[mb_start_idx];
            for (int i = mb_start_idx + 1; i < mb_end_idx; i++) {
                if (data[i] != first_val_in_mb) { 
                    is_rle_miniblock = false; 
                    break; 
                }
            }
            
            if (is_rle_miniblock) {
                miniblock_info_word |= (RLE_BITWIDTH_FLAG << (m * 8));
                rle_values_for_this_block_storage.push_back(first_val_in_mb); 
                rle_miniblock_counter_for_block++;
            } else { 
                UnsignedEquivalentT mb_max_delta_unsigned = 0;
                for (int i = mb_start_idx; i < mb_end_idx; i++) {
                    UnsignedEquivalentT delta_unsigned = static_cast<UnsignedEquivalentT>(data[i] - block_min_val);
                    if (delta_unsigned > mb_max_delta_unsigned) {
                        mb_max_delta_unsigned = delta_unsigned;
                    }
                }
                int bits_needed = get_bits_needed_cpu(mb_max_delta_unsigned);
                if (bits_needed < 0) bits_needed = 0; 
                if (bits_needed > 64) bits_needed = 64;  // Changed from 31 to 64
                miniblock_info_word |= (static_cast<uint32_t>(bits_needed) << (m * 8));
                
                if (bits_needed > 0) {
                    for (int i = mb_start_idx; i < mb_end_idx; i++) {
                        append_bits_to_byte_stream(for_packed_data_for_this_block_stream, 
                                                 current_for_bit_offset_in_block_for_stream, 
                                                 static_cast<uint64_t>(static_cast<UnsignedEquivalentT>(data[i] - block_min_val)), 
                                                 bits_needed);
                    }
                    for (int i = mb_end_idx; i < mb_start_idx + MINIBLOCK_SIZE; ++i) {
                        append_bits_to_byte_stream(for_packed_data_for_this_block_stream, 
                                                 current_for_bit_offset_in_block_for_stream, 0, bits_needed);
                    }
                }
            }
        }
        
        metadata_out[b].bitwidth_word = miniblock_info_word;
        rfor_metadata_out[b].rle_miniblock_count = rle_miniblock_counter_for_block;
        rfor_metadata_out[b].num_rle_values_in_block = rle_values_for_this_block_storage.size();
        rfor_metadata_out[b].for_packed_data_offset_bytes = rle_values_for_this_block_storage.size() * sizeof(T); 

        for (const T& val : rle_values_for_this_block_storage) {
            const uint8_t* val_bytes_ptr = reinterpret_cast<const uint8_t*>(&val);
            compressed_byte_stream_out.insert(compressed_byte_stream_out.end(), val_bytes_ptr, val_bytes_ptr + sizeof(T));
        }
        compressed_byte_stream_out.insert(compressed_byte_stream_out.end(), 
                                        for_packed_data_for_this_block_stream.begin(), 
                                        for_packed_data_for_this_block_stream.end());
        metadata_out[b].compressed_size_bytes = compressed_byte_stream_out.size() - byte_offset_start_of_this_block_data;
    }
}

// =============================================================================
//                       Serialization & Deserialization Functions
// =============================================================================
template<typename T>
std::vector<uint8_t> serialize_data_typed(
    CompressionAlgorithm algo, DataTypeId dtype_id, uint64_t total_original_values,
    const std::vector<uint8_t>& h_compressed_byte_stream, 
    const std::vector<BlockMetadata<T>>& h_block_metadata,
    const std::vector<DFORMetadata<T>>& h_dfor_metadata, 
    const std::vector<RFORMetadata>& h_rfor_metadata 
) { 
    SerializedDataHeader header = {};
    header.magic = GPCC_MAGIC; 
    header.version = GPCC_VERSION;
    header.algorithm_type = static_cast<uint32_t>(algo); 
    header.data_type_id = static_cast<uint32_t>(dtype_id);
    header.total_original_values = total_original_values; 
    header.num_blocks = h_block_metadata.size();
    header.block_metadata_entry_size = sizeof(BlockMetadata<T>);
    header.block_metadata_size_bytes = h_block_metadata.size() * sizeof(BlockMetadata<T>);
    header.algo_specific_metadata_size_bytes = 0;
    
    if (algo == ALGO_DFOR) {
        header.algo_specific_metadata_entry_size = sizeof(DFORMetadata<T>);
        header.algo_specific_metadata_size_bytes = h_dfor_metadata.size() * sizeof(DFORMetadata<T>);
    } else if (algo == ALGO_RFOR) {
        header.algo_specific_metadata_entry_size = sizeof(RFORMetadata);
        header.algo_specific_metadata_size_bytes = h_rfor_metadata.size() * sizeof(RFORMetadata);
    }
    
    header.compressed_data_stream_size_bytes = h_compressed_byte_stream.size();
    size_t current_offset = sizeof(SerializedDataHeader);
    header.block_metadata_offset = current_offset; 
    current_offset += header.block_metadata_size_bytes;
    header.algo_specific_metadata_offset = current_offset; 
    current_offset += header.algo_specific_metadata_size_bytes;
    header.compressed_data_stream_offset = current_offset; 
    current_offset += header.compressed_data_stream_size_bytes;
    
    std::vector<uint8_t> serialized_buffer(current_offset);
    memcpy(serialized_buffer.data(), &header, sizeof(SerializedDataHeader));
    if (!h_block_metadata.empty()) {
        memcpy(serialized_buffer.data() + header.block_metadata_offset, 
               h_block_metadata.data(), header.block_metadata_size_bytes);
    }
    if (algo == ALGO_DFOR && !h_dfor_metadata.empty()) {
        memcpy(serialized_buffer.data() + header.algo_specific_metadata_offset, 
               h_dfor_metadata.data(), header.algo_specific_metadata_size_bytes);
    } else if (algo == ALGO_RFOR && !h_rfor_metadata.empty()) {
        memcpy(serialized_buffer.data() + header.algo_specific_metadata_offset, 
               h_rfor_metadata.data(), header.algo_specific_metadata_size_bytes);
    }
    if (!h_compressed_byte_stream.empty()) {
        memcpy(serialized_buffer.data() + header.compressed_data_stream_offset, 
               h_compressed_byte_stream.data(), header.compressed_data_stream_size_bytes);
    }
    return serialized_buffer;
}

template<typename T>
bool deserialize_data_typed(
    const std::vector<uint8_t>& serialized_buffer, SerializedDataHeader& header, 
    CompressionAlgorithm& algo, DataTypeId& dtype_id, uint64_t& total_original_values,
    std::vector<uint8_t>& h_compressed_byte_stream, std::vector<BlockMetadata<T>>& h_block_metadata,
    std::vector<DFORMetadata<T>>& h_dfor_metadata, std::vector<RFORMetadata>& h_rfor_metadata
) { 
    if (serialized_buffer.size() < sizeof(SerializedDataHeader)) return false;
    memcpy(&header, serialized_buffer.data(), sizeof(SerializedDataHeader));
    if (header.magic != GPCC_MAGIC) return false;
    if (header.version != GPCC_VERSION) {
        std::cerr << "Warning: Version mismatch. File version: " << header.version 
                  << ", Expected: " << GPCC_VERSION << std::endl;
        // Allow older versions for backward compatibility
        if (header.version > GPCC_VERSION) return false;
    }
    
    algo = static_cast<CompressionAlgorithm>(header.algorithm_type);
    dtype_id = static_cast<DataTypeId>(header.data_type_id);
    total_original_values = header.total_original_values;
    if (header.block_metadata_entry_size != sizeof(BlockMetadata<T>)) return false;
    
    h_block_metadata.resize(header.num_blocks);
    if (header.block_metadata_size_bytes != h_block_metadata.size() * sizeof(BlockMetadata<T>)) return false;
    if (header.num_blocks > 0) {
        memcpy(h_block_metadata.data(), serialized_buffer.data() + header.block_metadata_offset, 
               header.block_metadata_size_bytes);
    }
    
    if (algo == ALGO_DFOR) {
        if (header.algo_specific_metadata_entry_size != sizeof(DFORMetadata<T>)) return false;
        h_dfor_metadata.resize(header.num_blocks);
        if (header.algo_specific_metadata_size_bytes != h_dfor_metadata.size() * sizeof(DFORMetadata<T>)) return false;
        if (header.num_blocks > 0) {
            memcpy(h_dfor_metadata.data(), serialized_buffer.data() + header.algo_specific_metadata_offset, 
                   header.algo_specific_metadata_size_bytes);
        }
    } else { 
        h_dfor_metadata.clear(); 
    }
    
    if (algo == ALGO_RFOR) {
        if (header.algo_specific_metadata_entry_size != sizeof(RFORMetadata)) return false;
        h_rfor_metadata.resize(header.num_blocks);
        if (header.algo_specific_metadata_size_bytes != h_rfor_metadata.size() * sizeof(RFORMetadata)) return false;
        if (header.num_blocks > 0) {
            memcpy(h_rfor_metadata.data(), serialized_buffer.data() + header.algo_specific_metadata_offset, 
                   header.algo_specific_metadata_size_bytes);
        }
    } else { 
        h_rfor_metadata.clear(); 
    }
    
    h_compressed_byte_stream.resize(header.compressed_data_stream_size_bytes);
    if (header.compressed_data_stream_size_bytes > 0) {
        memcpy(h_compressed_byte_stream.data(), serialized_buffer.data() + header.compressed_data_stream_offset, 
               header.compressed_data_stream_size_bytes);
    }
    return true;
}

// =============================================================================
//                       File I/O Utilities
// =============================================================================
class FileReader {
public:
    template<typename T_READ>
    static bool readTextFileGeneric(const std::string& filename, std::vector<T_READ>& data) { 
        std::ifstream file(filename);
        if (!file.is_open()) { 
            std::cerr << "Error: Unable to open file: " << filename << std::endl; 
            return false; 
        }
        data.clear(); 
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line); 
            T_READ value;
            while (iss >> value) { 
                data.push_back(value); 
                char c = 0; 
                iss.get(c); 
                if (c != ',' && c != ' ' && c != '\t' && !iss.eof()) iss.unget(); 
            }
        }
        file.close(); 
        return true;
    }
    
    template<typename T_READ>
    static bool readRawBinaryFileGeneric(const std::string& filename, std::vector<T_READ>& data) { 
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) { return false; }
        std::streamsize size_bytes = file.tellg(); 
        file.seekg(0, std::ios::beg);
        if (!file.good() || size_bytes < 0) { 
            file.close(); 
            return false; 
        }
        if (size_bytes > 0 && (static_cast<size_t>(size_bytes) % sizeof(T_READ) == 0)) {
            data.resize(static_cast<size_t>(size_bytes) / sizeof(T_READ));
            if (!data.empty()) { 
                file.read(reinterpret_cast<char*>(data.data()), size_bytes); 
                if(file.gcount() != size_bytes) data.resize(file.gcount() / sizeof(T_READ));
            }
        } else if (size_bytes == 0) { 
            data.clear(); 
        } else { 
            data.clear(); 
            file.close(); 
            return false;
        }
        file.close(); 
        return true;
    }
    
    static bool readSerializedGpccFile(const std::string& filename, std::vector<uint8_t>& data) { 
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) { return false; }
        std::streamsize size_bytes = file.tellg(); 
        file.seekg(0, std::ios::beg);
        if (!file.good() || size_bytes < 0) { 
            file.close(); 
            return false; 
        }
        if (size_bytes > 0) { 
            data.resize(static_cast<size_t>(size_bytes)); 
            if (!data.empty()) { 
                file.read(reinterpret_cast<char*>(data.data()), size_bytes); 
                if(file.gcount() != size_bytes) data.resize(file.gcount()); 
            }
        } else { 
            data.clear(); 
        } 
        file.close(); 
        return true;
    }
};

class FileWriter {
public:
    template<typename T_WRITE>
    static bool writeTextFileGeneric(const std::string& filename, const std::vector<T_WRITE>& data) { 
        std::ofstream file(filename); 
        if (!file.is_open()) return false;
        for (size_t i = 0; i < data.size(); i++) { 
            file << data[i] << (i == data.size() - 1 ? "" : "\n");
        } 
        file.close(); 
        return true;
    }
    
    template<typename T_WRITE>
    static bool writeRawBinaryFileGeneric(const std::string& filename, const std::vector<T_WRITE>& data) { 
        std::ofstream file(filename, std::ios::binary); 
        if (!file.is_open()) return false;
        if (!data.empty()) {
            file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T_WRITE)); 
        }
        file.close(); 
        return true;
    }
    
    static bool writeSerializedGpccFile(const std::string& filename, const std::vector<uint8_t>& data) { 
        std::ofstream file(filename, std::ios::binary); 
        if (!file.is_open()) return false;
        if (!data.empty()) {
            file.write(reinterpret_cast<const char*>(data.data()), data.size()); 
        }
        file.close(); 
        return true;
    }
};

// =============================================================================
//                       Performance Testing Framework
// =============================================================================
class PerformanceTester {
private: 
    cudaEvent_t start_event, stop_event; 
    std::vector<float> timings;
public:
    PerformanceTester() { 
        CUDA_CHECK(cudaEventCreate(&start_event)); 
        CUDA_CHECK(cudaEventCreate(&stop_event)); 
    }
    ~PerformanceTester() { 
        cudaEventDestroy(start_event); 
        cudaEventDestroy(stop_event); 
    }
    void startTimer() { CUDA_CHECK(cudaEventRecord(start_event)); }
    void stopTimer() { 
        CUDA_CHECK(cudaEventRecord(stop_event)); 
        CUDA_CHECK(cudaEventSynchronize(stop_event)); 
    }
    float getElapsedTime() { 
        float ms = 0; 
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event)); 
        return ms; 
    }
    void recordTiming(float time) { timings.push_back(time); }
    void clearTimings() { timings.clear(); }
    float getAverageTime() const { 
        if (timings.empty()) return 0; 
        float sum = 0; 
        for (float t : timings) sum += t; 
        return sum / timings.size(); 
    }
    void printStats(const std::string& name) const { 
        if (timings.empty()) { 
            std::cout << name << " Performance Statistics: No timings recorded." << std::endl; 
            return; 
        }
        float avg = getAverageTime(); 
        float min_t = timings[0], max_t = timings[0];
        for(float t : timings) { 
            if (t < min_t) min_t = t; 
            if (t > max_t) max_t = t;
        }
        std::cout << name << " Performance Statistics:" << std::endl;
        std::cout << "  Average: " << avg << " ms, Min: " << min_t << " ms, Max: " << max_t 
                  << " ms (" << timings.size() << " runs)" << std::endl;
    }
};

// =============================================================================
//                       Comprehensive Test Function (Templated)
// =============================================================================
template<typename T>
void testCompressionAlgorithmTyped(
    CompressionAlgorithm algo, DataTypeId dtype_id, const std::vector<T>& h_original_data,
    const std::string& algo_name, const std::string& dtype_name,
    std::vector<uint8_t>& out_serialized_data_for_saving,
    bool use_gpu_serialization = true  // New parameter to control GPU serialization
) {
    std::cout << "\n=== Testing " << algo_name << " for data type " << dtype_name << " ===" << std::endl;
    out_serialized_data_for_saving.clear();
    uint64_t N = h_original_data.size();
    if (N == 0) { 
        std::cout << "  No data to test. Skipping." << std::endl; 
        return; 
    }
    PerformanceTester perf_tester;
    
    // Start timing for compression (including serialization)
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<uint8_t> h_compressed_byte_stream_cpu;
    std::vector<BlockMetadata<T>> h_metadata_cpu;
    std::vector<DFORMetadata<T>> h_dfor_metadata_cpu;
    std::vector<RFORMetadata> h_rfor_metadata_cpu;

    // Compression
    switch (algo) {
        case ALGO_FOR: 
            compress_for_cpu<T>(h_original_data.data(), N, h_compressed_byte_stream_cpu, h_metadata_cpu); 
            break;
        case ALGO_DFOR: 
            compress_dfor_cpu<T>(h_original_data.data(), N, h_compressed_byte_stream_cpu, 
                               h_metadata_cpu, h_dfor_metadata_cpu); 
            break;
        case ALGO_RFOR: 
            compress_rfor_cpu<T>(h_original_data.data(), N, h_compressed_byte_stream_cpu, 
                               h_metadata_cpu, h_rfor_metadata_cpu); 
            break;
        default: 
            std::cerr << "Unsupported algo for CPU compression." << std::endl; 
            return;
    }
    
    if (algo == ALGO_DFOR && !h_dfor_metadata_cpu.empty()) { 
        std::cout << "  DEBUG CPU DFOR Metadata Block 0: first_value = " << h_dfor_metadata_cpu[0].first_value
                  << ", reference_delta = " << h_dfor_metadata_cpu[0].reference_delta << std::endl;
    }
    
    // Serialization (use GPU or CPU based on parameter)
    if (use_gpu_serialization) {
        std::cout << "  Using GPU serialization..." << std::endl;
        out_serialized_data_for_saving = serialize_data_gpu<T>(algo, dtype_id, N, h_compressed_byte_stream_cpu, 
                                                              h_metadata_cpu, h_dfor_metadata_cpu, h_rfor_metadata_cpu);
    } else {
        std::cout << "  Using CPU serialization..." << std::endl;
        out_serialized_data_for_saving = serialize_data_typed<T>(algo, dtype_id, N, h_compressed_byte_stream_cpu, 
                                                               h_metadata_cpu, h_dfor_metadata_cpu, h_rfor_metadata_cpu);
    }
    
    // End timing for compression + serialization
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    auto cpu_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end_time - cpu_start_time).count();
    
    // Calculate compression statistics
    CompressionStats stats; 
    stats.original_size_bytes = N * sizeof(T);
    stats.compressed_total_size_bytes = out_serialized_data_for_saving.size(); 
    stats.compression_ratio = stats.compressed_total_size_bytes > 0 ? 
                            (double)stats.original_size_bytes / stats.compressed_total_size_bytes : 0;
    stats.compression_time_ms = cpu_duration_ms; 
    stats.blocks_processed = h_metadata_cpu.size();
    
    std::cout << "Compression Statistics (CPU):" << std::endl; 
    std::cout << "  Original size: " << stats.original_size_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "  Total Serialized size (header+meta+data): " 
              << stats.compressed_total_size_bytes / (1024.0*1024.0) << " MB" << std::endl;
    std::cout << "  Compression ratio (vs total serialized): " << stats.compression_ratio << "x" << std::endl;
    std::cout << "  CPU Compression time (including serialization): " << stats.compression_time_ms << " ms" << std::endl;

    // Start timing for deserialization
    auto deserialize_start_time = std::chrono::high_resolution_clock::now();
    
    SerializedDataHeader loaded_header; 
    CompressionAlgorithm loaded_algo; 
    DataTypeId loaded_dtype; 
    uint64_t loaded_N;
    std::vector<uint8_t> h_compressed_loaded_stream; 
    std::vector<BlockMetadata<T>> h_metadata_loaded;
    std::vector<DFORMetadata<T>> h_dfor_metadata_loaded; 
    std::vector<RFORMetadata> h_rfor_metadata_loaded;
    
    bool deserialization_success;
    if (use_gpu_serialization) {
        std::cout << "  Using GPU deserialization..." << std::endl;
        deserialization_success = deserialize_data_gpu<T>(out_serialized_data_for_saving, loaded_header, loaded_algo, 
                                                         loaded_dtype, loaded_N, h_compressed_loaded_stream, h_metadata_loaded, 
                                                         h_dfor_metadata_loaded, h_rfor_metadata_loaded);
    } else {
        std::cout << "  Using CPU deserialization..." << std::endl;
        deserialization_success = deserialize_data_typed<T>(out_serialized_data_for_saving, loaded_header, loaded_algo, 
                                                           loaded_dtype, loaded_N, h_compressed_loaded_stream, h_metadata_loaded, 
                                                           h_dfor_metadata_loaded, h_rfor_metadata_loaded);
    }
    
    if (!deserialization_success) {
        std::cerr << "Deserialization FAILED!" << std::endl; 
        return;
    }
    
    // End timing for deserialization
    auto deserialize_end_time = std::chrono::high_resolution_clock::now();
    auto deserialize_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    deserialize_end_time - deserialize_start_time).count();
    
    if (loaded_algo == ALGO_DFOR && !h_dfor_metadata_loaded.empty()) { 
        std::cout << "  DEBUG Loaded DFOR Metadata Block 0: first_value = " << h_dfor_metadata_loaded[0].first_value
                  << ", reference_delta = " << h_dfor_metadata_loaded[0].reference_delta << std::endl;
    }
    
    // Integrity check
    bool integrity_ok = (loaded_algo == algo && loaded_dtype == dtype_id && loaded_N == N &&
                         h_compressed_loaded_stream.size() == h_compressed_byte_stream_cpu.size() &&
                         (h_compressed_loaded_stream.empty() || 
                          memcmp(h_compressed_loaded_stream.data(), h_compressed_byte_stream_cpu.data(), 
                                h_compressed_byte_stream_cpu.size()) == 0) &&
                         h_metadata_loaded.size() == h_metadata_cpu.size() &&
                         (h_metadata_loaded.empty() || 
                          memcmp(h_metadata_loaded.data(), h_metadata_cpu.data(), 
                                h_metadata_cpu.size() * sizeof(BlockMetadata<T>)) == 0) );
    if (algo == ALGO_DFOR) {
        integrity_ok = integrity_ok && (h_dfor_metadata_loaded.size() == h_dfor_metadata_cpu.size()) && 
                      (h_dfor_metadata_loaded.empty() || 
                       memcmp(h_dfor_metadata_loaded.data(), h_dfor_metadata_cpu.data(), 
                             h_dfor_metadata_cpu.size() * sizeof(DFORMetadata<T>)) == 0);
    }
    if (algo == ALGO_RFOR) {
        integrity_ok = integrity_ok && (h_rfor_metadata_loaded.size() == h_rfor_metadata_cpu.size()) && 
                      (h_rfor_metadata_loaded.empty() || 
                       memcmp(h_rfor_metadata_loaded.data(), h_rfor_metadata_cpu.data(), 
                             h_rfor_metadata_cpu.size() * sizeof(RFORMetadata)) == 0);
    }
    std::cout << "  Serialization/Deserialization Integrity Check: " << (integrity_ok ? "PASSED" : "FAILED") << std::endl;
    if (!integrity_ok) { return; }

    // GPU memory allocation and data transfer
    uint8_t *d_compressed_stream; 
    BlockMetadata<T> *d_block_metadata;
    DFORMetadata<T> *d_dfor_metadata = nullptr; 
    RFORMetadata *d_rfor_metadata_device = nullptr; 
    T *d_output;
    
    CUDA_CHECK(cudaMalloc(&d_compressed_stream, h_compressed_loaded_stream.size() > 0 ? 
                         h_compressed_loaded_stream.size() : 1));
    CUDA_CHECK(cudaMalloc(&d_block_metadata, h_metadata_loaded.size() > 0 ? 
                         h_metadata_loaded.size() * sizeof(BlockMetadata<T>) : 1));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(T)));
    
    if (!h_compressed_loaded_stream.empty()) 
        CUDA_CHECK(cudaMemcpy(d_compressed_stream, h_compressed_loaded_stream.data(), 
                            h_compressed_loaded_stream.size(), cudaMemcpyHostToDevice));
    if(!h_metadata_loaded.empty()) 
        CUDA_CHECK(cudaMemcpy(d_block_metadata, h_metadata_loaded.data(), 
                            h_metadata_loaded.size() * sizeof(BlockMetadata<T>), cudaMemcpyHostToDevice));
    
    if (algo == ALGO_DFOR) {
        CUDA_CHECK(cudaMalloc(&d_dfor_metadata, h_dfor_metadata_loaded.size() > 0 ? 
                            h_dfor_metadata_loaded.size() * sizeof(DFORMetadata<T>) : 1));
        if(!h_dfor_metadata_loaded.empty()) 
            CUDA_CHECK(cudaMemcpy(d_dfor_metadata, h_dfor_metadata_loaded.data(), 
                                h_dfor_metadata_loaded.size() * sizeof(DFORMetadata<T>), cudaMemcpyHostToDevice));
    } else if (algo == ALGO_RFOR) {
        CUDA_CHECK(cudaMalloc(&d_rfor_metadata_device, h_rfor_metadata_loaded.size() > 0 ? 
                            h_rfor_metadata_loaded.size() * sizeof(RFORMetadata) : 1));
        if(!h_rfor_metadata_loaded.empty()) 
            CUDA_CHECK(cudaMemcpy(d_rfor_metadata_device, h_rfor_metadata_loaded.data(), 
                                h_rfor_metadata_loaded.size() * sizeof(RFORMetadata), cudaMemcpyHostToDevice));
    }
    
    const int BLOCKS_PER_TB_VAL = 1; 
    int num_logical_blocks = h_metadata_loaded.size();
    float avg_decomp_time_ms = 0;
    
    if (num_logical_blocks > 0) {
        dim3 grid_dim((num_logical_blocks + BLOCKS_PER_TB_VAL - 1) / BLOCKS_PER_TB_VAL);
        dim3 block_dim(THREADS_PER_BLOCK); 
        
        // Warmup
        for (int i=0; i<10; ++i) {
            switch(algo) {
                case ALGO_FOR: 
                    decompress_gpu_for_kernel<T, BLOCKS_PER_TB_VAL><<<grid_dim, block_dim>>>(
                        d_compressed_stream, d_block_metadata, d_output, num_logical_blocks); 
                    break;
                case ALGO_DFOR: 
                    decompress_gpu_dfor_kernel<T, BLOCKS_PER_TB_VAL><<<grid_dim, block_dim>>>(
                        d_compressed_stream, d_block_metadata, d_dfor_metadata, d_output, num_logical_blocks, N); 
                    break;
                case ALGO_RFOR: 
                    decompress_gpu_rfor_kernel<T, BLOCKS_PER_TB_VAL><<<grid_dim, block_dim>>>(
                        d_compressed_stream, d_block_metadata, d_rfor_metadata_device, d_output, num_logical_blocks); 
                    break;
                default: break;
            }
        } 
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Timing loop for GPU decompression
        perf_tester.clearTimings(); 
        const int NUM_RUNS = 100;
        for (int run = 0; run < NUM_RUNS; run++) {
            perf_tester.startTimer();
            switch(algo) { 
                case ALGO_FOR: 
                    decompress_gpu_for_kernel<T, BLOCKS_PER_TB_VAL><<<grid_dim, block_dim>>>(
                        d_compressed_stream, d_block_metadata, d_output, num_logical_blocks); 
                    break;
                case ALGO_DFOR: 
                    decompress_gpu_dfor_kernel<T, BLOCKS_PER_TB_VAL><<<grid_dim, block_dim>>>(
                        d_compressed_stream, d_block_metadata, d_dfor_metadata, d_output, num_logical_blocks, N); 
                    break;
                case ALGO_RFOR: 
                    decompress_gpu_rfor_kernel<T, BLOCKS_PER_TB_VAL><<<grid_dim, block_dim>>>(
                        d_compressed_stream, d_block_metadata, d_rfor_metadata_device, d_output, num_logical_blocks); 
                    break;
                default: break;
            }
            perf_tester.stopTimer(); 
            perf_tester.recordTiming(perf_tester.getElapsedTime());
        }
        
        avg_decomp_time_ms = perf_tester.getAverageTime();  // 这里赋值
        // Add deserialization time to get total decompression time
        float total_decomp_time_ms = avg_decomp_time_ms + deserialize_duration_ms;
        stats.decompression_time_ms = total_decomp_time_ms;
        
        double throughput_gbps = (N * sizeof(T)) / (total_decomp_time_ms * 1e-3 * 1e9 + 1e-9); 
        std::cout << "  GPU Decompression (BLOCKS_PER_TB=" << BLOCKS_PER_TB_VAL << "):" << std::endl;
        std::cout << "    Deserialization time: " << deserialize_duration_ms << " ms" << std::endl;
        std::cout << "    GPU kernel time: " << avg_decomp_time_ms << " ms" << std::endl;
        std::cout << "    Total decompression time (deserialize + GPU): " << total_decomp_time_ms << " ms" << std::endl;
        std::cout << "    Throughput (based on total time): " << throughput_gbps << " GB/s" << std::endl;

        // Verification
        std::vector<T> h_output_gpu(N); 
        CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, N * sizeof(T), cudaMemcpyDeviceToHost));
        bool correct = true; 
        int error_count = 0;
        const int start_check = (N > 1000) ? N - 1000 : 0;  // Check last 1000 values
        for (uint64_t i = start_check; i < N; i++) {
            if (h_output_gpu[i] != h_original_data[i]) { 
                if (error_count < 10) 
                    std::cerr << "    Verification Error at index " << i << ": Expected " 
                              << h_original_data[i] << ", Got " << h_output_gpu[i] 
                              << " (diff: " << (int64_t)h_output_gpu[i] - (int64_t)h_original_data[i] << ")" << std::endl; 
                error_count++; 
                correct = false; 
            }
        }
        std::cout << "    Verification: " << (correct ? "PASSED" : "FAILED") 
                  << (error_count > 0 ? " (" + std::to_string(error_count) + " errors)" : "") << std::endl;
    } else {
        // 如果没有块要处理，设置默认值
        avg_decomp_time_ms = 0.001f;  // 设置一个小的默认值避免除零
    }

    // Random access test - Enhanced to match paper's description
    if (N > 0 && num_logical_blocks > 0) { 
        std::cout << "\n  Random Access Test:" << std::endl;
        std::vector<uint32_t> h_query_indices(NUM_RANDOM_QUERIES); 
        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<uint32_t> distrib(0, N > 0 ? N -1 : 0); 
        for(int i=0; i<NUM_RANDOM_QUERIES; ++i) 
            h_query_indices[i] = (N > 0 ? distrib(gen) : 0);

        uint32_t *d_query_indices; 
        T *d_query_output_rand; 
        CUDA_CHECK(cudaMalloc(&d_query_indices, NUM_RANDOM_QUERIES * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_query_output_rand, NUM_RANDOM_QUERIES * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_query_indices, h_query_indices.data(), 
                            NUM_RANDOM_QUERIES * sizeof(uint32_t), cudaMemcpyHostToDevice));
        
        dim3 rand_grid((NUM_RANDOM_QUERIES + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK); 
        dim3 rand_block(THREADS_PER_BLOCK);
        
        // 先测试原始的实现
        std::cout << "    Testing original (optimized) implementation:" << std::endl;
        perf_tester.clearTimings();
        for (int run=0; run<100; ++run) { 
            perf_tester.startTimer();
            decompress_random_access_kernel<T><<<rand_grid, rand_block>>>(
                d_compressed_stream, d_block_metadata, d_dfor_metadata, d_rfor_metadata_device, 
                d_query_indices, d_query_output_rand, NUM_RANDOM_QUERIES, algo, N);
            perf_tester.stopTimer(); 
            perf_tester.recordTiming(perf_tester.getElapsedTime());
        }
        float fast_avg_time = perf_tester.getAverageTime();
        std::cout << "      Average time: " << fast_avg_time << " ms" << std::endl;
        std::cout << "      QPS: " << NUM_RANDOM_QUERIES / (fast_avg_time * 1e-3 + 1e-9) << std::endl;
        
        // 测试按块解压的实现
        std::cout << "    Testing block-based (realistic) implementation:" << std::endl;
        
        // 使用较小的线程块，因为每个线程要做更多工作
        dim3 block_grid((NUM_RANDOM_QUERIES + 32 - 1) / 32); 
        dim3 block_block(32);
        
        perf_tester.clearTimings();
        for (int run=0; run<100; ++run) { 
            perf_tester.startTimer();
            decompress_random_access_block_based_kernel<T><<<block_grid, block_block>>>(
                d_compressed_stream, d_block_metadata, d_dfor_metadata, d_rfor_metadata_device, 
                d_query_indices, d_query_output_rand, NUM_RANDOM_QUERIES, algo, N);
            perf_tester.stopTimer(); 
            perf_tester.recordTiming(perf_tester.getElapsedTime());
        }
        
        float rand_avg_time = perf_tester.getAverageTime();
        std::cout << "      Average time: " << rand_avg_time << " ms for " << NUM_RANDOM_QUERIES << " queries." << std::endl;
        std::cout << "      QPS: " << NUM_RANDOM_QUERIES / (rand_avg_time * 1e-3 + 1e-9) << std::endl;
        std::cout << "      Slowdown vs optimized impl: " << rand_avg_time / fast_avg_time << "x" << std::endl;
        
        // Calculate and display selectivity-based performance
        std::set<uint32_t> unique_blocks;
        for (uint32_t idx : h_query_indices) {
            unique_blocks.insert(idx / BLOCK_SIZE);
        }
        float selectivity = (float)unique_blocks.size() / num_logical_blocks;
        std::cout << "    Selectivity: " << selectivity << " (" << unique_blocks.size() 
                  << " unique blocks accessed out of " << num_logical_blocks << ")" << std::endl;
        
        // 使用之前计算的平均解压时间
        float avg_decomp_time_per_element = avg_decomp_time_ms / N;  // 每个元素的平均解压时间
        float theoretical_min_time = unique_blocks.size() * avg_decomp_time_per_element * BLOCK_SIZE;
        std::cout << "    Theoretical minimum time (with perfect caching): " << theoretical_min_time << " ms" << std::endl;
        std::cout << "    Cache efficiency factor: " << rand_avg_time / theoretical_min_time << "x" << std::endl;
        
        // 验证结果正确性
        std::vector<T> h_query_output(NUM_RANDOM_QUERIES); 
        CUDA_CHECK(cudaMemcpy(h_query_output.data(), d_query_output_rand, 
                            NUM_RANDOM_QUERIES * sizeof(T), cudaMemcpyDeviceToHost));
        
        bool rand_correct = true; 
        int rand_error_count = 0;
        if (N > 0) { 
            for(int i=0; i<NUM_RANDOM_QUERIES; ++i) {
                if (h_query_output[i] != h_original_data[h_query_indices[i]]) {
                    rand_correct = false; 
                    if(rand_error_count < 5) 
                        std::cerr << "    Rand Access Mismatch qidx " << i << " (orig_idx " 
                                  << h_query_indices[i] << "): Exp " << h_original_data[h_query_indices[i]] 
                                  << " Got " << h_query_output[i] << std::endl; 
                    rand_error_count++;
                }
            }
        } else { 
            rand_correct = false; 
            std::cout << "    Skipping random access verification as N=0." << std::endl;
        } 
        std::cout << "    Random Access Verification: " << (rand_correct ? "PASSED" : "FAILED") 
                  << (rand_error_count > 0 ? " (" + std::to_string(rand_error_count) + " errors)" : "") << std::endl;
        
        CUDA_CHECK(cudaFree(d_query_indices)); 
        CUDA_CHECK(cudaFree(d_query_output_rand));
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_compressed_stream)); 
    CUDA_CHECK(cudaFree(d_block_metadata));
    CUDA_CHECK(cudaFree(d_output));
    if (d_dfor_metadata) CUDA_CHECK(cudaFree(d_dfor_metadata));
    if (d_rfor_metadata_device) CUDA_CHECK(cudaFree(d_rfor_metadata_device));
}

// =============================================================================
//                       Test Data Generation (Templated)
// =============================================================================
template<typename T>
void generateTestDataTyped(
    const std::string& distribution_type, uint64_t size, 
    std::vector<T>& data, uint32_t max_val_param = 0 
) {
    data.resize(size); 
    if (size == 0) return;
    std::mt19937 gen(std::random_device{}()); 
    T max_T = std::numeric_limits<T>::max(); 
    T min_T = std::numeric_limits<T>::min();

    if (distribution_type == "uniform") {
        T upper_b = max_val_param > 0 ? static_cast<T>(max_val_param) : static_cast<T>(65535);
        if (upper_b > max_T) upper_b = max_T; 
        T lower_b = static_cast<T>(0);
        if (std::is_signed<T>::value) { 
            lower_b = -upper_b / 2; 
            upper_b = upper_b / 2; 
            if (lower_b > upper_b) lower_b = upper_b; 
        }
        if (lower_b > upper_b && !std::is_signed<T>::value) upper_b = lower_b; 
        
        // For 64-bit types, use appropriate distribution
        if (sizeof(T) > 4) {
            std::uniform_int_distribution<int64_t> dist(static_cast<int64_t>(lower_b), 
                                                       static_cast<int64_t>(upper_b));
            for (uint64_t i = 0; i < size; i++) data[i] = static_cast<T>(dist(gen));
        } else {
            std::uniform_int_distribution<int32_t> dist(static_cast<int32_t>(lower_b), 
                                                       static_cast<int32_t>(upper_b));
            for (uint64_t i = 0; i < size; i++) data[i] = static_cast<T>(dist(gen));
        }
    } else if (distribution_type == "sorted") {
        T current = static_cast<T>(0);
        uint32_t max_inc = 10;
        if (sizeof(T) == 1) max_inc = 2; 
        std::uniform_int_distribution<uint32_t> increment_dist(1, max_inc); 
        for (uint64_t i = 0; i < size; i++) {
            T increment = static_cast<T>(increment_dist(gen));
            if (increment > 0 && current > max_T - increment) { 
                current = max_T;
            } else if (std::is_signed<T>::value && increment < 0 && current < min_T - increment) {
                current = min_T;
            } else {
                current += increment;
            }
            data[i] = current;
        }
    } else if (distribution_type == "runs") { 
        T val_upper_bound = static_cast<T>(1000); 
        if (val_upper_bound > max_T) val_upper_bound = max_T;
        
        // For 64-bit types, use appropriate distribution
        if (sizeof(T) > 4) {
            std::uniform_int_distribution<int64_t> value_dist(static_cast<int64_t>(0), 
                                                            static_cast<int64_t>(val_upper_bound));
            std::uniform_int_distribution<int> run_length_dist(10, 100); 
            uint64_t i = 0;
            while (i < size) { 
                T value = static_cast<T>(value_dist(gen)); 
                int run_length = std::min(run_length_dist(gen), (int)(size - i)); 
                for (int j = 0; j < run_length; j++) data[i++] = value; 
            }
        } else {
            std::uniform_int_distribution<int32_t> value_dist(static_cast<int32_t>(0), 
                                                            static_cast<int32_t>(val_upper_bound));
            std::uniform_int_distribution<int> run_length_dist(10, 100); 
            uint64_t i = 0;
            while (i < size) { 
                T value = static_cast<T>(value_dist(gen)); 
                int run_length = std::min(run_length_dist(gen), (int)(size - i)); 
                for (int j = 0; j < run_length; j++) data[i++] = value; 
            }
        }
    } else { 
        // For 64-bit types, use appropriate distribution
        if (sizeof(T) > 4) {
            std::uniform_int_distribution<int64_t> dist(static_cast<int64_t>(min_T), 
                                                       static_cast<int64_t>(max_T));
            for (uint64_t i = 0; i < size; i++) data[i] = static_cast<T>(dist(gen));
        } else {
            std::uniform_int_distribution<int32_t> dist(static_cast<int32_t>(min_T), 
                                                       static_cast<int32_t>(max_T));
            for (uint64_t i = 0; i < size; i++) data[i] = static_cast<T>(dist(gen));
        }
    }
}

// =============================================================================
//                               Main Function
// =============================================================================
DataTypeId getDataTypeIdFromString(const std::string& type_str) {
    if (type_str == "int") return TYPE_INT; 
    if (type_str == "uint") return TYPE_UINT;
    if (type_str == "long") return TYPE_LONG; 
    if (type_str == "ulong") return TYPE_ULONG;
    if (type_str == "longlong") return TYPE_LONGLONG; 
    if (type_str == "ulonglong") return TYPE_ULONGLONG;
    return TYPE_UNKNOWN;
}

std::string getStringFromDataTypeId(DataTypeId dtype) {
    switch(dtype) {
        case TYPE_INT: return "int"; 
        case TYPE_UINT: return "unsigned int";
        case TYPE_LONG: return "long"; 
        case TYPE_ULONG: return "unsigned long";
        case TYPE_LONGLONG: return "long long"; 
        case TYPE_ULONGLONG: return "unsigned long long";
        default: return "unknown";
    }
}

template<typename T>
void run_compression_tests_for_type(
    const std::string& input_file, const std::string& output_file, CompressionAlgorithm selected_algorithm,
    const std::string& test_distribution, uint64_t test_size, DataTypeId dtype_id, const std::string& dtype_name,
    bool use_gpu_serialization = true  // New parameter
) { 
    std::vector<T> h_data_typed; 
    std::vector<uint8_t> serialized_data_to_save; 
    
    if (!input_file.empty()) {
        std::cout << "\nReading input file for type " << dtype_name << ": " << input_file << std::endl;
        if (input_file.length() >= 5 && input_file.substr(input_file.length() - 5) == ".gpcc") { 
            std::vector<uint8_t> sis; 
            if (!FileReader::readSerializedGpccFile(input_file, sis)) { 
                std::cerr << "Failed to read .gpcc: " << input_file << std::endl; 
                return; 
            }
            SerializedDataHeader hdr; 
            CompressionAlgorithm la; 
            DataTypeId ldtid; 
            uint64_t lN; 
            std::vector<uint8_t> cc; 
            std::vector<BlockMetadata<T>> bm; 
            std::vector<DFORMetadata<T>> dm; 
            std::vector<RFORMetadata> rm;
            
            // Use GPU deserialization if enabled
            bool deserialize_success;
            if (use_gpu_serialization) {
                deserialize_success = deserialize_data_gpu<T>(sis, hdr, la, ldtid, lN, cc, bm, dm, rm);
            } else {
                deserialize_success = deserialize_data_typed<T>(sis, hdr, la, ldtid, lN, cc, bm, dm, rm);
            }
            
            if (!deserialize_success) { 
                std::cerr << "Failed to deserialize .gpcc for type " << dtype_name << std::endl; 
                return; 
            }
            if (ldtid != dtype_id) { 
                std::cerr << "Error: .gpcc type (" << getStringFromDataTypeId(ldtid) 
                          << ") != requested (" << dtype_name << ")" << std::endl; 
                return; 
            }
            std::cout << "  Deserialized .gpcc. Algo: " << la << ", N: " << lN 
                      << ". Verification needs original data." << std::endl; 
            return; 
        } else { 
            if (!FileReader::readRawBinaryFileGeneric<T>(input_file, h_data_typed)) { 
                std::cout << "  Raw binary read for '" << input_file << "' failed/not applicable for " 
                          << dtype_name << ", trying text..." << std::endl;
                if (!FileReader::readTextFileGeneric<T>(input_file, h_data_typed)) { 
                    std::cerr << "Failed to read input (raw/text) for " << dtype_name 
                              << ": " << input_file << std::endl; 
                    return; 
                }
            } 
            std::cout << "  Loaded " << h_data_typed.size() << " " << dtype_name << " values." << std::endl;
    
    // Debug: Check last few values for large datasets
    if (h_data_typed.size() > 200000000 && h_data_typed.size() < 200000010) {
        std::cout << "  DEBUG: Checking last few values:" << std::endl;
        for (size_t i = h_data_typed.size() - 5; i < h_data_typed.size(); i++) {
            std::cout << "    data[" << i << "] = " << h_data_typed[i] << std::endl;
        }
    }
        }
    } else {
        std::cout << "\nGenerating test data for type " << dtype_name << ":" << std::endl;
        std::cout << "  Distribution: " << test_distribution << ", Size: " << test_size << std::endl;
        generateTestDataTyped<T>(test_distribution, test_size, h_data_typed);
        std::cout << "  Generated " << h_data_typed.size() << " " << dtype_name << " values." << std::endl;
    }
    
    if (h_data_typed.empty() && test_size > 0) { 
        std::cerr << "Data empty for " << dtype_name << ". Exiting." << std::endl; 
        return; 
    }
    if (h_data_typed.empty() && test_size == 0) { 
        std::cout << "Test size 0, skipping for " << dtype_name << "." << std::endl; 
        return; 
    }

    if (selected_algorithm == ALGO_FOR || selected_algorithm == ALGO_NONE) {
        testCompressionAlgorithmTyped<T>(ALGO_FOR, dtype_id, h_data_typed, "GPU-FOR", 
                                       dtype_name, serialized_data_to_save, use_gpu_serialization);
    }
    if (selected_algorithm == ALGO_DFOR || selected_algorithm == ALGO_NONE) {
        testCompressionAlgorithmTyped<T>(ALGO_DFOR, dtype_id, h_data_typed, "GPU-DFOR", 
                                       dtype_name, serialized_data_to_save, use_gpu_serialization);
    }
    if (selected_algorithm == ALGO_RFOR || selected_algorithm == ALGO_NONE) {
        testCompressionAlgorithmTyped<T>(ALGO_RFOR, dtype_id, h_data_typed, "GPU-RFOR", 
                                       dtype_name, serialized_data_to_save, use_gpu_serialization);
    }

    if (!output_file.empty() && !serialized_data_to_save.empty()) { 
        std::cout << "\nSaving last tested (" << getStringFromDataTypeId(dtype_id) 
                  << ", algo " << selected_algorithm << ") to: " << output_file << std::endl;
        if (FileWriter::writeSerializedGpccFile(output_file, serialized_data_to_save)) {
            std::cout << "  Saved to " << output_file << std::endl;
        } else {
            std::cerr << "  Failed to save " << output_file << std::endl;
        }
    } else if (!output_file.empty()) {
        std::cout << "\nOutput file specified (" << output_file 
                  <<"), but no data to save for last algo." << std::endl;
    }
}

int main(int argc, char** argv) { 
    std::ios_base::sync_with_stdio(false); 
    std::cin.tie(NULL);
    std::cout << "GPU Compression Framework (Multi-Type Test with 64-bit Support and GPU Serialization)\n"
              << "=====================================================================================" << std::endl;
    
    std::string input_file = "", output_file = "", test_distribution = "uniform", type_str = "uint";
    CompressionAlgorithm selected_algorithm = ALGO_NONE; 
    bool gen_test_files = false, list_algos = false, use_gpu_serialization = true;
    uint64_t test_size = 1000000; 
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--generate-test-files") {
            gen_test_files = true;
        } else if (arg == "--algo" && i + 1 < argc) { 
            std::string algo_str = argv[++i]; 
            if (algo_str == "FOR") selected_algorithm = ALGO_FOR; 
            else if (algo_str == "DFOR") selected_algorithm = ALGO_DFOR; 
            else if (algo_str == "RFOR") selected_algorithm = ALGO_RFOR; 
            else if (algo_str == "ALL") selected_algorithm = ALGO_NONE; 
            else { 
                std::cerr << "Invalid algo: " << algo_str << std::endl; 
                return 1;
            } 
        } else if (arg == "--dist" && i + 1 < argc) {
            test_distribution = argv[++i];
        } else if (arg == "--size" && i + 1 < argc) {
            test_size = std::stoull(argv[++i]);
        } else if (arg == "--list-algorithms") {
            list_algos = true;
        } else if (arg == "--cpu-serialization") {
            use_gpu_serialization = false;
        } else if (arg == "--gpu-serialization") {
            use_gpu_serialization = true;
        } else if (arg == "--input" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "--dtype" && i + 1 < argc) {
            type_str = argv[++i];
        } else if (input_file.empty() && (arg[0] != '-' || arg.find('.') != std::string::npos)) {
            input_file = arg;
        } else if (!input_file.empty() && output_file.empty() && 
                   (arg[0] != '-' || arg.find('.') != std::string::npos) ) {
            output_file = arg;
        }
    }
    
    if (list_algos) { 
        std::cout << "Algos: FOR, DFOR, RFOR, ALL" << std::endl; 
        return 0; 
    }
    
    int dev_count; 
    CUDA_CHECK(cudaGetDeviceCount(&dev_count)); 
    if (dev_count == 0) { 
        std::cerr << "No CUDA devices!" << std::endl; 
        return 1; 
    }
    
    cudaDeviceProp prop; 
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0)); 
    std::cout << "\nDevice: " << prop.name << " (CC " << prop.major << "." << prop.minor << ")" << std::endl;
    
    DataTypeId dtype_id = getDataTypeIdFromString(type_str);
    if (dtype_id == TYPE_UNKNOWN) { 
        std::cerr << "Unsupported dtype: " << type_str 
                  << ". Use: int, uint, long, ulong, longlong, ulonglong" << std::endl; 
        return 1; 
    }
    std::cout << "Selected data type: " << type_str << " (ID: " << dtype_id << ")" << std::endl;
    std::cout << "Serialization mode: " << (use_gpu_serialization ? "GPU" : "CPU") << std::endl;
    
    if (gen_test_files) { 
        std::cout << "\nGenerating test data files (size: " << test_size 
                  << " elements type: " << type_str << "):" << std::endl;
        std::string gen_filename_base = "test_" + type_str + "_" + test_distribution + "_" 
                                      + std::to_string(test_size);
        switch (dtype_id) {
            case TYPE_INT: { 
                std::vector<int> d; 
                generateTestDataTyped<int>(test_distribution, test_size, d); 
                FileWriter::writeRawBinaryFileGeneric<int>(gen_filename_base + ".bin", d); 
                break; 
            }
            case TYPE_UINT: { 
                std::vector<unsigned int> d; 
                generateTestDataTyped<unsigned int>(test_distribution, test_size, d); 
                FileWriter::writeRawBinaryFileGeneric<unsigned int>(gen_filename_base + ".bin", d); 
                break; 
            }
            case TYPE_LONG: { 
                std::vector<long> d; 
                generateTestDataTyped<long>(test_distribution, test_size, d); 
                FileWriter::writeRawBinaryFileGeneric<long>(gen_filename_base + ".bin", d); 
                break; 
            }
            case TYPE_ULONG: { 
                std::vector<unsigned long> d; 
                generateTestDataTyped<unsigned long>(test_distribution, test_size, d); 
                FileWriter::writeRawBinaryFileGeneric<unsigned long>(gen_filename_base + ".bin", d); 
                break; 
            }
            case TYPE_LONGLONG: { 
                std::vector<long long> d; 
                generateTestDataTyped<long long>(test_distribution, test_size, d); 
                FileWriter::writeRawBinaryFileGeneric<long long>(gen_filename_base + ".bin", d); 
                break; 
            }
            case TYPE_ULONGLONG: { 
                std::vector<unsigned long long> d; 
                generateTestDataTyped<unsigned long long>(test_distribution, test_size, d); 
                FileWriter::writeRawBinaryFileGeneric<unsigned long long>(gen_filename_base + ".bin", d); 
                break; 
            }
            default: break;
        } 
        std::cout << "Generated " << gen_filename_base << ".bin" << std::endl; 
        return 0;
    }
    
    switch (dtype_id) {
        case TYPE_INT:      
            run_compression_tests_for_type<int>(input_file, output_file, selected_algorithm, 
                                              test_distribution, test_size, dtype_id, "int", use_gpu_serialization); 
            break;
        case TYPE_UINT:     
            run_compression_tests_for_type<unsigned int>(input_file, output_file, selected_algorithm, 
                                                        test_distribution, test_size, dtype_id, "unsigned int", use_gpu_serialization); 
            break;
        case TYPE_LONG:     
            run_compression_tests_for_type<long>(input_file, output_file, selected_algorithm, 
                                               test_distribution, test_size, dtype_id, "long", use_gpu_serialization); 
            break;
        case TYPE_ULONG:    
            run_compression_tests_for_type<unsigned long>(input_file, output_file, selected_algorithm, 
                                                         test_distribution, test_size, dtype_id, "unsigned long", use_gpu_serialization); 
            break;
        case TYPE_LONGLONG: 
            run_compression_tests_for_type<long long>(input_file, output_file, selected_algorithm, 
                                                    test_distribution, test_size, dtype_id, "long long", use_gpu_serialization); 
            break;
        case TYPE_ULONGLONG:
            run_compression_tests_for_type<unsigned long long>(input_file, output_file, selected_algorithm, 
                                                              test_distribution, test_size, dtype_id, "unsigned long long", use_gpu_serialization); 
            break;
        default: 
            std::cerr << "Internal error: unknown dtype in main dispatch." << std::endl; 
            return 1;
    }
    
    std::cout << "\nAll tests completed for type " << type_str << "!" << std::endl; 
    return 0;
}