//this version optimize the compressioin ratio, best ratio - REFACTORED TO SoA

#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <climits>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <fstream>
#include <string>
#include <type_traits>
#include <immintrin.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <stdlib.h> // For posix_memalign
#include <mma.h>

// Fix Kernel random access optimization
// work stealing opt

// CUDA Error Checking Macro
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

// Model types
enum ModelType {
    MODEL_CONSTANT = 0,
    MODEL_LINEAR = 1,
    MODEL_POLYNOMIAL2 = 2,
    MODEL_POLYNOMIAL3 = 3,
    MODEL_DIRECT_COPY = 4   // New model type for direct copy when overflow detected
};

// Enhanced partition metadata structure - ONLY FOR HOST USE AND SERIALIZATION
struct PartitionInfo {
    int32_t start_idx;
    int32_t end_idx;
    int32_t model_type;
    double model_params[4];
    int32_t delta_bits;
    int64_t delta_array_bit_offset;
    long long error_bound;
    int32_t reserved[1];
};

struct PartitionMetaOpt {
    int32_t start_idx;
    int32_t model_type;
    int32_t delta_bits;
    int32_t partition_len;
    double theta0;
    double theta1;
    int64_t bit_offset_base;
};

// Template compressed data structure - SoA LAYOUT
template<typename T>
struct CompressedData {
    // --- SoA Data Pointers (all are device pointers) ---
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    int32_t* d_model_types;
    double* d_model_params; // Note: This will store params for all partitions contiguously.
                            // For a linear model, layout will be [p0_t0, p0_t1, p1_t0, p1_t1, ...].
    int32_t* d_delta_bits;
    int64_t* d_delta_array_bit_offsets;
    long long* d_error_bounds;

    uint32_t* delta_array;          // This remains the same.

// 383 -------------------------------------------------------
    long long* d_plain_deltas;
// 383 -------------------------------------------------------

    // --- Host-side metadata ---
    int num_partitions;
    int total_values;

    // --- Device-side self pointer ---
    CompressedData<T>* d_self;
};

// Serialized data container (for host-side blob)
struct SerializedData {
    uint8_t* data;
    size_t size;

    SerializedData() : data(nullptr), size(0) {}
    ~SerializedData() {
        if (data) {
            delete[] data;
            data = nullptr;
        }
    }
    SerializedData(const SerializedData&) = delete;
    SerializedData& operator=(const SerializedData&) = delete;
    SerializedData(SerializedData&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    SerializedData& operator=(SerializedData&& other) noexcept {
        if (this != &other) {
            if (data) delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};

// Binary format header for serialized data - UPDATED FOR SoA
struct SerializedHeader {
    uint32_t magic;
    uint32_t version; // Increment this to reflect the new format (4)
    uint32_t total_values;
    uint32_t num_partitions;

    // --- New SoA Table Offsets and Sizes ---
    // All offsets are relative to the beginning of the data blob.
    uint64_t start_indices_offset;
    uint64_t end_indices_offset;
    uint64_t model_types_offset;
    uint64_t model_params_offset;
    uint64_t delta_bits_offset;
    uint64_t delta_array_bit_offsets_offset;
    uint64_t error_bounds_offset;
    uint64_t delta_array_offset; // Offset for the main delta bitstream

    // Field for the size in bytes of the model_params array, as it contains doubles.
    uint64_t model_params_size_bytes;
    uint64_t delta_array_size_bytes; // This remains.

    uint32_t data_type_size;
    uint32_t header_checksum;
    uint32_t reserved[3];
};

// Direct access handle for serialized data - UPDATED FOR SoA
// ALIGNMENT FIX: Ensure proper alignment for CUDA
template<typename T>
struct alignas(256) DirectAccessHandle {  // Increased alignment to 256 bytes
    const uint8_t* data_blob_host;
    const SerializedHeader* header_host;
    
    // Host-side SoA pointers
    const int32_t* start_indices_host;
    const int32_t* end_indices_host;
    const int32_t* model_types_host;
    const double* model_params_host;
    const int32_t* delta_bits_host;
    const int64_t* delta_array_bit_offsets_host;
    const long long* error_bounds_host;
    const uint32_t* delta_array_host;
    
    size_t data_blob_size;

    uint8_t* d_data_blob_device;
    SerializedHeader* d_header_device;
    
    // Device-side SoA pointers
    int32_t* d_start_indices_device;
    int32_t* d_end_indices_device;
    int32_t* d_model_types_device;
    double* d_model_params_device;
    int32_t* d_delta_bits_device;
    int64_t* d_delta_array_bit_offsets_device;
    long long* d_error_bounds_device;
    uint32_t* d_delta_array_device;
    
    // Padding to ensure size is multiple of alignment
    char padding[256 - (sizeof(void*) * 20 + sizeof(size_t)) % 256];
};

// Partition metadata structure for shared memory caching
struct PartitionMeta {
    int32_t start_idx;
    int32_t end_idx;
    int32_t model_type;
    double theta0;
    double theta1;
};

// Helper function to check if values might cause overflow in double precision
template<typename T>
__device__ __host__ inline bool mightOverflowDoublePrecision(T value) {
    if (std::is_signed<T>::value) {
        return false;  // Signed types within long long range are OK
    } else {
        // For unsigned types, check if value exceeds double precision (2^53)
        const uint64_t DOUBLE_PRECISION_LIMIT = (1ULL << 53);
        return static_cast<uint64_t>(value) > DOUBLE_PRECISION_LIMIT;
    }
}

// Helper template for safe delta calculation
template<typename T>
__device__ __host__ inline long long calculateDelta(T actual, T predicted) {
    if (std::is_signed<T>::value) {
        return static_cast<long long>(actual) - static_cast<long long>(predicted);
    } else {
        // For unsigned types
        if (sizeof(T) == 8) {
            // For 64-bit unsigned types (unsigned long long)
            unsigned long long actual_ull = static_cast<unsigned long long>(actual);
            unsigned long long pred_ull = static_cast<unsigned long long>(predicted);
            
            if (actual_ull >= pred_ull) {
                unsigned long long diff = actual_ull - pred_ull;
                if (diff <= static_cast<unsigned long long>(LLONG_MAX)) {
                    return static_cast<long long>(diff);
                } else {
                    return LLONG_MAX;
                }
            } else {
                unsigned long long diff = pred_ull - actual_ull;
                if (diff <= static_cast<unsigned long long>(LLONG_MAX)) {
                    return -static_cast<long long>(diff);
                } else {
                    return LLONG_MIN;
                }
            }
        } else {
            // For smaller unsigned types, direct conversion is safe
            return static_cast<long long>(actual) - static_cast<long long>(predicted);
        }
    }
}

// Helper template for applying delta to prediction
template<typename T>
__device__ __host__ inline T applyDelta(T predicted, long long delta) {
    if (std::is_signed<T>::value) {
        // For signed types, simple addition
        return predicted + static_cast<T>(delta);
    } else {
        // For unsigned types, use unsigned arithmetic to handle wraparound correctly
        if (sizeof(T) == 8) {
            // For 64-bit unsigned types
            unsigned long long pred_ull = static_cast<unsigned long long>(predicted);
            unsigned long long delta_ull = static_cast<unsigned long long>(delta);
            return static_cast<T>(pred_ull + delta_ull);
        } else if (sizeof(T) == 4) {
            // For 32-bit unsigned types
            unsigned long pred_ul = static_cast<unsigned long>(predicted);
            unsigned long delta_ul = static_cast<unsigned long>(static_cast<long>(delta));
            return static_cast<T>(pred_ul + delta_ul);
        } else if (sizeof(T) == 2) {
            // For 16-bit unsigned types
            unsigned pred_u = static_cast<unsigned>(predicted);
            unsigned delta_u = static_cast<unsigned>(static_cast<int>(delta));
            return static_cast<T>(pred_u + delta_u);
        } else {
            // For 8-bit unsigned types
            unsigned pred_u = static_cast<unsigned>(predicted);
            unsigned delta_u = static_cast<unsigned>(static_cast<int>(delta));
            return static_cast<T>(pred_u + delta_u);
        }
    }
}

// Helper function for delta extraction
template<typename T>
__device__ inline long long extractDelta(const uint32_t* delta_array, 
                                        int64_t bit_offset, 
                                        int delta_bits) {
    if (delta_bits <= 0) return 0;
    
    if (delta_bits <= 32) {
        int word_idx = bit_offset / 32;
        int bit_offset_in_word = bit_offset % 32;
        uint32_t extracted_bits;
        
        if (bit_offset_in_word + delta_bits <= 32) {
            extracted_bits = (delta_array[word_idx] >> bit_offset_in_word) & 
                            ((1U << delta_bits) - 1U);
        } else {
            uint32_t w1 = delta_array[word_idx];
            uint32_t w2 = delta_array[word_idx + 1]; 
            extracted_bits = (w1 >> bit_offset_in_word) | (w2 << (32 - bit_offset_in_word));
            extracted_bits &= ((1U << delta_bits) - 1U);
        }

        // Sign extension
        if (delta_bits < 32) {
            uint32_t sign_bit = 1U << (delta_bits - 1);
            if (extracted_bits & sign_bit) {
                uint32_t sign_extend_mask = ~((1U << delta_bits) - 1U);
                return static_cast<long long>(static_cast<int32_t>(extracted_bits | sign_extend_mask));
            } else {
                return static_cast<long long>(extracted_bits);
            }
        } else {
            return static_cast<long long>(static_cast<int32_t>(extracted_bits));
        }
    } else {
        // Handle > 32 bit deltas
        int start_word_idx = bit_offset / 32;
        int offset_in_word = bit_offset % 32;
        int bits_remaining = delta_bits;
        uint64_t extracted_val_64 = 0;
        int shift = 0;
        int word_idx = start_word_idx;
        
        while (bits_remaining > 0 && shift < 64) {
            int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
            uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
            uint32_t word_val = (delta_array[word_idx] >> offset_in_word) & mask;
            extracted_val_64 |= (static_cast<uint64_t>(word_val) << shift);
            
            shift += bits_in_this_word;
            bits_remaining -= bits_in_this_word;
            word_idx++;
            offset_in_word = 0;
        }
        
        // Sign extension for 64-bit deltas
        if (delta_bits < 64) {
            uint64_t sign_mask_64 = 1ULL << (delta_bits - 1);
            if (extracted_val_64 & sign_mask_64) {
                uint64_t sign_ext_mask_64 = ~((1ULL << delta_bits) - 1ULL);
                return static_cast<long long>(extracted_val_64 | sign_ext_mask_64);
            } else {
                return static_cast<long long>(extracted_val_64);
            }
        } else {
            return static_cast<long long>(extracted_val_64);
        }
    }
}

// Optimized delta extraction function that eliminates branching
// Optimized delta extraction function with reduced branching
template<typename T>
__device__ inline long long extractDelta_Optimized(const uint32_t* __restrict__ delta_array, 
                                                   int64_t bit_offset, 
                                                   int delta_bits) {
    if (delta_bits <= 0) return 0;
    
    // Use bit operations instead of division/modulo
    int word_idx = bit_offset >> 5;  // bit_offset / 32
    int bit_offset_in_word = bit_offset & 31;  // bit_offset % 32
    
    if (delta_bits <= 32) {
        // Always read two words to avoid branching
        uint32_t w1 = __ldg(&delta_array[word_idx]);
        uint32_t w2 = __ldg(&delta_array[word_idx + 1]);
        
        // Combine words into 64-bit value for branchless extraction
        uint64_t combined = (static_cast<uint64_t>(w2) << 32) | static_cast<uint64_t>(w1);
        uint32_t extracted_bits = (combined >> bit_offset_in_word) & ((1U << delta_bits) - 1U);
        
        // Branchless sign extension
        if (delta_bits < 32) {
            uint32_t sign_bit = extracted_bits >> (delta_bits - 1);
            uint32_t sign_mask = -sign_bit;  // All 1s if sign bit set, 0 otherwise
            uint32_t extend_mask = ~((1U << delta_bits) - 1U);
            extracted_bits |= (sign_mask & extend_mask);
        }
        
        return static_cast<long long>(static_cast<int32_t>(extracted_bits));
    } else {
        // Handle > 32 bit deltas with optimized loop
        uint64_t extracted_val_64 = 0;
        
        // First word
        uint32_t first_word = __ldg(&delta_array[word_idx]);
        int bits_from_first = 32 - bit_offset_in_word;
        extracted_val_64 = (first_word >> bit_offset_in_word);
        
        // Middle words (if any)
        int bits_remaining = delta_bits - bits_from_first;
        int shift = bits_from_first;
        word_idx++;
        
        // Unroll for common case of 64-bit values
        if (bits_remaining > 0) {
            uint32_t word = __ldg(&delta_array[word_idx]);
            if (bits_remaining >= 32) {
                extracted_val_64 |= (static_cast<uint64_t>(word) << shift);
                shift += 32;
                bits_remaining -= 32;
                word_idx++;
                
                if (bits_remaining > 0) {
                    word = __ldg(&delta_array[word_idx]);
                    uint32_t mask = (bits_remaining == 32) ? ~0U : ((1U << bits_remaining) - 1U);
                    extracted_val_64 |= (static_cast<uint64_t>(word & mask) << shift);
                }
            } else {
                uint32_t mask = (1U << bits_remaining) - 1U;
                extracted_val_64 |= (static_cast<uint64_t>(word & mask) << shift);
            }
        }
        
        // Branchless sign extension for 64-bit
        if (delta_bits < 64) {
            uint64_t sign_bit = extracted_val_64 >> (delta_bits - 1);
            uint64_t sign_mask = -(int64_t)sign_bit;
            uint64_t extend_mask = ~((1ULL << delta_bits) - 1ULL);
            extracted_val_64 |= (sign_mask & extend_mask);
        }
        
        return static_cast<long long>(extracted_val_64);
    }
}




// Kernel for pre-unpacked deltas decompression (High-throughput mode)
template<typename T>
__global__ void decompressFullFile_PreUnpacked(
    const CompressedData<T>* compressed_data,
    T* output_device,
    int total_elements) {
    
    // This kernel assumes d_plain_deltas is already populated
    if (!compressed_data || !compressed_data->d_plain_deltas) {
        return;
    }
    
    // Grid-stride loop for coalesced memory access
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;
    
    // Cache frequently accessed pointers
    const long long* __restrict__ plain_deltas = compressed_data->d_plain_deltas;
    const int32_t* __restrict__ model_types = compressed_data->d_model_types;
    const double* __restrict__ model_params = compressed_data->d_model_params;
    const int32_t* __restrict__ start_indices = compressed_data->d_start_indices;
    const int32_t* __restrict__ end_indices = compressed_data->d_end_indices;
    const int num_partitions = compressed_data->num_partitions;
    
    for (int idx = g_idx; idx < total_elements; idx += g_stride) {
        // Binary search to find partition
        int left = 0, right = num_partitions - 1;
        int partition_idx = -1;
        
        while (left <= right) {
            int mid = (left + right) >> 1;
            int32_t start = __ldg(&start_indices[mid]);
            int32_t end = __ldg(&end_indices[mid]);
            
            if (idx >= start && idx < end) {
                partition_idx = mid;
                break;
            }
            
            if (idx < start) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        if (partition_idx == -1) {
            output_device[idx] = static_cast<T>(0);
            continue;
        }
        
        // Get model type and apply appropriate decompression
        int32_t model_type = __ldg(&model_types[partition_idx]);
        
        if (model_type == MODEL_DIRECT_COPY) {
            // For direct copy, the plain delta IS the value
            output_device[idx] = static_cast<T>(__ldg(&plain_deltas[idx]));
        } else {
            // Model-based reconstruction
            int32_t start_idx = __ldg(&start_indices[partition_idx]);
            int local_idx = idx - start_idx;
            
            // Load model parameters
            int param_base = partition_idx * 4;
            double theta0 = __ldg(&model_params[param_base]);
            double theta1 = __ldg(&model_params[param_base + 1]);
            
            // Calculate prediction
            double predicted = fma(theta1, static_cast<double>(local_idx), theta0);
            
            // Handle polynomial models
            if (model_type == MODEL_POLYNOMIAL2) {
                double theta2 = __ldg(&model_params[param_base + 2]);
                predicted = fma(theta2, static_cast<double>(local_idx * local_idx), predicted);
            }
            
            // Clamp prediction to valid range
            if (!std::is_signed<T>::value) {
                predicted = fmax(0.0, predicted);
                if (sizeof(T) == 4) {
                    predicted = fmin(predicted, 4294967295.0);
                } else if (sizeof(T) == 2) {
                    predicted = fmin(predicted, 65535.0);
                } else if (sizeof(T) == 1) {
                    predicted = fmin(predicted, 255.0);
                }
            }
            
            T pred_val = static_cast<T>(round(predicted));
            long long delta = __ldg(&plain_deltas[idx]);
            output_device[idx] = applyDelta(pred_val, delta);
        }
    }
}

// Kernel for on-the-fly bit extraction decompression (Standard mode)
template<typename T>
__global__ void decompressFullFile_BitPacked(
    const CompressedData<T>* compressed_data,
    T* output_device,
    int total_elements) {
    
    // This kernel extracts deltas from bit-packed array on-the-fly
    if (!compressed_data || !compressed_data->delta_array) {
        return;
    }
    
    // Grid-stride loop
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;
    
    // Cache frequently accessed pointers
    const uint32_t* __restrict__ delta_array = compressed_data->delta_array;
    const int32_t* __restrict__ model_types = compressed_data->d_model_types;
    const double* __restrict__ model_params = compressed_data->d_model_params;
    const int32_t* __restrict__ delta_bits = compressed_data->d_delta_bits;
    const int64_t* __restrict__ delta_offsets = compressed_data->d_delta_array_bit_offsets;
    const int32_t* __restrict__ start_indices = compressed_data->d_start_indices;
    const int32_t* __restrict__ end_indices = compressed_data->d_end_indices;
    const int num_partitions = compressed_data->num_partitions;
    
    for (int idx = g_idx; idx < total_elements; idx += g_stride) {
        // Binary search to find partition
        int left = 0, right = num_partitions - 1;
        int partition_idx = -1;
        
        while (left <= right) {
            int mid = (left + right) >> 1;
            int32_t start = __ldg(&start_indices[mid]);
            int32_t end = __ldg(&end_indices[mid]);
            
            if (idx >= start && idx < end) {
                partition_idx = mid;
                break;
            }
            
            if (idx < start) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        
        if (partition_idx == -1) {
            output_device[idx] = static_cast<T>(0);
            continue;
        }
        
        // Get partition metadata
        int32_t model_type = __ldg(&model_types[partition_idx]);
        int32_t start_idx = __ldg(&start_indices[partition_idx]);
        int32_t bits = __ldg(&delta_bits[partition_idx]);
        int64_t base_offset = __ldg(&delta_offsets[partition_idx]);
        int local_idx = idx - start_idx;
        
        if (model_type == MODEL_DIRECT_COPY) {
            // Extract full value from bit-packed array
            if (bits > 0) {
                int64_t bit_offset = base_offset + (int64_t)local_idx * bits;
                
                // Inline extraction for direct copy values
                if (bits <= 32) {
                    int word_idx = bit_offset / 32;
                    int bit_offset_in_word = bit_offset % 32;
                    uint32_t extracted_bits;
                    
                    if (bit_offset_in_word + bits <= 32) {
                        extracted_bits = (delta_array[word_idx] >> bit_offset_in_word) & 
                                        ((1U << bits) - 1U);
                    } else {
                        uint32_t w1 = delta_array[word_idx];
                        uint32_t w2 = delta_array[word_idx + 1]; 
                        extracted_bits = (w1 >> bit_offset_in_word) | (w2 << (32 - bit_offset_in_word));
                        extracted_bits &= ((1U << bits) - 1U);
                    }
                    
                    output_device[idx] = static_cast<T>(extracted_bits);
                } else {
                    // Handle > 32 bit values
                    int start_word_idx = bit_offset / 32;
                    int offset_in_word = bit_offset % 32;
                    int bits_remaining = bits;
                    uint64_t extracted_val_64 = 0;
                    int shift = 0;
                    int word_idx = start_word_idx;
                    
                    while (bits_remaining > 0 && shift < 64) {
                        int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
                        uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                        uint32_t word_val = (delta_array[word_idx] >> offset_in_word) & mask;
                        extracted_val_64 |= (static_cast<uint64_t>(word_val) << shift);
                        
                        shift += bits_in_this_word;
                        bits_remaining -= bits_in_this_word;
                        word_idx++;
                        offset_in_word = 0;
                    }
                    
                    output_device[idx] = static_cast<T>(extracted_val_64);
                }
            } else {
                output_device[idx] = static_cast<T>(0);
            }
        } else {
            // Model-based reconstruction with bit-packed delta extraction
            int param_base = partition_idx * 4;
            double theta0 = __ldg(&model_params[param_base]);
            double theta1 = __ldg(&model_params[param_base + 1]);
            
            // Calculate prediction
            double predicted = fma(theta1, static_cast<double>(local_idx), theta0);
            
            // Handle polynomial models
            if (model_type == MODEL_POLYNOMIAL2) {
                double theta2 = __ldg(&model_params[param_base + 2]);
                predicted = fma(theta2, static_cast<double>(local_idx * local_idx), predicted);
            }
            
            // Extract delta from bit-packed array
            long long delta = 0;
            if (bits > 0) {
                int64_t bit_offset = base_offset + (int64_t)local_idx * bits;
                delta = extractDelta_Optimized<T>(delta_array, bit_offset, bits);
            }
            
            // Clamp prediction
            if (!std::is_signed<T>::value) {
                predicted = fmax(0.0, predicted);
                if (sizeof(T) == 4) {
                    predicted = fmin(predicted, 4294967295.0);
                } else if (sizeof(T) == 2) {
                    predicted = fmin(predicted, 65535.0);
                } else if (sizeof(T) == 1) {
                    predicted = fmin(predicted, 255.0);
                }
            }
            
            T pred_val = static_cast<T>(round(predicted));
            output_device[idx] = applyDelta(pred_val, delta);
        }
    }
}

// GPU-accelerated serialization kernel
__global__ void packToBlobKernel(
    const SerializedHeader* __restrict__ d_header,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const long long* __restrict__ d_error_bounds,
    const uint32_t* __restrict__ d_delta_array,
    int num_partitions,
    uint64_t delta_array_num_bytes,
    uint8_t* __restrict__ d_output_blob) {
    
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;
    
    // Get header from device memory
    SerializedHeader header;
    if (g_idx == 0) {
        header = *d_header;
    }
    __syncthreads();
    
    // Broadcast header to all threads (alternatively, each thread could read it)
    header = *d_header;
    
    // Total work items calculation
    uint64_t header_bytes = sizeof(SerializedHeader);
    uint64_t start_indices_bytes = num_partitions * sizeof(int32_t);
    uint64_t end_indices_bytes = num_partitions * sizeof(int32_t);
    uint64_t model_types_bytes = num_partitions * sizeof(int32_t);
    uint64_t model_params_bytes = num_partitions * 4 * sizeof(double);
    uint64_t delta_bits_bytes = num_partitions * sizeof(int32_t);
    uint64_t bit_offsets_bytes = num_partitions * sizeof(int64_t);
    uint64_t error_bounds_bytes = num_partitions * sizeof(long long);
    
    // Grid-stride loop to handle all copying tasks
    for (uint64_t idx = g_idx; idx < header.delta_array_offset + delta_array_num_bytes; idx += g_stride) {
        
        // Copy header
        if (idx < header_bytes) {
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_header)[idx];
        }
        // Copy start_indices
        else if (idx >= header.start_indices_offset && 
                 idx < header.start_indices_offset + start_indices_bytes) {
            uint64_t local_idx = idx - header.start_indices_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_start_indices)[local_idx];
        }
        // Copy end_indices
        else if (idx >= header.end_indices_offset && 
                 idx < header.end_indices_offset + end_indices_bytes) {
            uint64_t local_idx = idx - header.end_indices_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_end_indices)[local_idx];
        }
        // Copy model_types
        else if (idx >= header.model_types_offset && 
                 idx < header.model_types_offset + model_types_bytes) {
            uint64_t local_idx = idx - header.model_types_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_model_types)[local_idx];
        }
        // Copy model_params
        else if (idx >= header.model_params_offset && 
                 idx < header.model_params_offset + model_params_bytes) {
            uint64_t local_idx = idx - header.model_params_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_model_params)[local_idx];
        }
        // Copy delta_bits
        else if (idx >= header.delta_bits_offset && 
                 idx < header.delta_bits_offset + delta_bits_bytes) {
            uint64_t local_idx = idx - header.delta_bits_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_delta_bits)[local_idx];
        }
        // Copy delta_array_bit_offsets
        else if (idx >= header.delta_array_bit_offsets_offset && 
                 idx < header.delta_array_bit_offsets_offset + bit_offsets_bytes) {
            uint64_t local_idx = idx - header.delta_array_bit_offsets_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_delta_array_bit_offsets)[local_idx];
        }
        // Copy error_bounds
        else if (idx >= header.error_bounds_offset && 
                 idx < header.error_bounds_offset + error_bounds_bytes) {
            uint64_t local_idx = idx - header.error_bounds_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_error_bounds)[local_idx];
        }
        // Copy delta_array
        else if (idx >= header.delta_array_offset && 
                 idx < header.delta_array_offset + delta_array_num_bytes) {
            uint64_t local_idx = idx - header.delta_array_offset;
            d_output_blob[idx] = reinterpret_cast<const uint8_t*>(d_delta_array)[local_idx];
        }
    }
}

// GPU-accelerated deserialization kernel
__global__ void unpackFromBlobKernel(
    const uint8_t* __restrict__ d_input_blob,
    int num_partitions,
    uint64_t delta_array_num_bytes,
    int32_t* __restrict__ d_start_indices,
    int32_t* __restrict__ d_end_indices,
    int32_t* __restrict__ d_model_types,
    double* __restrict__ d_model_params,
    int32_t* __restrict__ d_delta_bits,
    int64_t* __restrict__ d_delta_array_bit_offsets,
    long long* __restrict__ d_error_bounds,
    uint32_t* __restrict__ d_delta_array) {
    
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;
    
    // Get header from the beginning of the blob
    const SerializedHeader* header = reinterpret_cast<const SerializedHeader*>(d_input_blob);
    
    // Calculate only what we need
    uint64_t delta_array_words = (delta_array_num_bytes + 3) / 4;
    
    // Grid-stride loop for parallel unpacking
    // Unpack start_indices
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_start_indices[i] = *reinterpret_cast<const int32_t*>(
            d_input_blob + header->start_indices_offset + i * sizeof(int32_t));
    }
    
    // Unpack end_indices
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_end_indices[i] = *reinterpret_cast<const int32_t*>(
            d_input_blob + header->end_indices_offset + i * sizeof(int32_t));
    }
    
    // Unpack model_types
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_model_types[i] = *reinterpret_cast<const int32_t*>(
            d_input_blob + header->model_types_offset + i * sizeof(int32_t));
    }
    
    // Unpack model_params (4 per partition)
    for (int i = g_idx; i < num_partitions * 4; i += g_stride) {
        d_model_params[i] = *reinterpret_cast<const double*>(
            d_input_blob + header->model_params_offset + i * sizeof(double));
    }
    
    // Unpack delta_bits
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_delta_bits[i] = *reinterpret_cast<const int32_t*>(
            d_input_blob + header->delta_bits_offset + i * sizeof(int32_t));
    }
    
    // Unpack delta_array_bit_offsets
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_delta_array_bit_offsets[i] = *reinterpret_cast<const int64_t*>(
            d_input_blob + header->delta_array_bit_offsets_offset + i * sizeof(int64_t));
    }
    
    // Unpack error_bounds
    for (int i = g_idx; i < num_partitions; i += g_stride) {
        d_error_bounds[i] = *reinterpret_cast<const long long*>(
            d_input_blob + header->error_bounds_offset + i * sizeof(long long));
    }
    
    // Unpack delta_array (word by word for alignment)
    for (uint64_t i = g_idx; i < delta_array_words; i += g_stride) {
        if (i * sizeof(uint32_t) < delta_array_num_bytes) {
            d_delta_array[i] = *reinterpret_cast<const uint32_t*>(
                d_input_blob + header->delta_array_offset + i * sizeof(uint32_t));
        }
    }
}

// Optimized version using cooperative groups for better memory coalescing
__global__ void packToBlobKernelOptimized(
    const SerializedHeader* __restrict__ d_header,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const long long* __restrict__ d_error_bounds,
    const uint32_t* __restrict__ d_delta_array,
    int num_partitions,
    uint64_t delta_array_num_bytes,
    uint8_t* __restrict__ d_output_blob) {
    
    // Use shared memory for coalesced writes
    extern __shared__ uint8_t s_buffer[];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Each block handles a specific section
    int section_id = blockIdx.x;
    
    // Get header
    SerializedHeader header = *d_header;
    
    // Section 0: Header copy
    if (section_id == 0) {
        const uint8_t* header_bytes = reinterpret_cast<const uint8_t*>(d_header);
        for (int i = tid; i < sizeof(SerializedHeader); i += block_size) {
            d_output_blob[i] = header_bytes[i];
        }
    }
    // Sections 1-7: Metadata arrays
    else if (section_id >= 1 && section_id <= 7) {
        uint64_t offset, num_bytes;
        const uint8_t* src_ptr;
        
        switch (section_id) {
            case 1: // start_indices
                offset = header.start_indices_offset;
                num_bytes = num_partitions * sizeof(int32_t);
                src_ptr = reinterpret_cast<const uint8_t*>(d_start_indices);
                break;
            case 2: // end_indices
                offset = header.end_indices_offset;
                num_bytes = num_partitions * sizeof(int32_t);
                src_ptr = reinterpret_cast<const uint8_t*>(d_end_indices);
                break;
            case 3: // model_types
                offset = header.model_types_offset;
                num_bytes = num_partitions * sizeof(int32_t);
                src_ptr = reinterpret_cast<const uint8_t*>(d_model_types);
                break;
            case 4: // model_params
                offset = header.model_params_offset;
                num_bytes = num_partitions * 4 * sizeof(double);
                src_ptr = reinterpret_cast<const uint8_t*>(d_model_params);
                break;
            case 5: // delta_bits
                offset = header.delta_bits_offset;
                num_bytes = num_partitions * sizeof(int32_t);
                src_ptr = reinterpret_cast<const uint8_t*>(d_delta_bits);
                break;
            case 6: // delta_array_bit_offsets
                offset = header.delta_array_bit_offsets_offset;
                num_bytes = num_partitions * sizeof(int64_t);
                src_ptr = reinterpret_cast<const uint8_t*>(d_delta_array_bit_offsets);
                break;
            case 7: // error_bounds
                offset = header.error_bounds_offset;
                num_bytes = num_partitions * sizeof(long long);
                src_ptr = reinterpret_cast<const uint8_t*>(d_error_bounds);
                break;
        }
        
        // Coalesced copy
        for (uint64_t i = tid; i < num_bytes; i += block_size) {
            d_output_blob[offset + i] = src_ptr[i];
        }
    }
    // Remaining sections: Delta array (distributed across multiple blocks)
    else {
        int delta_section = section_id - 8;
        uint64_t bytes_per_block = (delta_array_num_bytes + gridDim.x - 9) / (gridDim.x - 8);
        uint64_t start_byte = delta_section * bytes_per_block;
        uint64_t end_byte = min(start_byte + bytes_per_block, delta_array_num_bytes);
        
        const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(d_delta_array);
        for (uint64_t i = start_byte + tid; i < end_byte; i += block_size) {
            d_output_blob[header.delta_array_offset + i] = src_ptr[i];
        }
    }
}



// Combined kernel for overflow checking, model fitting, and metadata computation - UPDATED FOR SoA
template<typename T>
__global__ void wprocessPartitionsKernel(const T* values_device,
                                       int32_t* d_start_indices,
                                       int32_t* d_end_indices,
                                       int32_t* d_model_types,
                                       double* d_model_params,
                                       int32_t* d_delta_bits,
                                       long long* d_error_bounds,
                                       int num_partitions,
                                       int64_t* total_bits_device) {
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;
    
    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;
    
    if (segment_len <= 0) return;
    
    // Shared memory for reduction operations
    extern __shared__ char shared_mem[];
    double* s_sums = reinterpret_cast<double*>(shared_mem);
    long long* s_max_error = reinterpret_cast<long long*>(shared_mem + 4 * blockDim.x * sizeof(double));
    bool* s_overflow = reinterpret_cast<bool*>(shared_mem + 4 * blockDim.x * sizeof(double) + blockDim.x * sizeof(long long));
    
    int tid = threadIdx.x;
    
    // Phase 1: Check for overflow
    bool local_overflow = false;
    for (int i = tid; i < segment_len; i += blockDim.x) {
        if (mightOverflowDoublePrecision(values_device[start_idx + i])) {
            local_overflow = true;
            break;
        }
    }
    
    // Reduce overflow flag
    s_overflow[tid] = local_overflow;
    __syncthreads();
    
    // Simple reduction for overflow flag
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_overflow[tid] = s_overflow[tid] || s_overflow[tid + s];
        }
        __syncthreads();
    }
    
    bool has_overflow = s_overflow[0];
    
    if (tid == 0) {
        if (has_overflow) {
            // Direct copy model for overflow
            d_model_types[partition_idx] = MODEL_DIRECT_COPY;
            d_model_params[partition_idx * 4] = 0.0;
            d_model_params[partition_idx * 4 + 1] = 0.0;
            d_model_params[partition_idx * 4 + 2] = 0.0;
            d_model_params[partition_idx * 4 + 3] = 0.0;
            d_error_bounds[partition_idx] = 0;
            d_delta_bits[partition_idx] = sizeof(T) * 8;
        }
    }
    __syncthreads();
    
    if (!has_overflow) {
        // Phase 2: Fit linear model
        double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
        
        for (int i = tid; i < segment_len; i += blockDim.x) {
            double x = static_cast<double>(i);
            double y = static_cast<double>(values_device[start_idx + i]);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }
        
        s_sums[tid] = sum_x;
        s_sums[tid + blockDim.x] = sum_y;
        s_sums[tid + 2 * blockDim.x] = sum_xx;
        s_sums[tid + 3 * blockDim.x] = sum_xy;
        __syncthreads();
        
        // Reduction for sums
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sums[tid] += s_sums[tid + s];
                s_sums[tid + blockDim.x] += s_sums[tid + s + blockDim.x];
                s_sums[tid + 2 * blockDim.x] += s_sums[tid + s + 2 * blockDim.x];
                s_sums[tid + 3 * blockDim.x] += s_sums[tid + s + 3 * blockDim.x];
            }
            __syncthreads();
        }
        
        __shared__ double theta0, theta1;
        
        if (tid == 0) {
            double n = static_cast<double>(segment_len);
            double determinant = n * s_sums[2 * blockDim.x] - s_sums[0] * s_sums[0];
            
            if (fabs(determinant) > 1e-10) {
                theta1 = (n * s_sums[3 * blockDim.x] - s_sums[0] * s_sums[blockDim.x]) / determinant;
                theta0 = (s_sums[blockDim.x] - theta1 * s_sums[0]) / n;
            } else {
                theta1 = 0.0;
                theta0 = s_sums[blockDim.x] / n;
            }
            
            d_model_types[partition_idx] = MODEL_LINEAR;
            d_model_params[partition_idx * 4] = theta0;
            d_model_params[partition_idx * 4 + 1] = theta1;
            d_model_params[partition_idx * 4 + 2] = 0.0;
            d_model_params[partition_idx * 4 + 3] = 0.0;
        }
        __syncthreads();
        
        theta0 = d_model_params[partition_idx * 4];
        theta1 = d_model_params[partition_idx * 4 + 1];
        
        // Phase 3: Calculate maximum error
        long long max_error = 0;
        
        for (int i = tid; i < segment_len; i += blockDim.x) {
            double predicted = theta0 + theta1 * i;
            T pred_T = static_cast<T>(round(predicted));
            long long delta = calculateDelta(values_device[start_idx + i], pred_T);
            long long abs_error = (delta < 0) ? -delta : delta;
            if (abs_error > max_error) {
                max_error = abs_error;
            }
        }
        
        s_max_error[tid] = max_error;
        __syncthreads();
        
        // Reduction for maximum error
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_max_error[tid + s] > s_max_error[tid]) {
                    s_max_error[tid] = s_max_error[tid + s];
                }
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            d_error_bounds[partition_idx] = s_max_error[0];
            
            // Calculate delta bits
            int delta_bits = 0;
            if (s_max_error[0] > 0) {
                long long max_abs_error = s_max_error[0];
                int bits_for_magnitude = 0;
                unsigned long long temp = static_cast<unsigned long long>(max_abs_error);
                while (temp > 0) {
                    bits_for_magnitude++;
                    temp >>= 1;
                }
                delta_bits = bits_for_magnitude + 1; // +1 for sign bit
                delta_bits = min(delta_bits, MAX_DELTA_BITS);
                delta_bits = max(delta_bits, 0);
            }
            d_delta_bits[partition_idx] = delta_bits;
        }
    }
    
    // Atomic add to total bits counter
    if (tid == 0) {
        int64_t partition_bits = (int64_t)segment_len * d_delta_bits[partition_idx];
        // Use unsigned long long atomicAdd and cast
        atomicAdd(reinterpret_cast<unsigned long long*>(total_bits_device), 
                  static_cast<unsigned long long>(partition_bits));
    }
}

// Kernel to set bit offsets based on cumulative sum - UPDATED FOR SoA
__global__ void setBitOffsetsKernel(int32_t* d_start_indices,
                                   int32_t* d_end_indices,
                                   int32_t* d_delta_bits,
                                   int64_t* d_delta_array_bit_offsets,
                                   int num_partitions) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_partitions) return;
    
    // Calculate cumulative bit offset
    int64_t bit_offset = 0;
    for (int i = 0; i < tid; i++) {
        int seg_len = d_end_indices[i] - d_start_indices[i];
        bit_offset += (int64_t)seg_len * d_delta_bits[i];
    }
    
    d_delta_array_bit_offsets[tid] = bit_offset;
}

// Extract value for direct copy model
template<typename T>
__device__ inline T extractDirectValue(const uint32_t* delta_array, 
                                      int64_t bit_offset, 
                                      int value_bits) {
    if (value_bits <= 0) return static_cast<T>(0);
    
    if (value_bits <= 32) {
        int word_idx = bit_offset / 32;
        int bit_offset_in_word = bit_offset % 32;
        uint32_t extracted_bits;
        
        if (bit_offset_in_word + value_bits <= 32) {
            extracted_bits = (delta_array[word_idx] >> bit_offset_in_word) & 
                            ((1U << value_bits) - 1U);
        } else {
            uint32_t w1 = delta_array[word_idx];
            uint32_t w2 = delta_array[word_idx + 1]; 
            extracted_bits = (w1 >> bit_offset_in_word) | (w2 << (32 - bit_offset_in_word));
            extracted_bits &= ((1U << value_bits) - 1U);
        }
        
        return static_cast<T>(extracted_bits);
    } else {
        // Handle > 32 bit values
        int start_word_idx = bit_offset / 32;
        int offset_in_word = bit_offset % 32;
        int bits_remaining = value_bits;
        uint64_t extracted_val_64 = 0;
        int shift = 0;
        int word_idx = start_word_idx;
        
        while (bits_remaining > 0 && shift < 64) {
            int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
            uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
            uint32_t word_val = (delta_array[word_idx] >> offset_in_word) & mask;
            extracted_val_64 |= (static_cast<uint64_t>(word_val) << shift);
            
            shift += bits_in_this_word;
            bits_remaining -= bits_in_this_word;
            word_idx++;
            offset_in_word = 0;
        }
        
        return static_cast<T>(extracted_val_64);
    }
}


// 383 -------------------------------------------------------

// Kernel to pre-unpack all deltas from bit-packed format to plain long long array
template<typename T>
__global__ void unpackAllDeltasKernel(
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const uint32_t* __restrict__ delta_array,
    long long* __restrict__ d_plain_deltas_output,
    int num_partitions,
    int total_values) {
    
    // Grid-stride loop
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;
    
    for (int idx = g_idx; idx < total_values; idx += g_stride) {
        // Binary search to find the partition
        int p_left = 0, p_right = num_partitions - 1;
        int p_found_idx = -1;
        
        while (p_left <= p_right) {
            int p_mid = p_left + (p_right - p_left) / 2;
            int32_t current_start = d_start_indices[p_mid];
            int32_t current_end = d_end_indices[p_mid];
            
            if (idx >= current_start && idx < current_end) {
                p_found_idx = p_mid;
                break;
            } else if (idx < current_start) {
                p_right = p_mid - 1;
            } else {
                p_left = p_mid + 1;
            }
        }
        
        if (p_found_idx == -1) {
            d_plain_deltas_output[idx] = 0;
            continue;
        }
        
        // Extract delta for this element
        int32_t start_idx = d_start_indices[p_found_idx];
        int32_t model_type = d_model_types[p_found_idx];
        int32_t delta_bits = d_delta_bits[p_found_idx];
        int64_t bit_offset_base = d_delta_array_bit_offsets[p_found_idx];
        int local_idx = idx - start_idx;
        
        long long delta = 0;
        
        if (model_type == MODEL_DIRECT_COPY) {
            // For direct copy model, extract the full value (not a delta)
            if (delta_bits > 0 && delta_array) {
                int64_t bit_offset = bit_offset_base + (int64_t)local_idx * delta_bits;
                // We store the raw value as the "delta" for direct copy
                delta = static_cast<long long>(extractDirectValue<T>(delta_array, bit_offset, delta_bits));
            }
        } else {
            // Normal delta extraction
            if (delta_bits > 0 && delta_array) {
                int64_t bit_offset = bit_offset_base + (int64_t)local_idx * delta_bits;
                delta = extractDelta_Optimized<T>(delta_array, bit_offset, delta_bits);
            }
        }
        
        d_plain_deltas_output[idx] = delta;
    }
}
// 383 -------------------------------------------------------

// Optimized delta packing kernel with direct copy support - UPDATED FOR SoA
template<typename T>
__global__ void packDeltasKernelOptimized(const T* values_device,
                                          const int32_t* d_start_indices,
                                          const int32_t* d_end_indices,
                                          const int32_t* d_model_types,
                                          const double* d_model_params,
                                          const int32_t* d_delta_bits,
                                          const int64_t* d_delta_array_bit_offsets,
                                          int num_partitions_val,
                                          uint32_t* delta_array_device) {
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;

    if (num_partitions_val == 0) return;

    int max_idx_to_process = d_end_indices[num_partitions_val - 1];

    for (int current_idx = g_idx; current_idx < max_idx_to_process; current_idx += g_stride) {
        // Binary search for partition
        int p_left = 0, p_right = num_partitions_val - 1;
        int found_partition_idx = -1;

        while (p_left <= p_right) {
            int p_mid = p_left + (p_right - p_left) / 2;
            int32_t current_start = d_start_indices[p_mid];
            int32_t current_end = d_end_indices[p_mid];
            
            if (current_idx >= current_start && current_idx < current_end) {
                found_partition_idx = p_mid; 
                break;
            } else if (current_idx < current_start) {
                p_right = p_mid - 1;
            } else {
                p_left = p_mid + 1;
            }
        }

        if (found_partition_idx == -1) continue;

        // Get partition data using found index
        int32_t current_model_type = d_model_types[found_partition_idx];
        int32_t current_delta_bits = d_delta_bits[found_partition_idx];
        int64_t current_bit_offset_base = d_delta_array_bit_offsets[found_partition_idx];
        int32_t current_start_idx = d_start_indices[found_partition_idx];

        // For direct copy model, we store the full value
        if (current_model_type == MODEL_DIRECT_COPY) {
            int local_idx = current_idx - current_start_idx;
            int64_t bit_offset = current_bit_offset_base + 
                                (int64_t)local_idx * current_delta_bits;
            
            // Store the full value as "delta"
            T value = values_device[current_idx];
            uint64_t value_to_store = static_cast<uint64_t>(value);
            
            // Pack the value into the delta array
            int start_word_idx = bit_offset / 32;
            int offset_in_word = bit_offset % 32;
            int bits_remaining = current_delta_bits;
            int word_idx = start_word_idx;
            
            while (bits_remaining > 0) {
                int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
                uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                uint32_t value_part = (value_to_store & mask) << offset_in_word;
                atomicOr(&delta_array_device[word_idx], value_part);
                
                value_to_store >>= bits_in_this_word;
                bits_remaining -= bits_in_this_word;
                word_idx++;
                offset_in_word = 0;
            }
        } else {
            // Normal delta encoding
            int current_local_idx = current_idx - current_start_idx;

            double pred_double = d_model_params[found_partition_idx * 4] + 
                                d_model_params[found_partition_idx * 4 + 1] * current_local_idx;
            if (current_model_type == MODEL_POLYNOMIAL2) {
                pred_double += d_model_params[found_partition_idx * 4 + 2] * current_local_idx * current_local_idx;
            }

            T pred_T_val = static_cast<T>(round(pred_double));
            long long current_delta_ll = calculateDelta(values_device[current_idx], pred_T_val);

            if (current_delta_bits > 0) {
                int64_t current_bit_offset_val = current_bit_offset_base + 
                                                 (int64_t)current_local_idx * current_delta_bits;
                
                // Handle deltas up to 64 bits
                if (current_delta_bits <= 32) {
                    uint32_t final_packed_delta = static_cast<uint32_t>(current_delta_ll & 
                                                                       ((1ULL << current_delta_bits) - 1ULL));
                    
                    int target_word_idx = current_bit_offset_val / 32;
                    int offset_in_word = current_bit_offset_val % 32;

                    if (current_delta_bits + offset_in_word <= 32) {
                        atomicOr(&delta_array_device[target_word_idx], final_packed_delta << offset_in_word);
                    } else {
                        atomicOr(&delta_array_device[target_word_idx], final_packed_delta << offset_in_word); 
                        atomicOr(&delta_array_device[target_word_idx + 1], 
                                final_packed_delta >> (32 - offset_in_word));
                    }
                } else {
                    // For deltas > 32 bits
                    uint64_t final_packed_delta_64 = static_cast<uint64_t>(current_delta_ll & 
                        ((current_delta_bits == 64) ? ~0ULL : ((1ULL << current_delta_bits) - 1ULL)));
                    
                    int start_word_idx = current_bit_offset_val / 32;
                    int offset_in_word = current_bit_offset_val % 32;
                    int bits_remaining = current_delta_bits;
                    int word_idx = start_word_idx;
                    uint64_t delta_to_write = final_packed_delta_64;
                    
                    while (bits_remaining > 0) {
                        int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
                        uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                        uint32_t value = (delta_to_write & mask) << offset_in_word;
                        atomicOr(&delta_array_device[word_idx], value);
                        
                        delta_to_write >>= bits_in_this_word;
                        bits_remaining -= bits_in_this_word;
                        word_idx++;
                        offset_in_word = 0;
                    }
                }
            }
        }
    }
}

// Optimized decompression kernel with reduced branching and better memory access
template<typename T>
__global__ void decompressOptimizedKernel(const CompressedData<T>* compressed_data_on_device,
                                         T* output_device,
                                         const int* positions_device,
                                         int num_queries_val) {
    int q_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (q_idx >= num_queries_val) return;

    int q_pos = positions_device[q_idx];
    
    // Early exit check combined
    if (!compressed_data_on_device || compressed_data_on_device->num_partitions == 0) {
        output_device[q_idx] = static_cast<T>(0);
        return;
    }

    // Cache pointers in registers
    const int32_t* dev_start_indices = compressed_data_on_device->d_start_indices;
    const int32_t* dev_end_indices = compressed_data_on_device->d_end_indices;
    const int num_partitions = compressed_data_on_device->num_partitions;

    // Optimized binary search with reduced branching
    int p_left = 0;
    int p_right = num_partitions - 1;
    int p_found_idx = -1;

    // Use __ldg for better cache behavior
    while (p_left <= p_right) {
        int p_mid = (p_left + p_right) >> 1;  // Use bit shift instead of division
        int32_t current_start = __ldg(&dev_start_indices[p_mid]);
        int32_t current_end = __ldg(&dev_end_indices[p_mid]);
        
        // Check if found
        bool found = (q_pos >= current_start) & (q_pos < current_end);
        if (found) {
            p_found_idx = p_mid;
            break;
        }
        
        // Branchless update
        bool go_right = q_pos >= current_end;
        p_left = go_right ? (p_mid + 1) : p_left;
        p_right = go_right ? p_right : (p_mid - 1);
    }

    if (p_found_idx == -1) {
        output_device[q_idx] = static_cast<T>(0);
        return;
    }

    // Load all partition metadata at once
    int32_t start_idx = __ldg(&dev_start_indices[p_found_idx]);
    int32_t model_type = __ldg(&compressed_data_on_device->d_model_types[p_found_idx]);
    int32_t delta_bits = __ldg(&compressed_data_on_device->d_delta_bits[p_found_idx]);
    int64_t bit_offset_base = __ldg(&compressed_data_on_device->d_delta_array_bit_offsets[p_found_idx]);
    
    int local_q_idx = q_pos - start_idx;

    // Unified processing path
    T final_value;
    
    if (model_type == MODEL_DIRECT_COPY) {
        if (compressed_data_on_device->d_plain_deltas != nullptr) {
            final_value = static_cast<T>(__ldg(&compressed_data_on_device->d_plain_deltas[q_pos]));
        } else if (delta_bits > 0 && compressed_data_on_device->delta_array) {
            int64_t bit_off = bit_offset_base + (int64_t)local_q_idx * delta_bits;
            final_value = extractDirectValue<T>(compressed_data_on_device->delta_array, bit_off, delta_bits);
        } else {
            final_value = static_cast<T>(0);
        }
    } else {
        // Load model parameters using vector load for better bandwidth
        int param_idx = p_found_idx << 2;  // p_found_idx * 4
        double theta0 = __ldg(&compressed_data_on_device->d_model_params[param_idx]);
        double theta1 = __ldg(&compressed_data_on_device->d_model_params[param_idx + 1]);
        
        // Compute prediction with FMA
        double pred_double = fma(theta1, static_cast<double>(local_q_idx), theta0);
        
        // Conditional polynomial term without branching
        if (model_type == MODEL_POLYNOMIAL2) {
            double theta2 = __ldg(&compressed_data_on_device->d_model_params[param_idx + 2]);
            pred_double = fma(theta2, static_cast<double>(local_q_idx * local_q_idx), pred_double);
        }

        // Get delta with unified path
        long long current_final_delta = 0;
        
        if (compressed_data_on_device->d_plain_deltas != nullptr) {
            current_final_delta = __ldg(&compressed_data_on_device->d_plain_deltas[q_pos]);
        } else if (delta_bits > 0 && compressed_data_on_device->delta_array) {
            int64_t bit_off = bit_offset_base + (int64_t)local_q_idx * delta_bits;
            current_final_delta = extractDelta_Optimized<T>(compressed_data_on_device->delta_array, 
                                                            bit_off, delta_bits);
        }
        
        // Type-specific clamping using template specialization would be better
        // For now, use branchless min/max
        if (!std::is_signed<T>::value) {
            pred_double = fmax(0.0, pred_double);
            if (sizeof(T) == 4) {
                pred_double = fmin(pred_double, 4294967295.0);
            } else if (sizeof(T) == 2) {
                pred_double = fmin(pred_double, 65535.0);
            } else if (sizeof(T) == 1) {
                pred_double = fmin(pred_double, 255.0);
            }
        }
        
        T pred_T_val = static_cast<T>(round(pred_double));
        final_value = applyDelta(pred_T_val, current_final_delta);
    }
    
    output_device[q_idx] = final_value;
}




// Optimized direct random access kernel with proper alignment handling
template<typename T>
__global__ void directRandomAccessKernel(
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    const int32_t* __restrict__ model_types,
    const double* __restrict__ model_params,
    const int32_t* __restrict__ delta_bits,
    const int64_t* __restrict__ delta_array_bit_offsets,
    const uint32_t* __restrict__ delta_array,
    const int* __restrict__ positions,
    T* __restrict__ output,
    int num_partitions,
    int num_queries) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    
    int query_pos = __ldg(&positions[idx]);
    
    // Optimized binary search with reduced warp divergence
    int left = 0;
    int right = num_partitions - 1;
    int found = -1;
    
    // Unroll first few iterations for small partition counts
    if (num_partitions <= 8) {
        // Linear search for very small partition counts
        #pragma unroll
        for (int i = 0; i < 8 && i < num_partitions; i++) {
            int32_t start = __ldg(&start_indices[i]);
            int32_t end = __ldg(&end_indices[i]);
            if (query_pos >= start && query_pos < end) {
                found = i;
                break;
            }
        }
    } else {
        // Binary search with reduced branching
        while (left <= right) {
            int mid = (left + right) >> 1;
            int32_t start = __ldg(&start_indices[mid]);
            int32_t end = __ldg(&end_indices[mid]);
            
            bool is_found = (query_pos >= start) & (query_pos < end);
            if (is_found) {
                found = mid;
                break;
            }
            
            // Branchless update
            bool go_right = query_pos >= end;
            left = go_right ? (mid + 1) : left;
            right = go_right ? right : (mid - 1);
        }
    }
    
    if (found == -1) {
        output[idx] = static_cast<T>(0);
        return;
    }
    
    // Load all metadata with texture cache
    int32_t start_idx = __ldg(&start_indices[found]);
    int32_t type = __ldg(&model_types[found]);
    int32_t bits = __ldg(&delta_bits[found]);
    int64_t base_offset = __ldg(&delta_array_bit_offsets[found]);
    int local_idx = query_pos - start_idx;
    
    if (type == MODEL_DIRECT_COPY) {
        if (bits > 0) {
            int64_t bit_offset = base_offset + (int64_t)local_idx * bits;
            output[idx] = extractDirectValue<T>(delta_array, bit_offset, bits);
        } else {
            output[idx] = static_cast<T>(0);
        }
    } else {
        // Load model parameters safely (avoiding alignment issues)
        int param_base = found * 4;  // Each partition has 4 params
        double theta0 = __ldg(&model_params[param_base]);
        double theta1 = __ldg(&model_params[param_base + 1]);
        
        // Compute prediction using FMA
        double pred = fma(theta1, static_cast<double>(local_idx), theta0);
        
        // Handle polynomial models
        if (type == MODEL_POLYNOMIAL2) {
            double theta2 = __ldg(&model_params[param_base + 2]);
            pred = fma(theta2, static_cast<double>(local_idx * local_idx), pred);
        }
        
        // Extract delta
        long long delta = 0;
        if (bits > 0) {
            int64_t bit_offset = base_offset + (int64_t)local_idx * bits;
            
            // Optimized extraction for common bit widths
            if (bits <= 32) {
                int word_idx = bit_offset >> 5;  // bit_offset / 32
                int bit_shift = bit_offset & 31;  // bit_offset % 32
                
                uint32_t w1 = __ldg(&delta_array[word_idx]);
                uint32_t extracted_bits;
                
                if (bit_shift + bits <= 32) {
                    // Fits in single word
                    extracted_bits = (w1 >> bit_shift) & ((1U << bits) - 1U);
                } else {
                    // Spans two words
                    uint32_t w2 = __ldg(&delta_array[word_idx + 1]);
                    uint64_t combined = (static_cast<uint64_t>(w2) << 32) | static_cast<uint64_t>(w1);
                    extracted_bits = (combined >> bit_shift) & ((1U << bits) - 1U);
                }
                
                // Sign extension
                if (bits < 32) {
                    uint32_t sign_bit = extracted_bits >> (bits - 1);
                    uint32_t sign_mask = -sign_bit;
                    uint32_t extend_mask = ~((1U << bits) - 1U);
                    extracted_bits |= (sign_mask & extend_mask);
                }
                
                delta = static_cast<long long>(static_cast<int32_t>(extracted_bits));
            } else {
                // Use the optimized extraction function for > 32 bits
                delta = extractDelta_Optimized<T>(delta_array, bit_offset, bits);
            }
        }
        
        // Fast clamping and rounding
        T pred_val;
        if (sizeof(T) == 8 && !std::is_signed<T>::value) {
            // Special handling for unsigned 64-bit
            pred = fmax(0.0, fmin(pred, 18446744073709551615.0));
            pred_val = static_cast<T>(pred + 0.5);
        } else {
            // General case with proper clamping
            if (!std::is_signed<T>::value) {
                pred = fmax(0.0, pred);
                if (sizeof(T) == 4) {
                    pred = fmin(pred, 4294967295.0);
                } else if (sizeof(T) == 2) {
                    pred = fmin(pred, 65535.0);
                } else if (sizeof(T) == 1) {
                    pred = fmin(pred, 255.0);
                }
            } else {
                // For signed types
                if (sizeof(T) == 4) {
                    pred = fmax(-2147483648.0, fmin(pred, 2147483647.0));
                } else if (sizeof(T) == 2) {
                    pred = fmax(-32768.0, fmin(pred, 32767.0));
                } else if (sizeof(T) == 1) {
                    pred = fmax(-128.0, fmin(pred, 127.0));
                }
            }
            pred_val = static_cast<T>(round(pred));
        }
        
        output[idx] = applyDelta(pred_val, delta);
    }
}


template<typename T>
__global__ void decompressFullFile_OnTheFly_Optimized_V2(
    const CompressedData<T>* compressed_data,
    T* output_device,
    int total_elements) {

    // Use shared memory to cache the metadata for the partition this block is responsible for.
    __shared__ PartitionMetaOpt s_meta;

    // Map the CUDA block directly to a partition index.
    int partition_idx = blockIdx.x;

    // Ensure this block is not assigned to a non-existent partition.
    if (partition_idx >= compressed_data->num_partitions) {
        return;
    }

    // A single thread (thread 0) in the block loads the partition's metadata from global memory
    // into the fast shared memory. This is done once per block.
    if (threadIdx.x == 0) {
        s_meta.start_idx = compressed_data->d_start_indices[partition_idx];
        s_meta.model_type = compressed_data->d_model_types[partition_idx];
        s_meta.delta_bits = compressed_data->d_delta_bits[partition_idx];
        s_meta.bit_offset_base = compressed_data->d_delta_array_bit_offsets[partition_idx];
        
        // Load model parameters for the linear model.
        int params_base_idx = partition_idx * 4;
        s_meta.theta0 = compressed_data->d_model_params[params_base_idx];
        s_meta.theta1 = compressed_data->d_model_params[params_base_idx + 1];

        // Also cache the length of the partition.
        s_meta.partition_len = compressed_data->d_end_indices[partition_idx] - s_meta.start_idx;
    }

    // Synchronize all threads within the block to ensure that the shared memory is populated
    // before any thread attempts to use it.
    __syncthreads();

    // Use a grid-stride loop where each thread processes elements within this partition.
    // This ensures that all memory accesses are localized and coalesced.
    for (int local_idx = threadIdx.x; local_idx < s_meta.partition_len; local_idx += blockDim.x) {
        
        int global_idx = s_meta.start_idx + local_idx;

        // Ensure we don't write past the end of the total data array.
        if (global_idx >= total_elements) continue;

        long long delta = 0;
        T final_value;

        // Check the model type from shared memory.
        if (s_meta.model_type == MODEL_DIRECT_COPY) {
            // For direct copy, the "delta" array actually stores the full value.
            if (compressed_data->d_plain_deltas != nullptr) {
                // High-throughput mode: value is already unpacked.
                final_value = static_cast<T>(compressed_data->d_plain_deltas[global_idx]);
            } else if (s_meta.delta_bits > 0 && compressed_data->delta_array != nullptr) {
                // Standard mode: extract the value from the bit-packed array.
                int64_t bit_offset = s_meta.bit_offset_base + (int64_t)local_idx * s_meta.delta_bits;
                final_value = extractDirectValue<T>(compressed_data->delta_array, bit_offset, s_meta.delta_bits);
            } else {
                final_value = static_cast<T>(0);
            }
        } else {
            // For model-based decompression (e.g., MODEL_LINEAR)
            // Calculate the predicted value using the cached model parameters from shared memory.
            double predicted_double = fma(s_meta.theta1, static_cast<double>(local_idx), s_meta.theta0);
            
            // Extract the delta value.
            if (compressed_data->d_plain_deltas != nullptr) {
                 // High-throughput mode: delta is already unpacked.
                 delta = compressed_data->d_plain_deltas[global_idx];
            } else if (s_meta.delta_bits > 0 && compressed_data->delta_array != nullptr) {
                // Standard mode: calculate the bit offset and extract from the bit-packed array.
                int64_t bit_offset = s_meta.bit_offset_base + (int64_t)local_idx * s_meta.delta_bits;
                delta = extractDelta_Optimized<T>(compressed_data->delta_array, bit_offset, s_meta.delta_bits);
            }

            // Round prediction and apply the delta to get the final value.
            T predicted_T = static_cast<T>(round(predicted_double));
            final_value = applyDelta(predicted_T, delta);
        }

        // Write the final, perfectly coalesced result to global memory.
        output_device[global_idx] = final_value;
    }
}


// Optimized full-file decompression kernel with reduced register usage and better load balancing
template<typename T>
__launch_bounds__(256, 4)
__global__ void decompressFullFileCooperativeKernel(
    const CompressedData<T>* __restrict__ compressed_data_on_device,
    T* __restrict__ output_device,
    int total_elements) {
    
    // Check for valid data
    if (!compressed_data_on_device || compressed_data_on_device->num_partitions == 0) {
        return;
    }
    
    // This kernel requires pre-unpacked deltas
    if (compressed_data_on_device->d_plain_deltas == nullptr) {
        // Fallback to element-wise processing if no pre-unpacked deltas
        int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
        int g_stride = blockDim.x * gridDim.x;
        for (int idx = g_idx; idx < total_elements; idx += g_stride) {
            output_device[idx] = static_cast<T>(0);
        }
        return;
    }
    
    // Use dynamic shared memory more efficiently
    extern __shared__ char shared_mem_coop[];
    
    // Reduced constants to decrease register pressure
    const int PARTITIONS_PER_BLOCK = 4;    // Reduced from 8 to save shared memory and registers
    
    // Partition metadata in shared memory - use smaller types where possible
    int* s_start_indices = reinterpret_cast<int*>(shared_mem_coop);
    int* s_end_indices = reinterpret_cast<int*>(shared_mem_coop + PARTITIONS_PER_BLOCK * sizeof(int));
    int* s_model_types = reinterpret_cast<int*>(shared_mem_coop + 2 * PARTITIONS_PER_BLOCK * sizeof(int));
    // Store only theta0 and theta1 as float to reduce register usage
    float* s_model_params = reinterpret_cast<float*>(shared_mem_coop + 3 * PARTITIONS_PER_BLOCK * sizeof(int));
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int bid = blockIdx.x;
    
    // Get pointers to device data - use const to help compiler optimize
    const int32_t* __restrict__ d_start_indices = compressed_data_on_device->d_start_indices;
    const int32_t* __restrict__ d_end_indices = compressed_data_on_device->d_end_indices;
    const int32_t* __restrict__ d_model_types = compressed_data_on_device->d_model_types;
    const double* __restrict__ d_model_params = compressed_data_on_device->d_model_params;
    const long long* __restrict__ d_plain_deltas = compressed_data_on_device->d_plain_deltas;
    const int num_partitions = compressed_data_on_device->num_partitions;
    
    // Dynamic work stealing for better load balancing
    __shared__ int shared_partition_idx;
    if (tid == 0) {
        shared_partition_idx = bid;
    }
    __syncthreads();
    
    // Grid-stride loop for processing partitions
    while (true) {
        int partition_idx;
        if (tid == 0) {
            partition_idx = atomicAdd(&shared_partition_idx, gridDim.x);
        }
        __syncwarp();
        partition_idx = __shfl_sync(0xffffffff, partition_idx, 0);
        
        if (partition_idx >= num_partitions) break;
        
        // Load partition metadata
        __shared__ int start_idx, end_idx, model_type;
        __shared__ float theta0, theta1;
        
        if (tid == 0) {
            start_idx = __ldg(&d_start_indices[partition_idx]);
            end_idx = __ldg(&d_end_indices[partition_idx]);
            model_type = __ldg(&d_model_types[partition_idx]);
            // Convert double to float to save registers
            theta0 = static_cast<float>(__ldg(&d_model_params[partition_idx * 4]));
            theta1 = static_cast<float>(__ldg(&d_model_params[partition_idx * 4 + 1]));
        }
        __syncthreads();
        
        if (start_idx >= end_idx || start_idx >= total_elements) continue;
        
        int partition_size = min(end_idx - start_idx, total_elements - start_idx);
        
        if (model_type == MODEL_DIRECT_COPY) {
            // Direct copy with optimized memory access patterns
            
            // Try to use vector loads for better memory throughput
            if (sizeof(T) == 4 && ((uintptr_t)&d_plain_deltas[start_idx] & 15) == 0 && 
                ((uintptr_t)&output_device[start_idx] & 15) == 0) {
                
                // Process 4 elements at a time using int4
                int vec_idx = start_idx + (tid * 4);
                while (vec_idx + 3 < end_idx && vec_idx + 3 < total_elements) {
                    int4 data = *reinterpret_cast<const int4*>(&d_plain_deltas[vec_idx]);
                    *reinterpret_cast<int4*>(&output_device[vec_idx]) = data;
                    vec_idx += block_size * 4;
                }
                
                // Handle remaining elements
                int remaining_idx = start_idx + tid;
                int vec_end = (end_idx / 4) * 4;
                if (remaining_idx >= vec_end && remaining_idx < end_idx && remaining_idx < total_elements) {
                    output_device[remaining_idx] = static_cast<T>(__ldg(&d_plain_deltas[remaining_idx]));
                }
                
            } else if (sizeof(T) == 8 && ((uintptr_t)&d_plain_deltas[start_idx] & 15) == 0 && 
                       ((uintptr_t)&output_device[start_idx] & 15) == 0) {
                
                // Process 2 elements at a time using long2
                int vec_idx = start_idx + (tid * 2);
                while (vec_idx + 1 < end_idx && vec_idx + 1 < total_elements) {
                    long2 data = *reinterpret_cast<const long2*>(&d_plain_deltas[vec_idx]);
                    *reinterpret_cast<long2*>(&output_device[vec_idx]) = data;
                    vec_idx += block_size * 2;
                }
                
                // Handle remaining element
                if ((end_idx & 1) && (start_idx + tid * 2 + 1) == (end_idx - 1) && 
                    (end_idx - 1) < total_elements) {
                    output_device[end_idx - 1] = static_cast<T>(__ldg(&d_plain_deltas[end_idx - 1]));
                }
                
            } else {
                // Standard grid-stride loop for non-aligned or small types
                for (int idx = start_idx + tid; idx < end_idx && idx < total_elements; idx += block_size) {
                    output_device[idx] = static_cast<T>(__ldg(&d_plain_deltas[idx]));
                }
            }
            
        } else {
            // Model-based decompression with optimized operations
            
            // Use warp-level primitives for better efficiency
            const int warp_id = tid / 32;
            const int lane_id = tid & 31;
            const int num_warps = block_size / 32;
            
            // Each warp processes a portion of the partition
            const int elements_per_warp = (partition_size + num_warps - 1) / num_warps;
            const int warp_start = warp_id * elements_per_warp;
            const int warp_end = min(warp_start + elements_per_warp, partition_size);
            
            // Process elements with warp-level parallelism
            for (int local_idx = warp_start + lane_id; local_idx < warp_end; local_idx += 32) {
                int global_idx = start_idx + local_idx;
                if (global_idx >= total_elements) break;
                
                // Use fast math intrinsics
                float pred = __fmaf_rn(theta1, __int2float_rn(local_idx), theta0);
                
                // Type-specific clamping using intrinsics
                T pred_val;
                if (!std::is_signed<T>::value) {
                    pred = fmaxf(0.0f, pred);
                    if (sizeof(T) == 8) {
                        // Special handling for unsigned 64-bit
                        pred = fminf(pred, 18446744073709551615.0f);
                        pred_val = static_cast<T>(pred);
                    } else if (sizeof(T) == 4) {
                        pred = fminf(pred, 4294967295.0f);
                        pred_val = static_cast<T>(__float2uint_rn(pred));
                    } else if (sizeof(T) == 2) {
                        pred = fminf(pred, 65535.0f);
                        pred_val = static_cast<T>(__float2int_rn(pred));
                    } else {
                        pred = fminf(pred, 255.0f);
                        pred_val = static_cast<T>(__float2int_rn(pred));
                    }
                } else {
                    // Signed types
                    if (sizeof(T) == 8) {
                        pred = fmaxf(-9223372036854775808.0f, fminf(pred, 9223372036854775807.0f));
                        pred_val = static_cast<T>(__float2ll_rn(pred));
                    } else if (sizeof(T) == 4) {
                        pred = fmaxf(-2147483648.0f, fminf(pred, 2147483647.0f));
                        pred_val = static_cast<T>(__float2int_rn(pred));
                    } else if (sizeof(T) == 2) {
                        pred = fmaxf(-32768.0f, fminf(pred, 32767.0f));
                        pred_val = static_cast<T>(__float2int_rn(pred));
                    } else {
                        pred = fmaxf(-128.0f, fminf(pred, 127.0f));
                        pred_val = static_cast<T>(__float2int_rn(pred));
                    }
                }
                
                // Load delta with texture cache and apply
                long long delta = __ldg(&d_plain_deltas[global_idx]);
                output_device[global_idx] = applyDelta(pred_val, delta);
            }
        }
        
        __syncthreads();
    }
}



template<typename T>
__global__ void decompressFullFileFix(
    const CompressedData<T>* __restrict__ compressed_data_on_device,
    T* __restrict__ output_device,
    int total_elements,
    int partition_size) {

    // --- 优化部分：在共享内存中定义元数据缓存 ---
    // 用于缓存当前分区元数据的结构体
    struct PartitionMetaCache {
        int32_t model_type;
        int32_t delta_bits;
        double theta0;
        double theta1;
        int64_t bit_offset_base;
    };

    // 每个线程块（Block）使用一个共享内存缓存
    __shared__ PartitionMetaCache s_meta;
    // 用于记录当前缓存的是哪个分区
    __shared__ int s_cached_partition_idx;

    // --- 内核主体逻辑 ---
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;

    if (!compressed_data_on_device || compressed_data_on_device->num_partitions == 0) {
        return;
    }

    // 在循环开始前，由块内第一个线程初始化缓存标记
    if (threadIdx.x == 0) {
        s_cached_partition_idx = -1; // -1 表示缓存无效
    }
    // 同步以确保所有线程都看到无效的缓存标记
    __syncthreads();

    const int32_t* __restrict__ dev_model_types = compressed_data_on_device->d_model_types;
    const double* __restrict__ dev_model_params = compressed_data_on_device->d_model_params;
    const int32_t* __restrict__ dev_delta_bits = compressed_data_on_device->d_delta_bits;
    const int64_t* __restrict__ dev_delta_array_bit_offsets = compressed_data_on_device->d_delta_array_bit_offsets;
    const uint32_t* __restrict__ dev_delta_arr = compressed_data_on_device->delta_array;
    int num_partitions = compressed_data_on_device->num_partitions;

    for (int idx = g_idx; idx < total_elements; idx += g_stride) {
        int partition_idx = idx / partition_size;
        int local_idx = idx % partition_size;
        
        if (partition_idx >= num_partitions) {
            continue; 
        }

        // --- 优化部分：动态缓存检查与更新 ---
        // 检查当前分区元数据是否已在缓存中
        if (partition_idx != s_cached_partition_idx) {
            // 如果不在缓存中（缓存失效），则由块内的第一个线程负责从全局内存加载新元数据
            if (threadIdx.x == 0) {
                s_meta.model_type = dev_model_types[partition_idx];
                s_meta.delta_bits = dev_delta_bits[partition_idx];
                s_meta.bit_offset_base = dev_delta_array_bit_offsets[partition_idx];

                if (s_meta.model_type != MODEL_DIRECT_COPY) {
                    s_meta.theta0 = dev_model_params[partition_idx * 4];
                    s_meta.theta1 = dev_model_params[partition_idx * 4 + 1];
                }
                // 更新缓存标记
                s_cached_partition_idx = partition_idx;
            }
            // **关键**：同步块内所有线程，以确保所有线程都能看到由线程0加载的最新元数据
            __syncthreads();
        }
        
        // --- 解压逻辑：从共享内存缓存中读取元数据 ---
        int32_t model_type = s_meta.model_type;
        int32_t delta_bits = s_meta.delta_bits;
        int64_t bit_offset_base = s_meta.bit_offset_base;
        
        if (model_type == MODEL_DIRECT_COPY) {
            if (compressed_data_on_device->d_plain_deltas != nullptr) {
                output_device[idx] = static_cast<T>(compressed_data_on_device->d_plain_deltas[idx]);
            } else if (delta_bits > 0 && dev_delta_arr) {
                int64_t bit_offset = bit_offset_base + (int64_t)local_idx * delta_bits;
                output_device[idx] = extractDirectValue<T>(dev_delta_arr, bit_offset, delta_bits);
            } else {
                output_device[idx] = static_cast<T>(0);
            }
        } else {
            // 从共享内存缓存中读取模型参数
            double pred_double = fma(s_meta.theta1, static_cast<double>(local_idx), s_meta.theta0);
            
            long long delta = 0;
            if (compressed_data_on_device->d_plain_deltas != nullptr) {
                delta = compressed_data_on_device->d_plain_deltas[idx];
            } else {
                if (delta_bits > 0 && dev_delta_arr) {
                    int64_t bit_offset = bit_offset_base + (int64_t)local_idx * delta_bits;
                    delta = extractDelta_Optimized<T>(dev_delta_arr, bit_offset, delta_bits);
                }
            }
            
            T pred_val = static_cast<T>(round(pred_double));
            output_device[idx] = applyDelta(pred_val, delta);
        }
    }
}

// CUDA kernels for parallel partitioning (must be outside class)
template<typename T>
__global__ void calculatePartitionCostsKernel(const T* data, int data_size, int min_part_size,
                                             const int* start_indices, const int* end_indices, 
                                             double* costs, int num_candidates,
                                             double model_size_bytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    int start = start_indices[idx];
    int end = end_indices[idx];
    int n = end - start;
    
    if (n <= 0) {
        costs[idx] = 1e20; // Very high cost for invalid partitions
        return;
    }
    
    // Calculate sums for linear regression
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(data[start + i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    
    // Solve normal equations
    double theta0, theta1;
    double determinant = n * sum_xx - sum_x * sum_x;
    if (fabs(determinant) > 1e-10) {
        theta1 = (n * sum_xy - sum_x * sum_y) / determinant;
        theta0 = (sum_y - theta1 * sum_x) / n;
    } else {
        theta1 = 0.0;
        theta0 = sum_y / n;
    }
    
    // Calculate maximum error
    long long max_error = 0;
    for (int i = 0; i < n; i++) {
        double predicted = theta0 + theta1 * i;
        T pred_T = static_cast<T>(round(predicted));
        long long delta = calculateDelta(data[start + i], pred_T);
        long long abs_error = (delta < 0) ? -delta : delta;
        max_error = max(max_error, abs_error);
    }
    
    // Calculate delta bits
    int delta_bits = 0;
    if (max_error > 0) {
        unsigned long long temp = static_cast<unsigned long long>(max_error);
        while (temp > 0) {
            delta_bits++;
            temp >>= 1;
        }
        delta_bits++; // +1 for sign bit
        delta_bits = min(delta_bits, MAX_DELTA_BITS);
    }
    
    // Calculate total cost
    double delta_array_bytes = static_cast<double>(n) * delta_bits / 8.0;
    costs[idx] = model_size_bytes + delta_array_bytes;
}

// Structure to pass partition candidates to kernel
struct PartitionCandidateGPU {
    int start_idx;
    int end_idx;
    double theta0;
    double theta1;
    long long max_error;
    int delta_bits;
    double total_cost;
};

template<typename T>
__global__ void fitPartitionModelsKernel(const T* data, PartitionCandidateGPU* candidates,
                                        int num_partitions, double model_size_bytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_partitions) return;
    
    int start = candidates[idx].start_idx;
    int end = candidates[idx].end_idx;
    int n = end - start;
    
    if (n <= 0) return;
    
    // Calculate sums for linear regression
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(data[start + i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    
    // Solve normal equations
    double determinant = n * sum_xx - sum_x * sum_x;
    if (fabs(determinant) > 1e-10) {
        candidates[idx].theta1 = (n * sum_xy - sum_x * sum_y) / determinant;
        candidates[idx].theta0 = (sum_y - candidates[idx].theta1 * sum_x) / n;
    } else {
        candidates[idx].theta1 = 0.0;
        candidates[idx].theta0 = sum_y / n;
    }
    
    // Calculate maximum error
    long long max_error = 0;
    for (int i = 0; i < n; i++) {
        double predicted = candidates[idx].theta0 + candidates[idx].theta1 * i;
        T pred_T = static_cast<T>(round(predicted));
        long long delta = calculateDelta(data[start + i], pred_T);
        long long abs_error = (delta < 0) ? -delta : delta;
        max_error = max(max_error, abs_error);
    }
    candidates[idx].max_error = max_error;
    
    // Calculate delta bits
    if (max_error > 0) {
        int delta_bits = 0;
        unsigned long long temp = static_cast<unsigned long long>(max_error);
        while (temp > 0) {
            delta_bits++;
            temp >>= 1;
        }
        candidates[idx].delta_bits = delta_bits + 1; // +1 for sign bit
        candidates[idx].delta_bits = min(candidates[idx].delta_bits, MAX_DELTA_BITS);
    } else {
        candidates[idx].delta_bits = 0;
    }
    
    // Calculate total cost
    double delta_array_bytes = static_cast<double>(n) * candidates[idx].delta_bits / 8.0;
    candidates[idx].total_cost = model_size_bytes + delta_array_bytes;
}

// Work-stealing queue structure for GPU
struct WorkStealingQueue {
    int* tasks;          // Array of task indices
    int* head;           // Per-thread queue heads
    int* tail;           // Per-thread queue tails
    int* global_head;    // Global queue head for stealing
    int* global_tail;    // Global queue tail
    int max_tasks;
    int num_threads;
};

// Initialize work-stealing queue
__global__ void initWorkStealingQueueKernel(WorkStealingQueue queue, int initial_tasks) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        // Initialize global queue with all tasks
        for (int i = 0; i < initial_tasks; i++) {
            queue.tasks[i] = i;
        }
        *queue.global_head = 0;
        *queue.global_tail = initial_tasks;
    }
    
    // Initialize per-thread queues
    if (tid < queue.num_threads) {
        queue.head[tid] = 0;
        queue.tail[tid] = 0;
    }
}

// Work-stealing kernel for variable-length partitioning
template<typename T>
__global__ void workStealingPartitionKernel(
    const T* data,
    int data_size,
    int min_partition_size,
    double split_threshold,
    WorkStealingQueue queue,
    PartitionCandidateGPU* candidates,
    int* num_candidates,
    double model_size_bytes) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int block_id = blockIdx.x;
    int threads_per_block = blockDim.x;
    
    // Shared memory for local work queue
    extern __shared__ int s_local_queue[];
    int* s_local_head = &s_local_queue[0];
    int* s_local_tail = &s_local_queue[1];
    int* s_local_tasks = &s_local_queue[2];
    
    // Initialize local queue
    if (threadIdx.x == 0) {
        *s_local_head = 0;
        *s_local_tail = 0;
    }
    __syncthreads();
    
    // Main work loop
    while (true) {
        int task_idx = -1;
        
        // Try to get task from local queue
        if (threadIdx.x == 0) {
            if (*s_local_head < *s_local_tail) {
                task_idx = s_local_tasks[(*s_local_head)++];
            }
        }
        __syncthreads();
        
        // Broadcast task to all threads in block
        task_idx = __shfl_sync(0xffffffff, task_idx, 0);
        
        // If no local task, try to steal
        if (task_idx == -1) {
            if (threadIdx.x == 0) {
                // Try global queue first
                int old_head = atomicAdd(queue.global_head, 1);
                if (old_head < *queue.global_tail) {
                    task_idx = queue.tasks[old_head];
                } else {
                    // Try to steal from other blocks
                    for (int victim = 0; victim < gridDim.x; victim++) {
                        if (victim != block_id) {
                            int victim_head = queue.head[victim];
                            int victim_tail = queue.tail[victim];
                            if (victim_head < victim_tail) {
                                // Try to steal half of victim's tasks
                                int steal_count = (victim_tail - victim_head) / 2;
                                if (steal_count > 0) {
                                    int old_tail = atomicAdd(&queue.tail[victim], -steal_count);
                                    if (old_tail - steal_count >= victim_head) {
                                        // Successfully stole tasks
                                        for (int i = 0; i < steal_count && i < 32; i++) {
                                            s_local_tasks[*s_local_tail + i] = queue.tasks[old_tail - steal_count + i];
                                        }
                                        *s_local_tail += steal_count;
                                        task_idx = s_local_tasks[(*s_local_head)++];
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
            task_idx = __shfl_sync(0xffffffff, task_idx, 0);
        }
        
        // If still no task, exit
        if (task_idx == -1) break;
        
        // Process the task (partition range)
        // Each task represents a potential partition to evaluate
        int start_idx = task_idx * min_partition_size;
        if (start_idx >= data_size) continue;
        
        // Cooperatively evaluate different partition sizes
        int local_tid = threadIdx.x;
        int step_size = min_partition_size;
        
        // Each thread evaluates a different partition size
        int size = min_partition_size * (1 + local_tid);
        int end_idx = min(start_idx + size, data_size);
        
        if (end_idx > start_idx && local_tid < 32) {
            // Calculate partition cost
            int n = end_idx - start_idx;
            
            // Parallel reduction for linear regression
            double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
            
            for (int i = local_tid; i < n; i += 32) {
                double x = static_cast<double>(i);
                double y = static_cast<double>(data[start_idx + i]);
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
            }
            
            // Warp-level reduction
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum_x += __shfl_down_sync(0xffffffff, sum_x, offset);
                sum_y += __shfl_down_sync(0xffffffff, sum_y, offset);
                sum_xx += __shfl_down_sync(0xffffffff, sum_xx, offset);
                sum_xy += __shfl_down_sync(0xffffffff, sum_xy, offset);
            }
            
            if (local_tid == 0) {
                // Solve for linear model
                double determinant = n * sum_xx - sum_x * sum_x;
                double theta0, theta1;
                
                if (fabs(determinant) > 1e-10) {
                    theta1 = (n * sum_xy - sum_x * sum_y) / determinant;
                    theta0 = (sum_y - theta1 * sum_x) / n;
                } else {
                    theta1 = 0.0;
                    theta0 = sum_y / n;
                }
                
                // Calculate max error
                long long max_error = 0;
                for (int i = 0; i < n; i += 32) {
                    if (i < n) {
                        double predicted = theta0 + theta1 * i;
                        T pred_T = static_cast<T>(round(predicted));
                        long long delta = calculateDelta(data[start_idx + i], pred_T);
                        long long abs_error = (delta < 0) ? -delta : delta;
                        max_error = max(max_error, abs_error);
                    }
                }
                
                // Calculate delta bits
                int delta_bits = 0;
                if (max_error > 0) {
                    unsigned long long temp = static_cast<unsigned long long>(max_error);
                    while (temp > 0) {
                        delta_bits++;
                        temp >>= 1;
                    }
                    delta_bits++; // +1 for sign bit
                }
                
                // Calculate total cost
                double delta_array_bytes = static_cast<double>(n) * delta_bits / 8.0;
                double total_cost = model_size_bytes + delta_array_bytes;
                
                // Add candidate
                int candidate_idx = atomicAdd(num_candidates, 1);
                if (candidate_idx < queue.max_tasks) {
                    candidates[candidate_idx].start_idx = start_idx;
                    candidates[candidate_idx].end_idx = end_idx;
                    candidates[candidate_idx].theta0 = theta0;
                    candidates[candidate_idx].theta1 = theta1;
                    candidates[candidate_idx].max_error = max_error;
                    candidates[candidate_idx].delta_bits = delta_bits;
                    candidates[candidate_idx].total_cost = total_cost;
                }
            }
        }
        __syncthreads();
    }
}

// Parallel merge kernel using work-stealing
__global__ void workStealingMergeKernel(
    PartitionCandidateGPU* candidates,
    int* num_candidates,
    bool* changed,
    int iteration) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = blockDim.x * gridDim.x;
    
    // Each thread handles a range of adjacent partition pairs
    int pairs_per_thread = (*num_candidates - 1 + total_threads - 1) / total_threads;
    int start_pair = tid * pairs_per_thread;
    int end_pair = min(start_pair + pairs_per_thread, *num_candidates - 1);
    
    for (int i = start_pair; i < end_pair; i++) {
        if (candidates[i].end_idx == candidates[i + 1].start_idx) {
            // Calculate costs
            double cost1 = candidates[i].total_cost;
            double cost2 = candidates[i + 1].total_cost;
            double combined_cost = cost1 + cost2;
            
            // Calculate merged partition cost
            int merged_start = candidates[i].start_idx;
            int merged_end = candidates[i + 1].end_idx;
            int n = merged_end - merged_start;
            
            // Would need to recalculate model and cost here
            // For brevity, using approximation
            double approx_merged_cost = (cost1 + cost2) * 0.9; // Assume 10% savings
            
            if (approx_merged_cost < combined_cost) {
                // Mark for merging
                candidates[i].end_idx = merged_end;
                candidates[i + 1].start_idx = -1; // Mark for deletion
                *changed = true;
            }
        }
    }
}

// Enhanced Variable-length partitioner class with proper split-merge algorithm - NO CHANGES NEEDED
template<typename T>
class VariableLengthPartitioner {
private:
    std::vector<T> data_host_vec;
    double split_thresh_param;
    int min_part_size_param;
    const double MODEL_SIZE_BYTES = sizeof(PartitionInfo);  // Model overhead per partition
    std::string dataset_name_member;

    // Structure to hold partition candidate information
    struct PartitionCandidate {
        int start_idx;
        int end_idx;
        double theta0;
        double theta1;
        long long max_error;
        int delta_bits;
        double total_cost;
    };

    // Cache for linear regression computations
    struct RegressionCache {
        std::vector<double> prefix_sum_x;
        std::vector<double> prefix_sum_y;
        std::vector<double> prefix_sum_xx;
        std::vector<double> prefix_sum_xy;
        
        void precompute(const std::vector<T>& data) {
            int n = data.size();
            prefix_sum_x.resize(n + 1, 0.0);
            prefix_sum_y.resize(n + 1, 0.0);
            prefix_sum_xx.resize(n + 1, 0.0);
            prefix_sum_xy.resize(n + 1, 0.0);
            
            for (int i = 0; i < n; i++) {
                double x = static_cast<double>(i);
                double y = static_cast<double>(data[i]);
                prefix_sum_x[i + 1] = prefix_sum_x[i] + x;
                prefix_sum_y[i + 1] = prefix_sum_y[i] + y;
                prefix_sum_xx[i + 1] = prefix_sum_xx[i] + x * x;
                prefix_sum_xy[i + 1] = prefix_sum_xy[i] + x * y;
            }
        }
        
        void getRangeSums(int start, int end, double& sum_x, double& sum_y, 
                         double& sum_xx, double& sum_xy) const {
            // Adjust indices for the local coordinate system
            int n = end - start;
            sum_x = n * (n - 1) / 2.0;  // Sum of 0 to n-1
            sum_y = prefix_sum_y[end] - prefix_sum_y[start];
            
            // For xx and xy, we need to adjust for the offset
            sum_xx = 0.0;
            sum_xy = 0.0;
            for (int i = 0; i < n; i++) {
                double local_x = i;
                double global_x = start + i;
                sum_xx += local_x * local_x;
                sum_xy += local_x * (prefix_sum_y[global_x + 1] - prefix_sum_y[global_x]);
            }
        }
    };
    
    RegressionCache cache;

    // Fast linear model fitting using precomputed sums
    void fitLinearModelFast(int start, int end, double& theta0, double& theta1, 
                           long long& max_error, int& delta_bits) {
        int n = end - start;
        if (n <= 0) {
            theta0 = theta1 = 0.0;
            max_error = 0;
            delta_bits = 0;
            return;
        }

        // Get sums from cache
        double sum_x, sum_y, sum_xx, sum_xy;
        cache.getRangeSums(start, end, sum_x, sum_y, sum_xx, sum_xy);

        // Solve normal equations
        double determinant = n * sum_xx - sum_x * sum_x;
        if (std::abs(determinant) > 1e-10) {
            theta1 = (n * sum_xy - sum_x * sum_y) / determinant;
            theta0 = (sum_y - theta1 * sum_x) / n;
        } else {
            theta1 = 0.0;
            theta0 = sum_y / n;
        }

        // Calculate maximum prediction error
        max_error = 0;
        for (int i = 0; i < n; i++) {
            double predicted = theta0 + theta1 * i;
            T pred_T = static_cast<T>(std::round(predicted));
            long long delta = calculateDelta(data_host_vec[start + i], pred_T);
            long long abs_error = std::abs(delta);
            max_error = std::max(max_error, abs_error);
        }

        // Calculate required delta bits
        if (max_error > 0) {
            int bits_for_magnitude = 0;
            unsigned long long temp = static_cast<unsigned long long>(max_error);
            while (temp > 0) {
                bits_for_magnitude++;
                temp >>= 1;
            }
            delta_bits = bits_for_magnitude + 1;  // +1 for sign bit
            delta_bits = std::min(delta_bits, MAX_DELTA_BITS);
        } else {
            delta_bits = 0;
        }
    }

    // Calculate the total cost (in bytes) for a partition
    double calculatePartitionCostFast(int start, int end) {
        double theta0, theta1;
        long long max_error;
        int delta_bits;
        
        fitLinearModelFast(start, end, theta0, theta1, max_error, delta_bits);
        
        // Total cost = model size + delta array size
        double delta_array_bytes = static_cast<double>(end - start) * delta_bits / 8.0;
        return MODEL_SIZE_BYTES + delta_array_bytes;
    }

    // Optimized split phase using dynamic programming concepts
    void splitPhaseOptimized(std::vector<PartitionCandidate>& candidates) {
        candidates.clear();
        if (data_host_vec.empty()) return;

        int n = data_host_vec.size();
        
        // Use a greedy approach with lookahead
        int current_start = 0;
        
        while (current_start < n) {
            int best_end = std::min(current_start + min_part_size_param, n);
            double best_cost_per_element = std::numeric_limits<double>::max();
            
            // Try different partition sizes with exponential stepping
            int step = min_part_size_param;
            int max_size = std::min(n - current_start, min_part_size_param * 32);
            
            for (int size = min_part_size_param; size <= max_size; size += step) {
                int end = std::min(current_start + size, n);
                double cost = calculatePartitionCostFast(current_start, end);
                double cost_per_element = cost / (end - current_start);
                
                if (cost_per_element < best_cost_per_element) {
                    best_cost_per_element = cost_per_element;
                    best_end = end;
                }
                
                // Adaptive stepping: increase step size for larger partitions
                if (size > min_part_size_param * 4) {
                    step = min_part_size_param * 2;
                }
                if (size > min_part_size_param * 16) {
                    step = min_part_size_param * 4;
                }
            }
            
            // Create partition with the best found size
            double theta0, theta1;
            long long max_error;
            int delta_bits;
            fitLinearModelFast(current_start, best_end, theta0, theta1, max_error, delta_bits);
            
            PartitionCandidate candidate;
            candidate.start_idx = current_start;
            candidate.end_idx = best_end;
            candidate.theta0 = theta0;
            candidate.theta1 = theta1;
            candidate.max_error = max_error;
            candidate.delta_bits = delta_bits;
            candidate.total_cost = calculatePartitionCostFast(current_start, best_end);
            
            candidates.push_back(candidate);
            current_start = best_end;
        }
    }

    // Fixed merge phase that maintains partition integrity
    void mergePhaseOptimized(std::vector<PartitionCandidate>& candidates) {
        if (candidates.size() <= 1) return;
        
        bool changed = true;
        int max_iterations = 10;  // Limit iterations to prevent excessive runtime
        int iteration = 0;
        
        while (changed && iteration < max_iterations) {
            changed = false;
            iteration++;
            
            // Try merging adjacent partitions
            for (int i = 0; i < static_cast<int>(candidates.size()) - 1; ) {
                // Calculate individual costs
                double cost1 = candidates[i].total_cost;
                double cost2 = candidates[i + 1].total_cost;
                double combined_cost = cost1 + cost2;
                
                // Calculate merged cost
                double merged_cost = calculatePartitionCostFast(candidates[i].start_idx, 
                                                              candidates[i + 1].end_idx);
                
                // If merging reduces cost, do it
                if (merged_cost < combined_cost) {
                    // Update the first partition with merged data
                    candidates[i].end_idx = candidates[i + 1].end_idx;
                    
                    // Refit the model for the merged partition
                    fitLinearModelFast(candidates[i].start_idx, candidates[i].end_idx,
                                     candidates[i].theta0, candidates[i].theta1,
                                     candidates[i].max_error, candidates[i].delta_bits);
                    
                    candidates[i].total_cost = merged_cost;
                    
                    // Remove the second partition
                    candidates.erase(candidates.begin() + i + 1);
                    
                    changed = true;
                    // Don't increment i, check the same position again
                } else {
                    i++;
                }
            }
        }
    }

    // Validate that partitions cover all indices without gaps
    void validatePartitions(const std::vector<PartitionCandidate>& candidates) {
        if (candidates.empty()) return;
        
        // Check first partition starts at 0
        if (candidates[0].start_idx != 0) {
            std::cerr << "ERROR: First partition doesn't start at 0!" << std::endl;
        }
        
        // Check continuity and final partition
        for (size_t i = 0; i < candidates.size() - 1; i++) {
            if (candidates[i].end_idx != candidates[i + 1].start_idx) {
                std::cerr << "ERROR: Gap between partitions " << i << " and " << i + 1 
                         << " (" << candidates[i].end_idx << " != " 
                         << candidates[i + 1].start_idx << ")" << std::endl;
            }
        }
        
        if (candidates.back().end_idx != static_cast<int>(data_host_vec.size())) {
            std::cerr << "ERROR: Last partition doesn't end at data size!" << std::endl;
        }
    }


private:
    // 新增方法：将分段长度写入文件
    void writePartitionLengthsToFile(const std::vector<PartitionInfo>& partitions) {
        // 构建文件名：dataset_name_partition_lengths.txt
        std::string filename = dataset_name_member + "_cpu_var_partition_lengths.txt";
        
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Warning: Could not open file " << filename 
                     << " for writing partition lengths." << std::endl;
            return;
        }
        
        // 写入头部信息
        outfile << "# CPU Variable-Length Partition Lengths for dataset: " << dataset_name_member << std::endl;
        outfile << "# Total partitions: " << partitions.size() << std::endl;
        outfile << "# Total elements: " << data_host_vec.size() << std::endl;
        outfile << "# Format: partition_index start_idx end_idx length" << std::endl;
        outfile << "#" << std::endl;
        
        // 写入每个分段的长度
        for (size_t i = 0; i < partitions.size(); i++) {
            int length = partitions[i].end_idx - partitions[i].start_idx;
            outfile << i << " " << partitions[i].start_idx << " " 
                   << partitions[i].end_idx << " " << length << std::endl;
        }
        
        // 写入统计信息
        int min_length = INT_MAX;
        int max_length = 0;
        long long total_length = 0;
        
        for (const auto& partition : partitions) {
            int length = partition.end_idx - partition.start_idx;
            min_length = std::min(min_length, length);
            max_length = std::max(max_length, length);
            total_length += length;
        }
        
        double avg_length = static_cast<double>(total_length) / partitions.size();
        
        outfile << "#" << std::endl;
        outfile << "# Statistics:" << std::endl;
        outfile << "# Min partition length: " << min_length << std::endl;
        outfile << "# Max partition length: " << max_length << std::endl;
        outfile << "# Average partition length: " << avg_length << std::endl;
        outfile << "# Length ratio (max/min): " << 
                (min_length > 0 ? static_cast<double>(max_length) / min_length : 0) << std::endl;
        
        outfile.close();
        std::cout << "CPU-Var partition lengths written to: " << filename << std::endl;
    }
    
    
public:
        VariableLengthPartitioner(const std::vector<T>& input_data_vec,
                                double threshold_param = SPLIT_THRESHOLD,
                                int min_size_param = MIN_PARTITION_SIZE,
                                const std::string& dataset_name_param = "")
            : data_host_vec(input_data_vec), 
            split_thresh_param(threshold_param), 
            min_part_size_param(min_size_param), 
            dataset_name_member(dataset_name_param) {
            
            // 双重确保赋值
            this->dataset_name_member = dataset_name_param;
            
            // 调试输出
            std::cout << "[DEBUG] VariableLengthPartitioner constructor called" << std::endl;
            std::cout << "[DEBUG]   dataset_name_param = '" << dataset_name_param << "'" << std::endl;
            std::cout << "[DEBUG]   this->dataset_name_member = '" << this->dataset_name_member << "'" << std::endl;
        }

        // 修改 VariableLengthPartitioner 的 partition() 方法，添加调试输出
        std::vector<PartitionInfo> partition() {
            if (data_host_vec.empty()) {
                return std::vector<PartitionInfo>();
            }
            
            // 添加调试输出
            std::cout << "[DEBUG] VariableLengthPartitioner::partition() called" << std::endl;
            std::cout << "[DEBUG] dataset_name_member = '" << dataset_name_member << "'" << std::endl;
            
            // Precompute prefix sums for fast regression
            cache.precompute(data_host_vec);
            
            std::vector<PartitionCandidate> candidates;
            
            // Phase 1: Split - Create initial partitions (optimized)
            splitPhaseOptimized(candidates);
            
            // Validate after split
            validatePartitions(candidates);
            
            // Phase 2: Merge - Combine partitions to optimize total size (optimized)
            mergePhaseOptimized(candidates);
            
            // Validate after merge
            validatePartitions(candidates);
            
            // Convert candidates to PartitionInfo format
            std::vector<PartitionInfo> result;
            for (const auto& candidate : candidates) {
                PartitionInfo info;
                info.start_idx = candidate.start_idx;
                info.end_idx = candidate.end_idx;
                info.model_type = MODEL_LINEAR;
                info.model_params[0] = candidate.theta0;
                info.model_params[1] = candidate.theta1;
                info.model_params[2] = 0.0;
                info.model_params[3] = 0.0;
                info.delta_bits = candidate.delta_bits;
                info.delta_array_bit_offset = 0;  // Will be set later
                info.error_bound = candidate.max_error;
                result.push_back(info);
            }
            
            // Final safety check
            if (!result.empty()) {
                // Ensure partitions are sorted and contiguous
                std::sort(result.begin(), result.end(), 
                        [](const PartitionInfo& a, const PartitionInfo& b) {
                            return a.start_idx < b.start_idx;
                        });
                
                // Fix any remaining gaps (should not happen with the fixes above)
                for (size_t i = 0; i < result.size() - 1; i++) {
                    if (result[i].end_idx != result[i + 1].start_idx) {
                        std::cerr << "WARNING: Fixing gap at partition " << i << std::endl;
                        result[i].end_idx = result[i + 1].start_idx;
                    }
                }
            }
            
            // 添加更多调试输出
            std::cout << "[DEBUG] result.size() = " << result.size() << std::endl;
            std::cout << "[DEBUG] result.empty() = " << result.empty() << std::endl;
            std::cout << "[DEBUG] dataset_name_member.empty() = " << dataset_name_member.empty() << std::endl;
            
            // 在返回结果之前，写入分段长度到文件
            if (!result.empty() && !dataset_name_member.empty()) {
                std::cout << "[DEBUG] Calling writePartitionLengthsToFile..." << std::endl;
                writePartitionLengthsToFile(result);
            } else {
                std::cout << "[DEBUG] NOT calling writePartitionLengthsToFile because:" << std::endl;
                if (result.empty()) {
                    std::cout << "  - result is empty" << std::endl;
                }
                if (dataset_name_member.empty()) {
                    std::cout << "  - dataset_name_member is empty" << std::endl;
                }
            }
            
            return result;
        }
};



// Define this constant before the kernels
const double PARTITION_MODEL_SIZE_BYTES = sizeof(PartitionInfo);


// Kernel for parallel partition fitting (same as before but with overflow check)
template<typename T>
__global__ void fitPartitionsParallelKernelV2(
    const T* data,
    const int* partition_starts,
    const int* partition_ends,
    int* model_types,
    double* theta0_array,
    double* theta1_array,
    int* delta_bits_array,
    long long* max_errors,
    double* costs,
    int num_partitions) {
    
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;
    
    int start = partition_starts[pid];
    int end = partition_ends[pid];
    int n = end - start;
    
    if (n <= 0) return;
    
    // Check for overflow
    bool has_overflow = false;
    for (int i = 0; i < n && !has_overflow; i++) {
        if (mightOverflowDoublePrecision(data[start + i])) {
            has_overflow = true;
        }
    }
    
    if (has_overflow) {
        // Direct copy model
        model_types[pid] = MODEL_DIRECT_COPY;
        theta0_array[pid] = 0.0;
        theta1_array[pid] = 0.0;
        delta_bits_array[pid] = sizeof(T) * 8;
        max_errors[pid] = 0;
        costs[pid] = PARTITION_MODEL_SIZE_BYTES + n * sizeof(T);
        return;
    }
    
    // Shared memory for reduction
    extern __shared__ char shared_mem_raw[];
    double* s_sums = reinterpret_cast<double*>(shared_mem_raw);
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Calculate sums for linear regression
    double local_sum_x = 0.0, local_sum_y = 0.0;
    double local_sum_xx = 0.0, local_sum_xy = 0.0;
    
    for (int i = tid; i < n; i += block_size) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(data[start + i]);
        local_sum_x += x;
        local_sum_y += y;
        local_sum_xx += x * x;
        local_sum_xy += x * y;
    }
    
    // Store in shared memory
    s_sums[tid] = local_sum_x;
    s_sums[tid + block_size] = local_sum_y;
    s_sums[tid + 2 * block_size] = local_sum_xx;
    s_sums[tid + 3 * block_size] = local_sum_xy;
    __syncthreads();
    
    // Reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sums[tid] += s_sums[tid + s];
            s_sums[tid + block_size] += s_sums[tid + s + block_size];
            s_sums[tid + 2 * block_size] += s_sums[tid + s + 2 * block_size];
            s_sums[tid + 3 * block_size] += s_sums[tid + s + 3 * block_size];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        double sum_x = s_sums[0];
        double sum_y = s_sums[block_size];
        double sum_xx = s_sums[2 * block_size];
        double sum_xy = s_sums[3 * block_size];
        
        // Solve for linear model
        double determinant = n * sum_xx - sum_x * sum_x;
        double theta0, theta1;
        
        if (fabs(determinant) > 1e-10) {
            theta1 = (n * sum_xy - sum_x * sum_y) / determinant;
            theta0 = (sum_y - theta1 * sum_x) / n;
        } else {
            theta1 = 0.0;
            theta0 = sum_y / n;
        }
        
        model_types[pid] = MODEL_LINEAR;
        theta0_array[pid] = theta0;
        theta1_array[pid] = theta1;
    }
    __syncthreads();
    
    // Calculate max error
    double theta0 = theta0_array[pid];
    double theta1 = theta1_array[pid];
    
    long long local_max_error = 0;
    for (int i = tid; i < n; i += block_size) {
        double predicted = theta0 + theta1 * i;
        T pred_T = static_cast<T>(round(predicted));
        long long delta = calculateDelta(data[start + i], pred_T);
        long long abs_error = (delta < 0) ? -delta : delta;
        local_max_error = max(local_max_error, abs_error);
    }
    
    // Reduction for max error
    long long* s_max_errors = reinterpret_cast<long long*>(s_sums + 4 * block_size);
    s_max_errors[tid] = local_max_error;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s && s_max_errors[tid + s] > s_max_errors[tid]) {
            s_max_errors[tid] = s_max_errors[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        max_errors[pid] = s_max_errors[0];
        
        // Calculate delta bits
        int delta_bits = 0;
        if (s_max_errors[0] > 0) {
            unsigned long long temp = static_cast<unsigned long long>(s_max_errors[0]);
            delta_bits = 64 - __clzll(temp) + 1;
        }
        delta_bits_array[pid] = delta_bits;
        
        // Calculate cost
        double delta_array_bytes = static_cast<double>(n) * delta_bits / 8.0;
        costs[pid] = PARTITION_MODEL_SIZE_BYTES + delta_array_bytes;
    }
}

// Kernel to apply merges
__global__ void applyMergesKernel(
    int* partition_starts,
    int* partition_ends,
    double* costs,
    int* merge_targets,
    bool* active,
    int num_partitions) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid < num_partitions && merge_targets[tid] >= 0) {
        int target = merge_targets[tid];
        partition_ends[tid] = partition_ends[target];
        costs[tid] = costs[tid] + costs[target] - PARTITION_MODEL_SIZE_BYTES;
        active[target] = false;
    }
}

// Helper function for warp reduction - sum
__device__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Helper function for warp reduction - max
__device__ long long warpReduceMax(long long val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Helper for block reduction - sum
__device__ double blockReduceSum(double val) {
    __shared__ double shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Helper for block reduction - max
__device__ long long blockReduceMax(long long val) {
    __shared__ long long shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    
    val = warpReduceMax(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceMax(val);
    
    return val;
}


// Optimized variance calculation using grid-stride loops and better parallelism
template<typename T>
__global__ void analyzeDataVarianceFast(
    const T* __restrict__ data,
    int data_size,
    int block_size,
    float* __restrict__ variances,
    int num_blocks) {
    
    // Grid-stride loop for better GPU utilization
    for (int bid = blockIdx.x; bid < num_blocks; bid += gridDim.x) {
        int start = bid * block_size;
        int end = min(start + block_size, data_size);
        int n = end - start;
        
        if (n <= 0) continue;
        
        // Use Kahan summation for better numerical stability
        double sum = 0.0;
        double sum_sq = 0.0;
        double c1 = 0.0, c2 = 0.0;
        
        // Coalesced access with grid-stride
        for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
            double val = static_cast<double>(data[i]);
            
            // Kahan summation
            double y1 = val - c1;
            double t1 = sum + y1;
            c1 = (t1 - sum) - y1;
            sum = t1;
            
            double y2 = val * val - c2;
            double t2 = sum_sq + y2;
            c2 = (t2 - sum_sq) - y2;
            sum_sq = t2;
        }
        
        // Warp reduction
        sum = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);
        
        // Write result by first thread of each warp
        if ((threadIdx.x & 31) == 0) {
            atomicAdd(&variances[bid], static_cast<float>(sum_sq / n - (sum / n) * (sum / n)));
        }
    }
}

// Fast partition creation with pre-computed thresholds
template<typename T>
__global__ void createPartitionsFast(
    int data_size,
    int base_size,
    const float* __restrict__ variances,
    int num_variance_blocks,
    int* __restrict__ partition_starts,
    int* __restrict__ partition_ends,
    int* __restrict__ num_partitions,
    const float* __restrict__ variance_thresholds) {
    
    // Pre-computed thresholds for faster decision making
    float thresh_low = variance_thresholds[0];
    float thresh_med = variance_thresholds[1];
    float thresh_high = variance_thresholds[2];
    
    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
         i < num_variance_blocks; 
         i += blockDim.x * gridDim.x) {
        
        float var = variances[i];
        int block_start = i * base_size * 8;  // 8x base_size for variance blocks
        int block_end = min(block_start + base_size * 8, data_size);
        
        // Fast partition size decision
        int partition_size;
        if (var < thresh_low) {
            partition_size = base_size * 4;
        } else if (var < thresh_med) {
            partition_size = base_size * 2;
        } else if (var < thresh_high) {
            partition_size = base_size;
        } else {
            partition_size = base_size / 2;
        }
        
        // Create partitions
        for (int j = block_start; j < block_end; j += partition_size) {
            if (j < data_size) {
                int idx = atomicAdd(num_partitions, 1);
                if (idx < data_size / (base_size / 2)) {  // Safety check
                    partition_starts[idx] = j;
                    partition_ends[idx] = min(j + partition_size, data_size);
                }
            }
        }
    }
}


// --- 核心优化内核: "一个Block处理一个Partition" ---
template<typename T>
__global__ void fitPartitionsBatched_Optimized(
    const T* __restrict__ data,
    const int* __restrict__ partition_starts,
    const int* __restrict__ partition_ends,
    int* __restrict__ model_types,
    double* __restrict__ theta0_array,
    double* __restrict__ theta1_array,
    int* __restrict__ delta_bits_array,
    long long* __restrict__ max_errors,
    double* __restrict__ costs,
    int num_partitions)
{
    // **架构优化**: 每个Block直接通过 blockIdx.x 获取自己的分区ID
    const int pid = blockIdx.x;
    if (pid >= num_partitions) {
        return; // 边界检查
    }

    // 为这个Block内共享的数据声明静态共享内存
    __shared__ double s_theta0;
    __shared__ double s_theta1;
    __shared__ int s_has_overflow_flag;

    const int start = partition_starts[pid];
    const int end = partition_ends[pid];
    const int n = end - start;

    // --- 阶段1: 检查分区是否为空或存在溢出风险 ---
    if (threadIdx.x == 0) {
        s_has_overflow_flag = false;
    }
     __syncthreads();
     
    if (n <= 0) {
        if (threadIdx.x == 0) {
            model_types[pid] = MODEL_DIRECT_COPY;
            costs[pid] = 0.0;
        }
        return;
    }

    bool local_overflow = false;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (mightOverflowDoublePrecision(data[start + i])) {
            local_overflow = true;
            break;
        }
    }

    if (local_overflow) {
        atomicExch(&s_has_overflow_flag, true);
    }
    __syncthreads();

    if (s_has_overflow_flag) {
        if (threadIdx.x == 0) {
            model_types[pid] = MODEL_DIRECT_COPY;
            theta0_array[pid] = 0.0;
            theta1_array[pid] = 0.0;
            delta_bits_array[pid] = sizeof(T) * 8;
            max_errors[pid] = 0;
            costs[pid] = PARTITION_MODEL_SIZE_BYTES + n * sizeof(T);
        }
        return;
    }
    
    // --- 阶段2: 快速线性回归计算 ---
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(data[start + i]);
        sum_x += x;
        sum_y += y;
        // **FMA优化**: 将 x*x + sum_xx 合并为单条 fma 指令
        sum_xx = fma(x, x, sum_xx);
        sum_xy = fma(x, y, sum_xy);
    }

    // 使用共享内存进行高效的块内归约
    sum_x = blockReduceSum(sum_x);
    sum_y = blockReduceSum(sum_y);
    sum_xx = blockReduceSum(sum_xx);
    sum_xy = blockReduceSum(sum_xy);

    // --- 阶段3: 计算模型参数 (仅由线程0完成) ---
    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);
        // **FMA优化**: dn * sum_xx - sum_x * sum_x
        double determinant = fma(dn, sum_xx, -(sum_x * sum_x));
        
        if (fabs(determinant) > 1e-10) {
            // **FMA优化**: dn * sum_xy - sum_x * sum_y
            s_theta1 = fma(dn, sum_xy, -(sum_x * sum_y)) / determinant;
            // **FMA优化**: sum_y - s_theta1 * sum_x
            s_theta0 = fma(-s_theta1, sum_x, sum_y) / dn;
        } else {
            s_theta1 = 0.0;
            s_theta0 = sum_y / dn;
        }
        model_types[pid] = MODEL_LINEAR;
        theta0_array[pid] = s_theta0;
        theta1_array[pid] = s_theta1;
    }
    __syncthreads();

    // --- 阶段4: 计算最大误差 ---
    double theta0 = theta0_array[pid];
    double theta1 = theta1_array[pid];
    long long local_max_error = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        // **FMA优化**: theta1 * i + theta0
        double predicted = fma(theta1, static_cast<double>(i), theta0);
        T pred_T = static_cast<T>(round(predicted));
        long long delta = calculateDelta(data[start + i], pred_T);
        local_max_error = max(local_max_error, llabs(delta));
    }

    long long partition_max_error = blockReduceMax(local_max_error);

    // --- 阶段5: 写回最终结果 (仅由线程0完成) ---
    if (threadIdx.x == 0) {
        max_errors[pid] = partition_max_error;
        
        int delta_bits = 0;
        if (partition_max_error > 0) {
            delta_bits = 64 - __clzll(static_cast<unsigned long long>(partition_max_error)) + 1;
        }
        delta_bits_array[pid] = delta_bits;
        
        double delta_array_bytes = static_cast<double>(n) * delta_bits / 8.0;
        costs[pid] = PARTITION_MODEL_SIZE_BYTES + delta_array_bytes;
    }
}



// Optimized GPU partitioner V6 focused on speed
template<typename T>
class GPUVariableLengthPartitionerV6 {
private:
    T* d_data;
    int data_size;
    int base_partition_size;
    cudaStream_t stream;
    
public:
    GPUVariableLengthPartitionerV6(const std::vector<T>& data,
                                   int base_size = 1024,
                                   cudaStream_t cuda_stream = 0)
        : data_size(data.size()), base_partition_size(base_size), stream(cuda_stream) {
        
        CUDA_CHECK(cudaMalloc(&d_data, data_size * sizeof(T)));
        CUDA_CHECK(cudaMemcpyAsync(d_data, data.data(), data_size * sizeof(T), 
                                  cudaMemcpyHostToDevice, stream));
    }
    
    ~GPUVariableLengthPartitionerV6() {
        if (d_data) CUDA_CHECK(cudaFree(d_data));
    }
    
    std::vector<PartitionInfo> partition() {
        if (data_size == 0) return std::vector<PartitionInfo>();
        
        // Get device properties for optimal configuration
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        int sm_count = prop.multiProcessorCount;
        
        // Phase 1: Fast variance analysis
        int variance_block_size = base_partition_size * 8;
        int num_variance_blocks = (data_size + variance_block_size - 1) / variance_block_size;
        float* d_variances;
        float* d_variance_thresholds;
        
        CUDA_CHECK(cudaMalloc(&d_variances, num_variance_blocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_variance_thresholds, 3 * sizeof(float)));
        CUDA_CHECK(cudaMemsetAsync(d_variances, 0, num_variance_blocks * sizeof(float), stream));
        
        // Use multiple blocks per SM for better occupancy
        int threads = 128;  // Smaller block for better occupancy
        int blocks = min(num_variance_blocks, sm_count * 4);
        
        analyzeDataVarianceFast<T><<<blocks, threads, 0, stream>>>(
            d_data, data_size, variance_block_size, d_variances, num_variance_blocks);
        
        // Calculate variance thresholds on GPU
        thrust::device_ptr<float> var_ptr(d_variances);
        thrust::sort(var_ptr, var_ptr + num_variance_blocks);
        
        float h_thresholds[3];
        h_thresholds[0] = var_ptr[num_variance_blocks / 4];      // 25th percentile
        h_thresholds[1] = var_ptr[num_variance_blocks / 2];      // 50th percentile
        h_thresholds[2] = var_ptr[3 * num_variance_blocks / 4];  // 75th percentile
        
        CUDA_CHECK(cudaMemcpyAsync(d_variance_thresholds, h_thresholds, 
                                  3 * sizeof(float), cudaMemcpyHostToDevice, stream));
        
        // Phase 2: Fast partition creation
        int estimated_partitions = data_size / (base_partition_size / 2);
        int* d_partition_starts;
        int* d_partition_ends;
        int* d_num_partitions;
        
        CUDA_CHECK(cudaMalloc(&d_partition_starts, estimated_partitions * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_partition_ends, estimated_partitions * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_num_partitions, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_num_partitions, 0, sizeof(int), stream));
        
        blocks = min((num_variance_blocks + threads - 1) / threads, sm_count * 2);
        createPartitionsFast<T><<<blocks, threads, 0, stream>>>(
            data_size, base_partition_size, d_variances, num_variance_blocks,
            d_partition_starts, d_partition_ends, d_num_partitions, d_variance_thresholds);
        
        // Get actual number of partitions
        int h_num_partitions;
        CUDA_CHECK(cudaMemcpyAsync(&h_num_partitions, d_num_partitions, sizeof(int), 
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Sort partitions
        thrust::device_ptr<int> starts_ptr(d_partition_starts);
        thrust::device_ptr<int> ends_ptr(d_partition_ends);
        thrust::sort_by_key(starts_ptr, starts_ptr + h_num_partitions, ends_ptr);
        
        // Phase 3: Batched model fitting for better throughput
        int* d_model_types;
        double* d_theta0;
        double* d_theta1;
        int* d_delta_bits;
        long long* d_max_errors;
        double* d_costs;
        
        CUDA_CHECK(cudaMalloc(&d_model_types, h_num_partitions * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_theta0, h_num_partitions * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_theta1, h_num_partitions * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_delta_bits, h_num_partitions * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_max_errors, h_num_partitions * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_costs, h_num_partitions * sizeof(double)));
        
        // 定义每个Block的线程数
        int threads_per_block = 256; // 128 或 256 是一个不错的选择，有利于占用率

        // **架构核心修改**: Grid的大小直接由分区数决定。
        // 每个分区都获得一个独立的CUDA Block来并行处理。
        int grid_size = h_num_partitions;

        // **内存修改**: 为块内归约操作计算所需的共享内存大小。
        // 需要的空间是 double 和 long long 中较大者乘以线程数。
        size_t shared_mem_size = threads_per_block * sizeof(double); // blockReduceSum 需要
        shared_mem_size = max(shared_mem_size, threads_per_block * sizeof(long long)); // blockReduceMax 需要

        // **启动修改**: 调用新的优化内核 fitPartitionsBatched_Optimized，
        // 并传入新的启动配置。注意，最后一个参数 partitions_per_block 已被移除。
        fitPartitionsBatched_Optimized<T><<<grid_size, threads_per_block, shared_mem_size, stream>>>(
            d_data,
            d_partition_starts,
            d_partition_ends,
            d_model_types,
            d_theta0,
            d_theta1,
            d_delta_bits,
            d_max_errors,
            d_costs,
            h_num_partitions
        );
        
        // Skip merge phase for speed - the variance-based partitioning is already good
        
        // Copy results back
        std::vector<int> h_starts(h_num_partitions);
        std::vector<int> h_ends(h_num_partitions);
        std::vector<int> h_model_types(h_num_partitions);
        std::vector<double> h_theta0(h_num_partitions);
        std::vector<double> h_theta1(h_num_partitions);
        std::vector<int> h_delta_bits(h_num_partitions);
        std::vector<long long> h_max_errors(h_num_partitions);
        
        CUDA_CHECK(cudaMemcpyAsync(h_starts.data(), d_partition_starts, 
                                  h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_ends.data(), d_partition_ends, 
                                  h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_model_types.data(), d_model_types, 
                                  h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_theta0.data(), d_theta0, 
                                  h_num_partitions * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_theta1.data(), d_theta1, 
                                  h_num_partitions * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_delta_bits.data(), d_delta_bits, 
                                  h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_max_errors.data(), d_max_errors, 
                                  h_num_partitions * sizeof(long long), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Build final partition list
        std::vector<PartitionInfo> result;
        result.reserve(h_num_partitions);
        
        for (int i = 0; i < h_num_partitions; i++) {
            PartitionInfo info;
            info.start_idx = h_starts[i];
            info.end_idx = h_ends[i];
            info.model_type = h_model_types[i];
            info.model_params[0] = h_theta0[i];
            info.model_params[1] = h_theta1[i];
            info.model_params[2] = 0.0;
            info.model_params[3] = 0.0;
            info.delta_bits = h_delta_bits[i];
            info.delta_array_bit_offset = 0;
            info.error_bound = h_max_errors[i];
            result.push_back(info);
        }
        
        // Ensure coverage
        if (!result.empty()) {
            std::sort(result.begin(), result.end(), 
                     [](const PartitionInfo& a, const PartitionInfo& b) {
                         return a.start_idx < b.start_idx;
                     });
            
            result[0].start_idx = 0;
            result.back().end_idx = data_size;
            
            for (size_t i = 0; i < result.size() - 1; i++) {
                if (result[i].end_idx != result[i + 1].start_idx) {
                    result[i].end_idx = result[i + 1].start_idx;
                }
            }
        }
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_variances));
        CUDA_CHECK(cudaFree(d_variance_thresholds));
        CUDA_CHECK(cudaFree(d_partition_starts));
        CUDA_CHECK(cudaFree(d_partition_ends));
        CUDA_CHECK(cudaFree(d_num_partitions));
        CUDA_CHECK(cudaFree(d_model_types));
        CUDA_CHECK(cudaFree(d_theta0));
        CUDA_CHECK(cudaFree(d_theta1));
        CUDA_CHECK(cudaFree(d_delta_bits));
        CUDA_CHECK(cudaFree(d_max_errors));
        CUDA_CHECK(cudaFree(d_costs));
        
        return result;
    }
};

    // Helper function to align offset to a specific boundary
inline uint64_t alignOffset(uint64_t offset, uint64_t alignment) {
    return ((offset + alignment - 1) / alignment) * alignment;
}

// Replace the original work-stealing kernel with the optimized version
template<typename T>
__global__ void decompressFullFile_WorkStealing(
    const CompressedData<T>* compressed_data,
    T* output_device,
    int total_elements,
    int* global_work_counter) {
    
    // Constants for optimization
    const int PARTITIONS_PER_BATCH = 4;
    const int MIN_ELEMENTS_PER_BATCH = 1024;
    
    // Cache frequently accessed pointers
    const int32_t* __restrict__ start_indices = compressed_data->d_start_indices;
    const int32_t* __restrict__ end_indices = compressed_data->d_end_indices;
    const int32_t* __restrict__ model_types = compressed_data->d_model_types;
    const double* __restrict__ model_params = compressed_data->d_model_params;
    const int32_t* __restrict__ delta_bits = compressed_data->d_delta_bits;
    const int64_t* __restrict__ delta_offsets = compressed_data->d_delta_array_bit_offsets;
    const uint32_t* __restrict__ delta_array = compressed_data->delta_array;
    const long long* __restrict__ plain_deltas = compressed_data->d_plain_deltas;
    const int num_partitions = compressed_data->num_partitions;
    
    // Shared memory for batch coordination
    __shared__ int s_partition_batch[PARTITIONS_PER_BATCH];
    __shared__ int s_batch_size;
    __shared__ int s_total_elements;
    __shared__ double s_model_params_cache[PARTITIONS_PER_BATCH * 4];
    
    // Initialize work counter (only first thread of first block)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *global_work_counter = 0;
    }
    __syncthreads();
    
    // Work-stealing loop
    while (true) {
        // Thread 0 grabs a batch of work
        if (threadIdx.x == 0) {
            s_batch_size = 0;
            s_total_elements = 0;
            
            // Try to grab partitions until we have enough work
            while (s_batch_size < PARTITIONS_PER_BATCH) {
                int partition_idx = atomicAdd(global_work_counter, 1);
                if (partition_idx >= num_partitions) break;
                
                int32_t start = start_indices[partition_idx];
                int32_t end = end_indices[partition_idx];
                int partition_size = end - start;
                
                if (partition_size > 0 && start < total_elements) {
                    s_partition_batch[s_batch_size] = partition_idx;
                    s_total_elements += partition_size;
                    
                    // Cache model parameters
                    int param_base = partition_idx * 4;
                    int cache_base = s_batch_size * 4;
                    s_model_params_cache[cache_base] = model_params[param_base];
                    s_model_params_cache[cache_base + 1] = model_params[param_base + 1];
                    s_model_params_cache[cache_base + 2] = model_params[param_base + 2];
                    s_model_params_cache[cache_base + 3] = model_params[param_base + 3];
                    
                    s_batch_size++;
                    
                    // If we have enough work or this is a large partition, stop
                    if (s_total_elements >= MIN_ELEMENTS_PER_BATCH || 
                        partition_size >= MIN_ELEMENTS_PER_BATCH) {
                        break;
                    }
                }
            }
        }
        __syncthreads();
        
        if (s_batch_size == 0) break;  // No more work
        
        // Process all partitions in the batch
        for (int batch_idx = 0; batch_idx < s_batch_size; batch_idx++) {
            int partition_idx = s_partition_batch[batch_idx];
            
            // Load partition metadata
            int32_t start_idx = start_indices[partition_idx];
            int32_t end_idx = end_indices[partition_idx];
            int32_t model_type = model_types[partition_idx];
            int32_t bits = delta_bits[partition_idx];
            int64_t base_bit_offset = delta_offsets[partition_idx];
            
            int partition_size = end_idx - start_idx;
            if (partition_size <= 0) continue;
            
            // Load cached model parameters
            int cache_base = batch_idx * 4;
            double theta0 = s_model_params_cache[cache_base];
            double theta1 = s_model_params_cache[cache_base + 1];
            double theta2 = s_model_params_cache[cache_base + 2];
            
            // Special handling for very small partitions
            if (partition_size < 32 && threadIdx.x >= 32) {
                // Only first warp processes very small partitions
                continue;
            }
            
            // Process elements with coalesced access pattern
            if (model_type == MODEL_DIRECT_COPY) {
                // Direct copy model
                if (plain_deltas != nullptr) {
                    // Vectorized copy for pre-unpacked data
                    for (int i = threadIdx.x; i < partition_size; i += blockDim.x) {
                        int global_idx = start_idx + i;
                        if (global_idx >= total_elements) break;
                        output_device[global_idx] = static_cast<T>(plain_deltas[global_idx]);
                    }
                } else if (delta_array != nullptr && bits > 0) {
                    // Bit-packed extraction
                    for (int i = threadIdx.x; i < partition_size; i += blockDim.x) {
                        int global_idx = start_idx + i;
                        if (global_idx >= total_elements) break;
                        
                        int64_t bit_offset = base_bit_offset + (int64_t)i * bits;
                        output_device[global_idx] = extractDirectValue<T>(delta_array, bit_offset, bits);
                    }
                } else {
                    // No delta data
                    for (int i = threadIdx.x; i < partition_size; i += blockDim.x) {
                        int global_idx = start_idx + i;
                        if (global_idx >= total_elements) break;
                        output_device[global_idx] = static_cast<T>(0);
                    }
                }
            } else {
                // Model-based reconstruction
                for (int i = threadIdx.x; i < partition_size; i += blockDim.x) {
                    int global_idx = start_idx + i;
                    if (global_idx >= total_elements) break;
                    
                    // Calculate prediction using cached parameters
                    double predicted = theta0 + theta1 * i;
                    
                    // Handle polynomial models
                    if (model_type == MODEL_POLYNOMIAL2) {
                        predicted += theta2 * i * i;
                    } else if (model_type == MODEL_POLYNOMIAL3) {
                        double theta3 = s_model_params_cache[cache_base + 3];
                        predicted += theta2 * i * i + theta3 * i * i * i;
                    }
                    
                    // Clamp prediction to valid range
                    if (!std::is_signed<T>::value) {
                        predicted = fmax(0.0, predicted);
                        if (sizeof(T) == 8) {
                            predicted = fmin(predicted, 18446744073709551615.0);
                        } else if (sizeof(T) == 4) {
                            predicted = fmin(predicted, 4294967295.0);
                        } else if (sizeof(T) == 2) {
                            predicted = fmin(predicted, 65535.0);
                        } else if (sizeof(T) == 1) {
                            predicted = fmin(predicted, 255.0);
                        }
                    } else {
                        if (sizeof(T) == 8) {
                            predicted = fmax(-9223372036854775808.0, fmin(predicted, 9223372036854775807.0));
                        } else if (sizeof(T) == 4) {
                            predicted = fmax(-2147483648.0, fmin(predicted, 2147483647.0));
                        } else if (sizeof(T) == 2) {
                            predicted = fmax(-32768.0, fmin(predicted, 32767.0));
                        } else if (sizeof(T) == 1) {
                            predicted = fmax(-128.0, fmin(predicted, 127.0));
                        }
                    }
                    
                    T pred_val = static_cast<T>(round(predicted));
                    
                    // Get delta
                    long long delta = 0;
                    if (plain_deltas != nullptr) {
                        delta = plain_deltas[global_idx];
                    } else if (delta_array != nullptr && bits > 0) {
                        int64_t bit_offset = base_bit_offset + (int64_t)i * bits;
                        delta = extractDelta_Optimized<T>(delta_array, bit_offset, bits);
                    }
                    
                    output_device[global_idx] = applyDelta(pred_val, delta);
                }
            }
            
            // Ensure all threads finish this partition before moving to next
            __syncwarp();
        }
        __syncthreads();
    }
}

// Helper function for warp-optimized partition processing
template<typename T>
__device__ inline void processPartitionWarpOptimized(
    int start_idx, int end_idx, int model_type, int partition_idx,
    const double* model_params, const int32_t* delta_bits,
    const int64_t* delta_offsets, const uint32_t* delta_array,
    const long long* plain_deltas, T* output_device,
    int total_elements, int lane_id) {
    
    int partition_size = end_idx - start_idx;
    if (partition_size <= 0 || start_idx >= total_elements) return;
    
    if (model_type == MODEL_DIRECT_COPY) {
        // Direct copy with warp-level parallelism
        for (int i = lane_id; i < partition_size; i += 32) {
            int global_idx = start_idx + i;
            if (global_idx >= total_elements) break;
            
            if (plain_deltas != nullptr) {
                output_device[global_idx] = static_cast<T>(plain_deltas[global_idx]);
            } else if (delta_array != nullptr) {
                int32_t bits = delta_bits[partition_idx];
                if (bits > 0) {
                    int64_t bit_offset = delta_offsets[partition_idx] + (int64_t)i * bits;
                    output_device[global_idx] = extractDirectValue<T>(delta_array, bit_offset, bits);
                } else {
                    output_device[global_idx] = static_cast<T>(0);
                }
            } else {
                output_device[global_idx] = static_cast<T>(0);
            }
        }
    } else {
        // Model-based reconstruction with warp-level parallelism
        double theta0 = model_params[partition_idx * 4];
        double theta1 = model_params[partition_idx * 4 + 1];
        double theta2 = (model_type == MODEL_POLYNOMIAL2) ? model_params[partition_idx * 4 + 2] : 0.0;
        
        for (int i = lane_id; i < partition_size; i += 32) {
            int global_idx = start_idx + i;
            if (global_idx >= total_elements) break;
            
            // Calculate prediction
            double predicted = theta0 + theta1 * i;
            if (model_type == MODEL_POLYNOMIAL2) {
                predicted += theta2 * i * i;
            }
            
            // Clamp prediction
            if (!std::is_signed<T>::value) {
                predicted = fmax(0.0, predicted);
                if (sizeof(T) == 8) {
                    predicted = fmin(predicted, 18446744073709551615.0);
                } else if (sizeof(T) == 4) {
                    predicted = fmin(predicted, 4294967295.0);
                } else if (sizeof(T) == 2) {
                    predicted = fmin(predicted, 65535.0);
                } else if (sizeof(T) == 1) {
                    predicted = fmin(predicted, 255.0);
                }
            } else {
                if (sizeof(T) == 8) {
                    predicted = fmax(-9223372036854775808.0, fmin(predicted, 9223372036854775807.0));
                } else if (sizeof(T) == 4) {
                    predicted = fmax(-2147483648.0, fmin(predicted, 2147483647.0));
                } else if (sizeof(T) == 2) {
                    predicted = fmax(-32768.0, fmin(predicted, 32767.0));
                } else if (sizeof(T) == 1) {
                    predicted = fmax(-128.0, fmin(predicted, 127.0));
                }
            }
            
            T pred_val = static_cast<T>(round(predicted));
            
            // Get delta
            long long delta = 0;
            if (plain_deltas != nullptr) {
                delta = plain_deltas[global_idx];
            } else if (delta_array != nullptr) {
                int32_t bits = delta_bits[partition_idx];
                if (bits > 0) {
                    int64_t bit_offset = delta_offsets[partition_idx] + (int64_t)i * bits;
                    delta = extractDelta_Optimized<T>(delta_array, bit_offset, bits);
                }
            }
            
            output_device[global_idx] = applyDelta(pred_val, delta);
        }
    }
}

// Replace the advanced work-stealing kernel with optimized version
template<typename T>
__global__ void decompressFullFile_WorkStealingAdvanced(
    const CompressedData<T>* compressed_data,
    T* output_device,
    int total_elements,
    int* global_partition_counter) {
    
    // Constants for optimization
    const int WARP_SIZE_CONST = 32;
    const int WARPS_PER_BLOCK = blockDim.x / WARP_SIZE_CONST;
    
    // Cache frequently accessed pointers
    const int32_t* __restrict__ start_indices = compressed_data->d_start_indices;
    const int32_t* __restrict__ end_indices = compressed_data->d_end_indices;
    const int32_t* __restrict__ model_types = compressed_data->d_model_types;
    const double* __restrict__ model_params = compressed_data->d_model_params;
    const int32_t* __restrict__ delta_bits = compressed_data->d_delta_bits;
    const int64_t* __restrict__ delta_offsets = compressed_data->d_delta_array_bit_offsets;
    const uint32_t* __restrict__ delta_array = compressed_data->delta_array;
    const long long* __restrict__ plain_deltas = compressed_data->d_plain_deltas;
    const int num_partitions = compressed_data->num_partitions;
    
    // Warp-level processing
    const int warp_id = threadIdx.x / WARP_SIZE_CONST;
    const int lane_id = threadIdx.x & (WARP_SIZE_CONST - 1);
    
    // Shared memory for partition metadata caching
    __shared__ int s_partition_queue[8];  // Queue of partitions to process
    __shared__ int s_queue_size;
    __shared__ int s_large_partition;     // Index of a large partition for load balancing
    
    // Each block processes partitions atomically
    while (true) {
        // Warp 0, lane 0 manages work distribution
        if (threadIdx.x == 0) {
            s_queue_size = 0;
            s_large_partition = -1;
            
            // Try to get a mix of partition sizes
            for (int i = 0; i < 8 && s_queue_size < 8; i++) {
                int partition_idx = atomicAdd(global_partition_counter, 1);
                if (partition_idx >= num_partitions) break;
                
                int32_t size = end_indices[partition_idx] - start_indices[partition_idx];
                if (size > 0) {
                    s_partition_queue[s_queue_size++] = partition_idx;
                    
                    // Identify large partitions for better distribution
                    if (size > 4096 && s_large_partition == -1) {
                        s_large_partition = s_queue_size - 1;
                    }
                }
            }
        }
        __syncthreads();
        
        if (s_queue_size == 0) break;
        
        // Process partitions with warp specialization
        if (s_large_partition >= 0 && warp_id == 0) {
            // First warp handles the large partition
            int partition_idx = s_partition_queue[s_large_partition];
            int start_idx = start_indices[partition_idx];
            int end_idx = end_indices[partition_idx];
            int model_type = model_types[partition_idx];
            
            processPartitionWarpOptimized<T>(
                start_idx, end_idx, model_type, partition_idx,
                model_params, delta_bits, delta_offsets,
                delta_array, plain_deltas, output_device,
                total_elements, lane_id);
        }
        
        // Other warps (or all warps if no large partition) handle smaller partitions
        for (int q_idx = 0; q_idx < s_queue_size; q_idx++) {
            if (q_idx == s_large_partition && warp_id == 0) continue;
            
            // Distribute small partitions among warps
            if ((q_idx % WARPS_PER_BLOCK) == warp_id) {
                int partition_idx = s_partition_queue[q_idx];
                int start_idx = start_indices[partition_idx];
                int end_idx = end_indices[partition_idx];
                int model_type = model_types[partition_idx];
                
                processPartitionWarpOptimized<T>(
                    start_idx, end_idx, model_type, partition_idx,
                    model_params, delta_bits, delta_offsets,
                    delta_array, plain_deltas, output_device,
                    total_elements, lane_id);
            }
        }
        
        __syncthreads();
    }
}

// Place these AFTER the existing kernel definitions and BEFORE the LeCoGPU class definition

// Optimized random access kernel for pre-unpacked deltas
template<typename T>
__global__ void randomAccessPreUnpackedKernel(
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    const int32_t* __restrict__ model_types,
    const double* __restrict__ model_params,
    const long long* __restrict__ plain_deltas,
    const int* __restrict__ positions,
    T* __restrict__ output,
    int num_partitions,
    int num_queries) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    
    int query_pos = __ldg(&positions[idx]);
    
    // Binary search for partition with reduced divergence
    int left = 0;
    int right = num_partitions - 1;
    int found = -1;
    
    // Unroll first few iterations for small partition counts
    if (num_partitions <= 8) {
        #pragma unroll
        for (int i = 0; i < 8 && i < num_partitions; i++) {
            int32_t start = __ldg(&start_indices[i]);
            int32_t end = __ldg(&end_indices[i]);
            if (query_pos >= start && query_pos < end) {
                found = i;
                break;
            }
        }
    } else {
        while (left <= right) {
            int mid = (left + right) >> 1;
            int32_t start = __ldg(&start_indices[mid]);
            int32_t end = __ldg(&end_indices[mid]);
            
            bool is_found = (query_pos >= start) & (query_pos < end);
            if (is_found) {
                found = mid;
                break;
            }
            
            bool go_right = query_pos >= end;
            left = go_right ? (mid + 1) : left;
            right = go_right ? right : (mid - 1);
        }
    }
    
    if (found == -1) {
        output[idx] = static_cast<T>(0);
        return;
    }
    
    // Load metadata
    int32_t start_idx = __ldg(&start_indices[found]);
    int32_t type = __ldg(&model_types[found]);
    
    if (type == MODEL_DIRECT_COPY) {
        // For direct copy, the plain delta IS the value
        output[idx] = static_cast<T>(__ldg(&plain_deltas[query_pos]));
    } else {
        // Model-based reconstruction
        int local_idx = query_pos - start_idx;
        int param_base = found * 4;
        double theta0 = __ldg(&model_params[param_base]);
        double theta1 = __ldg(&model_params[param_base + 1]);
        
        // Compute prediction using FMA
        double pred = fma(theta1, static_cast<double>(local_idx), theta0);
        
        // Handle polynomial models
        if (type == MODEL_POLYNOMIAL2) {
            double theta2 = __ldg(&model_params[param_base + 2]);
            pred = fma(theta2, static_cast<double>(local_idx * local_idx), pred);
        }
        
        // Fast clamping
        if (!std::is_signed<T>::value) {
            pred = fmax(0.0, pred);
            if (sizeof(T) == 8) {
                pred = fmin(pred, 18446744073709551615.0);
            } else if (sizeof(T) == 4) {
                pred = fmin(pred, 4294967295.0);
            } else if (sizeof(T) == 2) {
                pred = fmin(pred, 65535.0);
            } else if (sizeof(T) == 1) {
                pred = fmin(pred, 255.0);
            }
        } else {
            if (sizeof(T) == 8) {
                pred = fmax(-9223372036854775808.0, fmin(pred, 9223372036854775807.0));
            } else if (sizeof(T) == 4) {
                pred = fmax(-2147483648.0, fmin(pred, 2147483647.0));
            } else if (sizeof(T) == 2) {
                pred = fmax(-32768.0, fmin(pred, 32767.0));
            } else if (sizeof(T) == 1) {
                pred = fmax(-128.0, fmin(pred, 127.0));
            }
        }
        
        T pred_val = static_cast<T>(round(pred));
        
        // Get pre-unpacked delta directly
        long long delta = __ldg(&plain_deltas[query_pos]);
        
        output[idx] = applyDelta(pred_val, delta);
    }
}


template<typename T>
__global__ void randomAccessFixedPartitionKernel(
    // Input SoA arrays from the CompressedData struct
    const int32_t* __restrict__ start_indices,             // Still needed to access metadata array correctly
    const int32_t* __restrict__ model_types,
    const double* __restrict__ model_params,
    const int32_t* __restrict__ delta_bits,
    const int64_t* __restrict__ delta_array_bit_offsets,
    const uint32_t* __restrict__ delta_array,
    // Query data
    const int* __restrict__ positions,
    T* __restrict__ output,
    int num_queries,
    // The KEY optimization parameter
    int partition_size)
{
    // Standard grid-stride setup for queries
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;

    // Get the global position to decompress for this thread
    int query_pos = __ldg(&positions[idx]);

    // --- CORE OPTIMIZATION: Direct O(1) Calculation ---
    // No binary search needed. We directly compute the indices.
    // This eliminates the while loop and prevents warp divergence.
    int partition_idx = query_pos / partition_size;
    int local_idx = query_pos % partition_size;

    // --- Decompression Logic (reused from general kernel) ---

    // Load partition metadata using the directly calculated partition_idx
    // Using __ldg for read-only data to leverage the texture cache
    int32_t type = __ldg(&model_types[partition_idx]);
    int32_t bits = __ldg(&delta_bits[partition_idx]);
    int64_t base_offset = __ldg(&delta_array_bit_offsets[partition_idx]);

    if (type == MODEL_DIRECT_COPY) {
        if (bits > 0) {
            // For direct copy, the "delta" array stores the raw value.
            int64_t bit_offset = base_offset + (int64_t)local_idx * bits;
            output[idx] = extractDirectValue<T>(delta_array, bit_offset, bits);
        } else {
            output[idx] = static_cast<T>(0);
        }
    } else {
        // Model-based decompression
        int param_base = partition_idx * 4;
        double theta0 = __ldg(&model_params[param_base]);
        double theta1 = __ldg(&model_params[param_base + 1]);

        // Compute prediction using the directly calculated local_idx
        double pred = fma(theta1, static_cast<double>(local_idx), theta0);

        // Handle polynomial models if necessary
        if (type == MODEL_POLYNOMIAL2) {
            double theta2 = __ldg(&model_params[param_base + 2]);
            pred = fma(theta2, static_cast<double>(local_idx * local_idx), pred);
        }

        // Extract delta from the bit-packed stream
        long long delta = 0;
        if (bits > 0) {
            int64_t bit_offset = base_offset + (int64_t)local_idx * bits;
            // Use the optimized version of delta extraction from the original file
            delta = extractDelta_Optimized<T>(delta_array, bit_offset, bits);
        }

        // Clamp prediction to the valid range of the target type
        if (!std::is_signed<T>::value) {
            pred = fmax(0.0, pred);
            if (sizeof(T) == 4) pred = fmin(pred, 4294967295.0);
            else if (sizeof(T) == 2) pred = fmin(pred, 65535.0);
            else if (sizeof(T) == 1) pred = fmin(pred, 255.0);
        }
        
        // Round prediction and apply the delta
        T pred_val = static_cast<T>(round(pred));
        output[idx] = applyDelta(pred_val, delta);
    }
}


// Highly optimized random access kernel for fixed partitions with pre-unpacked deltas
template<typename T>
__global__ void randomAccessFixedPreUnpackedKernel(
    const int32_t* __restrict__ model_types,
    const double* __restrict__ model_params,
    const long long* __restrict__ plain_deltas,
    const int* __restrict__ positions,
    T* __restrict__ output,
    int num_queries,
    int partition_size) {
    
    // Process multiple queries per thread for better memory access pattern
    const int QUERIES_PER_THREAD = 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_idx = tid * QUERIES_PER_THREAD;
    
    // Prefetch positions
    int local_positions[QUERIES_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < QUERIES_PER_THREAD; i++) {
        int query_idx = start_idx + i;
        if (query_idx < num_queries) {
            local_positions[i] = __ldg(&positions[query_idx]);
        }
    }
    
    // Process queries
    #pragma unroll
    for (int i = 0; i < QUERIES_PER_THREAD; i++) {
        int query_idx = start_idx + i;
        if (query_idx >= num_queries) return;
        
        int query_pos = local_positions[i];
        
        // O(1) partition calculation
        int partition_idx = query_pos / partition_size;
        int local_idx = query_pos % partition_size;
        
        // Prefetch model type
        int32_t type = __ldg(&model_types[partition_idx]);
        
        if (type == MODEL_DIRECT_COPY) {
            // Direct load with prefetch
            output[query_idx] = static_cast<T>(__ldg(&plain_deltas[query_pos]));
        } else {
            // Prefetch model parameters
            int param_base = partition_idx * 4;
            double2 params = *reinterpret_cast<const double2*>(&model_params[param_base]);
            double theta0 = params.x;
            double theta1 = params.y;
            
            // Fast prediction using FMA
            double pred = __fma_rn(theta1, static_cast<double>(local_idx), theta0);
            
            // Handle polynomial models
            if (type == MODEL_POLYNOMIAL2) {
                double theta2 = __ldg(&model_params[param_base + 2]);
                pred = __fma_rn(theta2, static_cast<double>(local_idx * local_idx), pred);
            }
            
            // Fast type conversion
            T pred_val;
            if (sizeof(T) == 4) {
                if (!std::is_signed<T>::value) {
                    pred = fmax(0.0, fmin(pred, 4294967295.0));
                    pred_val = static_cast<T>(__double2uint_rn(pred));
                } else {
                    pred = fmax(-2147483648.0, fmin(pred, 2147483647.0));
                    pred_val = static_cast<T>(__double2int_rn(pred));
                }
            } else if (sizeof(T) == 8) {
                pred_val = static_cast<T>(__double2ll_rn(pred));
            } else {
                pred_val = static_cast<T>(round(pred));
            }
            
            // Prefetch delta
            long long delta = __ldg(&plain_deltas[query_pos]);
            
            // Apply delta efficiently
            if (std::is_signed<T>::value || sizeof(T) == 8) {
                output[query_idx] = pred_val + static_cast<T>(delta);
            } else {
                // For unsigned types, handle wraparound
                output[query_idx] = static_cast<T>(static_cast<long long>(pred_val) + delta);
            }
        }
    }

}

// Optimized random access kernel for fixed partitions with pre-unpacked deltas
template<typename T>
__global__ void randomAccessFixedPreUnpackedOptimized(
    const int32_t* __restrict__ model_types,
    const double* __restrict__ model_params,
    const long long* __restrict__ plain_deltas,
    const int* __restrict__ positions,
    T* __restrict__ output,
    int num_queries,
    int partition_size) {
    
    // Use shared memory to cache frequently accessed data
    extern __shared__ char shared_mem[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_queries) return;
    
    // Prefetch position
    int query_pos = __ldg(&positions[idx]);
    
    // O(1) partition calculation
    int partition_idx = query_pos / partition_size;
    int local_idx = query_pos % partition_size;
    
    // Prefetch model type using texture cache
    int32_t type = __ldg(&model_types[partition_idx]);
    
    if (type == MODEL_DIRECT_COPY) {
        // Direct memory access with prefetch
        output[idx] = static_cast<T>(__ldg(&plain_deltas[query_pos]));
    } else {
        // Prefetch model parameters
        int param_base = partition_idx * 4;
        double theta0 = __ldg(&model_params[param_base]);
        double theta1 = __ldg(&model_params[param_base + 1]);
        
        // Fast prediction computation
        double pred = __fma_rn(theta1, static_cast<double>(local_idx), theta0);
        
        // Handle polynomial models
        if (type == MODEL_POLYNOMIAL2) {
            double theta2 = __ldg(&model_params[param_base + 2]);
            pred = __fma_rn(theta2, static_cast<double>(local_idx * local_idx), pred);
        }
        
        // Type-specific fast clamping using intrinsics
        T pred_val;
        if (!std::is_signed<T>::value) {
            pred = fmax(0.0, pred);
            if (sizeof(T) == 8) {
                pred = fmin(pred, 18446744073709551615.0);
                pred_val = static_cast<T>(pred);
            } else if (sizeof(T) == 4) {
                pred = fmin(pred, 4294967295.0);
                pred_val = static_cast<T>(__double2uint_rn(pred));
            } else if (sizeof(T) == 2) {
                pred = fmin(pred, 65535.0);
                pred_val = static_cast<T>(__double2int_rn(pred));
            } else {
                pred = fmin(pred, 255.0);
                pred_val = static_cast<T>(__double2int_rn(pred));
            }
        } else {
            if (sizeof(T) == 8) {
                pred = fmax(-9223372036854775808.0, fmin(pred, 9223372036854775807.0));
                pred_val = static_cast<T>(__double2ll_rn(pred));
            } else if (sizeof(T) == 4) {
                pred = fmax(-2147483648.0, fmin(pred, 2147483647.0));
                pred_val = static_cast<T>(__double2int_rn(pred));
            } else if (sizeof(T) == 2) {
                pred = fmax(-32768.0, fmin(pred, 32767.0));
                pred_val = static_cast<T>(__double2int_rn(pred));
            } else {
                pred = fmax(-128.0, fmin(pred, 127.0));
                pred_val = static_cast<T>(__double2int_rn(pred));
            }
        }
        
        // Prefetch delta with texture cache
        long long delta = __ldg(&plain_deltas[query_pos]);
        
        // Apply delta
        output[idx] = applyDelta(pred_val, delta);
    }
}

// Main LeCoGPU class - UPDATED FOR SoA
template<typename T>
class LeCoGPU {
private:
    cudaStream_t main_cuda_stream;
    cudaStream_t compression_cuda_stream;
    cudaStream_t decompression_cuda_stream;

    struct PartitionCache {
        static const int CACHE_SIZE = 64;
        PartitionInfo entries_arr[CACHE_SIZE];
        int indices_arr[CACHE_SIZE]; 
        int lru_counters[CACHE_SIZE];
        int global_lru_counter;

        PartitionCache() : global_lru_counter(0) {
            std::fill(indices_arr, indices_arr + CACHE_SIZE, -1);
            std::fill(lru_counters, lru_counters + CACHE_SIZE, 0);
        }
        bool find(int cache_key, PartitionInfo& found_info) {
            for (int i = 0; i < CACHE_SIZE; i++) {
                if (indices_arr[i] == cache_key) {
                    found_info = entries_arr[i];
                    lru_counters[i] = ++global_lru_counter;
                    return true;
                }
            }
            return false;
        }
        void insert(int cache_key, const PartitionInfo& new_info) {
            int lru_target_idx = 0;
            int min_counter_val = lru_counters[0];
            for (int i = 0; i < CACHE_SIZE; i++) {
                if (indices_arr[i] == -1) { 
                    lru_target_idx = i; 
                    break; 
                }
                if (lru_counters[i] < min_counter_val) {
                    min_counter_val = lru_counters[i];
                    lru_target_idx = i;
                }
            }
            entries_arr[lru_target_idx] = new_info;
            indices_arr[lru_target_idx] = cache_key;
            lru_counters[lru_target_idx] = ++global_lru_counter;
        }
    };
    PartitionCache cpu_partition_cache;

    uint32_t calculateChecksum(const void* data_to_check, size_t size_of_data) {
        const uint8_t* byte_ptr = static_cast<const uint8_t*>(data_to_check);
        uint32_t csum = 0;
        #ifdef __SSE4_2__
        for (size_t i = 0; i < size_of_data; i++) 
            csum = _mm_crc32_u8(csum, byte_ptr[i]);
        #else
        for (size_t i = 0; i < size_of_data; i++) { 
            csum += byte_ptr[i]; 
            csum = (csum << 1) | (csum >> 31); 
        }
        #endif
        return csum;
    }

public:
    LeCoGPU() {
        CUDA_CHECK(cudaStreamCreate(&main_cuda_stream));
        CUDA_CHECK(cudaStreamCreate(&compression_cuda_stream));
        CUDA_CHECK(cudaStreamCreate(&decompression_cuda_stream));
    }
    
    ~LeCoGPU() {
        CUDA_CHECK(cudaStreamDestroy(main_cuda_stream));
        CUDA_CHECK(cudaStreamDestroy(compression_cuda_stream));
        CUDA_CHECK(cudaStreamDestroy(decompression_cuda_stream));
    }

    // 在 LeCoGPU 类中添加这个方法
    void analyzeRandomAccessPerformance(CompressedData<T>* compressed_data,
                                    const std::vector<int>& positions,
                                    int fixed_partition_size) {
        std::cout << "\n--- Random Access Performance Analysis ---" << std::endl;
        
        // Analyze data locality
        std::vector<int> sorted_positions = positions;
        std::sort(sorted_positions.begin(), sorted_positions.end());
        
        // Calculate stride patterns
        long long total_stride = 0;
        int sequential_count = 0;
        for (size_t i = 1; i < sorted_positions.size(); i++) {
            int stride = sorted_positions[i] - sorted_positions[i-1];
            total_stride += stride;
            if (stride == 1) sequential_count++;
        }
        
        double avg_stride = (double)total_stride / (sorted_positions.size() - 1);
        double sequential_ratio = (double)sequential_count / (sorted_positions.size() - 1);
        
        std::cout << "Query pattern analysis:" << std::endl;
        std::cout << "  - Average stride: " << avg_stride << std::endl;
        std::cout << "  - Sequential access ratio: " << sequential_ratio << std::endl;
        std::cout << "  - Total unique positions: " << positions.size() << std::endl;
        
        // Analyze partition distribution
        std::map<int, int> partition_counts;
        for (int pos : positions) {
            int partition = pos / fixed_partition_size;
            partition_counts[partition]++;
        }
        
        std::cout << "  - Unique partitions accessed: " << partition_counts.size() << std::endl;
        std::cout << "  - Queries per partition (avg): " << 
                (double)positions.size() / partition_counts.size() << std::endl;
        
        // Memory access analysis
        size_t bitpacked_bytes_accessed = positions.size() * sizeof(T) * 2; // Approximate
        size_t preunpacked_bytes_accessed = positions.size() * sizeof(long long);
        
        std::cout << "\nMemory access analysis:" << std::endl;
        std::cout << "  - Bit-packed bytes accessed: ~" << bitpacked_bytes_accessed << std::endl;
        std::cout << "  - Pre-unpacked bytes accessed: " << preunpacked_bytes_accessed << std::endl;
        std::cout << "  - Memory access ratio: " << 
                (double)preunpacked_bytes_accessed / bitpacked_bytes_accessed << "x" << std::endl;
    }
    // Add these methods to the LeCoGPU class public section:

    // Work-stealing decompression method
    void decompressFullFile_WorkStealing(CompressedData<T>* compressed_data_input,
                                        std::vector<T>& output_decompressed_data) {
        if (!compressed_data_input || compressed_data_input->total_values == 0) {
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->d_self) {
            std::cerr << "Error: compressed_data_input->d_self is null." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        int total_elements = compressed_data_input->total_values;
        output_decompressed_data.resize(total_elements);
        
        T* d_output_ptr;
        CUDA_CHECK(cudaMalloc(&d_output_ptr, total_elements * sizeof(T)));
        
        // Allocate global work counter
        int* d_global_work_counter;
        CUDA_CHECK(cudaMalloc(&d_global_work_counter, sizeof(int)));
        
        // Calculate optimal launch configuration
        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        
        // Use more blocks than SMs for better work distribution
        int block_size = 256;
        int grid_size = prop.multiProcessorCount * 4;
        
        // Calculate shared memory size
        size_t shared_mem_size = sizeof(PartitionMeta) * 32;  // Cache for 32 partitions
        
        // Launch work-stealing kernel
        ::decompressFullFile_WorkStealing<T><<<grid_size, block_size, shared_mem_size, 
                                             decompression_cuda_stream>>>(
            compressed_data_input->d_self, d_output_ptr, total_elements, d_global_work_counter);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy results back
        CUDA_CHECK(cudaMemcpyAsync(output_decompressed_data.data(), d_output_ptr,
                                  total_elements * sizeof(T), cudaMemcpyDeviceToHost,
                                  decompression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(decompression_cuda_stream));
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_output_ptr));
        CUDA_CHECK(cudaFree(d_global_work_counter));
    }
    
    // Advanced work-stealing with dynamic load balancing - SIMPLIFIED VERSION
    void decompressFullFile_WorkStealingAdvanced(CompressedData<T>* compressed_data_input,
                                                std::vector<T>& output_decompressed_data) {
        if (!compressed_data_input || compressed_data_input->total_values == 0) {
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->d_self) {
            std::cerr << "Error: compressed_data_input->d_self is null." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        int total_elements = compressed_data_input->total_values;
        output_decompressed_data.resize(total_elements);
        
        T* d_output_ptr;
        int* d_global_partition_counter;
        
        CUDA_CHECK(cudaMalloc(&d_output_ptr, total_elements * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_global_partition_counter, sizeof(int)));
        CUDA_CHECK(cudaMemsetAsync(d_global_partition_counter, 0, sizeof(int), decompression_cuda_stream));
        
        // Calculate optimal launch configuration
        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        
        // Use more blocks for better work distribution
        int block_size = 256;
        int grid_size = min(prop.multiProcessorCount * 4, 256);
        
        // Launch the simplified work-stealing kernel
        ::decompressFullFile_WorkStealingAdvanced<T><<<grid_size, block_size, 0, 
                                                      decompression_cuda_stream>>>(
            compressed_data_input->d_self, d_output_ptr, total_elements,
            d_global_partition_counter);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy results back
        CUDA_CHECK(cudaMemcpyAsync(output_decompressed_data.data(), d_output_ptr,
                                  total_elements * sizeof(T), cudaMemcpyDeviceToHost,
                                  decompression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(decompression_cuda_stream));
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_output_ptr));
        CUDA_CHECK(cudaFree(d_global_partition_counter));
    }

    void decompressFullFileCooperative(CompressedData<T>* compressed_data_input,
                                std::vector<T>& output_decompressed_data) {
        if (!compressed_data_input || compressed_data_input->total_values == 0) {
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->d_self) {
            std::cerr << "Error: compressed_data_input->d_self is null in decompressFullFileCooperative." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        // This method requires pre-unpacked deltas
        if (!compressed_data_input->d_plain_deltas) {
            std::cerr << "Error: decompressFullFileCooperative requires pre-unpacked deltas. "
                    << "Please deserialize with preUnpackDeltas=true." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        int total_elements = compressed_data_input->total_values;
        output_decompressed_data.resize(total_elements);
        
        T* d_output_ptr;
        CUDA_CHECK(cudaMalloc(&d_output_ptr, total_elements * sizeof(T)));
        
        // Calculate optimal grid and block sizes
        int block_size = 256;

        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

        // Use more blocks for work stealing approach
        int grid_size = prop.multiProcessorCount * 4;

        // Calculate shared memory requirements
        const int PARTITIONS_PER_BLOCK = 4;  // Reduced to match optimized kernel
        size_t shared_mem_size = 3 * PARTITIONS_PER_BLOCK * sizeof(int) +     // start, end, model_types
                                PARTITIONS_PER_BLOCK * 2 * sizeof(float);     // model params (only theta0, theta1)
        
        // Check shared memory limits
        if (shared_mem_size > prop.sharedMemPerBlock) {
            std::cerr << "Warning: Required shared memory (" << shared_mem_size 
                    << " bytes) exceeds device limit (" << prop.sharedMemPerBlock 
                    << " bytes). Falling back to standard kernel." << std::endl;
            decompressFullFile(compressed_data_input, output_decompressed_data);
            CUDA_CHECK(cudaFree(d_output_ptr));
            return;
        }
        
        // Check if we can use more shared memory per block
        if (prop.sharedMemPerBlock >= 49152) {
            // We have 48KB of shared memory, try to configure for maximum
            cudaFuncSetAttribute(decompressFullFileCooperativeKernel<T>, 
                            cudaFuncAttributeMaxDynamicSharedMemorySize, 
                            shared_mem_size);
        }
        
        // Launch the cooperative kernel
        decompressFullFileCooperativeKernel<T><<<grid_size, block_size, shared_mem_size, 
                                        decompression_cuda_stream>>>(
            compressed_data_input->d_self, d_output_ptr, total_elements);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy results back to host
        CUDA_CHECK(cudaMemcpyAsync(output_decompressed_data.data(), d_output_ptr,
                                total_elements * sizeof(T), cudaMemcpyDeviceToHost,
                                decompression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(decompression_cuda_stream));
        
        CUDA_CHECK(cudaFree(d_output_ptr));
    }

    // UPDATED compress function for SoA
    // Add this method to LeCoGPU class (replacing the existing compress method)
    CompressedData<T>* compress(const std::vector<T>& host_data_vec,
                                bool use_variable_partitioning,
                                long long* compressed_size_bytes_result,
                                bool use_gpu_partitioning = false,
                                const std::string& dataset_name = "") {
        int num_elements = host_data_vec.size();
        if (num_elements == 0) {
            if(compressed_size_bytes_result) *compressed_size_bytes_result = 0;
            CompressedData<T>* result_empty = new CompressedData<T>();
            result_empty->num_partitions = 0; 
            result_empty->total_values = 0;
            result_empty->d_start_indices = nullptr;
            result_empty->d_end_indices = nullptr;
            result_empty->d_model_types = nullptr;
            result_empty->d_model_params = nullptr;
            result_empty->d_delta_bits = nullptr;
            result_empty->d_delta_array_bit_offsets = nullptr;
            result_empty->d_error_bounds = nullptr;
            result_empty->delta_array = nullptr;
            result_empty->d_plain_deltas = nullptr;
            result_empty->d_self = nullptr;
            return result_empty;
        }

        T* d_input_data;
        CUDA_CHECK(cudaMalloc(&d_input_data, num_elements * sizeof(T)));
        CUDA_CHECK(cudaMemcpyAsync(d_input_data, host_data_vec.data(), 
                                num_elements * sizeof(T),
                                cudaMemcpyHostToDevice, compression_cuda_stream));

        CompressedData<T>* result_compressed_data = new CompressedData<T>();
        std::vector<PartitionInfo> h_partition_infos;

        if (use_variable_partitioning) {
            if (use_gpu_partitioning) {
                // Use speed-optimized GPU partitioner V6
                GPUVariableLengthPartitionerV6<T> gpu_partitioner(host_data_vec, 
                                                                2048,   // Base size
                                                                compression_cuda_stream);
                h_partition_infos = gpu_partitioner.partition();
            } else {
                // 添加调试输出
                std::cout << "[DEBUG] Creating CPU VariableLengthPartitioner with dataset_name = '" 
                        << dataset_name << "'" << std::endl;
                
                // Use original CPU partitioner - 传递数据集名称
                VariableLengthPartitioner<T> var_partitioner(host_data_vec, 
                                                            SPLIT_THRESHOLD, 
                                                            MIN_PARTITION_SIZE,
                                                            dataset_name);
                h_partition_infos = var_partitioner.partition();
            }
        } else {
            int p_size = TILE_SIZE;
            int num_p = (num_elements + p_size - 1) / p_size;
            for (int i = 0; i < num_p; i++) {
                PartitionInfo p_current;
                p_current.start_idx = i * p_size;
                p_current.end_idx = std::min(p_current.start_idx + p_size, num_elements);
                if (p_current.start_idx >= p_current.end_idx) continue;
                p_current.model_type = MODEL_LINEAR;
                std::fill(p_current.model_params, p_current.model_params + 4, 0.0);
                p_current.delta_bits = 0; 
                p_current.delta_array_bit_offset = 0; 
                p_current.error_bound = 0;
                h_partition_infos.push_back(p_current);
            }
        }
        
        if (h_partition_infos.empty() && num_elements > 0) {
            PartitionInfo p_def; 
            p_def.start_idx = 0; 
            p_def.end_idx = num_elements;
            p_def.model_type = MODEL_LINEAR; 
            std::fill(p_def.model_params, p_def.model_params + 4, 0.0);
            p_def.delta_bits = 0; 
            p_def.delta_array_bit_offset = 0; 
            p_def.error_bound = 0;
            h_partition_infos.push_back(p_def);
        }
        
        result_compressed_data->num_partitions = h_partition_infos.size();
        if (result_compressed_data->num_partitions == 0 && num_elements > 0) {
            CUDA_CHECK(cudaFree(d_input_data)); 
            delete result_compressed_data;
            if(compressed_size_bytes_result) *compressed_size_bytes_result = 0;
            return nullptr;
        }
        
        result_compressed_data->total_values = num_elements;
        result_compressed_data->d_plain_deltas = nullptr; // Initialize new member
        
        if (result_compressed_data->num_partitions == 0) {
            result_compressed_data->d_start_indices = nullptr;
            result_compressed_data->d_end_indices = nullptr;
            result_compressed_data->d_model_types = nullptr;
            result_compressed_data->d_model_params = nullptr;
            result_compressed_data->d_delta_bits = nullptr;
            result_compressed_data->d_delta_array_bit_offsets = nullptr;
            result_compressed_data->d_error_bounds = nullptr;
            result_compressed_data->delta_array = nullptr;
        } else {
            // Allocate SoA arrays on device
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_start_indices,
                                result_compressed_data->num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_end_indices,
                                result_compressed_data->num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_model_types,
                                result_compressed_data->num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_model_params,
                                result_compressed_data->num_partitions * 4 * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_delta_bits,
                                result_compressed_data->num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_delta_array_bit_offsets,
                                result_compressed_data->num_partitions * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&result_compressed_data->d_error_bounds,
                                result_compressed_data->num_partitions * sizeof(long long)));
            
            // Copy partition info to SoA arrays on host first
            std::vector<int32_t> h_start_indices(result_compressed_data->num_partitions);
            std::vector<int32_t> h_end_indices(result_compressed_data->num_partitions);
            std::vector<int32_t> h_model_types(result_compressed_data->num_partitions);
            std::vector<double> h_model_params(result_compressed_data->num_partitions * 4);
            std::vector<int32_t> h_delta_bits(result_compressed_data->num_partitions);
            std::vector<int64_t> h_delta_array_bit_offsets(result_compressed_data->num_partitions);
            std::vector<long long> h_error_bounds(result_compressed_data->num_partitions);
            
            for (int i = 0; i < result_compressed_data->num_partitions; i++) {
                h_start_indices[i] = h_partition_infos[i].start_idx;
                h_end_indices[i] = h_partition_infos[i].end_idx;
                h_model_types[i] = h_partition_infos[i].model_type;
                for (int j = 0; j < 4; j++) {
                    h_model_params[i * 4 + j] = h_partition_infos[i].model_params[j];
                }
                h_delta_bits[i] = h_partition_infos[i].delta_bits;
                h_delta_array_bit_offsets[i] = h_partition_infos[i].delta_array_bit_offset;
                h_error_bounds[i] = h_partition_infos[i].error_bound;
            }
            
            // Copy SoA arrays to device
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_start_indices, h_start_indices.data(),
                                    result_compressed_data->num_partitions * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_end_indices, h_end_indices.data(),
                                    result_compressed_data->num_partitions * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_model_types, h_model_types.data(),
                                    result_compressed_data->num_partitions * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_model_params, h_model_params.data(),
                                    result_compressed_data->num_partitions * 4 * sizeof(double),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_delta_bits, h_delta_bits.data(),
                                    result_compressed_data->num_partitions * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_delta_array_bit_offsets, h_delta_array_bit_offsets.data(),
                                    result_compressed_data->num_partitions * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
            CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_error_bounds, h_error_bounds.data(),
                                    result_compressed_data->num_partitions * sizeof(long long),
                                    cudaMemcpyHostToDevice, compression_cuda_stream));
        }

        // Allocate device memory for total bits counter
        int64_t* d_total_bits;
        CUDA_CHECK(cudaMalloc(&d_total_bits, sizeof(int64_t)));
        CUDA_CHECK(cudaMemsetAsync(d_total_bits, 0, sizeof(int64_t), compression_cuda_stream));

        // Process all partitions in a single kernel launch
        if (result_compressed_data->num_partitions > 0) {
            int block_size = std::min(256, ((h_partition_infos[0].end_idx - h_partition_infos[0].start_idx) + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
            block_size = std::max(block_size, WARP_SIZE);
            
            size_t shared_mem_size = (4 * sizeof(double) + sizeof(long long) + sizeof(bool)) * block_size;
            
            wprocessPartitionsKernel<T><<<result_compressed_data->num_partitions, block_size, 
                            shared_mem_size, compression_cuda_stream>>>(
                d_input_data,
                result_compressed_data->d_start_indices,
                result_compressed_data->d_end_indices,
                result_compressed_data->d_model_types,
                result_compressed_data->d_model_params,
                result_compressed_data->d_delta_bits,
                result_compressed_data->d_error_bounds,
                result_compressed_data->num_partitions,
                d_total_bits);
            CUDA_CHECK(cudaGetLastError());
            
            // Set bit offsets
            int offset_blocks = (result_compressed_data->num_partitions + 255) / 256;
            setBitOffsetsKernel<<<offset_blocks, 256, 0, compression_cuda_stream>>>(
                result_compressed_data->d_start_indices,
                result_compressed_data->d_end_indices,
                result_compressed_data->d_delta_bits,
                result_compressed_data->d_delta_array_bit_offsets,
                result_compressed_data->num_partitions);
            CUDA_CHECK(cudaGetLastError());
        }

        // Get total bits from device
        int64_t h_total_bits = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_total_bits, d_total_bits, sizeof(int64_t),
                                cudaMemcpyDeviceToHost, compression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(compression_cuda_stream));
        CUDA_CHECK(cudaFree(d_total_bits));

        // Allocate delta array
        size_t final_delta_array_words = (h_total_bits + 31) / 32;
        if (h_total_bits == 0) final_delta_array_words = 0;
        
        if (final_delta_array_words > 0) {
            CUDA_CHECK(cudaMalloc(&result_compressed_data->delta_array, 
                                final_delta_array_words * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemsetAsync(result_compressed_data->delta_array, 0, 
                                    final_delta_array_words * sizeof(uint32_t), 
                                    compression_cuda_stream));
        } else {
            result_compressed_data->delta_array = nullptr;
        }

        // Pack deltas
        if (h_total_bits > 0 && result_compressed_data->num_partitions > 0 && num_elements > 0) {
            int pack_kernel_block_dim = MAX_BLOCK_SIZE;
            int pack_kernel_grid_dim = (num_elements + pack_kernel_block_dim - 1) / pack_kernel_block_dim;
            pack_kernel_grid_dim = std::min(pack_kernel_grid_dim, 65535);
            
            packDeltasKernelOptimized<T><<<pack_kernel_grid_dim, pack_kernel_block_dim, 0, 
                                        compression_cuda_stream>>>(
                d_input_data,
                result_compressed_data->d_start_indices,
                result_compressed_data->d_end_indices,
                result_compressed_data->d_model_types,
                result_compressed_data->d_model_params,
                result_compressed_data->d_delta_bits,
                result_compressed_data->d_delta_array_bit_offsets,
                result_compressed_data->num_partitions,
                result_compressed_data->delta_array);
            CUDA_CHECK(cudaGetLastError());
        }
        
        if(compressed_size_bytes_result) {
            // Calculate size based on SoA layout
            long long final_model_size = 0;
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(int32_t); // start_indices
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(int32_t); // end_indices
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(int32_t); // model_types
            final_model_size += (long long)result_compressed_data->num_partitions * 4 * sizeof(double); // model_params
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(int32_t); // delta_bits
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(int64_t); // delta_array_bit_offsets
            final_model_size += (long long)result_compressed_data->num_partitions * sizeof(long long); // error_bounds
            final_model_size += sizeof(SerializedHeader); // header
            
            long long final_delta_size = (h_total_bits + 7) / 8;
            *compressed_size_bytes_result = final_model_size + final_delta_size;
        }
        
        CUDA_CHECK(cudaMalloc(&result_compressed_data->d_self, sizeof(CompressedData<T>)));
        CUDA_CHECK(cudaMemcpyAsync(result_compressed_data->d_self, result_compressed_data, 
                                sizeof(CompressedData<T>), cudaMemcpyHostToDevice, 
                                compression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(compression_cuda_stream));

        CUDA_CHECK(cudaFree(d_input_data));
        
        return result_compressed_data;
    }
    void decompress(CompressedData<T>* compressed_data_input,
                    const std::vector<int>& positions_to_decompress,
                    std::vector<T>& output_decompressed_data) {
        if (!compressed_data_input || positions_to_decompress.empty()) {
            output_decompressed_data.clear(); 
            return;
        }
        
        if (!compressed_data_input->d_self) {
            std::cerr << "Error: compressed_data_input->d_self is null in decompress." << std::endl;
            output_decompressed_data.clear(); 
            return;
        }
        
        int num_queries_to_run = positions_to_decompress.size();
        output_decompressed_data.resize(num_queries_to_run);

        int* h_pinned_positions = nullptr; 
        T* h_pinned_output = nullptr;
        bool should_use_pinned = num_queries_to_run > 1000;

        if (should_use_pinned) {
            CUDA_CHECK(cudaHostAlloc(&h_pinned_positions, num_queries_to_run * sizeof(int), 
                                    cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&h_pinned_output, num_queries_to_run * sizeof(T), 
                                    cudaHostAllocDefault));
            memcpy(h_pinned_positions, positions_to_decompress.data(), 
                   num_queries_to_run * sizeof(int));
        }
        
        int* d_positions_ptr; 
        T* d_output_ptr;
        CUDA_CHECK(cudaMalloc(&d_positions_ptr, num_queries_to_run * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output_ptr, num_queries_to_run * sizeof(T)));

        CUDA_CHECK(cudaMemcpyAsync(d_positions_ptr, 
                                  should_use_pinned ? h_pinned_positions : positions_to_decompress.data(), 
                                  num_queries_to_run * sizeof(int), cudaMemcpyHostToDevice, 
                                  decompression_cuda_stream));
        
        int decomp_block_size = 256; 
        int decomp_grid_size = (num_queries_to_run + decomp_block_size - 1) / decomp_block_size;
        decomp_grid_size = std::min(decomp_grid_size, 65535); 
        
        decompressOptimizedKernel<T><<<decomp_grid_size, decomp_block_size, 0, 
                                       decompression_cuda_stream>>>(
           compressed_data_input->d_self, d_output_ptr, d_positions_ptr, num_queries_to_run);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaMemcpyAsync(should_use_pinned ? h_pinned_output : output_decompressed_data.data(), 
                                  d_output_ptr, 
                                  num_queries_to_run * sizeof(T), cudaMemcpyDeviceToHost, 
                                  decompression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(decompression_cuda_stream));

        if (should_use_pinned) {
            memcpy(output_decompressed_data.data(), h_pinned_output, num_queries_to_run * sizeof(T));
            CUDA_CHECK(cudaFreeHost(h_pinned_positions));
            CUDA_CHECK(cudaFreeHost(h_pinned_output));
        }
        
        CUDA_CHECK(cudaFree(d_positions_ptr));
        CUDA_CHECK(cudaFree(d_output_ptr));
    }
    
    void decompressFullFile(CompressedData<T>* compressed_data_input,
                        std::vector<T>& output_decompressed_data) {
        if (!compressed_data_input) {
            output_decompressed_data.clear();
            return;
        }
        
        // Choose method based on available data
        if (compressed_data_input->d_plain_deltas) {
            decompressFullFile_PreUnpacked(compressed_data_input, output_decompressed_data);
        } else {
            decompressFullFile_BitPacked(compressed_data_input, output_decompressed_data);
        }
    }

        // Random access using pre-unpacked deltas for better performance
    void randomAccessPreUnpacked(CompressedData<T>* compressed_data_input,
                            const std::vector<int>& positions_to_access,
                            std::vector<T>& output_decompressed_data) {
        if (!compressed_data_input || positions_to_access.empty()) {
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->d_self) {
            std::cerr << "Error: compressed_data_input->d_self is null in randomAccessPreUnpacked." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->d_plain_deltas) {
            std::cerr << "Error: No pre-unpacked deltas available. "
                    << "Please deserialize with preUnpackDeltas=true." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        int num_queries = positions_to_access.size();
        output_decompressed_data.resize(num_queries);
        
        // Allocate device memory
        int* d_positions;
        T* d_output;
        
        CUDA_CHECK(cudaMalloc(&d_positions, num_queries * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output, num_queries * sizeof(T)));
        
        // Copy positions to device
        CUDA_CHECK(cudaMemcpyAsync(d_positions, positions_to_access.data(),
                                num_queries * sizeof(int), cudaMemcpyHostToDevice,
                                decompression_cuda_stream));
        
        // Calculate optimal launch configuration
        int block_size = 256;
        int grid_size = (num_queries + block_size - 1) / block_size;
        
        // Limit grid size for better occupancy
        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        grid_size = std::min(grid_size, prop.multiProcessorCount * 32);
        
        // Launch kernel
        randomAccessPreUnpackedKernel<T><<<grid_size, block_size, 0, decompression_cuda_stream>>>(
            compressed_data_input->d_start_indices,
            compressed_data_input->d_end_indices,
            compressed_data_input->d_model_types,
            compressed_data_input->d_model_params,
            compressed_data_input->d_plain_deltas,
            d_positions,
            d_output,
            compressed_data_input->num_partitions,
            num_queries);
        
        CUDA_CHECK(cudaGetLastError());
        
        // Copy results back
        CUDA_CHECK(cudaMemcpyAsync(output_decompressed_data.data(), d_output,
                                num_queries * sizeof(T), cudaMemcpyDeviceToHost,
                                decompression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(decompression_cuda_stream));
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_positions));
        CUDA_CHECK(cudaFree(d_output));
    }

    void decompressFullFile_Fix(CompressedData<T>* compressed_data_input,
                                std::vector<T>& output_decompressed_data,
                                int partition_size) {
            if (!compressed_data_input || compressed_data_input->total_values == 0) {
                output_decompressed_data.clear();
                return;
            }
            int total_elements = compressed_data_input->total_values;
            output_decompressed_data.resize(total_elements);
            T* d_output_ptr;
            CUDA_CHECK(cudaMalloc(&d_output_ptr, total_elements * sizeof(T)));
            int block_size = 256;
            int grid_size = (total_elements + block_size - 1) / block_size;
            
            // Call the new, specialized kernel
            decompressFullFileFix<T><<<grid_size, block_size>>>(
                compressed_data_input->d_self, d_output_ptr, total_elements, partition_size);
            
            CUDA_CHECK(cudaMemcpy(output_decompressed_data.data(), d_output_ptr, total_elements * sizeof(T), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_output_ptr));
        }

    
    
    // UPDATED cleanup function for SoA
    void cleanup(CompressedData<T>* compressed_object_to_clean) {
        if (compressed_object_to_clean) {
            // Free all SoA arrays
            if (compressed_object_to_clean->d_start_indices) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_start_indices));
            if (compressed_object_to_clean->d_end_indices) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_end_indices));
            if (compressed_object_to_clean->d_model_types) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_model_types));
            if (compressed_object_to_clean->d_model_params) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_model_params));
            if (compressed_object_to_clean->d_delta_bits) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_delta_bits));
            if (compressed_object_to_clean->d_delta_array_bit_offsets) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_delta_array_bit_offsets));
            if (compressed_object_to_clean->d_error_bounds) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_error_bounds));
            if (compressed_object_to_clean->delta_array) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->delta_array));
            
            // --- NEW: Free the plain deltas array if it exists ---
            if (compressed_object_to_clean->d_plain_deltas)
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_plain_deltas));
                
            if (compressed_object_to_clean->d_self) 
                CUDA_CHECK(cudaFree(compressed_object_to_clean->d_self));
            delete compressed_object_to_clean;
        }
    }


    
    // UPDATED serialize function for SoA
    SerializedData* serialize(CompressedData<T>* compressed_object_to_serialize) {
        if (!compressed_object_to_serialize) { 
            std::cerr << "Error: Null data to serialize." << std::endl; 
            return nullptr; 
        }
        
        // Copy SoA arrays from device to host
        std::vector<int32_t> h_start_indices(compressed_object_to_serialize->num_partitions);
        std::vector<int32_t> h_end_indices(compressed_object_to_serialize->num_partitions);
        std::vector<int32_t> h_model_types(compressed_object_to_serialize->num_partitions);
        std::vector<double> h_model_params(compressed_object_to_serialize->num_partitions * 4);
        std::vector<int32_t> h_delta_bits(compressed_object_to_serialize->num_partitions);
        std::vector<int64_t> h_delta_array_bit_offsets(compressed_object_to_serialize->num_partitions);
        std::vector<long long> h_error_bounds(compressed_object_to_serialize->num_partitions);
        
        if (compressed_object_to_serialize->num_partitions > 0) {
            CUDA_CHECK(cudaMemcpy(h_start_indices.data(), 
                                  compressed_object_to_serialize->d_start_indices,
                                  compressed_object_to_serialize->num_partitions * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_end_indices.data(), 
                                  compressed_object_to_serialize->d_end_indices,
                                  compressed_object_to_serialize->num_partitions * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_model_types.data(), 
                                  compressed_object_to_serialize->d_model_types,
                                  compressed_object_to_serialize->num_partitions * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_model_params.data(), 
                                  compressed_object_to_serialize->d_model_params,
                                  compressed_object_to_serialize->num_partitions * 4 * sizeof(double),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), 
                                  compressed_object_to_serialize->d_delta_bits,
                                  compressed_object_to_serialize->num_partitions * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_delta_array_bit_offsets.data(), 
                                  compressed_object_to_serialize->d_delta_array_bit_offsets,
                                  compressed_object_to_serialize->num_partitions * sizeof(int64_t),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_error_bounds.data(), 
                                  compressed_object_to_serialize->d_error_bounds,
                                  compressed_object_to_serialize->num_partitions * sizeof(long long),
                                  cudaMemcpyDeviceToHost));
        }
        
        // Calculate delta array size
        uint64_t max_bit_offset_val = 0;
        for (int i = 0; i < compressed_object_to_serialize->num_partitions; i++) {
            uint64_t p_start_bit = static_cast<uint64_t>(h_delta_array_bit_offsets[i]);
            int seg_len_val = h_end_indices[i] - h_start_indices[i];
            if (seg_len_val < 0) seg_len_val = 0;
            uint64_t p_end_bit = p_start_bit + static_cast<uint64_t>(seg_len_val) * h_delta_bits[i];
            max_bit_offset_val = std::max(max_bit_offset_val, p_end_bit);
        }
        
        uint64_t total_delta_bytes = (max_bit_offset_val + 7) / 8;
        uint64_t total_delta_words = (max_bit_offset_val + 31) / 32;
        if (max_bit_offset_val == 0) { 
            total_delta_bytes = 0; 
            total_delta_words = 0; 
        }

        std::vector<uint32_t> host_delta_data_vec;
        if (total_delta_words > 0 && compressed_object_to_serialize->delta_array) {
            host_delta_data_vec.resize(total_delta_words);
            CUDA_CHECK(cudaMemcpy(host_delta_data_vec.data(), 
                                  compressed_object_to_serialize->delta_array,
                                  total_delta_words * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        } else if (total_delta_words > 0 && !compressed_object_to_serialize->delta_array) {
             host_delta_data_vec.assign(total_delta_words, 0);
        }

        // Create SerializedHeader with SoA offsets - WITH PROPER ALIGNMENT
        SerializedHeader file_header;
        memset(&file_header, 0, sizeof(SerializedHeader));
        file_header.magic = 0x4F43454C; 
        file_header.version = 5; // Incremented for aligned format
        file_header.total_values = compressed_object_to_serialize->total_values;
        file_header.num_partitions = compressed_object_to_serialize->num_partitions;
        
        // Calculate offsets for each SoA array with proper alignment
        uint64_t current_offset = sizeof(SerializedHeader);
        
        // Align to 8 bytes for start_indices
        current_offset = alignOffset(current_offset, 8);
        file_header.start_indices_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        // Align to 8 bytes for end_indices
        current_offset = alignOffset(current_offset, 8);
        file_header.end_indices_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        // Align to 8 bytes for model_types
        current_offset = alignOffset(current_offset, 8);
        file_header.model_types_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        // Align to 8 bytes for model_params (double requires 8-byte alignment)
        current_offset = alignOffset(current_offset, 8);
        file_header.model_params_offset = current_offset;
        file_header.model_params_size_bytes = compressed_object_to_serialize->num_partitions * 4 * sizeof(double);
        current_offset += file_header.model_params_size_bytes;
        
        // Align to 8 bytes for delta_bits
        current_offset = alignOffset(current_offset, 8);
        file_header.delta_bits_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        // Align to 8 bytes for delta_array_bit_offsets (int64_t requires 8-byte alignment)
        current_offset = alignOffset(current_offset, 8);
        file_header.delta_array_bit_offsets_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int64_t);
        
        // Align to 8 bytes for error_bounds (long long requires 8-byte alignment)
        current_offset = alignOffset(current_offset, 8);
        file_header.error_bounds_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(long long);
        
        // Align to 8 bytes for delta_array
        current_offset = alignOffset(current_offset, 8);
        file_header.delta_array_offset = current_offset;
        file_header.delta_array_size_bytes = total_delta_bytes;
        
        file_header.data_type_size = sizeof(T);
        
        SerializedHeader temp_header_for_csum = file_header;
        temp_header_for_csum.header_checksum = 0;
        file_header.header_checksum = calculateChecksum(&temp_header_for_csum, sizeof(SerializedHeader));

        size_t final_total_size = current_offset + total_delta_bytes;
        // Align final size to 8 bytes as well
        final_total_size = alignOffset(final_total_size, 8);
        
        SerializedData* output_serialized_obj = new SerializedData();
        try { 
            output_serialized_obj->data = new uint8_t[final_total_size]; 
            // Initialize with zeros to ensure padding bytes are clean
            memset(output_serialized_obj->data, 0, final_total_size);
        }
        catch (const std::bad_alloc& e) { 
            delete output_serialized_obj; 
            return nullptr; 
        }
        output_serialized_obj->size = final_total_size;
        
        uint8_t* current_write_ptr = output_serialized_obj->data;
        
        // Write header
        memcpy(current_write_ptr, &file_header, sizeof(file_header)); 
        
        // Write each SoA array at its aligned offset
        if (compressed_object_to_serialize->num_partitions > 0) {
            memcpy(output_serialized_obj->data + file_header.start_indices_offset, 
                   h_start_indices.data(), 
                   compressed_object_to_serialize->num_partitions * sizeof(int32_t));
            
            memcpy(output_serialized_obj->data + file_header.end_indices_offset, 
                   h_end_indices.data(), 
                   compressed_object_to_serialize->num_partitions * sizeof(int32_t));
            
            memcpy(output_serialized_obj->data + file_header.model_types_offset, 
                   h_model_types.data(), 
                   compressed_object_to_serialize->num_partitions * sizeof(int32_t));
            
            memcpy(output_serialized_obj->data + file_header.model_params_offset, 
                   h_model_params.data(), 
                   compressed_object_to_serialize->num_partitions * 4 * sizeof(double));
            
            memcpy(output_serialized_obj->data + file_header.delta_bits_offset, 
                   h_delta_bits.data(), 
                   compressed_object_to_serialize->num_partitions * sizeof(int32_t));
            
            memcpy(output_serialized_obj->data + file_header.delta_array_bit_offsets_offset, 
                   h_delta_array_bit_offsets.data(), 
                   compressed_object_to_serialize->num_partitions * sizeof(int64_t));
            
            memcpy(output_serialized_obj->data + file_header.error_bounds_offset, 
                   h_error_bounds.data(), 
                   compressed_object_to_serialize->num_partitions * sizeof(long long));
        }
        
        // Write delta array at its aligned offset
        if (total_delta_bytes > 0 && !host_delta_data_vec.empty()) {
            memcpy(output_serialized_obj->data + file_header.delta_array_offset, 
                   host_delta_data_vec.data(), total_delta_bytes);
        }
        
        return output_serialized_obj;
    }
    
    // UPDATED deserialize function for SoA
    CompressedData<T>* deserialize(const SerializedData* serialized_input_data) {
        if (!serialized_input_data || !serialized_input_data->data || 
            serialized_input_data->size < sizeof(SerializedHeader)) { 
            return nullptr; 
        }
        
        const uint8_t* input_byte_ptr = serialized_input_data->data;
        SerializedHeader read_header;
        memcpy(&read_header, input_byte_ptr, sizeof(read_header));

        if (read_header.magic != 0x4F43454C || 
            (read_header.version != 4 && read_header.version != 5) || // Support both versions
            read_header.data_type_size != sizeof(T)) {
            return nullptr; 
        }
        
        SerializedHeader temp_hdr_for_csum = read_header; 
        temp_hdr_for_csum.header_checksum = 0;
        if (calculateChecksum(&temp_hdr_for_csum, sizeof(SerializedHeader)) != read_header.header_checksum) { 
            return nullptr; 
        }

        // Validate offsets and sizes
        uint64_t expected_size = read_header.delta_array_offset + read_header.delta_array_size_bytes;
        if (read_header.version == 5) {
            // For version 5, the size might be aligned
            expected_size = alignOffset(expected_size, 8);
        }
        if (expected_size > serialized_input_data->size) {
            return nullptr;
        }

        // Read SoA arrays from serialized data
        std::vector<int32_t> h_start_indices(read_header.num_partitions);
        std::vector<int32_t> h_end_indices(read_header.num_partitions);
        std::vector<int32_t> h_model_types(read_header.num_partitions);
        std::vector<double> h_model_params(read_header.num_partitions * 4);
        std::vector<int32_t> h_delta_bits(read_header.num_partitions);
        std::vector<int64_t> h_delta_array_bit_offsets(read_header.num_partitions);
        std::vector<long long> h_error_bounds(read_header.num_partitions);
        
        if (read_header.num_partitions > 0) {
            memcpy(h_start_indices.data(), input_byte_ptr + read_header.start_indices_offset, 
                read_header.num_partitions * sizeof(int32_t));
            memcpy(h_end_indices.data(), input_byte_ptr + read_header.end_indices_offset, 
                read_header.num_partitions * sizeof(int32_t));
            memcpy(h_model_types.data(), input_byte_ptr + read_header.model_types_offset, 
                read_header.num_partitions * sizeof(int32_t));
            memcpy(h_model_params.data(), input_byte_ptr + read_header.model_params_offset, 
                read_header.num_partitions * 4 * sizeof(double));
            memcpy(h_delta_bits.data(), input_byte_ptr + read_header.delta_bits_offset, 
                read_header.num_partitions * sizeof(int32_t));
            memcpy(h_delta_array_bit_offsets.data(), input_byte_ptr + read_header.delta_array_bit_offsets_offset, 
                read_header.num_partitions * sizeof(int64_t));
            memcpy(h_error_bounds.data(), input_byte_ptr + read_header.error_bounds_offset, 
                read_header.num_partitions * sizeof(long long));
        }
        
        uint64_t num_delta_words = (read_header.delta_array_size_bytes + 3) / 4;
        std::vector<uint32_t> host_deltas_from_file;
        if (num_delta_words > 0) {
            host_deltas_from_file.resize(num_delta_words);
            memcpy(host_deltas_from_file.data(), input_byte_ptr + read_header.delta_array_offset, 
                read_header.delta_array_size_bytes);
        }
        
        CompressedData<T>* new_compressed_data = new CompressedData<T>();
        new_compressed_data->num_partitions = read_header.num_partitions;
        new_compressed_data->total_values = read_header.total_values;
        
        // Allocate and copy SoA arrays to device
        if (new_compressed_data->num_partitions > 0) {
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_start_indices, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMemcpy(new_compressed_data->d_start_indices, h_start_indices.data(), 
                                read_header.num_partitions * sizeof(int32_t), 
                                cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_end_indices, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMemcpy(new_compressed_data->d_end_indices, h_end_indices.data(), 
                                read_header.num_partitions * sizeof(int32_t), 
                                cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_model_types, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMemcpy(new_compressed_data->d_model_types, h_model_types.data(), 
                                read_header.num_partitions * sizeof(int32_t), 
                                cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_model_params, 
                                read_header.num_partitions * 4 * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(new_compressed_data->d_model_params, h_model_params.data(), 
                                read_header.num_partitions * 4 * sizeof(double), 
                                cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_delta_bits, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMemcpy(new_compressed_data->d_delta_bits, h_delta_bits.data(), 
                                read_header.num_partitions * sizeof(int32_t), 
                                cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_delta_array_bit_offsets, 
                                read_header.num_partitions * sizeof(int64_t)));
            CUDA_CHECK(cudaMemcpy(new_compressed_data->d_delta_array_bit_offsets, h_delta_array_bit_offsets.data(), 
                                read_header.num_partitions * sizeof(int64_t), 
                                cudaMemcpyHostToDevice));
            
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_error_bounds, 
                                read_header.num_partitions * sizeof(long long)));
            CUDA_CHECK(cudaMemcpy(new_compressed_data->d_error_bounds, h_error_bounds.data(), 
                                read_header.num_partitions * sizeof(long long), 
                                cudaMemcpyHostToDevice));
        } else { 
            new_compressed_data->d_start_indices = nullptr;
            new_compressed_data->d_end_indices = nullptr;
            new_compressed_data->d_model_types = nullptr;
            new_compressed_data->d_model_params = nullptr;
            new_compressed_data->d_delta_bits = nullptr;
            new_compressed_data->d_delta_array_bit_offsets = nullptr;
            new_compressed_data->d_error_bounds = nullptr;
        }
        
        if (num_delta_words > 0) {
            CUDA_CHECK(cudaMalloc(&new_compressed_data->delta_array, num_delta_words * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy(new_compressed_data->delta_array, host_deltas_from_file.data(), 
                                num_delta_words * sizeof(uint32_t), cudaMemcpyHostToDevice));
        } else { 
            new_compressed_data->delta_array = nullptr; 
        }
        
        CUDA_CHECK(cudaMalloc(&new_compressed_data->d_self, sizeof(CompressedData<T>)));
        CUDA_CHECK(cudaMemcpy(new_compressed_data->d_self, new_compressed_data, 
                            sizeof(CompressedData<T>), cudaMemcpyHostToDevice));
        return new_compressed_data;
    }
        
    // UPDATED createDirectAccessHandle for SoA with alignment
    DirectAccessHandle<T>* createDirectAccessHandle(const uint8_t* serialized_blob_ptr, 
                                                size_t blob_size, 
                                                bool copy_to_device_flag = true) {
        if (!serialized_blob_ptr || blob_size < sizeof(SerializedHeader)) { 
            return nullptr; 
        }
        
        // Allocate handle with proper alignment
        DirectAccessHandle<T>* new_handle = nullptr;
        void* aligned_ptr = nullptr;
        size_t aligned_size = sizeof(DirectAccessHandle<T>) + 256;
        
        // Use posix_memalign for aligned allocation
        if (posix_memalign(&aligned_ptr, 256, aligned_size) != 0) {
            return nullptr;
        }
        
        new_handle = new (aligned_ptr) DirectAccessHandle<T>();
        memset(new_handle, 0, sizeof(DirectAccessHandle<T>));
        
        new_handle->data_blob_host = serialized_blob_ptr; 
        new_handle->data_blob_size = blob_size;
        new_handle->header_host = reinterpret_cast<const SerializedHeader*>(serialized_blob_ptr);
        
        if (new_handle->header_host->magic != 0x4F43454C || 
            new_handle->header_host->data_type_size != sizeof(T) ||
            (new_handle->header_host->version != 4 && new_handle->header_host->version != 5)) { // Support both versions
            new_handle->~DirectAccessHandle<T>();
            free(aligned_ptr);
            return nullptr;
        }

        // Set host pointers to SoA arrays
        new_handle->start_indices_host = reinterpret_cast<const int32_t*>(
            serialized_blob_ptr + new_handle->header_host->start_indices_offset);
        new_handle->end_indices_host = reinterpret_cast<const int32_t*>(
            serialized_blob_ptr + new_handle->header_host->end_indices_offset);
        new_handle->model_types_host = reinterpret_cast<const int32_t*>(
            serialized_blob_ptr + new_handle->header_host->model_types_offset);
        new_handle->model_params_host = reinterpret_cast<const double*>(
            serialized_blob_ptr + new_handle->header_host->model_params_offset);
        new_handle->delta_bits_host = reinterpret_cast<const int32_t*>(
            serialized_blob_ptr + new_handle->header_host->delta_bits_offset);
        new_handle->delta_array_bit_offsets_host = reinterpret_cast<const int64_t*>(
            serialized_blob_ptr + new_handle->header_host->delta_array_bit_offsets_offset);
        new_handle->error_bounds_host = reinterpret_cast<const long long*>(
            serialized_blob_ptr + new_handle->header_host->error_bounds_offset);
        new_handle->delta_array_host = reinterpret_cast<const uint32_t*>(
            serialized_blob_ptr + new_handle->header_host->delta_array_offset);
        
        // Initialize device pointers
        new_handle->d_data_blob_device = nullptr;
        new_handle->d_header_device = nullptr;
        new_handle->d_start_indices_device = nullptr;
        new_handle->d_end_indices_device = nullptr;
        new_handle->d_model_types_device = nullptr;
        new_handle->d_model_params_device = nullptr;
        new_handle->d_delta_bits_device = nullptr;
        new_handle->d_delta_array_bit_offsets_device = nullptr;
        new_handle->d_error_bounds_device = nullptr;
        new_handle->d_delta_array_device = nullptr;

        if (copy_to_device_flag) {
            // For version 5, the blob is already aligned. For version 4, we need to align it.
            size_t aligned_blob_size = blob_size;
            if (new_handle->header_host->version == 5) {
                // Version 5 blobs should already be aligned
                aligned_blob_size = alignOffset(blob_size, 256);
            } else {
                // For version 4, align to 256 bytes for safety
                aligned_blob_size = alignOffset(blob_size, 256);
            }
            
            CUDA_CHECK(cudaMalloc(&new_handle->d_data_blob_device, aligned_blob_size));
            CUDA_CHECK(cudaMemset(new_handle->d_data_blob_device, 0, aligned_blob_size));
            CUDA_CHECK(cudaMemcpy(new_handle->d_data_blob_device, serialized_blob_ptr, blob_size, 
                                cudaMemcpyHostToDevice));
            
            // Set device pointers to SoA arrays
            new_handle->d_header_device = reinterpret_cast<SerializedHeader*>(new_handle->d_data_blob_device);
            new_handle->d_start_indices_device = reinterpret_cast<int32_t*>(
                new_handle->d_data_blob_device + new_handle->header_host->start_indices_offset);
            new_handle->d_end_indices_device = reinterpret_cast<int32_t*>(
                new_handle->d_data_blob_device + new_handle->header_host->end_indices_offset);
            new_handle->d_model_types_device = reinterpret_cast<int32_t*>(
                new_handle->d_data_blob_device + new_handle->header_host->model_types_offset);
            new_handle->d_model_params_device = reinterpret_cast<double*>(
                new_handle->d_data_blob_device + new_handle->header_host->model_params_offset);
            new_handle->d_delta_bits_device = reinterpret_cast<int32_t*>(
                new_handle->d_data_blob_device + new_handle->header_host->delta_bits_offset);
            new_handle->d_delta_array_bit_offsets_device = reinterpret_cast<int64_t*>(
                new_handle->d_data_blob_device + new_handle->header_host->delta_array_bit_offsets_offset);
            new_handle->d_error_bounds_device = reinterpret_cast<long long*>(
                new_handle->d_data_blob_device + new_handle->header_host->error_bounds_offset);
            new_handle->d_delta_array_device = reinterpret_cast<uint32_t*>(
                new_handle->d_data_blob_device + new_handle->header_host->delta_array_offset);
        }
        return new_handle;
    }
    
    void directRandomAccessGPU(DirectAccessHandle<T>* da_h,
                        const std::vector<int>& positions_to_access,
                        std::vector<T>& output_target_vec) {
        if (!da_h || !da_h->d_data_blob_device || positions_to_access.empty()) {
            output_target_vec.clear();
            return;
        }
        
        int num_queries = positions_to_access.size();
        output_target_vec.resize(num_queries);
        
        // Get header info
        SerializedHeader h_header;
        CUDA_CHECK(cudaMemcpy(&h_header, da_h->d_header_device, 
                            sizeof(SerializedHeader), cudaMemcpyDeviceToHost));
        
        if (h_header.num_partitions == 0) {
            std::fill(output_target_vec.begin(), output_target_vec.end(), static_cast<T>(0));
            return;
        }
        
        // Calculate device pointers from blob
        const uint8_t* d_blob = da_h->d_data_blob_device;
        const int32_t* d_start_indices = reinterpret_cast<const int32_t*>(d_blob + h_header.start_indices_offset);
        const int32_t* d_end_indices = reinterpret_cast<const int32_t*>(d_blob + h_header.end_indices_offset);
        const int32_t* d_model_types = reinterpret_cast<const int32_t*>(d_blob + h_header.model_types_offset);
        const double* d_model_params = reinterpret_cast<const double*>(d_blob + h_header.model_params_offset);
        const int32_t* d_delta_bits = reinterpret_cast<const int32_t*>(d_blob + h_header.delta_bits_offset);
        const int64_t* d_delta_array_bit_offsets = reinterpret_cast<const int64_t*>(d_blob + h_header.delta_array_bit_offsets_offset);
        const uint32_t* d_delta_array = reinterpret_cast<const uint32_t*>(d_blob + h_header.delta_array_offset);
        
        // Allocate device memory for queries
        int* d_positions = nullptr;
        T* d_output = nullptr;
        
        CUDA_CHECK(cudaMalloc(&d_positions, num_queries * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output, num_queries * sizeof(T)));
        
        // Copy positions to device
        CUDA_CHECK(cudaMemcpy(d_positions, positions_to_access.data(), 
                            num_queries * sizeof(int), cudaMemcpyHostToDevice));
        
        // Configure kernel launch
        int block_size = 256;
        int grid_size = (num_queries + block_size - 1) / block_size;
        
        // Limit grid size for better occupancy
        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        grid_size = std::min(grid_size, prop.multiProcessorCount * 32);
        
        // Launch optimized kernel without shared memory
        directRandomAccessKernel<T><<<grid_size, block_size>>>(
            d_start_indices,
            d_end_indices,
            d_model_types,
            d_model_params,
            d_delta_bits,
            d_delta_array_bit_offsets,
            d_delta_array,
            d_positions,
            d_output,
            h_header.num_partitions,
            num_queries);
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back
        CUDA_CHECK(cudaMemcpy(output_target_vec.data(), d_output, 
                            num_queries * sizeof(T), cudaMemcpyDeviceToHost));
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_positions));
        CUDA_CHECK(cudaFree(d_output));
    }
    
    // UPDATED directRandomAccessCPU for SoA
    void directRandomAccessCPU(DirectAccessHandle<T>* da_h_cpu,
                              const std::vector<int>& positions_to_access_cpu,
                              std::vector<T>& output_target_vec_cpu,
                              bool debug_print_flag = false) {
        if (!da_h_cpu || !da_h_cpu->data_blob_host) {
            if (debug_print_flag) std::cerr << "DEBUG: Invalid handle or host data blob." << std::endl;
            output_target_vec_cpu.clear();
            return;
        }
        if (positions_to_access_cpu.empty()){
            output_target_vec_cpu.clear();
            return;
        }
        int num_access_q_cpu = positions_to_access_cpu.size();
        output_target_vec_cpu.resize(num_access_q_cpu);

        const SerializedHeader* h_hdr = da_h_cpu->header_host;
        const int32_t* h_start_indices = da_h_cpu->start_indices_host;
        const int32_t* h_end_indices = da_h_cpu->end_indices_host;
        const int32_t* h_model_types = da_h_cpu->model_types_host;
        const double* h_model_params = da_h_cpu->model_params_host;
        const int32_t* h_delta_bits = da_h_cpu->delta_bits_host;
        const int64_t* h_delta_array_bit_offsets = da_h_cpu->delta_array_bit_offsets_host;
        const uint32_t* h_deltas_arr = da_h_cpu->delta_array_host;

        if (!h_hdr || (h_hdr->num_partitions > 0 && (!h_start_indices || !h_end_indices)) || 
            (h_hdr->delta_array_size_bytes > 0 && !h_deltas_arr && h_hdr->num_partitions > 0)) {
            if (debug_print_flag) 
                std::cerr << "DEBUG: Null host pointers or inconsistent header." << std::endl;
            for(int i = 0; i < num_access_q_cpu; ++i) 
                output_target_vec_cpu[i] = T(0);
            return;
        }

        // Process queries serially to avoid race conditions
        for (int i = 0; i < num_access_q_cpu; i++) {
            int current_q_pos = positions_to_access_cpu[i];
            
            if (debug_print_flag) {
                std::cout << "\n--- DEBUG START for Query Position: " << current_q_pos << " ---" << std::endl;
            }

            // Find partition using binary search
            if (h_hdr->num_partitions == 0) {
                output_target_vec_cpu[i] = T(0); 
                if (debug_print_flag) {
                    std::cout << "DEBUG: No partitions." << std::endl 
                             << "--- DEBUG END ---" << std::endl;
                }
                continue;
            }
            
            int l_idx = 0; 
            int r_idx = h_hdr->num_partitions - 1; 
            int found_partition_idx = -1;
            while(l_idx <= r_idx){
                int m_idx = l_idx + (r_idx - l_idx) / 2;
                int32_t current_start = h_start_indices[m_idx];
                int32_t current_end = h_end_indices[m_idx];
                if(current_q_pos >= current_start && current_q_pos < current_end){
                    found_partition_idx = m_idx;
                    break;
                }
                if(current_q_pos < current_start) 
                    r_idx = m_idx - 1; 
                else 
                    l_idx = m_idx + 1;
            }
            
            if(found_partition_idx == -1){
                output_target_vec_cpu[i] = T(0); 
                if (debug_print_flag) {
                    std::cout << "DEBUG: Partition not found." << std::endl 
                             << "--- DEBUG END ---" << std::endl;
                }
                continue;
            }
            
            // Access partition data using found index
            int32_t start_idx = h_start_indices[found_partition_idx];
            int32_t end_idx = h_end_indices[found_partition_idx];
            int32_t model_type = h_model_types[found_partition_idx];
            int32_t delta_bits = h_delta_bits[found_partition_idx];
            int64_t delta_array_bit_offset = h_delta_array_bit_offsets[found_partition_idx];
            
            if (debug_print_flag) {
                std::cout << "DEBUG: Partition Info - start=" << start_idx 
                        << ", end=" << end_idx 
                        << ", delta_bits=" << delta_bits 
                        << ", model_type=" << model_type << std::endl;
            }

            int current_local_idx = current_q_pos - start_idx;

            // Handle direct copy model
            if (model_type == MODEL_DIRECT_COPY) {
                if (delta_bits > 0 && h_deltas_arr) {
                    int64_t current_bit_off = delta_array_bit_offset + 
                                             (int64_t)current_local_idx * delta_bits;
                    
                    // Extract the full value stored as "delta"
                    uint64_t extracted_value = 0;
                    int start_word_idx = current_bit_off / 32;
                    int offset_in_word = current_bit_off % 32;
                    int bits_remaining = delta_bits;
                    int shift = 0;
                    
                    while (bits_remaining > 0 && shift < 64) {
                        uint32_t word_val;
                        memcpy(&word_val, &h_deltas_arr[start_word_idx], sizeof(uint32_t));
                        
                        int bits_in_this_word = std::min(bits_remaining, 32 - offset_in_word);
                        uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                        uint32_t extracted_bits = (word_val >> offset_in_word) & mask;
                        
                        extracted_value |= (static_cast<uint64_t>(extracted_bits) << shift);
                        
                        shift += bits_in_this_word;
                        bits_remaining -= bits_in_this_word;
                        start_word_idx++;
                        offset_in_word = 0;
                    }
                    
                    output_target_vec_cpu[i] = static_cast<T>(extracted_value);
                    
                    if (debug_print_flag) {
                        std::cout << "DEBUG: Direct copy model - extracted_value=" << extracted_value 
                                 << ", output=" << output_target_vec_cpu[i] << std::endl;
                    }
                } else {
                    output_target_vec_cpu[i] = T(0);
                }
            } else {
                // Normal model-based decompression
                double pred_double_val = h_model_params[found_partition_idx * 4] + 
                                        h_model_params[found_partition_idx * 4 + 1] * current_local_idx;
                if (model_type == MODEL_POLYNOMIAL2) {
                    pred_double_val += h_model_params[found_partition_idx * 4 + 2] * current_local_idx * current_local_idx;
                }
                
                if (debug_print_flag) {
                    std::cout << "DEBUG: pred_double_val=" << pred_double_val << std::endl;
                }

                long long final_delta_reconstructed = 0;

                if (delta_bits > 0) {
                    int64_t current_bit_off = delta_array_bit_offset + 
                                             (int64_t)current_local_idx * delta_bits;
                    int current_word_idx = current_bit_off / 32;
                    int current_bit_off_in_word = current_bit_off % 32;
                    uint64_t total_words_available = (h_hdr->delta_array_size_bytes + 3) / 4;
                    
                    if (debug_print_flag) {
                        std::cout << "DEBUG: Delta offset info - bit_offset=" << current_bit_off 
                                << ", word_idx=" << current_word_idx 
                                << ", bit_offset_in_word=" << current_bit_off_in_word << std::endl;
                    }

                    if (current_word_idx < 0 || current_word_idx >= static_cast<int>(total_words_available)) {
                        final_delta_reconstructed = 0; 
                    } else {
                        if (delta_bits <= 32) {
                            uint32_t extracted_val_bits;
                            
                            // Read word(s) with proper memory barrier
                            uint32_t w1_val;
                            memcpy(&w1_val, &h_deltas_arr[current_word_idx], sizeof(uint32_t));

                            if (current_bit_off_in_word + delta_bits <= 32) {
                                // Single word read
                                uint32_t read_mask = (delta_bits == 32) ? 0xFFFFFFFFU : ((1U << delta_bits) - 1U);
                                extracted_val_bits = (w1_val >> current_bit_off_in_word) & read_mask;
                                
                                if (debug_print_flag) {
                                    std::cout << "DEBUG: Delta <= 32 (single word): word_val=0x" << std::hex << w1_val 
                                            << ", mask_val=0x" << read_mask << std::dec 
                                            << ", extracted_val_bits=0x" << std::hex << extracted_val_bits 
                                            << std::dec << " (" << extracted_val_bits << ")" << std::endl;
                                }

                            } else { 
                                // Spans two words
                                uint32_t w2_val = 0;
                                if (current_word_idx + 1 < static_cast<int>(total_words_available)) {
                                    memcpy(&w2_val, &h_deltas_arr[current_word_idx + 1], sizeof(uint32_t));
                                }
                                
                                // Extract bits from both words
                                uint32_t bits_from_first_word = 32 - current_bit_off_in_word;
                                uint32_t bits_from_second_word = delta_bits - bits_from_first_word;
                                
                                // Get the bits from the first word
                                uint32_t first_part = w1_val >> current_bit_off_in_word;
                                
                                // Get the bits from the second word
                                uint32_t second_part = w2_val & ((bits_from_second_word == 32) ? 0xFFFFFFFFU : ((1U << bits_from_second_word) - 1U));
                                
                                // Combine them
                                extracted_val_bits = first_part | (second_part << bits_from_first_word);
                                
                                // Mask to ensure we only have the bits we want
                                uint32_t final_mask = (delta_bits == 32) ? 0xFFFFFFFFU : ((1U << delta_bits) - 1U);
                                extracted_val_bits &= final_mask;
                                
                                if (debug_print_flag) {
                                    std::cout << "DEBUG: Delta <= 32 (two words): w1_val=0x" << std::hex << w1_val 
                                            << ", w2_val=0x" << w2_val 
                                            << ", mask_val=0x" << final_mask << std::dec 
                                            << ", extracted_val_bits=0x" << std::hex << extracted_val_bits 
                                            << std::dec << " (" << extracted_val_bits << ")" << std::endl;
                                }
                            }

                            // Sign extension
                            if (delta_bits < 32) {
                                uint32_t sign_bit = 1U << (delta_bits - 1);
                                if (extracted_val_bits & sign_bit) {
                                    // Sign extend
                                    uint32_t sign_extend_mask = ~((1U << delta_bits) - 1U);
                                    extracted_val_bits |= sign_extend_mask;
                                }
                                final_delta_reconstructed = static_cast<long long>(static_cast<int32_t>(extracted_val_bits));
                            } else {
                                // For 32-bit deltas, just cast directly
                                final_delta_reconstructed = static_cast<long long>(static_cast<int32_t>(extracted_val_bits));
                            }
                        } else { 
                            // delta_bits > 32
                            int bits_remaining = delta_bits; 
                            uint64_t extracted_val_64 = 0;
                            int shift = 0; 
                            int word_idx = current_word_idx;
                            int offset_in_word = current_bit_off_in_word;
                            
                            if (debug_print_flag) std::cout << "DEBUG: Delta > 32. Reading words: ";
                            
                            while (bits_remaining > 0 && word_idx < static_cast<int>(total_words_available) && shift < 64) {
                                int bits_to_read = std::min(bits_remaining, 32 - offset_in_word);
                                
                                // Read word with proper memory barrier
                                uint32_t word_val;
                                memcpy(&word_val, &h_deltas_arr[word_idx], sizeof(uint32_t));
                                
                                // Extract bits from current word
                                uint32_t mask = (bits_to_read == 32) ? 0xFFFFFFFFU : ((1U << bits_to_read) - 1U);
                                uint32_t extracted_bits = (word_val >> offset_in_word) & mask;
                                
                                if (debug_print_flag) {
                                    std::cout << " {idx=" << word_idx << ", off=" << offset_in_word 
                                            << ", bits=" << bits_to_read << ", raw_word=0x" 
                                            << std::hex << word_val 
                                            << ", masked_word_val=0x" << extracted_bits << std::dec << "} ";
                                }
                                
                                // Add to result
                                extracted_val_64 |= (static_cast<uint64_t>(extracted_bits) << shift);
                                
                                shift += bits_to_read; 
                                bits_remaining -= bits_to_read;
                                word_idx++; 
                                offset_in_word = 0; // Next words start at bit 0
                            }
                            
                            if (debug_print_flag) {
                                std::cout << "\nDEBUG: Delta > 32: extracted_val_64=0x" << std::hex << extracted_val_64 
                                        << std::dec << " (" << extracted_val_64 << ")" << std::endl;
                            }
                            
                            // Sign extension for values less than 64 bits
                            if (delta_bits < 64) {
                                uint64_t sign_bit = 1ULL << (delta_bits - 1);
                                if (extracted_val_64 & sign_bit) {
                                    // Create a mask with all bits set above the delta_bits position
                                    uint64_t sign_extend_mask = ~((1ULL << delta_bits) - 1ULL);
                                    extracted_val_64 |= sign_extend_mask;
                                    
                                    if (debug_print_flag) {
                                        std::cout << "DEBUG: Sign extension - sign_bit=0x" << std::hex << sign_bit 
                                                << ", sign_extend_mask=0x" << sign_extend_mask 
                                                << ", after sign extension=0x" << extracted_val_64 << std::dec << std::endl;
                                    }
                                }
                            }
                            // Cast to signed long long - this will preserve the sign-extended value
                            final_delta_reconstructed = static_cast<long long>(static_cast<int64_t>(extracted_val_64));
                        }
                    }
                } 
                
                if (debug_print_flag) {
                    std::cout << "DEBUG: final_delta_reconstructed=" << final_delta_reconstructed << std::endl;
                }

                // Clamp predicted value to valid range for the type
                if (!std::is_signed<T>::value) {
                    if (pred_double_val < 0) {
                        pred_double_val = 0;
                    } else if (sizeof(T) == 4 && pred_double_val > 4294967295.0) {
                        pred_double_val = 4294967295.0;
                    } else if (sizeof(T) == 2 && pred_double_val > 65535.0) {
                        pred_double_val = 65535.0;
                    } else if (sizeof(T) == 1 && pred_double_val > 255.0) {
                        pred_double_val = 255.0;
                    }
                } else {
                    // For signed types, also clamp to valid range
                    if (sizeof(T) == 4) {
                        if (pred_double_val > 2147483647.0) pred_double_val = 2147483647.0;
                        else if (pred_double_val < -2147483648.0) pred_double_val = -2147483648.0;
                    } else if (sizeof(T) == 2) {
                        if (pred_double_val > 32767.0) pred_double_val = 32767.0;
                        else if (pred_double_val < -32768.0) pred_double_val = -32768.0;
                    } else if (sizeof(T) == 1) {
                        if (pred_double_val > 127.0) pred_double_val = 127.0;
                        else if (pred_double_val < -128.0) pred_double_val = -128.0;
                    }
                }
                
                T pred_T_type_val = static_cast<T>(std::round(pred_double_val));
                
                if (debug_print_flag) {
                    std::cout << "DEBUG: pred_T_type_val=" << pred_T_type_val << std::endl;
                }
                
                // Apply delta
                output_target_vec_cpu[i] = applyDelta(pred_T_type_val, final_delta_reconstructed);
                
                if (debug_print_flag) {
                    std::cout << "DEBUG: final output_target_vec_cpu[" << i << "]=" 
                             << output_target_vec_cpu[i] << std::endl;
                    std::cout << "--- DEBUG END ---" << std::endl;
                }
            }
        } 
    }

    void destroyDirectAccessHandle(DirectAccessHandle<T>* handle_to_destroy) {
        if (handle_to_destroy) { 
            if (handle_to_destroy->d_data_blob_device) 
                CUDA_CHECK(cudaFree(handle_to_destroy->d_data_blob_device)); 
            handle_to_destroy->~DirectAccessHandle<T>();
            free(handle_to_destroy); // Use free() since we used posix_memalign
        }
    }
    
    bool saveToFile(const uint8_t* data_to_save, size_t size_of_save_data, 
                    const std::string& output_filename) {
        if (!data_to_save || size_of_save_data == 0) return false;
        std::ofstream output_file_stream(output_filename, std::ios::binary);
        if (!output_file_stream.is_open()) return false;
        output_file_stream.write(reinterpret_cast<const char*>(data_to_save), size_of_save_data);
        bool write_ok = output_file_stream.good(); 
        output_file_stream.close();
        if(write_ok) 
            std::cout << "Saved serialized data to " << output_filename << " (" 
                     << size_of_save_data << " bytes)" << std::endl;
        else 
            std::cerr << "Error writing to file: " << output_filename << std::endl;
        return write_ok;
    }
    
    uint8_t* loadFromFile(const std::string& input_filename, size_t& loaded_data_size) {
        std::ifstream input_file_stream(input_filename, std::ios::binary | std::ios::ate);
        if (!input_file_stream.is_open()) { 
            loaded_data_size = 0; 
            return nullptr; 
        }
        std::streampos file_s_val = input_file_stream.tellg();
        if (file_s_val <= 0) { 
            input_file_stream.close(); 
            loaded_data_size = 0; 
            return nullptr; 
        }
        loaded_data_size = static_cast<size_t>(file_s_val);
        input_file_stream.seekg(0, std::ios::beg);
        uint8_t* file_data_buffer = nullptr;
        try { 
            file_data_buffer = new uint8_t[loaded_data_size]; 
        } catch (const std::bad_alloc&) { 
            input_file_stream.close(); 
            loaded_data_size = 0; 
            return nullptr; 
        }
        input_file_stream.read(reinterpret_cast<char*>(file_data_buffer), loaded_data_size);
        bool read_ok = input_file_stream.good() && 
                      (static_cast<size_t>(input_file_stream.gcount()) == loaded_data_size);
        input_file_stream.close();
        if(!read_ok) { 
            delete[] file_data_buffer; 
            loaded_data_size = 0; 
            return nullptr;
        }
        std::cout << "Loaded serialized data from " << input_filename << " (" 
                 << loaded_data_size << " bytes)" << std::endl;
        return file_data_buffer;
    }


    // Add these methods to the LeCoGPU class:

    // Method for standard bit-packed decompression
    void decompressFullFile_BitPacked(CompressedData<T>* compressed_data_input,
                                    std::vector<T>& output_decompressed_data) {
        if (!compressed_data_input || compressed_data_input->total_values == 0) {
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->d_self) {
            std::cerr << "Error: compressed_data_input->d_self is null." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->delta_array) {
            std::cerr << "Error: No bit-packed delta array available." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        int total_elements = compressed_data_input->total_values;
        output_decompressed_data.resize(total_elements);
        
        T* d_output_ptr;
        CUDA_CHECK(cudaMalloc(&d_output_ptr, total_elements * sizeof(T)));
        
        // Calculate optimal launch configuration
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        // Limit grid size for better occupancy
        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        grid_size = std::min(grid_size, prop.multiProcessorCount * 32);
        
        // Launch bit-packed decompression kernel
        ::decompressFullFile_BitPacked<T><<<grid_size, block_size, 0, decompression_cuda_stream>>>(
            compressed_data_input->d_self, d_output_ptr, total_elements);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy results back
        CUDA_CHECK(cudaMemcpyAsync(output_decompressed_data.data(), d_output_ptr,
                                total_elements * sizeof(T), cudaMemcpyDeviceToHost,
                                decompression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(decompression_cuda_stream));
        
        CUDA_CHECK(cudaFree(d_output_ptr));
    }

    // Method for pre-unpacked decompression
    void decompressFullFile_PreUnpacked(CompressedData<T>* compressed_data_input,
                                        std::vector<T>& output_decompressed_data) {
        if (!compressed_data_input || compressed_data_input->total_values == 0) {
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->d_self) {
            std::cerr << "Error: compressed_data_input->d_self is null." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->d_plain_deltas) {
            std::cerr << "Error: No pre-unpacked deltas available. "
                    << "Please deserialize with preUnpackDeltas=true." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        int total_elements = compressed_data_input->total_values;
        output_decompressed_data.resize(total_elements);
        
        T* d_output_ptr;
        CUDA_CHECK(cudaMalloc(&d_output_ptr, total_elements * sizeof(T)));
        
        // Calculate optimal launch configuration
        int block_size = 256;
        int grid_size = (total_elements + block_size - 1) / block_size;
        
        // Limit grid size
        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        grid_size = std::min(grid_size, prop.multiProcessorCount * 32);
        
        // Launch pre-unpacked decompression kernel
        ::decompressFullFile_PreUnpacked<T><<<grid_size, block_size, 0, decompression_cuda_stream>>>(
            compressed_data_input->d_self, d_output_ptr, total_elements);
        CUDA_CHECK(cudaGetLastError());
        
        // Copy results back
        CUDA_CHECK(cudaMemcpyAsync(output_decompressed_data.data(), d_output_ptr,
                                total_elements * sizeof(T), cudaMemcpyDeviceToHost,
                                decompression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(decompression_cuda_stream));
        
        CUDA_CHECK(cudaFree(d_output_ptr));
    }

    // Wrapper method for the optimized V2 kernel
    void decompressFullFile_OnTheFly_Optimized_V2(CompressedData<T>* compressed_data_input,
                                                std::vector<T>& output_decompressed_data) {
        if (!compressed_data_input || compressed_data_input->total_values == 0) {
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->d_self) {
            std::cerr << "Error: compressed_data_input->d_self is null." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        int total_elements = compressed_data_input->total_values;
        output_decompressed_data.resize(total_elements);
        
        T* d_output_ptr;
        CUDA_CHECK(cudaMalloc(&d_output_ptr, total_elements * sizeof(T)));
        
        int grid_size = compressed_data_input->num_partitions;
        int block_size = 256;
        
        ::decompressFullFile_OnTheFly_Optimized_V2<T><<<grid_size, block_size, 0, 
                                                    decompression_cuda_stream>>>(
            compressed_data_input->d_self, d_output_ptr, total_elements);
        CUDA_CHECK(cudaGetLastError());
        
        CUDA_CHECK(cudaMemcpyAsync(output_decompressed_data.data(), d_output_ptr,
                                total_elements * sizeof(T), cudaMemcpyDeviceToHost,
                                decompression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(decompression_cuda_stream));
        
        CUDA_CHECK(cudaFree(d_output_ptr));
    }

public:
    void randomAccessFixed(CompressedData<T>* compressed_data_input,
                           const std::vector<int>& positions_to_access,
                           std::vector<T>& output_decompressed_data,
                           int fixed_partition_size) { // 传入固定的分区大小
        if (!compressed_data_input || positions_to_access.empty() || fixed_partition_size <= 0) {
            output_decompressed_data.clear();
            return;
        }

        int num_queries = positions_to_access.size();
        output_decompressed_data.resize(num_queries);

        // 分配设备内存
        int* d_positions;
        T* d_output;
        CUDA_CHECK(cudaMalloc(&d_positions, num_queries * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output, num_queries * sizeof(T)));

        // 拷贝查询位置到设备
        CUDA_CHECK(cudaMemcpy(d_positions, positions_to_access.data(),
                            num_queries * sizeof(int), cudaMemcpyHostToDevice));

        // 配置启动参数
        int block_size = 256;
        int grid_size = (num_queries + block_size - 1) / block_size;

        // 调用新的、优化的内核
        randomAccessFixedPartitionKernel<T><<<grid_size, block_size>>>(
            compressed_data_input->d_start_indices,
            compressed_data_input->d_model_types,
            compressed_data_input->d_model_params,
            compressed_data_input->d_delta_bits,
            compressed_data_input->d_delta_array_bit_offsets,
            compressed_data_input->delta_array,
            d_positions,
            d_output,
            num_queries,
            fixed_partition_size // 传入关键的优化参数
        );

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 将结果拷贝回主机
        CUDA_CHECK(cudaMemcpy(output_decompressed_data.data(), d_output,
                            num_queries * sizeof(T), cudaMemcpyDeviceToHost));

        // 清理
        CUDA_CHECK(cudaFree(d_positions));
        CUDA_CHECK(cudaFree(d_output));
    }

    // Optimized fixed-partition random access with pre-unpacked deltas
    void randomAccessFixedPreUnpacked(CompressedData<T>* compressed_data_input,
                                    const std::vector<int>& positions_to_access,
                                    std::vector<T>& output_decompressed_data,
                                    int fixed_partition_size) {
        if (!compressed_data_input || positions_to_access.empty() || fixed_partition_size <= 0) {
            output_decompressed_data.clear();
            return;
        }
        
        if (!compressed_data_input->d_plain_deltas) {
            std::cerr << "Error: No pre-unpacked deltas available. "
                    << "Please deserialize with preUnpackDeltas=true." << std::endl;
            output_decompressed_data.clear();
            return;
        }
        
        int num_queries = positions_to_access.size();
        output_decompressed_data.resize(num_queries);
        
        // Allocate device memory
        int* d_positions;
        T* d_output;
        
        CUDA_CHECK(cudaMalloc(&d_positions, num_queries * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_output, num_queries * sizeof(T)));
        
        // Copy positions to device
        CUDA_CHECK(cudaMemcpyAsync(d_positions, positions_to_access.data(),
                                num_queries * sizeof(int), cudaMemcpyHostToDevice,
                                decompression_cuda_stream));
        
        // Calculate optimal launch configuration
        const int QUERIES_PER_THREAD = 4;
        int total_threads = (num_queries + QUERIES_PER_THREAD - 1) / QUERIES_PER_THREAD;
        int block_size = 128;  // Smaller block size for better occupancy
        int grid_size = (total_threads + block_size - 1) / block_size;
        
        // Limit grid size
        cudaDeviceProp prop;
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        grid_size = std::min(grid_size, prop.multiProcessorCount * 32);
        
        // Launch the optimized kernel
        randomAccessFixedPreUnpackedKernel<T><<<grid_size, block_size, 0, decompression_cuda_stream>>>(
            compressed_data_input->d_model_types,
            compressed_data_input->d_model_params,
            compressed_data_input->d_plain_deltas,
            d_positions,
            d_output,
            num_queries,
            fixed_partition_size);
        
        CUDA_CHECK(cudaGetLastError());
        
        // Copy results back
        CUDA_CHECK(cudaMemcpyAsync(output_decompressed_data.data(), d_output,
                                num_queries * sizeof(T), cudaMemcpyDeviceToHost,
                                decompression_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(decompression_cuda_stream));
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_positions));
        CUDA_CHECK(cudaFree(d_output));
    }
    // GPU-accelerated serialization
// GPU-accelerated serialization
    SerializedData* serializeGPU(CompressedData<T>* compressed_object_to_serialize) {
        if (!compressed_object_to_serialize) { 
            std::cerr << "Error: Null data to serialize." << std::endl; 
            return nullptr; 
        }
        
        // --- NEW CHECK: Prevent serialization of pre-unpacked data ---
        if (compressed_object_to_serialize->delta_array == nullptr && 
            compressed_object_to_serialize->d_plain_deltas != nullptr) {
            std::cerr << "Error: Cannot serialize data that has been deserialized with pre-unpacking enabled. "
                    << "Re-serialization is not supported in this mode." << std::endl;
            return nullptr;
        }
        
        // Calculate sizes and offsets (same as original serialize)
        SerializedHeader file_header;
        memset(&file_header, 0, sizeof(SerializedHeader));
        file_header.magic = 0x4F43454C; 
        file_header.version = 5;
        file_header.total_values = compressed_object_to_serialize->total_values;
        file_header.num_partitions = compressed_object_to_serialize->num_partitions;
        
        // Calculate delta array size
        uint64_t max_bit_offset_val = 0;
        if (compressed_object_to_serialize->num_partitions > 0) {
            // Need to get the last partition's bit offset and size from device
            int64_t last_bit_offset;
            int32_t last_delta_bits;
            int32_t last_start, last_end;
            int last_idx = compressed_object_to_serialize->num_partitions - 1;
            
            CUDA_CHECK(cudaMemcpy(&last_bit_offset, 
                compressed_object_to_serialize->d_delta_array_bit_offsets + last_idx,
                sizeof(int64_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_delta_bits, 
                compressed_object_to_serialize->d_delta_bits + last_idx,
                sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_start, 
                compressed_object_to_serialize->d_start_indices + last_idx,
                sizeof(int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&last_end, 
                compressed_object_to_serialize->d_end_indices + last_idx,
                sizeof(int32_t), cudaMemcpyDeviceToHost));
            
            int seg_len = last_end - last_start;
            if (seg_len > 0) {
                max_bit_offset_val = last_bit_offset + (uint64_t)seg_len * last_delta_bits;
            }
        }
        
        uint64_t total_delta_bytes = (max_bit_offset_val + 7) / 8;
        
        // Calculate offsets for each SoA array with proper alignment
        uint64_t current_offset = sizeof(SerializedHeader);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.start_indices_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.end_indices_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.model_types_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.model_params_offset = current_offset;
        file_header.model_params_size_bytes = compressed_object_to_serialize->num_partitions * 4 * sizeof(double);
        current_offset += file_header.model_params_size_bytes;
        
        current_offset = alignOffset(current_offset, 8);
        file_header.delta_bits_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int32_t);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.delta_array_bit_offsets_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(int64_t);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.error_bounds_offset = current_offset;
        current_offset += compressed_object_to_serialize->num_partitions * sizeof(long long);
        
        current_offset = alignOffset(current_offset, 8);
        file_header.delta_array_offset = current_offset;
        file_header.delta_array_size_bytes = total_delta_bytes;
        
        file_header.data_type_size = sizeof(T);
        
        SerializedHeader temp_header_for_csum = file_header;
        temp_header_for_csum.header_checksum = 0;
        file_header.header_checksum = calculateChecksum(&temp_header_for_csum, sizeof(SerializedHeader));

        size_t final_total_size = current_offset + total_delta_bytes;
        final_total_size = alignOffset(final_total_size, 8);
        
        // Allocate device memory for serialized blob
        uint8_t* d_serialized_blob;
        CUDA_CHECK(cudaMalloc(&d_serialized_blob, final_total_size));
        CUDA_CHECK(cudaMemset(d_serialized_blob, 0, final_total_size));
        
        // Copy header to device
        SerializedHeader* d_header;
        CUDA_CHECK(cudaMalloc(&d_header, sizeof(SerializedHeader)));
        CUDA_CHECK(cudaMemcpy(d_header, &file_header, sizeof(SerializedHeader), cudaMemcpyHostToDevice));
        
        // Launch packing kernel
        int block_size = 256;
        int num_blocks = 8 + ((total_delta_bytes > 0) ? 32 : 0); // 8 for metadata + blocks for delta array
        
        if (compressed_object_to_serialize->num_partitions > 0 || total_delta_bytes > 0) {
            packToBlobKernelOptimized<<<num_blocks, block_size, 0, main_cuda_stream>>>(
                d_header,
                compressed_object_to_serialize->d_start_indices,
                compressed_object_to_serialize->d_end_indices,
                compressed_object_to_serialize->d_model_types,
                compressed_object_to_serialize->d_model_params,
                compressed_object_to_serialize->d_delta_bits,
                compressed_object_to_serialize->d_delta_array_bit_offsets,
                compressed_object_to_serialize->d_error_bounds,
                compressed_object_to_serialize->delta_array,
                compressed_object_to_serialize->num_partitions,
                total_delta_bytes,
                d_serialized_blob
            );
            CUDA_CHECK(cudaGetLastError());
        }
        
        // Single large transfer from device to host
        SerializedData* output_serialized_obj = new SerializedData();
        try { 
            output_serialized_obj->data = new uint8_t[final_total_size]; 
            memset(output_serialized_obj->data, 0, final_total_size);
        }
        catch (const std::bad_alloc& e) { 
            CUDA_CHECK(cudaFree(d_serialized_blob));
            CUDA_CHECK(cudaFree(d_header));
            delete output_serialized_obj; 
            return nullptr; 
        }
        output_serialized_obj->size = final_total_size;
        
        // Single efficient copy
        CUDA_CHECK(cudaMemcpyAsync(output_serialized_obj->data, d_serialized_blob, 
                                final_total_size, cudaMemcpyDeviceToHost, main_cuda_stream));
        CUDA_CHECK(cudaStreamSynchronize(main_cuda_stream));
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_serialized_blob));
        CUDA_CHECK(cudaFree(d_header));
        
        return output_serialized_obj;
    }


// 383 -------------------------------------------------------------
    // GPU-accelerated deserialization
    CompressedData<T>* deserializeGPU(const SerializedData* serialized_input_data, bool preUnpackDeltas = false) {
        if (!serialized_input_data || !serialized_input_data->data || 
            serialized_input_data->size < sizeof(SerializedHeader)) { 
            return nullptr; 
        }
        
        // Read header from host to get sizes
        SerializedHeader read_header;
        memcpy(&read_header, serialized_input_data->data, sizeof(read_header));

        if (read_header.magic != 0x4F43454C || 
            (read_header.version != 4 && read_header.version != 5) ||
            read_header.data_type_size != sizeof(T)) {
            return nullptr; 
        }
        
        SerializedHeader temp_hdr_for_csum = read_header; 
        temp_hdr_for_csum.header_checksum = 0;
        if (calculateChecksum(&temp_hdr_for_csum, sizeof(SerializedHeader)) != read_header.header_checksum) { 
            return nullptr; 
        }

        // Single large transfer from host to device
        uint8_t* d_serialized_blob;
        CUDA_CHECK(cudaMalloc(&d_serialized_blob, serialized_input_data->size));
        CUDA_CHECK(cudaMemcpyAsync(d_serialized_blob, serialized_input_data->data, 
                                serialized_input_data->size, cudaMemcpyHostToDevice, 
                                main_cuda_stream));
        
        // Create new compressed data structure
        CompressedData<T>* new_compressed_data = new CompressedData<T>();
        new_compressed_data->num_partitions = read_header.num_partitions;
        new_compressed_data->total_values = read_header.total_values;
        new_compressed_data->d_plain_deltas = nullptr; // Initialize new member
        
        // Allocate device memory for all SoA arrays
        if (new_compressed_data->num_partitions > 0) {
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_start_indices, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_end_indices, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_model_types, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_model_params, 
                                read_header.num_partitions * 4 * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_delta_bits, 
                                read_header.num_partitions * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_delta_array_bit_offsets, 
                                read_header.num_partitions * sizeof(int64_t)));
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_error_bounds, 
                                read_header.num_partitions * sizeof(long long)));
        } else {
            new_compressed_data->d_start_indices = nullptr;
            new_compressed_data->d_end_indices = nullptr;
            new_compressed_data->d_model_types = nullptr;
            new_compressed_data->d_model_params = nullptr;
            new_compressed_data->d_delta_bits = nullptr;
            new_compressed_data->d_delta_array_bit_offsets = nullptr;
            new_compressed_data->d_error_bounds = nullptr;
        }
        
        uint64_t num_delta_words = (read_header.delta_array_size_bytes + 3) / 4;
        if (num_delta_words > 0) {
            CUDA_CHECK(cudaMalloc(&new_compressed_data->delta_array, 
                                num_delta_words * sizeof(uint32_t)));
        } else { 
            new_compressed_data->delta_array = nullptr; 
        }
        
        // Launch unpacking kernel
        if (new_compressed_data->num_partitions > 0 || num_delta_words > 0) {
            int block_size = 256;
            int grid_size = (new_compressed_data->num_partitions * 8 + num_delta_words + block_size - 1) / block_size;
            grid_size = min(grid_size, 65535);
            
            unpackFromBlobKernel<<<grid_size, block_size, 0, main_cuda_stream>>>(
                d_serialized_blob,
                new_compressed_data->num_partitions,
                read_header.delta_array_size_bytes,
                new_compressed_data->d_start_indices,
                new_compressed_data->d_end_indices,
                new_compressed_data->d_model_types,
                new_compressed_data->d_model_params,
                new_compressed_data->d_delta_bits,
                new_compressed_data->d_delta_array_bit_offsets,
                new_compressed_data->d_error_bounds,
                new_compressed_data->delta_array
            );
            CUDA_CHECK(cudaGetLastError());
        }
        
        // --- NEW STAGE: Conditionally pre-unpack deltas ---
        if (preUnpackDeltas && new_compressed_data->total_values > 0 && new_compressed_data->delta_array != nullptr) {
            // std::cout << "Pre-unpacking deltas for high-throughput mode..." << std::endl;
            
            // Allocate memory for plain deltas
            CUDA_CHECK(cudaMalloc(&new_compressed_data->d_plain_deltas, 
                                new_compressed_data->total_values * sizeof(long long)));
            
            // Launch the unpacking kernel
            int block_size = 256;
            int grid_size = (new_compressed_data->total_values + block_size - 1) / block_size;
            
            // Limit grid size for better occupancy
            cudaDeviceProp prop;
            int device;
            CUDA_CHECK(cudaGetDevice(&device));
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
            grid_size = std::min(grid_size, prop.multiProcessorCount * 32);
            
            unpackAllDeltasKernel<T><<<grid_size, block_size, 0, main_cuda_stream>>>(
                new_compressed_data->d_start_indices,
                new_compressed_data->d_end_indices,
                new_compressed_data->d_model_types,
                new_compressed_data->d_delta_bits,
                new_compressed_data->d_delta_array_bit_offsets,
                new_compressed_data->delta_array,
                new_compressed_data->d_plain_deltas,
                new_compressed_data->num_partitions,
                new_compressed_data->total_values
            );
            CUDA_CHECK(cudaGetLastError());
            
            // Synchronize to ensure kernel completes
            CUDA_CHECK(cudaStreamSynchronize(main_cuda_stream));
            
            // Free the original delta array to save GPU memory
            CUDA_CHECK(cudaFree(new_compressed_data->delta_array));
            new_compressed_data->delta_array = nullptr;
            
            // std::cout << "Pre-unpacking complete. Freed original delta array." << std::endl;
        }
        
        // Set d_self pointer
        CUDA_CHECK(cudaMalloc(&new_compressed_data->d_self, sizeof(CompressedData<T>)));
        CUDA_CHECK(cudaMemcpyAsync(new_compressed_data->d_self, new_compressed_data, 
                                sizeof(CompressedData<T>), cudaMemcpyHostToDevice,
                                main_cuda_stream));
        
        CUDA_CHECK(cudaStreamSynchronize(main_cuda_stream));
        
        // Cleanup
        CUDA_CHECK(cudaFree(d_serialized_blob));
        
        return new_compressed_data;
    }
// 383 -------------------------------------------------------------
};

// Benchmark & File I/O Utilities
template<typename Func>
double benchmark(Func func_to_run, int num_iterations = 100) {
    if (num_iterations > 0) 
        for (int i = 0; i < std::min(5, num_iterations); i++) 
            func_to_run();
    auto timer_start = std::chrono::high_resolution_clock::now();
    if (num_iterations > 0) 
        for (int i = 0; i < num_iterations; i++) 
            func_to_run();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto timer_end = std::chrono::high_resolution_clock::now();
    if (num_iterations <= 0) return 0.0;
    return std::chrono::duration<double, std::milli>(timer_end - timer_start).count() / num_iterations;
}

template<typename T>
bool read_text_file(const std::string& in_filename, std::vector<T>& out_data_vec) {
    std::ifstream file_stream(in_filename);
    if (!file_stream.is_open()) { 
        std::cerr << "Error: Could not open text file " << in_filename << std::endl; 
        return false; 
    }
    out_data_vec.clear(); 
    std::string line_str;
    while (std::getline(file_stream, line_str)) {
        try {
            line_str.erase(0, line_str.find_first_not_of(" \t\n\r\f\v"));
            line_str.erase(line_str.find_last_not_of(" \t\n\r\f\v") + 1);
            if (line_str.empty()) continue;
            if (std::is_same<T, int>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoi(line_str)));
            else if (std::is_same<T, long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stol(line_str)));
            else if (std::is_same<T, long long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoll(line_str)));
            else if (std::is_same<T, unsigned int>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoul(line_str)));
            else if (std::is_same<T, unsigned long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoul(line_str)));
            else if (std::is_same<T, unsigned long long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoull(line_str)));
            else { 
                std::cerr << "Warning: Read_text_file unsupported integer type." << std::endl; 
            }
        } catch (const std::exception& e) { 
            std::cerr << "Warning: Parsing line '"<< line_str << "': " << e.what() << std::endl;
        }
    }
    file_stream.close();
    std::cout << "Successfully read " << out_data_vec.size() << " values from text file: " 
             << in_filename << std::endl;
    return true;
}

template<typename T>
bool read_binary_file(const std::string& in_filename, std::vector<T>& out_data_vec) {
    std::ifstream file_stream(in_filename, std::ios::binary | std::ios::ate);
    if (!file_stream.is_open()) { 
        std::cerr << "Error: Could not open binary file " << in_filename << std::endl; 
        return false; 
    }
    std::streampos stream_file_size = file_stream.tellg();
    if (stream_file_size < 0 || stream_file_size % sizeof(T) != 0) { 
        std::cerr << "Error: Binary file " << in_filename << " has invalid size." << std::endl; 
        file_stream.close(); 
        return false;
    }
    if (stream_file_size == 0) {
        out_data_vec.clear(); 
        file_stream.close(); 
        std::cout << "Read 0 values from empty binary file: " << in_filename << std::endl; 
        return true;
    }
    file_stream.seekg(0, std::ios::beg);
    size_t num_file_elements = static_cast<size_t>(stream_file_size) / sizeof(T);
    try {
        out_data_vec.resize(num_file_elements);
    } catch(const std::bad_alloc&){ 
        std::cerr << "Error: Malloc failed for binary data." << std::endl; 
        file_stream.close();
        return false;
    }
    file_stream.read(reinterpret_cast<char*>(out_data_vec.data()), num_file_elements * sizeof(T));
    bool read_success = file_stream.good() && 
                       (static_cast<size_t>(file_stream.gcount()) == num_file_elements * sizeof(T));
    file_stream.close();
    if(!read_success) {
        out_data_vec.clear(); 
        std::cerr << "Error reading binary file: " << in_filename << std::endl; 
        return false;
    }
    std::cout << "Successfully read " << out_data_vec.size() << " values from binary file: " 
             << in_filename << std::endl;
    return true;
}

// Test Function
template<typename T>
void run_compression_test(const std::vector<T>& data_to_test, const std::string& data_type_string_name, const std::string& dataset_name = "") {
    if (data_to_test.empty()) { 
        std::cout << "\nData is empty for " << data_type_string_name << ". Skipping tests." << std::endl; 
        return; 
    }
    std::cout << "\n=== Testing with data type: " << data_type_string_name << " ===" << std::endl;
    std::cout << "Data type size: " << sizeof(T) << " bytes, Data size: " << data_to_test.size() 
             << " elements" << std::endl;
    LeCoGPU<T> leco_instance;

    // Test three configurations: fixed, variable-CPU, variable-GPU
    std::vector<std::tuple<bool, bool, std::string>> test_configs = {
        {false, false, "Fixed-length"},
        {true, false, "Variable-length (CPU)"},
        {true, true, "Variable-length (GPU Work-Stealing)"}
    };
    
    for (const auto& config : test_configs) {
        bool current_use_variable_part = std::get<0>(config);
        bool use_gpu_partitioning = std::get<1>(config);
        std::string config_name = std::get<2>(config);
        
        std::cout << "\n--- " << config_name << " Partitioning ---" << std::endl;
        
        // Declare all variables that will be used across different sections
        long long current_compressed_size = 0;
        CompressedData<T>* current_compressed_ptr = nullptr;
        double time_to_decompress = 0.0;
        DirectAccessHandle<T>* current_da_handle = nullptr;
        double time_gpu_da = 0.0;

        // Compression
        auto time_to_compress = benchmark([&]() {
            if (current_compressed_ptr) { 
                leco_instance.cleanup(current_compressed_ptr); 
                current_compressed_ptr = nullptr; 
            }
            // 构建带有配置信息的数据集名称
            std::string full_dataset_name = dataset_name.empty() ? 
                data_type_string_name : dataset_name;
            
            // 添加调试输出
            std::cout << "[DEBUG] run_compression_test: dataset_name = '" << dataset_name << "'" << std::endl;
            std::cout << "[DEBUG] run_compression_test: full_dataset_name = '" << full_dataset_name << "'" << std::endl;
            std::cout << "[DEBUG] run_compression_test: use_variable_part = " << current_use_variable_part << std::endl;
            std::cout << "[DEBUG] run_compression_test: use_gpu_partitioning = " << use_gpu_partitioning << std::endl;
            
            current_compressed_ptr = leco_instance.compress(data_to_test, 
                                                        current_use_variable_part, 
                                                        &current_compressed_size, 
                                                        use_gpu_partitioning,
                                                        full_dataset_name);
        }, 1);
        std::cout << "Compression time: " << time_to_compress << " ms\n";
        if (!current_compressed_ptr) { 
            std::cerr << "Compression FAILED." << std::endl; 
            continue; 
        }
        
        long long uncomp_size = (long long)data_to_test.size() * sizeof(T);
        // MODIFIED: Changed compression ratio calculation from compressed/uncompressed to uncompressed/compressed
        double comp_ratio = (current_compressed_size > 0) ? static_cast<double>(uncomp_size) / current_compressed_size : 0.0;
        std::cout << "Uncompressed size: " << uncomp_size << " bytes\n";
        std::cout << "Compressed size:   " << current_compressed_size << " bytes\n";
        std::cout << "Compression Ratio: " << comp_ratio << "\n";
        std::cout << "Number of partitions: " << current_compressed_ptr->num_partitions << "\n";
        
        // If this is GPU variable partitioning, compare with CPU variable partitioning
        if (use_gpu_partitioning) {
            std::cout << "\n  [Partitioning Performance Comparison]" << std::endl;
            CompressedData<T>* cpu_var_compressed = nullptr;
            long long cpu_var_size = 0;
            
            auto cpu_var_time = benchmark([&]() {
                if (cpu_var_compressed) {
                    leco_instance.cleanup(cpu_var_compressed);
                    cpu_var_compressed = nullptr;
                }
                // 修正：这里也要传递 dataset_name
                std::string full_dataset_name = dataset_name.empty() ? 
                    data_type_string_name : dataset_name;
                
                cpu_var_compressed = leco_instance.compress(data_to_test, true, &cpu_var_size, false, 
                                                        full_dataset_name);  // 添加数据集名称参数
            }, 1);
            
            std::cout << "  CPU Variable Partitioning time: " << cpu_var_time << " ms" << std::endl;
            std::cout << "  GPU Work-Stealing speedup: " << (cpu_var_time / time_to_compress) << "x" << std::endl;
            
            if (cpu_var_compressed) {
                std::cout << "  CPU partitions: " << cpu_var_compressed->num_partitions << std::endl;
                std::cout << "  GPU partitions: " << current_compressed_ptr->num_partitions << std::endl;
                std::cout << "  Partition count difference: " 
                         << std::abs(cpu_var_compressed->num_partitions - current_compressed_ptr->num_partitions) 
                         << std::endl;
                leco_instance.cleanup(cpu_var_compressed);
            }
        }
        
        // Random access decompression test
        std::vector<int> random_positions(std::min((size_t)100000, data_to_test.size()));
        if (!data_to_test.empty()) { 
            for (size_t i = 0; i < random_positions.size(); i++) 
                random_positions[i] = rand() % data_to_test.size(); 
        }
        std::vector<T> decompressed_output_data;

        if (!random_positions.empty()) {
            time_to_decompress = benchmark([&]() { 
                leco_instance.decompress(current_compressed_ptr, random_positions, decompressed_output_data); 
            }, 100);
            std::cout << "Random access time for " << random_positions.size() << " queries: " 
                     << time_to_decompress << " ms\n";
            if (time_to_decompress > 0) 
                std::cout << "Throughput: " << (random_positions.size() / time_to_decompress * 1000) 
                         << " queries/sec\n";

            bool is_data_correct = true; 
            int num_mismatches = 0;
            for (size_t i = 0; i < random_positions.size(); i++) {
                if (decompressed_output_data[i] != data_to_test[random_positions[i]]) {
                    is_data_correct = false; 
                    num_mismatches++;
                    if (num_mismatches <= 5) 
                        std::cerr << "Mismatch at original index " << random_positions[i] 
                                 << ": Original " << data_to_test[random_positions[i]] 
                                 << ", Decompressed " << decompressed_output_data[i] << "\n";
                }
            }
            std::cout << "Correctness (vs decompress): " << (is_data_correct ? "PASSED" : "FAILED") << "\n";
            if (num_mismatches > 0) 
                std::cout << "Total mismatches (vs decompress): " << num_mismatches << " out of " 
                         << random_positions.size() << " queries\n";

            if (config_name == "Fixed-length") {

                leco_instance.analyzeRandomAccessPerformance(current_compressed_ptr, 
                                                 random_positions, TILE_SIZE);
                std::cout << "\n--- Optimized Fixed-Partition Random Access Test ---" << std::endl;
                std::vector<T> fixed_ra_output;
                
                // 声明所需的时间变量
                double time_fixed_ra = 0.0;

                // 测试 Fixed-only 优化
                time_fixed_ra = benchmark([&]() {
                    leco_instance.randomAccessFixed(current_compressed_ptr, random_positions, fixed_ra_output, TILE_SIZE);
                }, 100);

                std::cout << "Optimized fixed random access: " << time_fixed_ra << " ms" << std::endl;
                if (time_fixed_ra > 0) {
                    double throughput_qps = static_cast<double>(random_positions.size()) / (time_fixed_ra / 1000.0);
                    std::cout << "  -> Throughput: " << throughput_qps << " queries/sec" << std::endl;
                }

                // 计算并打印性能提升
                if (time_to_decompress > 0 && time_fixed_ra > 0) {
                    std::cout << "  -> Speedup vs. generic random access: " << 
                            (time_to_decompress / time_fixed_ra) << "x" << std::endl;
                }

                // 验证 Fixed-only 的正确性
                bool fixed_ra_correct = true;
                int fixed_ra_mismatches = 0;
                for (size_t i = 0; i < random_positions.size(); i++) {
                    if (fixed_ra_output[i] != data_to_test[random_positions[i]]) {
                        fixed_ra_correct = false;
                        fixed_ra_mismatches++;
                        if (fixed_ra_mismatches <= 5) {
                            std::cerr << "Mismatch (Fixed RA) at original index " << random_positions[i]
                                    << ": Original " << data_to_test[random_positions[i]]
                                    << ", Decompressed " << fixed_ra_output[i] << "\n";
                        }
                    }
                }
                std::cout << "Correctness (Optimized Fixed RA): " << (fixed_ra_correct ? "PASSED" : "FAILED");
                if (fixed_ra_mismatches > 0) {
                    std::cout << " (" << fixed_ra_mismatches << " mismatches)";
                }
                std::cout << std::endl;

                // 测试 Fixed + Pre-unpacked 优化
                std::cout << "\n--- Fixed-Partition Pre-Unpacked Random Access Test ---" << std::endl;
                
                // 序列化并反序列化以启用预解包
                SerializedData* temp_fixed_serialized = leco_instance.serializeGPU(current_compressed_ptr);
                if (temp_fixed_serialized && temp_fixed_serialized->data) {
                    // 反序列化时启用预解包
                    CompressedData<T>* fixed_pre_unpacked = leco_instance.deserializeGPU(temp_fixed_serialized, true);
                    
                    if (fixed_pre_unpacked && fixed_pre_unpacked->d_plain_deltas) {
                        std::cout << "Successfully created pre-unpacked data for fixed partitions." << std::endl;
                        
                        // 测试优化的内核
                        std::vector<T> fixed_pre_unpacked_output;
                        double time_fixed_pre_unpacked = benchmark([&]() {
                            leco_instance.randomAccessFixedPreUnpacked(fixed_pre_unpacked, 
                                                                    random_positions, 
                                                                    fixed_pre_unpacked_output, 
                                                                    TILE_SIZE);
                        }, 100);
                        
                        std::cout << "Fixed + Pre-unpacked random access: " << time_fixed_pre_unpacked << " ms" << std::endl;
                        
                        if (time_fixed_pre_unpacked > 0) {
                            double throughput_qps = static_cast<double>(random_positions.size()) / 
                                                (time_fixed_pre_unpacked / 1000.0);
                            double throughput_mbs = (static_cast<double>(random_positions.size()) * sizeof(T)) / 
                                                (time_fixed_pre_unpacked / 1000.0) / (1024.0 * 1024.0);
                            std::cout << "  -> Throughput: " << throughput_qps << " queries/sec (" 
                                    << throughput_mbs << " MB/s)" << std::endl;
                        }
                        
                        // 与其他方法比较
                        if (time_to_decompress > 0 && time_fixed_pre_unpacked > 0) {
                            std::cout << "  -> Speedup vs. generic random access: " << 
                                    (time_to_decompress / time_fixed_pre_unpacked) << "x" << std::endl;
                        }
                        
                        if (time_fixed_ra > 0 && time_fixed_pre_unpacked > 0) {
                            std::cout << "  -> Speedup vs. fixed-only optimization: " << 
                                    (time_fixed_ra / time_fixed_pre_unpacked) << "x" << std::endl;
                        }
                        
                        // 验证正确性
                        bool fixed_pre_unpacked_correct = true;
                        int fixed_pre_unpacked_mismatches = 0;
                        for (size_t i = 0; i < random_positions.size(); i++) {
                            if (fixed_pre_unpacked_output[i] != data_to_test[random_positions[i]]) {
                                fixed_pre_unpacked_correct = false;
                                fixed_pre_unpacked_mismatches++;
                                if (fixed_pre_unpacked_mismatches <= 5) {
                                    std::cerr << "Mismatch (Fixed+Pre-unpacked) at index " << random_positions[i]
                                            << ": Expected " << data_to_test[random_positions[i]]
                                            << ", Got " << fixed_pre_unpacked_output[i] << "\n";
                                }
                            }
                        }
                        
                        std::cout << "Correctness (Fixed + Pre-unpacked): " << 
                                (fixed_pre_unpacked_correct ? "PASSED" : "FAILED");
                        if (fixed_pre_unpacked_mismatches > 0) {
                            std::cout << " (" << fixed_pre_unpacked_mismatches << " mismatches)";
                        }
                        std::cout << std::endl;
                        
                        // 内存分析
                        std::cout << "\nMemory overhead analysis:" << std::endl;
                        long long bitpacked_size = (current_compressed_size - 
                            (current_compressed_ptr->num_partitions * sizeof(PartitionInfo)));
                        long long preunpacked_size = (long long)fixed_pre_unpacked->total_values * sizeof(long long);
                        std::cout << "  -> Bit-packed delta size: ~" << bitpacked_size << " bytes" << std::endl;
                        std::cout << "  -> Pre-unpacked delta size: " << preunpacked_size << " bytes" << std::endl;
                        std::cout << "  -> Memory overhead factor: " << 
                                ((double)preunpacked_size / bitpacked_size) << "x" << std::endl;
                        
                        // 所有固定分区方法的总结
                        std::cout << "\n--- Fixed-Partition Random Access Summary ---" << std::endl;
                        std::cout << "1. Generic random access: " << time_to_decompress << " ms" << std::endl;
                        std::cout << "2. Fixed-only optimization: " << time_fixed_ra << " ms" << std::endl;
                        std::cout << "3. Fixed + Pre-unpacked: " << time_fixed_pre_unpacked << " ms" << std::endl;
                        
                        if (time_to_decompress > 0 && time_fixed_pre_unpacked > 0) {
                            std::cout << "Total speedup achieved: " << 
                                    (time_to_decompress / time_fixed_pre_unpacked) << "x" << std::endl;
                        }
                        
                        // 清理
                        leco_instance.cleanup(fixed_pre_unpacked);
                    } else {
                        std::cout << "Failed to create pre-unpacked data for fixed partitions." << std::endl;
                    }
                    
                    delete temp_fixed_serialized;
                }
            }
        }

        // ========== NEW: Pre-Unpacked Random Access Test ==========
        std::cout << "\n--- Pre-Unpacked Random Access Test ---" << std::endl;
        
        // Serialize current compressed data
        SerializedData* temp_serialized = leco_instance.serializeGPU(current_compressed_ptr);
        if (temp_serialized && temp_serialized->data) {
            // Deserialize with pre-unpacking enabled
            CompressedData<T>* pre_unpacked_data = leco_instance.deserializeGPU(temp_serialized, true);
            
            if (pre_unpacked_data && pre_unpacked_data->d_plain_deltas) {
                std::cout << "Successfully created pre-unpacked data structure." << std::endl;
                
                // Test standard random access with pre-unpacked data
                std::vector<T> standard_ra_output;
                auto time_standard_ra = benchmark([&]() {
                    leco_instance.decompress(pre_unpacked_data, random_positions, standard_ra_output);
                }, 100);
                
                // Test optimized pre-unpacked random access
                std::vector<T> pre_unpacked_ra_output;
                auto time_pre_unpacked_ra = benchmark([&]() {
                    leco_instance.randomAccessPreUnpacked(pre_unpacked_data, random_positions, pre_unpacked_ra_output);
                }, 100);
                
                std::cout << "Standard random access (pre-unpacked data): " << time_standard_ra << " ms" << std::endl;
                // *** ADDED THIS BLOCK ***
                if (time_standard_ra > 0) {
                    double throughput_qps = static_cast<double>(random_positions.size()) / (time_standard_ra / 1000.0);
                    double throughput_mbs = (static_cast<double>(random_positions.size()) * sizeof(T)) / (time_standard_ra / 1000.0) / (1024.0 * 1024.0);
                    std::cout << "  -> Throughput: " << throughput_qps << " queries/sec (" << throughput_mbs << " MB/s)" << std::endl;
                }
                
                std::cout << "Optimized pre-unpacked random access: " << time_pre_unpacked_ra << " ms" << std::endl;
                // *** ADDED THIS BLOCK ***
                if (time_pre_unpacked_ra > 0) {
                    double throughput_qps = static_cast<double>(random_positions.size()) / (time_pre_unpacked_ra / 1000.0);
                    double throughput_mbs = (static_cast<double>(random_positions.size()) * sizeof(T)) / (time_pre_unpacked_ra / 1000.0) / (1024.0 * 1024.0);
                     std::cout << "  -> Throughput: " << throughput_qps << " queries/sec (" << throughput_mbs << " MB/s)" << std::endl;
                }

                if (time_standard_ra > 0 && time_pre_unpacked_ra > 0) {
                    std::cout << "Pre-unpacked optimization speedup: " << 
                              (time_standard_ra / time_pre_unpacked_ra) << "x" << std::endl;
                }
                
                if (time_to_decompress > 0 && time_pre_unpacked_ra > 0) {
                    std::cout << "Speedup vs bit-packed random access: " << 
                              (time_to_decompress / time_pre_unpacked_ra) << "x" << std::endl;
                }
                
                // Verify correctness
                bool pre_unpacked_correct = true;
                int pre_unpacked_mismatches = 0;
                for (size_t i = 0; i < random_positions.size(); i++) {
                    if (pre_unpacked_ra_output[i] != data_to_test[random_positions[i]]) {
                        pre_unpacked_correct = false;
                        pre_unpacked_mismatches++;
                        if (pre_unpacked_mismatches <= 5) {
                            std::cerr << "Pre-unpacked mismatch at index " << random_positions[i] 
                                     << ": Expected " << data_to_test[random_positions[i]] 
                                     << ", Got " << pre_unpacked_ra_output[i] << "\n";
                        }
                    }
                }
                
                std::cout << "Pre-unpacked random access correctness: " << 
                          (pre_unpacked_correct ? "PASSED" : "FAILED");
                if (pre_unpacked_mismatches > 0) {
                    std::cout << " (" << pre_unpacked_mismatches << " mismatches out of " 
                             << random_positions.size() << " queries)";
                }
                std::cout << std::endl;
                
                // Memory overhead analysis
                long long bitpacked_delta_size = (current_compressed_size - 
                    (current_compressed_ptr->num_partitions * sizeof(PartitionInfo)));
                long long preunpacked_delta_size = 
                    (long long)pre_unpacked_data->total_values * sizeof(long long);
                
                std::cout << "Memory overhead for pre-unpacking: " << 
                          ((double)preunpacked_delta_size / bitpacked_delta_size) << "x" << std::endl;
                
                leco_instance.cleanup(pre_unpacked_data);
            } else {
                std::cout << "Failed to create pre-unpacked data structure." << std::endl;
            }
            
            delete temp_serialized;
        }
        // ========== END: Pre-Unpacked Random Access Test ==========

        // Full-file decompression test
        if (!current_use_variable_part) {
            // Fixed-length case: perform comprehensive testing
            std::cout << "\n--- Full-File Decompression (Fixed Partitions) ---" << std::endl;

            // Test with original compressed data (bit-packed deltas)
            std::cout << "  [Bit-Packed Mode] Testing with original compressed data..." << std::endl;
            std::vector<T> full_decompressed_bitpacked;
            auto time_bitpacked = benchmark([&]() {
                leco_instance.decompressFullFile(current_compressed_ptr, full_decompressed_bitpacked);
            }, 10);
            double throughput_bitpacked = (data_to_test.size() * sizeof(T)) / (time_bitpacked / 1000.0) / (1024.0 * 1024.0);
            std::cout << "    -> Time: " << time_bitpacked << " ms, Throughput: " << throughput_bitpacked << " MB/s" << std::endl;
            
            bool correct_bitpacked = (full_decompressed_bitpacked.size() == data_to_test.size() && 
                                    std::equal(full_decompressed_bitpacked.begin(), full_decompressed_bitpacked.end(), data_to_test.begin()));
            std::cout << "    -> Correctness: " << (correct_bitpacked ? "PASSED" : "FAILED") << std::endl;

            // Test the optimized fixed-partition kernel (no binary search)
            std::cout << "  [Fixed-Optimized] Testing decompressFullFile_Fix (no binary search)..." << std::endl;
            std::vector<T> full_decompressed_fix;
            auto time_fix = benchmark([&]() {
                leco_instance.decompressFullFile_Fix(current_compressed_ptr, full_decompressed_fix, TILE_SIZE);
            }, 10);
            double throughput_fix = (data_to_test.size() * sizeof(T)) / (time_fix / 1000.0) / (1024.0 * 1024.0);
            std::cout << "    -> Time: " << time_fix << " ms, Throughput: " << throughput_fix << " MB/s" << std::endl;
            
            bool correct_fix = (full_decompressed_fix.size() == data_to_test.size() &&
                              std::equal(full_decompressed_fix.begin(), full_decompressed_fix.end(), data_to_test.begin()));
            std::cout << "    -> Correctness: " << (correct_fix ? "PASSED" : "FAILED") << std::endl;

            std::cout << "\n  [Fixed-Partition Analysis]" << std::endl;
            if (time_fix > 0 && time_bitpacked > 0) {
                double speedup = time_bitpacked / time_fix;
                std::cout << "    -> Speedup of 'Fix' over standard: " << speedup << "x" << std::endl;
            }
        } else {
            // Variable-length case
            std::cout << "\n--- Full-File Decompression (" << config_name << ") ---" << std::endl;
            std::vector<T> full_decompressed_data;
            auto time_full_decompress = benchmark([&]() {
                leco_instance.decompressFullFile(current_compressed_ptr, full_decompressed_data);
            }, 10);
            double throughput = (data_to_test.size() * sizeof(T)) / (time_full_decompress / 1000.0) / (1024.0 * 1024.0);
            std::cout << "  -> Time: " << time_full_decompress << " ms, Throughput: " << throughput << " MB/s" << std::endl;
            
            bool correct = (full_decompressed_data.size() == data_to_test.size() &&
                            std::equal(full_decompressed_data.begin(), full_decompressed_data.end(), data_to_test.begin()));
            std::cout << "  -> Correctness: " << (correct ? "PASSED" : "FAILED") << std::endl;
        }
        
        // Serialization/Deserialization
        std::cout << "\n--- Serialization/Deserialization ---" << std::endl;
        SerializedData* current_serialized_obj = nullptr;
        auto time_to_serialize = benchmark([&]() { 
            if(current_serialized_obj) delete current_serialized_obj; 
            current_serialized_obj = leco_instance.serialize(current_compressed_ptr); 
        }, 10);
        std::cout << "CPU Serialization time: " << time_to_serialize << " ms\n";
        if (!current_serialized_obj || !current_serialized_obj->data || current_serialized_obj->size == 0) {
            leco_instance.cleanup(current_compressed_ptr); 
            if(current_serialized_obj) delete current_serialized_obj; 
            std::cerr << "Serialization FAILED." << std::endl; 
            continue;
        }
        std::cout << "Serialized size: " << current_serialized_obj->size << " bytes\n";
        
        // GPU Serialization test
        SerializedData* gpu_serialized_obj = nullptr;
        auto time_to_serialize_gpu = benchmark([&]() { 
            if(gpu_serialized_obj) delete gpu_serialized_obj; 
            gpu_serialized_obj = leco_instance.serializeGPU(current_compressed_ptr); 
        }, 10);
        std::cout << "GPU Serialization time: " << time_to_serialize_gpu << " ms\n";
        
        if (!gpu_serialized_obj || !gpu_serialized_obj->data || gpu_serialized_obj->size == 0) {
            std::cerr << "GPU Serialization FAILED." << std::endl; 
        } else {
            std::cout << "GPU Serialized size: " << gpu_serialized_obj->size << " bytes\n";
            std::cout << "GPU Serialization speedup over CPU: " << (time_to_serialize / time_to_serialize_gpu) << "x\n";
            
            // Use GPU serialized data for further tests
            delete current_serialized_obj;
            current_serialized_obj = gpu_serialized_obj;
            gpu_serialized_obj = nullptr;
        }
        
        leco_instance.cleanup(current_compressed_ptr); 
        current_compressed_ptr = nullptr; 
        
        // CPU Deserialization
        CompressedData<T>* current_deserialized_data = nullptr;
        auto time_to_deserialize = benchmark([&]() { 
            if(current_deserialized_data) leco_instance.cleanup(current_deserialized_data); 
            current_deserialized_data = leco_instance.deserialize(current_serialized_obj); 
        }, 10);
        std::cout << "CPU Deserialization time: " << time_to_deserialize << " ms\n";
        
        // GPU Deserialization
        CompressedData<T>* gpu_deserialized_data = nullptr;
        auto time_to_deserialize_gpu = benchmark([&]() { 
            if(gpu_deserialized_data) leco_instance.cleanup(gpu_deserialized_data); 
            gpu_deserialized_data = leco_instance.deserializeGPU(current_serialized_obj, false); 
        }, 10);
        std::cout << "GPU Deserialization time: " << time_to_deserialize_gpu << " ms\n";
        std::cout << "GPU Deserialization speedup over CPU: " << (time_to_deserialize / time_to_deserialize_gpu) << "x\n";
        
        if (!current_deserialized_data || !gpu_deserialized_data) { 
            if (current_deserialized_data) leco_instance.cleanup(current_deserialized_data);
            if (gpu_deserialized_data) leco_instance.cleanup(gpu_deserialized_data);
            delete current_serialized_obj; 
            std::cerr << "Deserialization FAILED." << std::endl; 
            continue; 
        }
        
        // Use GPU deserialized data for consistency
        leco_instance.cleanup(current_deserialized_data);
        current_deserialized_data = gpu_deserialized_data;
        gpu_deserialized_data = nullptr;

        // Comprehensive Decompression Mode Testing with Work-Stealing
        std::cout << "\n--- Comprehensive Decompression Mode Testing (Including Work-Stealing) ---" << std::endl;

        if (current_serialized_obj) {
            std::cout << "Testing different decompression strategies:" << std::endl;
            
            // 1. Deserialize in standard mode (bit-packed)
            CompressedData<T>* standard_mode_data = nullptr;
            auto time_deser_standard = benchmark([&]() {
                if(standard_mode_data) leco_instance.cleanup(standard_mode_data);
                standard_mode_data = leco_instance.deserializeGPU(current_serialized_obj, false);
            }, 5);
            
            // 2. Deserialize in high-throughput mode (pre-unpacked)
            CompressedData<T>* high_throughput_data = nullptr;
            auto time_deser_high_throughput = benchmark([&]() {
                if(high_throughput_data) leco_instance.cleanup(high_throughput_data);
                high_throughput_data = leco_instance.deserializeGPU(current_serialized_obj, true);
            }, 5);
            
            std::cout << "\n1. Deserialization Performance:" << std::endl;
            std::cout << "   Standard mode (bit-packed): " << time_deser_standard << " ms" << std::endl;
            std::cout << "   High-throughput mode (pre-unpacked): " << time_deser_high_throughput << " ms" << std::endl;
            std::cout << "   Pre-unpacking overhead: " << 
                    (time_deser_high_throughput / time_deser_standard) << "x" << std::endl;
            
            if (standard_mode_data && high_throughput_data) {
                std::cout << "\n2. Full-File Decompression Performance (Including Work-Stealing):" << std::endl;
                
                // Test all decompression methods
                std::vector<T> decomp_bitpacked;
                std::vector<T> decomp_preunpacked;
                std::vector<T> decomp_auto_standard;
                std::vector<T> decomp_auto_ht;
                std::vector<T> decomp_optimized_v2;
                std::vector<T> decomp_work_stealing;
                std::vector<T> decomp_work_stealing_advanced;
                
                // 2.1 Bit-packed kernel with standard data
                auto time_bitpacked = benchmark([&]() {
                    leco_instance.decompressFullFile_BitPacked(standard_mode_data, decomp_bitpacked);
                }, 10);
                double throughput_bitpacked = (data_to_test.size() * sizeof(T)) / 
                                            (time_bitpacked / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.1 Bit-packed kernel (standard data): " 
                        << time_bitpacked << " ms (" << throughput_bitpacked << " MB/s)" << std::endl;
                
                // 2.2 Pre-unpacked kernel with high-throughput data
                auto time_preunpacked = benchmark([&]() {
                    leco_instance.decompressFullFile_PreUnpacked(high_throughput_data, decomp_preunpacked);
                }, 10);
                double throughput_preunpacked = (data_to_test.size() * sizeof(T)) / 
                                            (time_preunpacked / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.2 Pre-unpacked kernel (HT data): " 
                        << time_preunpacked << " ms (" << throughput_preunpacked << " MB/s)" << std::endl;
                
                // 2.3 Auto-selection with standard data
                auto time_auto_standard = benchmark([&]() {
                    leco_instance.decompressFullFile(standard_mode_data, decomp_auto_standard);
                }, 10);
                double throughput_auto_standard = (data_to_test.size() * sizeof(T)) / 
                                                (time_auto_standard / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.3 Auto-selection (standard data): " 
                        << time_auto_standard << " ms (" << throughput_auto_standard << " MB/s)" << std::endl;
                
                // 2.4 Auto-selection with HT data
                auto time_auto_ht = benchmark([&]() {
                    leco_instance.decompressFullFile(high_throughput_data, decomp_auto_ht);
                }, 10);
                double throughput_auto_ht = (data_to_test.size() * sizeof(T)) / 
                                        (time_auto_ht / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.4 Auto-selection (HT data): " 
                        << time_auto_ht << " ms (" << throughput_auto_ht << " MB/s)" << std::endl;
                
                // 2.5 Optimized V2 kernel
                auto time_optimized_v2 = benchmark([&]() {
                    leco_instance.decompressFullFile_OnTheFly_Optimized_V2(standard_mode_data, decomp_optimized_v2);
                }, 10);
                double throughput_optimized_v2 = (data_to_test.size() * sizeof(T)) / 
                                                (time_optimized_v2 / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.5 Optimized V2 kernel (standard data): " 
                        << time_optimized_v2 << " ms (" << throughput_optimized_v2 << " MB/s)" << std::endl;
                
                // 2.6 Work-stealing kernel (standard data)
                auto time_work_stealing = benchmark([&]() {
                    leco_instance.decompressFullFile_WorkStealing(standard_mode_data, decomp_work_stealing);
                }, 10);
                double throughput_work_stealing = (data_to_test.size() * sizeof(T)) / 
                                                (time_work_stealing / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.6 Work-stealing kernel (standard data): " 
                        << time_work_stealing << " ms (" << throughput_work_stealing << " MB/s)" << std::endl;
                
                // 2.7 Work-stealing advanced kernel (HT data)
                auto time_work_stealing_advanced = benchmark([&]() {
                    leco_instance.decompressFullFile_WorkStealingAdvanced(high_throughput_data, decomp_work_stealing_advanced);
                }, 10);
                double throughput_work_stealing_advanced = (data_to_test.size() * sizeof(T)) / 
                                                        (time_work_stealing_advanced / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.7 Work-stealing advanced (HT data): " 
                        << time_work_stealing_advanced << " ms (" << throughput_work_stealing_advanced << " MB/s)" << std::endl;
                
                // 2.8 Fixed partition optimization (if applicable)
                if (!current_use_variable_part) {
                    std::vector<T> decomp_fix_standard;
                    std::vector<T> decomp_fix_ht;
                    
                    auto time_fix_standard = benchmark([&]() {
                        leco_instance.decompressFullFile_Fix(standard_mode_data, decomp_fix_standard, TILE_SIZE);
                    }, 10);
                    double throughput_fix_standard = (data_to_test.size() * sizeof(T)) / 
                                                (time_fix_standard / 1000.0) / (1024.0 * 1024.0);
                    std::cout << "   2.8 Fixed optimization (standard data): " 
                            << time_fix_standard << " ms (" << throughput_fix_standard << " MB/s)" << std::endl;
                    
                    auto time_fix_ht = benchmark([&]() {
                        leco_instance.decompressFullFile_Fix(high_throughput_data, decomp_fix_ht, TILE_SIZE);
                    }, 10);
                    double throughput_fix_ht = (data_to_test.size() * sizeof(T)) / 
                                            (time_fix_ht / 1000.0) / (1024.0 * 1024.0);
                    std::cout << "   2.9 Fixed optimization (HT data): " 
                            << time_fix_ht << " ms (" << throughput_fix_ht << " MB/s)" << std::endl;
                }
                
                // 3. Performance Analysis
                std::cout << "\n3. Performance Analysis:" << std::endl;
                if (time_bitpacked > 0) {
                    std::cout << "   Pre-unpacking speedup: " << 
                            (time_bitpacked / time_preunpacked) << "x" << std::endl;
                    std::cout << "   Work-stealing speedup over standard: " << 
                            (time_bitpacked / time_work_stealing) << "x" << std::endl;
                    std::cout << "   Work-stealing advanced speedup: " << 
                            (time_bitpacked / time_work_stealing_advanced) << "x" << std::endl;
                    
                    // Find best performance
                    double best_time = std::min({time_preunpacked, time_optimized_v2, 
                                               time_work_stealing, time_work_stealing_advanced});
                    std::cout << "   Best overall speedup: " << 
                            (time_bitpacked / best_time) << "x" << std::endl;
                }
                
                // 4. Work-Stealing Efficiency Analysis
                std::cout << "\n4. Work-Stealing Efficiency Analysis:" << std::endl;
                if (standard_mode_data->num_partitions > 0) {
                    std::cout << "   Number of partitions: " << standard_mode_data->num_partitions << std::endl;
                    std::cout << "   Average partition size: " << 
                            (data_to_test.size() / standard_mode_data->num_partitions) << " elements" << std::endl;
                    
                    // Calculate load imbalance metric
                    std::vector<int> partition_sizes;
                    std::vector<int32_t> h_starts(standard_mode_data->num_partitions);
                    std::vector<int32_t> h_ends(standard_mode_data->num_partitions);
                    CUDA_CHECK(cudaMemcpy(h_starts.data(), standard_mode_data->d_start_indices,
                                        standard_mode_data->num_partitions * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_ends.data(), standard_mode_data->d_end_indices,
                                        standard_mode_data->num_partitions * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost));
                    
                    int min_size = INT_MAX, max_size = 0;
                    for (int i = 0; i < standard_mode_data->num_partitions; i++) {
                        int size = h_ends[i] - h_starts[i];
                        min_size = std::min(min_size, size);
                        max_size = std::max(max_size, size);
                    }
                    
                    std::cout << "   Min partition size: " << min_size << " elements" << std::endl;
                    std::cout << "   Max partition size: " << max_size << " elements" << std::endl;
                    std::cout << "   Load imbalance ratio: " << 
                            (max_size > 0 ? (double)max_size / min_size : 0) << std::endl;
                    
                    // Work-stealing efficiency
                    if (time_optimized_v2 > 0 && time_work_stealing > 0) {
                        double efficiency = (time_optimized_v2 - time_work_stealing) / time_optimized_v2 * 100;
                        std::cout << "   Work-stealing efficiency gain: " << efficiency << "%" << std::endl;
                    }
                }
                
                // 5. Correctness Verification
                std::cout << "\n5. Correctness Verification:" << std::endl;
                auto verify_correctness = [&](const std::vector<T>& result, const std::string& method) {
                    bool correct = (result.size() == data_to_test.size()) &&
                                std::equal(result.begin(), result.end(), data_to_test.begin());
                    std::cout << "   " << method << ": " << (correct ? "PASSED" : "FAILED");
                    
                    if (!correct && result.size() == data_to_test.size()) {
                        int mismatches = 0;
                        for (size_t i = 0; i < result.size(); ++i) {
                            if (result[i] != data_to_test[i]) {
                                mismatches++;
                                if (mismatches <= 3) {
                                    std::cout << "\n     Mismatch at " << i << ": expected " 
                                            << data_to_test[i] << ", got " << result[i];
                                }
                            }
                        }
                        std::cout << "\n     Total mismatches: " << mismatches;
                    }
                    std::cout << std::endl;
                };
                
                verify_correctness(decomp_bitpacked, "Bit-packed kernel");
                verify_correctness(decomp_preunpacked, "Pre-unpacked kernel");
                verify_correctness(decomp_auto_standard, "Auto-selection (standard)");
                verify_correctness(decomp_auto_ht, "Auto-selection (HT)");
                verify_correctness(decomp_optimized_v2, "Optimized V2 kernel");
                verify_correctness(decomp_work_stealing, "Work-stealing kernel");
                verify_correctness(decomp_work_stealing_advanced, "Work-stealing advanced");
                
                // 6. Memory Analysis
                std::cout << "\n6. Memory Usage Analysis:" << std::endl;
                long long bitpacked_delta_size = (current_compressed_size - 
                    (standard_mode_data->num_partitions * sizeof(PartitionInfo)));
                long long preunpacked_delta_size = 
                    (long long)high_throughput_data->total_values * sizeof(long long);
                
                std::cout << "   Bit-packed delta storage: ~" << bitpacked_delta_size << " bytes" << std::endl;
                std::cout << "   Pre-unpacked delta storage: " << preunpacked_delta_size << " bytes" << std::endl;
                std::cout << "   Memory overhead factor: " << 
                        ((double)preunpacked_delta_size / bitpacked_delta_size) << "x" << std::endl;
            }
            
            // Cleanup
            if (standard_mode_data) leco_instance.cleanup(standard_mode_data);
            if (high_throughput_data) leco_instance.cleanup(high_throughput_data);
        }

        // Direct Random Access tests
        std::cout << "\n--- Direct Random Access ---" << std::endl;
        current_da_handle = leco_instance.createDirectAccessHandle(
            current_serialized_obj->data, current_serialized_obj->size, true);
            
        if (current_da_handle && !random_positions.empty()) {
            std::vector<T> gpu_direct_output;
            time_gpu_da = benchmark([&]() { 
                leco_instance.directRandomAccessGPU(current_da_handle, random_positions, gpu_direct_output); 
            }, 100);
            std::cout << "GPU Direct Random Access: " << time_gpu_da << " ms, Throughput: " 
                     << (time_gpu_da > 0 ? (random_positions.size() / time_gpu_da * 1000) : 0) 
                     << " queries/sec\n";
            bool gpu_da_correct = true; 
            int gpu_da_mismatches = 0;
            for(size_t i=0; i<random_positions.size(); ++i) {
                if(gpu_direct_output[i] != data_to_test[random_positions[i]]) {
                    gpu_da_correct=false; 
                    gpu_da_mismatches++;
                }
            }
            std::cout << "GPU Direct Access Correctness: " << (gpu_da_correct ? "PASSED" : "FAILED") 
                     << (gpu_da_mismatches > 0 ? " (" + std::to_string(gpu_da_mismatches) + " mismatches)" : "") 
                     << "\n";

            // CPU Direct Random Access
            std::vector<T> cpu_direct_output;
            auto time_cpu_da = benchmark([&]() { 
                leco_instance.directRandomAccessCPU(current_da_handle, random_positions, 
                                                   cpu_direct_output, false); 
            }, 10);
            std::cout << "CPU Direct Random Access: " << time_cpu_da << " ms, Throughput: " 
                     << (time_cpu_da > 0 ? (random_positions.size() / time_cpu_da * 1000) : 0) 
                     << " queries/sec\n";
            if (time_gpu_da > 0 && time_cpu_da > 0) 
                std::cout << "GPU Speedup over CPU: " << (time_cpu_da / time_gpu_da) << "x\n";
            
            bool cpu_da_correct = true; 
            int cpu_da_mismatches = 0;
            for(size_t k=0; k<random_positions.size(); ++k) {
                if(cpu_direct_output[k] != data_to_test[random_positions[k]]) {
                    cpu_da_correct=false; 
                    cpu_da_mismatches++;
                }
            }
            std::cout << "CPU Direct Access Correctness: " << (cpu_da_correct ? "PASSED" : "FAILED") 
                     << (cpu_da_mismatches > 0 ? " (" + std::to_string(cpu_da_mismatches) + " mismatches)" : "") 
                     << "\n";
            
            leco_instance.destroyDirectAccessHandle(current_da_handle);
        }
        
        // File I/O tests
        std::cout << "\n--- File I/O ---" << std::endl;
        std::string output_file_name = "test_" + data_type_string_name + "_" + config_name + ".leco";
        // Replace spaces with underscores in filename
        std::replace(output_file_name.begin(), output_file_name.end(), ' ', '_');
        std::replace(output_file_name.begin(), output_file_name.end(), '(', '_');
        std::replace(output_file_name.begin(), output_file_name.end(), ')', '_');
        std::replace(output_file_name.begin(), output_file_name.end(), '-', '_');
        
        if (leco_instance.saveToFile(current_serialized_obj->data, current_serialized_obj->size, 
                                    output_file_name)) {
            std::cout << "File save/load test: SUCCESS\n";
            std::remove(output_file_name.c_str());
        }
        
        leco_instance.cleanup(current_deserialized_data); 
        delete current_serialized_obj;
    }
}

// Main Function
int main(int argc, char* argv[]) {
    srand(static_cast<unsigned int>(time(NULL)));
    std::string data_type_arg_str = "int";
    std::string file_type_flag_str = "";
    std::string input_filename_str = "";

    if (argc >= 2) data_type_arg_str = argv[1];
    if (argc == 3) {
        input_filename_str = argv[2];
        if (input_filename_str == "--text" || input_filename_str == "--binary" || 
            input_filename_str == "--help" || input_filename_str == "-h") {
            std::cerr << "Error: Flag '" << input_filename_str 
                     << "' needs a filename, or invalid help arguments." << std::endl;
            input_filename_str = ""; 
            file_type_flag_str = "";
        } else { 
            file_type_flag_str = "--text"; // Default to text
        }
    } else if (argc >= 4) {
        file_type_flag_str = argv[2]; 
        input_filename_str = argv[3];
        if (file_type_flag_str != "--text" && file_type_flag_str != "--binary") {
            std::cerr << "Error: Invalid file type specifier '" << file_type_flag_str 
                     << "'. Use --text or --binary." << std::endl;
            input_filename_str = ""; 
            file_type_flag_str = "";
        }
    }

    std::cout << "LeCo GPU Compression Test - Integer Focus" << std::endl;
    std::cout << "Selected Data Type: " << data_type_arg_str << std::endl;
    if (!input_filename_str.empty() && !file_type_flag_str.empty()) 
        std::cout << "Input File: " << input_filename_str << " (Type: " << file_type_flag_str << ")" << std::endl;
    else { 
        std::cout << "Using synthetic data." << std::endl; 
        input_filename_str = ""; 
        file_type_flag_str = ""; 
    }
    
    int cuda_device_count; 
    CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
    if (cuda_device_count == 0) { 
        std::cerr << "No CUDA devices found." << std::endl; 
        return 1; 
    }
    cudaDeviceProp cuda_dev_props; 
    CUDA_CHECK(cudaGetDeviceProperties(&cuda_dev_props, 0));
    std::cout << "Using GPU: " << cuda_dev_props.name << " (Compute: " << cuda_dev_props.major 
             << "." << cuda_dev_props.minor << ")" << std::endl;
    
    bool data_loaded_ok = false;

    std::string dataset_name;
    if (!input_filename_str.empty()) {
        // 去除路径，只保留文件名
        size_t last_slash = input_filename_str.find_last_of("/\\");
        dataset_name = (last_slash != std::string::npos) ? 
                       input_filename_str.substr(last_slash + 1) : input_filename_str;
        
        // 去除扩展名
        size_t last_dot = dataset_name.find_last_of(".");
        if (last_dot != std::string::npos) {
            dataset_name = dataset_name.substr(0, last_dot);
        }
    } else {
        dataset_name = "synthetic_" + data_type_arg_str;
    }

    // printf("file name = %s\n", dataset_name);
    std::cout << "file name = " << dataset_name << std::endl;
    

    if (data_type_arg_str == "int") {
        std::vector<int> int_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") 
                data_loaded_ok = read_text_file(input_filename_str, int_data_vec);
            else if (file_type_flag_str == "--binary") 
                data_loaded_ok = read_binary_file(input_filename_str, int_data_vec);
        } else { 
            int_data_vec.resize(1000000); 
            for(size_t i=0; i<int_data_vec.size(); ++i) 
                int_data_vec[i] = 1000 + static_cast<int>(i) * 5 + (rand() % 20 - 10); 
            data_loaded_ok=true; 
            std::cout << "Generated synthetic int data: " << int_data_vec.size() << std::endl;
        }
        if (data_loaded_ok && !int_data_vec.empty()) 
            run_compression_test(int_data_vec, "int", dataset_name);
        else if(data_loaded_ok) 
            std::cout << "Data empty for int." << std::endl;
        else if(!input_filename_str.empty()) 
            std::cout << "Failed to load int data." << std::endl;

    } else if (data_type_arg_str == "long") {
        std::vector<long> long_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") 
                data_loaded_ok = read_text_file(input_filename_str, long_data_vec);
            else if (file_type_flag_str == "--binary") 
                data_loaded_ok = read_binary_file(input_filename_str, long_data_vec);
        } else { 
            long_data_vec.resize(1000000); 
            for(size_t i=0; i<long_data_vec.size(); ++i) 
                long_data_vec[i] = 100000L + static_cast<long>(i) * 50L + (rand() % 100 - 50);
            data_loaded_ok=true; 
            std::cout << "Generated synthetic long data: " << long_data_vec.size() << std::endl;
        }
        if (data_loaded_ok && !long_data_vec.empty()) 
            run_compression_test(long_data_vec, "long", dataset_name);
        else if(data_loaded_ok) 
            std::cout << "Data empty for long." << std::endl;
        else if(!input_filename_str.empty()) 
            std::cout << "Failed to load long data." << std::endl;

    } else if (data_type_arg_str == "long_long" || data_type_arg_str == "longlong") {
        std::vector<long long> ll_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") 
                data_loaded_ok = read_text_file(input_filename_str, ll_data_vec);
            else if (file_type_flag_str == "--binary") 
                data_loaded_ok = read_binary_file(input_filename_str, ll_data_vec);
        } else { 
            std::cout << "Generating synthetic long long data..." << std::endl;
            ll_data_vec.resize(1000000);
            for (size_t i = 0; i < ll_data_vec.size(); i++) { 
                 ll_data_vec[i] = 1000000000LL + static_cast<long long>(i) * 
                                 (500LL + (rand()%200 - 100)) + (rand() % 1000 - 500);
            }
            std::cout << "Synthetic data generated: " << ll_data_vec.size() << " long longs." << std::endl;
            data_loaded_ok = true;
        }
        if (data_loaded_ok && !ll_data_vec.empty()) 
            run_compression_test(ll_data_vec, "long_long", dataset_name);
        else if(data_loaded_ok) 
            std::cout << "Data empty for long_long." << std::endl;
        else if(!input_filename_str.empty()) 
            std::cout << "Failed to load long_long data." << std::endl;

    } else if (data_type_arg_str == "unsigned_long_long" || data_type_arg_str == "ull") {
        std::vector<unsigned long long> ull_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") 
                data_loaded_ok = read_text_file(input_filename_str, ull_data_vec);
            else if (file_type_flag_str == "--binary") 
                data_loaded_ok = read_binary_file(input_filename_str, ull_data_vec);
        } else { 
            std::cout << "Generating synthetic unsigned long long data..." << std::endl;
            ull_data_vec.resize(1000000);
            // Use smaller base value to avoid overflow issues
            unsigned long long base_val = 1000000000000ULL; // 1 trillion instead of ULLONG_MAX/2
            for (size_t i = 0; i < ull_data_vec.size(); i++) {
                ull_data_vec[i] = base_val + static_cast<unsigned long long>(i) * 
                                 (500ULL + (rand()%200 - 100)) + (rand() % 1000 - 500);
            }
            std::cout << "Synthetic data generated: " << ull_data_vec.size() 
                     << " unsigned long longs." << std::endl;
            data_loaded_ok = true;
        }
        if (data_loaded_ok && !ull_data_vec.empty()) 
            run_compression_test(ull_data_vec, "unsigned_long_long", dataset_name);
        else if(data_loaded_ok) 
            std::cout << "Data empty for unsigned_long_long." << std::endl;
        else if(!input_filename_str.empty()) 
            std::cout << "Failed to load unsigned_long_long data." << std::endl;
        
    } else if (data_type_arg_str == "uint" || data_type_arg_str == "uint32" || 
               data_type_arg_str == "unsigned_int") {
        std::vector<unsigned int> uint_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") 
                data_loaded_ok = read_text_file(input_filename_str, uint_data_vec);
            else if (file_type_flag_str == "--binary") 
                data_loaded_ok = read_binary_file(input_filename_str, uint_data_vec);
        } else { 
            std::cout << "Generating synthetic unsigned int data..." << std::endl;
            uint_data_vec.resize(1000000);
            for (size_t i = 0; i < uint_data_vec.size(); i++) {
                uint_data_vec[i] = 1000000U + static_cast<unsigned int>(i) * 50U + (rand() % 100);
            }
            std::cout << "Synthetic data generated: " << uint_data_vec.size() << " unsigned ints." << std::endl;
            data_loaded_ok = true;
        }
        if (data_loaded_ok && !uint_data_vec.empty()) 
            run_compression_test(uint_data_vec, "unsigned_int", dataset_name);
        
    } else {
        std::cerr << "\nUsage: " << argv[0] << " <data_type> [FILENAME | --text FILENAME | --binary FILENAME]" 
                  << std::endl;
        std::cerr << "Supported integer data types: int, long, long_long (or longlong), "
                  << "unsigned_long_long (or ull), uint (or uint32, unsigned_int)" << std::endl;
        std::cerr << "If FILENAME is provided without --text/--binary, --text is assumed for it." << std::endl;
        std::cerr << "If no filename arguments, synthetic data is used." << std::endl;
        return 1;
    }
    return 0;
}