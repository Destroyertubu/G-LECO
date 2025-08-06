#include <cuda_runtime.h>
#include "core/InternalTypes.cuh"
#include "core/MathHelpers.cuh"
#include "core/BitManipulation.cuh"
#include <cmath>

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