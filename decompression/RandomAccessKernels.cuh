#ifndef DECOMPRESSION_RANDOM_ACCESS_KERNELS_CUH
#define DECOMPRESSION_RANDOM_ACCESS_KERNELS_CUH

#include "api/G-LeCo_Types.cuh"

// Optimized kernel for random access queries with reduced branching
template<typename T>
__global__ void decompressOptimizedKernel(const CompressedData<T>* compressed_data_on_device,
                                         T* output_device,
                                         const int* positions_device,
                                         int num_queries_val);

// Optimized kernel for direct random access from a serialized data blob on the GPU
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
    int num_queries);

// Kernel for random access queries when deltas have been pre-unpacked
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
    int num_queries);




#include <cuda_runtime.h>
#include "api/G-LeCo_Types.cuh"       
#include "core/InternalTypes.cuh"   
#include "core/MathHelpers.cuh"      
#include "core/BitManipulation.cuh" 
#include <type_traits>             




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


#endif // DECOMPRESSION_RANDOM_ACCESS_KERNELS_CUH
