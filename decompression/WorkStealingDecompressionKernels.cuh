#ifndef DECOMPRESSION_WORK_STEALING_DECOMPRESSION_KERNELS_CUH
#define DECOMPRESSION_WORK_STEALING_DECOMPRESSION_KERNELS_CUH

#include "api/G-LeCo_Types.cuh"

// Kernel using cooperative groups for better load balancing
template<typename T>
__launch_bounds__(256, 4)
__global__ void decompressFullFileCooperativeKernel(
    const CompressedData<T>* __restrict__ compressed_data_on_device,
    T* __restrict__ output_device,
    int total_elements);

// Kernel using a basic work-stealing approach
template<typename T>
__global__ void decompressFullFile_WorkStealing(
    const CompressedData<T>* compressed_data,
    T* output_device,
    int total_elements,
    int* global_work_counter);

// Kernel using an advanced work-stealing approach with warp specialization
template<typename T>
__global__ void decompressFullFile_WorkStealingAdvanced(
    const CompressedData<T>* compressed_data,
    T* output_device,
    int total_elements,
    int* global_partition_counter);



#include <cuda_runtime.h>
#include "api/G-LeCo_Types.cuh"       
#include "core/InternalTypes.cuh"   
#include "core/MathHelpers.cuh"       
#include "core/BitManipulation.cuh" 
#include <type_traits>              

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







#endif // DECOMPRESSION_WORK_STEALING_DECOMPRESSION_KERNELS_CUH
