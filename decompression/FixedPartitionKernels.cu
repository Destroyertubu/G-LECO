#include <cuda_runtime.h>
#include "api/G-LeCo_Types.cuh"       // 需要操作 CompressedData<T>
#include "core/InternalTypes.cuh"   // 需要内部元数据结构
#include "core/MathHelpers.cuh"       // 需要 applyDelta, etc.
#include "core/BitManipulation.cuh" // 需要 extractDelta, etc.
#include <type_traits>              // for std::is_signed


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