#ifndef DECOMPRESSION_UNPACK_KERNEL_CUH
#define DECOMPRESSION_UNPACK_KERNEL_CUH

#include "api/G-LeCo_Types.cuh"

// Kernel to pre-unpack all deltas from the bit-packed format into a plain long long array
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
    int total_values);



#include <cuda_runtime.h>
#include "api/G-LeCo_Types.cuh"       
#include "core/InternalTypes.cuh"   
#include "core/MathHelpers.cuh"       
#include "core/BitManipulation.cuh" 
#include <type_traits>              



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


#endif // DECOMPRESSION_UNPACK_KERNEL_CUH
