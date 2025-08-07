#ifndef DECOMPRESSION_FULL_DECOMPRESSION_KERNELS_CUH
#define DECOMPRESSION_FULL_DECOMPRESSION_KERNELS_CUH

#include "api/G-LeCo_Types.cuh"

// Kernel for full-file decompression when deltas are pre-unpacked
template<typename T>
__global__ void decompressFullFile_PreUnpacked(
    const CompressedData<T>* compressed_data,
    T* output_device,
    int total_elements);

// Kernel for full-file decompression, extracting deltas from the bit-packed array on the fly
template<typename T>
__global__ void decompressFullFile_BitPacked(
    const CompressedData<T>* compressed_data,
    T* output_device,
    int total_elements);

// Optimized kernel where one CUDA block is mapped to one partition
template<typename T>
__global__ void decompressFullFile_OnTheFly_Optimized_V2(
    const CompressedData<T>* compressed_data,
    T* output_device,
    int total_elements);

// Optimized kernel for fixed-size partitions that avoids binary search
template<typename T>
__global__ void decompressFullFileFix(
    const CompressedData<T>* __restrict__ compressed_data_on_device,
    T* __restrict__ output_device,
    int total_elements,
    int partition_size);



#include <cuda_runtime.h>
#include "api/G-LeCo_Types.cuh"       
#include "core/InternalTypes.cuh"   
#include "core/MathHelpers.cuh"       
#include "core/BitManipulation.cuh" 
#include <type_traits>              



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



template<typename T>
__global__ void decompressFullFileFix(
    const CompressedData<T>* __restrict__ compressed_data_on_device,
    T* __restrict__ output_device,
    int total_elements,
    int partition_size) {


    struct PartitionMetaCache {
        int32_t model_type;
        int32_t delta_bits;
        double theta0;
        double theta1;
        int64_t bit_offset_base;
    };


    __shared__ PartitionMetaCache s_meta;
    __shared__ int s_cached_partition_idx;

    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;

    if (!compressed_data_on_device || compressed_data_on_device->num_partitions == 0) {
        return;
    }

    if (threadIdx.x == 0) {
        s_cached_partition_idx = -1; 
    }
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


        if (partition_idx != s_cached_partition_idx) {
            if (threadIdx.x == 0) {
                s_meta.model_type = dev_model_types[partition_idx];
                s_meta.delta_bits = dev_delta_bits[partition_idx];
                s_meta.bit_offset_base = dev_delta_array_bit_offsets[partition_idx];

                if (s_meta.model_type != MODEL_DIRECT_COPY) {
                    s_meta.theta0 = dev_model_params[partition_idx * 4];
                    s_meta.theta1 = dev_model_params[partition_idx * 4 + 1];
                }
                s_cached_partition_idx = partition_idx;
            }

            __syncthreads();
        }
        
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


#endif // DECOMPRESSION_FULL_DECOMPRESSION_KERNELS_CUH
