#include <cuda_runtime.h>
#include "core/InternalTypes.cuh"

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