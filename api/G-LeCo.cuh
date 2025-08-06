#include "api/G-LeCo_Types.cuh" // 引入公共数据结构
#include <cuda_runtime.h>       // for cudaStream_t
#include <vector>               // for std::vector
#include <string>               // for std::string

#include "api/G-LeCo.cuh"       // 包含自身类的声明
#include "core/InternalTypes.cuh" // 使用内部数据结构
#include "compression/CpuPartitioner.cuh" // 使用CPU分区器
#include "compression/GpuPartitioner.cuh" // 使用GPU分区器
#include "io/FileUtils.cuh"         // for calculateChecksum

// 概念上，需要包含所有将要调用的内核文件 (在实际构建系统中，这通常通过链接或包含声明头文件完成)
#include "compression/CompressionKernels.cu"
#include "compression/PartitioningKernels.cu"
#include "decompression/FullDecompressionKernels.cu"
#include "decompression/RandomAccessKernels.cu"
#include "decompression/FixedPartitionKernels.cu"
#include "decompression/WorkStealingDecompressionKernels.cu"
#include "decompression/UnpackKernel.cu"
#include "io/SerializationKernels.cu"

#include <iostream>             // for std::cout, std::cerr
#include <algorithm>            // for std::min, std::max
#include <cmath>                // for std::round
#include <map>                  // for performance analysis

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
