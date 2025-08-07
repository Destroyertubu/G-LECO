#include "core/InternalTypes.cuh"
#include "core/CudaUtils.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include "compression/PartitioningKernels.cuh"

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
        

        int threads_per_block = 256; 


        int grid_size = h_num_partitions;


        size_t shared_mem_size = threads_per_block * sizeof(double);
        shared_mem_size = max(shared_mem_size, threads_per_block * sizeof(long long));


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
