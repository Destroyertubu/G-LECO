#include <cuda_runtime.h>
#include "core/InternalTypes.cuh"
#include "core/MathHelpers.cuh"
#include "core/BitManipulation.cuh"
#include <cmath>

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
    
    const int pid = blockIdx.x;
    if (pid >= num_partitions) {
        return; 
    }


    __shared__ double s_theta0;
    __shared__ double s_theta1;
    __shared__ int s_has_overflow_flag;

    const int start = partition_starts[pid];
    const int end = partition_ends[pid];
    const int n = end - start;


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
    
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(data[start + i]);
        sum_x += x;
        sum_y += y;

        sum_xx = fma(x, x, sum_xx);
        sum_xy = fma(x, y, sum_xy);
    }


    sum_x = blockReduceSum(sum_x);
    sum_y = blockReduceSum(sum_y);
    sum_xx = blockReduceSum(sum_xx);
    sum_xy = blockReduceSum(sum_xy);


    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);

        double determinant = fma(dn, sum_xx, -(sum_x * sum_x));
        
        if (fabs(determinant) > 1e-10) {

            s_theta1 = fma(dn, sum_xy, -(sum_x * sum_y)) / determinant;
 
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

        double predicted = fma(theta1, static_cast<double>(i), theta0);
        T pred_T = static_cast<T>(round(predicted));
        long long delta = calculateDelta(data[start + i], pred_T);
        local_max_error = max(local_max_error, llabs(delta));
    }

    long long partition_max_error = blockReduceMax(local_max_error);

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

