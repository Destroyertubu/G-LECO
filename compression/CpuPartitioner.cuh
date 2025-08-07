#include "core/InternalTypes.cuh"
#include "core/MathHelpers.cuh"
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <climits>

// Enhanced Variable-length partitioner class with proper split-merge algorithm - NO CHANGES NEEDED
template<typename T>
class VariableLengthPartitioner {
private:
    std::vector<T> data_host_vec;
    double split_thresh_param;
    int min_part_size_param;
    const double MODEL_SIZE_BYTES = sizeof(PartitionInfo);  // Model overhead per partition
    std::string dataset_name_member;

    // Structure to hold partition candidate information
    struct PartitionCandidate {
        int start_idx;
        int end_idx;
        double theta0;
        double theta1;
        long long max_error;
        int delta_bits;
        double total_cost;
    };

    // Cache for linear regression computations
    struct RegressionCache {
        std::vector<double> prefix_sum_x;
        std::vector<double> prefix_sum_y;
        std::vector<double> prefix_sum_xx;
        std::vector<double> prefix_sum_xy;
        
        void precompute(const std::vector<T>& data) {
            int n = data.size();
            prefix_sum_x.resize(n + 1, 0.0);
            prefix_sum_y.resize(n + 1, 0.0);
            prefix_sum_xx.resize(n + 1, 0.0);
            prefix_sum_xy.resize(n + 1, 0.0);
            
            for (int i = 0; i < n; i++) {
                double x = static_cast<double>(i);
                double y = static_cast<double>(data[i]);
                prefix_sum_x[i + 1] = prefix_sum_x[i] + x;
                prefix_sum_y[i + 1] = prefix_sum_y[i] + y;
                prefix_sum_xx[i + 1] = prefix_sum_xx[i] + x * x;
                prefix_sum_xy[i + 1] = prefix_sum_xy[i] + x * y;
            }
        }
        
        void getRangeSums(int start, int end, double& sum_x, double& sum_y, 
                         double& sum_xx, double& sum_xy) const {
            // Adjust indices for the local coordinate system
            int n = end - start;
            sum_x = n * (n - 1) / 2.0;  // Sum of 0 to n-1
            sum_y = prefix_sum_y[end] - prefix_sum_y[start];
            
            // For xx and xy, we need to adjust for the offset
            sum_xx = 0.0;
            sum_xy = 0.0;
            for (int i = 0; i < n; i++) {
                double local_x = i;
                double global_x = start + i;
                sum_xx += local_x * local_x;
                sum_xy += local_x * (prefix_sum_y[global_x + 1] - prefix_sum_y[global_x]);
            }
        }
    };
    
    RegressionCache cache;

    // Fast linear model fitting using precomputed sums
    void fitLinearModelFast(int start, int end, double& theta0, double& theta1, 
                           long long& max_error, int& delta_bits) {
        int n = end - start;
        if (n <= 0) {
            theta0 = theta1 = 0.0;
            max_error = 0;
            delta_bits = 0;
            return;
        }

        // Get sums from cache
        double sum_x, sum_y, sum_xx, sum_xy;
        cache.getRangeSums(start, end, sum_x, sum_y, sum_xx, sum_xy);

        // Solve normal equations
        double determinant = n * sum_xx - sum_x * sum_x;
        if (std::abs(determinant) > 1e-10) {
            theta1 = (n * sum_xy - sum_x * sum_y) / determinant;
            theta0 = (sum_y - theta1 * sum_x) / n;
        } else {
            theta1 = 0.0;
            theta0 = sum_y / n;
        }

        // Calculate maximum prediction error
        max_error = 0;
        for (int i = 0; i < n; i++) {
            double predicted = theta0 + theta1 * i;
            T pred_T = static_cast<T>(std::round(predicted));
            long long delta = calculateDelta(data_host_vec[start + i], pred_T);
            long long abs_error = std::abs(delta);
            max_error = std::max(max_error, abs_error);
        }

        // Calculate required delta bits
        if (max_error > 0) {
            int bits_for_magnitude = 0;
            unsigned long long temp = static_cast<unsigned long long>(max_error);
            while (temp > 0) {
                bits_for_magnitude++;
                temp >>= 1;
            }
            delta_bits = bits_for_magnitude + 1;  // +1 for sign bit
            delta_bits = std::min(delta_bits, MAX_DELTA_BITS);
        } else {
            delta_bits = 0;
        }
    }

    // Calculate the total cost (in bytes) for a partition
    double calculatePartitionCostFast(int start, int end) {
        double theta0, theta1;
        long long max_error;
        int delta_bits;
        
        fitLinearModelFast(start, end, theta0, theta1, max_error, delta_bits);
        
        // Total cost = model size + delta array size
        double delta_array_bytes = static_cast<double>(end - start) * delta_bits / 8.0;
        return MODEL_SIZE_BYTES + delta_array_bytes;
    }

    // Optimized split phase using dynamic programming concepts
    void splitPhaseOptimized(std::vector<PartitionCandidate>& candidates) {
        candidates.clear();
        if (data_host_vec.empty()) return;

        int n = data_host_vec.size();
        
        // Use a greedy approach with lookahead
        int current_start = 0;
        
        while (current_start < n) {
            int best_end = std::min(current_start + min_part_size_param, n);
            double best_cost_per_element = std::numeric_limits<double>::max();
            
            // Try different partition sizes with exponential stepping
            int step = min_part_size_param;
            int max_size = std::min(n - current_start, min_part_size_param * 32);
            
            for (int size = min_part_size_param; size <= max_size; size += step) {
                int end = std::min(current_start + size, n);
                double cost = calculatePartitionCostFast(current_start, end);
                double cost_per_element = cost / (end - current_start);
                
                if (cost_per_element < best_cost_per_element) {
                    best_cost_per_element = cost_per_element;
                    best_end = end;
                }
                
                // Adaptive stepping: increase step size for larger partitions
                if (size > min_part_size_param * 4) {
                    step = min_part_size_param * 2;
                }
                if (size > min_part_size_param * 16) {
                    step = min_part_size_param * 4;
                }
            }
            
            // Create partition with the best found size
            double theta0, theta1;
            long long max_error;
            int delta_bits;
            fitLinearModelFast(current_start, best_end, theta0, theta1, max_error, delta_bits);
            
            PartitionCandidate candidate;
            candidate.start_idx = current_start;
            candidate.end_idx = best_end;
            candidate.theta0 = theta0;
            candidate.theta1 = theta1;
            candidate.max_error = max_error;
            candidate.delta_bits = delta_bits;
            candidate.total_cost = calculatePartitionCostFast(current_start, best_end);
            
            candidates.push_back(candidate);
            current_start = best_end;
        }
    }

    // Fixed merge phase that maintains partition integrity
    void mergePhaseOptimized(std::vector<PartitionCandidate>& candidates) {
        if (candidates.size() <= 1) return;
        
        bool changed = true;
        int max_iterations = 10;  // Limit iterations to prevent excessive runtime
        int iteration = 0;
        
        while (changed && iteration < max_iterations) {
            changed = false;
            iteration++;
            
            // Try merging adjacent partitions
            for (int i = 0; i < static_cast<int>(candidates.size()) - 1; ) {
                // Calculate individual costs
                double cost1 = candidates[i].total_cost;
                double cost2 = candidates[i + 1].total_cost;
                double combined_cost = cost1 + cost2;
                
                // Calculate merged cost
                double merged_cost = calculatePartitionCostFast(candidates[i].start_idx, 
                                                              candidates[i + 1].end_idx);
                
                // If merging reduces cost, do it
                if (merged_cost < combined_cost) {
                    // Update the first partition with merged data
                    candidates[i].end_idx = candidates[i + 1].end_idx;
                    
                    // Refit the model for the merged partition
                    fitLinearModelFast(candidates[i].start_idx, candidates[i].end_idx,
                                     candidates[i].theta0, candidates[i].theta1,
                                     candidates[i].max_error, candidates[i].delta_bits);
                    
                    candidates[i].total_cost = merged_cost;
                    
                    // Remove the second partition
                    candidates.erase(candidates.begin() + i + 1);
                    
                    changed = true;
                    // Don't increment i, check the same position again
                } else {
                    i++;
                }
            }
        }
    }

    // Validate that partitions cover all indices without gaps
    void validatePartitions(const std::vector<PartitionCandidate>& candidates) {
        if (candidates.empty()) return;
        
        // Check first partition starts at 0
        if (candidates[0].start_idx != 0) {
            std::cerr << "ERROR: First partition doesn't start at 0!" << std::endl;
        }
        
        // Check continuity and final partition
        for (size_t i = 0; i < candidates.size() - 1; i++) {
            if (candidates[i].end_idx != candidates[i + 1].start_idx) {
                std::cerr << "ERROR: Gap between partitions " << i << " and " << i + 1 
                         << " (" << candidates[i].end_idx << " != " 
                         << candidates[i + 1].start_idx << ")" << std::endl;
            }
        }
        
        if (candidates.back().end_idx != static_cast<int>(data_host_vec.size())) {
            std::cerr << "ERROR: Last partition doesn't end at data size!" << std::endl;
        }
    }


private:

    void writePartitionLengthsToFile(const std::vector<PartitionInfo>& partitions) {

        std::string filename = dataset_name_member + "_cpu_var_partition_lengths.txt";
        
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Warning: Could not open file " << filename 
                     << " for writing partition lengths." << std::endl;
            return;
        }
        

        outfile << "# CPU Variable-Length Partition Lengths for dataset: " << dataset_name_member << std::endl;
        outfile << "# Total partitions: " << partitions.size() << std::endl;
        outfile << "# Total elements: " << data_host_vec.size() << std::endl;
        outfile << "# Format: partition_index start_idx end_idx length" << std::endl;
        outfile << "#" << std::endl;
        

        for (size_t i = 0; i < partitions.size(); i++) {
            int length = partitions[i].end_idx - partitions[i].start_idx;
            outfile << i << " " << partitions[i].start_idx << " " 
                   << partitions[i].end_idx << " " << length << std::endl;
        }
        

        int min_length = INT_MAX;
        int max_length = 0;
        long long total_length = 0;
        
        for (const auto& partition : partitions) {
            int length = partition.end_idx - partition.start_idx;
            min_length = std::min(min_length, length);
            max_length = std::max(max_length, length);
            total_length += length;
        }
        
        double avg_length = static_cast<double>(total_length) / partitions.size();
        
        outfile << "#" << std::endl;
        outfile << "# Statistics:" << std::endl;
        outfile << "# Min partition length: " << min_length << std::endl;
        outfile << "# Max partition length: " << max_length << std::endl;
        outfile << "# Average partition length: " << avg_length << std::endl;
        outfile << "# Length ratio (max/min): " << 
                (min_length > 0 ? static_cast<double>(max_length) / min_length : 0) << std::endl;
        
        outfile.close();
        std::cout << "CPU-Var partition lengths written to: " << filename << std::endl;
    }
    
    
public:
        VariableLengthPartitioner(const std::vector<T>& input_data_vec,
                                double threshold_param = SPLIT_THRESHOLD,
                                int min_size_param = MIN_PARTITION_SIZE,
                                const std::string& dataset_name_param = "")
            : data_host_vec(input_data_vec), 
            split_thresh_param(threshold_param), 
            min_part_size_param(min_size_param), 
            dataset_name_member(dataset_name_param) {
            

            this->dataset_name_member = dataset_name_param;
            

            std::cout << "[DEBUG] VariableLengthPartitioner constructor called" << std::endl;
            std::cout << "[DEBUG]   dataset_name_param = '" << dataset_name_param << "'" << std::endl;
            std::cout << "[DEBUG]   this->dataset_name_member = '" << this->dataset_name_member << "'" << std::endl;
        }


        std::vector<PartitionInfo> partition() {
            if (data_host_vec.empty()) {
                return std::vector<PartitionInfo>();
            }
            

            std::cout << "[DEBUG] VariableLengthPartitioner::partition() called" << std::endl;
            std::cout << "[DEBUG] dataset_name_member = '" << dataset_name_member << "'" << std::endl;
            
            // Precompute prefix sums for fast regression
            cache.precompute(data_host_vec);
            
            std::vector<PartitionCandidate> candidates;
            
            // Phase 1: Split - Create initial partitions (optimized)
            splitPhaseOptimized(candidates);
            
            // Validate after split
            validatePartitions(candidates);
            
            // Phase 2: Merge - Combine partitions to optimize total size (optimized)
            mergePhaseOptimized(candidates);
            
            // Validate after merge
            validatePartitions(candidates);
            
            // Convert candidates to PartitionInfo format
            std::vector<PartitionInfo> result;
            for (const auto& candidate : candidates) {
                PartitionInfo info;
                info.start_idx = candidate.start_idx;
                info.end_idx = candidate.end_idx;
                info.model_type = MODEL_LINEAR;
                info.model_params[0] = candidate.theta0;
                info.model_params[1] = candidate.theta1;
                info.model_params[2] = 0.0;
                info.model_params[3] = 0.0;
                info.delta_bits = candidate.delta_bits;
                info.delta_array_bit_offset = 0;  // Will be set later
                info.error_bound = candidate.max_error;
                result.push_back(info);
            }
            
            // Final safety check
            if (!result.empty()) {
                // Ensure partitions are sorted and contiguous
                std::sort(result.begin(), result.end(), 
                        [](const PartitionInfo& a, const PartitionInfo& b) {
                            return a.start_idx < b.start_idx;
                        });
                
                // Fix any remaining gaps (should not happen with the fixes above)
                for (size_t i = 0; i < result.size() - 1; i++) {
                    if (result[i].end_idx != result[i + 1].start_idx) {
                        std::cerr << "WARNING: Fixing gap at partition " << i << std::endl;
                        result[i].end_idx = result[i + 1].start_idx;
                    }
                }
            }
            

            std::cout << "[DEBUG] result.size() = " << result.size() << std::endl;
            std::cout << "[DEBUG] result.empty() = " << result.empty() << std::endl;
            std::cout << "[DEBUG] dataset_name_member.empty() = " << dataset_name_member.empty() << std::endl;
            

            if (!result.empty() && !dataset_name_member.empty()) {
                std::cout << "[DEBUG] Calling writePartitionLengthsToFile..." << std::endl;
                writePartitionLengthsToFile(result);
            } else {
                std::cout << "[DEBUG] NOT calling writePartitionLengthsToFile because:" << std::endl;
                if (result.empty()) {
                    std::cout << "  - result is empty" << std::endl;
                }
                if (dataset_name_member.empty()) {
                    std::cout << "  - dataset_name_member is empty" << std::endl;
                }
            }
            
            return result;
        }
};