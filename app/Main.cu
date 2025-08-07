#include "api/G-LeCo.cuh"           // 使用库的主类
#include "app/BenchmarkUtils.cuh"   // 使用测试工具
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>              // for srand, rand
#include <ctime>                // for time
#include <tuple>                // for test configurations
#include <algorithm>            // for std::replace
#include <map>


// Test Function
template<typename T>
void run_compression_test(const std::vector<T>& data_to_test, const std::string& data_type_string_name, const std::string& dataset_name = "") {
    if (data_to_test.empty()) { 
        std::cout << "\nData is empty for " << data_type_string_name << ". Skipping tests." << std::endl; 
        return; 
    }
    std::cout << "\n=== Testing with data type: " << data_type_string_name << " ===" << std::endl;
    std::cout << "Data type size: " << sizeof(T) << " bytes, Data size: " << data_to_test.size() 
             << " elements" << std::endl;
    LeCoGPU<T> leco_instance;

    // Test three configurations: fixed, variable-CPU, variable-GPU
    std::vector<std::tuple<bool, bool, std::string>> test_configs = {
        {false, false, "Fixed-length"},
        {true, false, "Variable-length (CPU)"},
        {true, true, "Variable-length (GPU Work-Stealing)"}
    };
    
    for (const auto& config : test_configs) {
        bool current_use_variable_part = std::get<0>(config);
        bool use_gpu_partitioning = std::get<1>(config);
        std::string config_name = std::get<2>(config);
        
        std::cout << "\n--- " << config_name << " Partitioning ---" << std::endl;
        
        // Declare all variables that will be used across different sections
        long long current_compressed_size = 0;
        CompressedData<T>* current_compressed_ptr = nullptr;
        double time_to_decompress = 0.0;
        DirectAccessHandle<T>* current_da_handle = nullptr;
        double time_gpu_da = 0.0;

        // Compression
        auto time_to_compress = benchmark([&]() {
            if (current_compressed_ptr) { 
                leco_instance.cleanup(current_compressed_ptr); 
                current_compressed_ptr = nullptr; 
            }
            // 构建带有配置信息的数据集名称
            std::string full_dataset_name = dataset_name.empty() ? 
                data_type_string_name : dataset_name;
            
            // 添加调试输出
            std::cout << "[DEBUG] run_compression_test: dataset_name = '" << dataset_name << "'" << std::endl;
            std::cout << "[DEBUG] run_compression_test: full_dataset_name = '" << full_dataset_name << "'" << std::endl;
            std::cout << "[DEBUG] run_compression_test: use_variable_part = " << current_use_variable_part << std::endl;
            std::cout << "[DEBUG] run_compression_test: use_gpu_partitioning = " << use_gpu_partitioning << std::endl;
            
            current_compressed_ptr = leco_instance.compress(data_to_test, 
                                                        current_use_variable_part, 
                                                        &current_compressed_size, 
                                                        use_gpu_partitioning,
                                                        full_dataset_name);
        }, 1);
        std::cout << "Compression time: " << time_to_compress << " ms\n";
        if (!current_compressed_ptr) { 
            std::cerr << "Compression FAILED." << std::endl; 
            continue; 
        }
        
        long long uncomp_size = (long long)data_to_test.size() * sizeof(T);
        // MODIFIED: Changed compression ratio calculation from compressed/uncompressed to uncompressed/compressed
        double comp_ratio = (current_compressed_size > 0) ? static_cast<double>(uncomp_size) / current_compressed_size : 0.0;
        std::cout << "Uncompressed size: " << uncomp_size << " bytes\n";
        std::cout << "Compressed size:   " << current_compressed_size << " bytes\n";
        std::cout << "Compression Ratio: " << comp_ratio << "\n";
        std::cout << "Number of partitions: " << current_compressed_ptr->num_partitions << "\n";
        
        // If this is GPU variable partitioning, compare with CPU variable partitioning
        if (use_gpu_partitioning) {
            std::cout << "\n  [Partitioning Performance Comparison]" << std::endl;
            CompressedData<T>* cpu_var_compressed = nullptr;
            long long cpu_var_size = 0;
            
            auto cpu_var_time = benchmark([&]() {
                if (cpu_var_compressed) {
                    leco_instance.cleanup(cpu_var_compressed);
                    cpu_var_compressed = nullptr;
                }
                // 修正：这里也要传递 dataset_name
                std::string full_dataset_name = dataset_name.empty() ? 
                    data_type_string_name : dataset_name;
                
                cpu_var_compressed = leco_instance.compress(data_to_test, true, &cpu_var_size, false, 
                                                        full_dataset_name);  // 添加数据集名称参数
            }, 1);
            
            std::cout << "  CPU Variable Partitioning time: " << cpu_var_time << " ms" << std::endl;
            std::cout << "  GPU Work-Stealing speedup: " << (cpu_var_time / time_to_compress) << "x" << std::endl;
            
            if (cpu_var_compressed) {
                std::cout << "  CPU partitions: " << cpu_var_compressed->num_partitions << std::endl;
                std::cout << "  GPU partitions: " << current_compressed_ptr->num_partitions << std::endl;
                std::cout << "  Partition count difference: " 
                         << std::abs(cpu_var_compressed->num_partitions - current_compressed_ptr->num_partitions) 
                         << std::endl;
                leco_instance.cleanup(cpu_var_compressed);
            }
        }
        
        // Random access decompression test
        std::vector<int> random_positions(std::min((size_t)100000, data_to_test.size()));
        if (!data_to_test.empty()) { 
            for (size_t i = 0; i < random_positions.size(); i++) 
                random_positions[i] = rand() % data_to_test.size(); 
        }
        std::vector<T> decompressed_output_data;

        if (!random_positions.empty()) {
            time_to_decompress = benchmark([&]() { 
                leco_instance.decompress(current_compressed_ptr, random_positions, decompressed_output_data); 
            }, 100);
            std::cout << "Random access time for " << random_positions.size() << " queries: " 
                     << time_to_decompress << " ms\n";
            if (time_to_decompress > 0) 
                std::cout << "Throughput: " << (random_positions.size() / time_to_decompress * 1000) 
                         << " queries/sec\n";

            bool is_data_correct = true; 
            int num_mismatches = 0;
            for (size_t i = 0; i < random_positions.size(); i++) {
                if (decompressed_output_data[i] != data_to_test[random_positions[i]]) {
                    is_data_correct = false; 
                    num_mismatches++;
                    if (num_mismatches <= 5) 
                        std::cerr << "Mismatch at original index " << random_positions[i] 
                                 << ": Original " << data_to_test[random_positions[i]] 
                                 << ", Decompressed " << decompressed_output_data[i] << "\n";
                }
            }
            std::cout << "Correctness (vs decompress): " << (is_data_correct ? "PASSED" : "FAILED") << "\n";
            if (num_mismatches > 0) 
                std::cout << "Total mismatches (vs decompress): " << num_mismatches << " out of " 
                         << random_positions.size() << " queries\n";

            if (config_name == "Fixed-length") {

                leco_instance.analyzeRandomAccessPerformance(current_compressed_ptr, 
                                                 random_positions, TILE_SIZE);
                std::cout << "\n--- Optimized Fixed-Partition Random Access Test ---" << std::endl;
                std::vector<T> fixed_ra_output;
                
                // 声明所需的时间变量
                double time_fixed_ra = 0.0;

                // 测试 Fixed-only 优化
                time_fixed_ra = benchmark([&]() {
                    leco_instance.randomAccessFixed(current_compressed_ptr, random_positions, fixed_ra_output, TILE_SIZE);
                }, 100);

                std::cout << "Optimized fixed random access: " << time_fixed_ra << " ms" << std::endl;
                if (time_fixed_ra > 0) {
                    double throughput_qps = static_cast<double>(random_positions.size()) / (time_fixed_ra / 1000.0);
                    std::cout << "  -> Throughput: " << throughput_qps << " queries/sec" << std::endl;
                }

                // 计算并打印性能提升
                if (time_to_decompress > 0 && time_fixed_ra > 0) {
                    std::cout << "  -> Speedup vs. generic random access: " << 
                            (time_to_decompress / time_fixed_ra) << "x" << std::endl;
                }

                // 验证 Fixed-only 的正确性
                bool fixed_ra_correct = true;
                int fixed_ra_mismatches = 0;
                for (size_t i = 0; i < random_positions.size(); i++) {
                    if (fixed_ra_output[i] != data_to_test[random_positions[i]]) {
                        fixed_ra_correct = false;
                        fixed_ra_mismatches++;
                        if (fixed_ra_mismatches <= 5) {
                            std::cerr << "Mismatch (Fixed RA) at original index " << random_positions[i]
                                    << ": Original " << data_to_test[random_positions[i]]
                                    << ", Decompressed " << fixed_ra_output[i] << "\n";
                        }
                    }
                }
                std::cout << "Correctness (Optimized Fixed RA): " << (fixed_ra_correct ? "PASSED" : "FAILED");
                if (fixed_ra_mismatches > 0) {
                    std::cout << " (" << fixed_ra_mismatches << " mismatches)";
                }
                std::cout << std::endl;

                // 测试 Fixed + Pre-unpacked 优化
                std::cout << "\n--- Fixed-Partition Pre-Unpacked Random Access Test ---" << std::endl;
                
                // 序列化并反序列化以启用预解包
                SerializedData* temp_fixed_serialized = leco_instance.serializeGPU(current_compressed_ptr);
                if (temp_fixed_serialized && temp_fixed_serialized->data) {
                    // 反序列化时启用预解包
                    CompressedData<T>* fixed_pre_unpacked = leco_instance.deserializeGPU(temp_fixed_serialized, true);
                    
                    if (fixed_pre_unpacked && fixed_pre_unpacked->d_plain_deltas) {
                        std::cout << "Successfully created pre-unpacked data for fixed partitions." << std::endl;
                        
                        // 测试优化的内核
                        std::vector<T> fixed_pre_unpacked_output;
                        double time_fixed_pre_unpacked = benchmark([&]() {
                            leco_instance.randomAccessFixedPreUnpacked(fixed_pre_unpacked, 
                                                                    random_positions, 
                                                                    fixed_pre_unpacked_output, 
                                                                    TILE_SIZE);
                        }, 100);
                        
                        std::cout << "Fixed + Pre-unpacked random access: " << time_fixed_pre_unpacked << " ms" << std::endl;
                        
                        if (time_fixed_pre_unpacked > 0) {
                            double throughput_qps = static_cast<double>(random_positions.size()) / 
                                                (time_fixed_pre_unpacked / 1000.0);
                            double throughput_mbs = (static_cast<double>(random_positions.size()) * sizeof(T)) / 
                                                (time_fixed_pre_unpacked / 1000.0) / (1024.0 * 1024.0);
                            std::cout << "  -> Throughput: " << throughput_qps << " queries/sec (" 
                                    << throughput_mbs << " MB/s)" << std::endl;
                        }
                        
                        // 与其他方法比较
                        if (time_to_decompress > 0 && time_fixed_pre_unpacked > 0) {
                            std::cout << "  -> Speedup vs. generic random access: " << 
                                    (time_to_decompress / time_fixed_pre_unpacked) << "x" << std::endl;
                        }
                        
                        if (time_fixed_ra > 0 && time_fixed_pre_unpacked > 0) {
                            std::cout << "  -> Speedup vs. fixed-only optimization: " << 
                                    (time_fixed_ra / time_fixed_pre_unpacked) << "x" << std::endl;
                        }
                        
                        // 验证正确性
                        bool fixed_pre_unpacked_correct = true;
                        int fixed_pre_unpacked_mismatches = 0;
                        for (size_t i = 0; i < random_positions.size(); i++) {
                            if (fixed_pre_unpacked_output[i] != data_to_test[random_positions[i]]) {
                                fixed_pre_unpacked_correct = false;
                                fixed_pre_unpacked_mismatches++;
                                if (fixed_pre_unpacked_mismatches <= 5) {
                                    std::cerr << "Mismatch (Fixed+Pre-unpacked) at index " << random_positions[i]
                                            << ": Expected " << data_to_test[random_positions[i]]
                                            << ", Got " << fixed_pre_unpacked_output[i] << "\n";
                                }
                            }
                        }
                        
                        std::cout << "Correctness (Fixed + Pre-unpacked): " << 
                                (fixed_pre_unpacked_correct ? "PASSED" : "FAILED");
                        if (fixed_pre_unpacked_mismatches > 0) {
                            std::cout << " (" << fixed_pre_unpacked_mismatches << " mismatches)";
                        }
                        std::cout << std::endl;
                        
                        // 内存分析
                        std::cout << "\nMemory overhead analysis:" << std::endl;
                        long long bitpacked_size = (current_compressed_size - 
                            (current_compressed_ptr->num_partitions * sizeof(PartitionInfo)));
                        long long preunpacked_size = (long long)fixed_pre_unpacked->total_values * sizeof(long long);
                        std::cout << "  -> Bit-packed delta size: ~" << bitpacked_size << " bytes" << std::endl;
                        std::cout << "  -> Pre-unpacked delta size: " << preunpacked_size << " bytes" << std::endl;
                        std::cout << "  -> Memory overhead factor: " << 
                                ((double)preunpacked_size / bitpacked_size) << "x" << std::endl;
                        
                        // 所有固定分区方法的总结
                        std::cout << "\n--- Fixed-Partition Random Access Summary ---" << std::endl;
                        std::cout << "1. Generic random access: " << time_to_decompress << " ms" << std::endl;
                        std::cout << "2. Fixed-only optimization: " << time_fixed_ra << " ms" << std::endl;
                        std::cout << "3. Fixed + Pre-unpacked: " << time_fixed_pre_unpacked << " ms" << std::endl;
                        
                        if (time_to_decompress > 0 && time_fixed_pre_unpacked > 0) {
                            std::cout << "Total speedup achieved: " << 
                                    (time_to_decompress / time_fixed_pre_unpacked) << "x" << std::endl;
                        }
                        
                        // 清理
                        leco_instance.cleanup(fixed_pre_unpacked);
                    } else {
                        std::cout << "Failed to create pre-unpacked data for fixed partitions." << std::endl;
                    }
                    
                    delete temp_fixed_serialized;
                }
            }
        }

        // ========== NEW: Pre-Unpacked Random Access Test ==========
        std::cout << "\n--- Pre-Unpacked Random Access Test ---" << std::endl;
        
        // Serialize current compressed data
        SerializedData* temp_serialized = leco_instance.serializeGPU(current_compressed_ptr);
        if (temp_serialized && temp_serialized->data) {
            // Deserialize with pre-unpacking enabled
            CompressedData<T>* pre_unpacked_data = leco_instance.deserializeGPU(temp_serialized, true);
            
            if (pre_unpacked_data && pre_unpacked_data->d_plain_deltas) {
                std::cout << "Successfully created pre-unpacked data structure." << std::endl;
                
                // Test standard random access with pre-unpacked data
                std::vector<T> standard_ra_output;
                auto time_standard_ra = benchmark([&]() {
                    leco_instance.decompress(pre_unpacked_data, random_positions, standard_ra_output);
                }, 100);
                
                // Test optimized pre-unpacked random access
                std::vector<T> pre_unpacked_ra_output;
                auto time_pre_unpacked_ra = benchmark([&]() {
                    leco_instance.randomAccessPreUnpacked(pre_unpacked_data, random_positions, pre_unpacked_ra_output);
                }, 100);
                
                std::cout << "Standard random access (pre-unpacked data): " << time_standard_ra << " ms" << std::endl;
                // *** ADDED THIS BLOCK ***
                if (time_standard_ra > 0) {
                    double throughput_qps = static_cast<double>(random_positions.size()) / (time_standard_ra / 1000.0);
                    double throughput_mbs = (static_cast<double>(random_positions.size()) * sizeof(T)) / (time_standard_ra / 1000.0) / (1024.0 * 1024.0);
                    std::cout << "  -> Throughput: " << throughput_qps << " queries/sec (" << throughput_mbs << " MB/s)" << std::endl;
                }
                
                std::cout << "Optimized pre-unpacked random access: " << time_pre_unpacked_ra << " ms" << std::endl;
                // *** ADDED THIS BLOCK ***
                if (time_pre_unpacked_ra > 0) {
                    double throughput_qps = static_cast<double>(random_positions.size()) / (time_pre_unpacked_ra / 1000.0);
                    double throughput_mbs = (static_cast<double>(random_positions.size()) * sizeof(T)) / (time_pre_unpacked_ra / 1000.0) / (1024.0 * 1024.0);
                     std::cout << "  -> Throughput: " << throughput_qps << " queries/sec (" << throughput_mbs << " MB/s)" << std::endl;
                }

                if (time_standard_ra > 0 && time_pre_unpacked_ra > 0) {
                    std::cout << "Pre-unpacked optimization speedup: " << 
                              (time_standard_ra / time_pre_unpacked_ra) << "x" << std::endl;
                }
                
                if (time_to_decompress > 0 && time_pre_unpacked_ra > 0) {
                    std::cout << "Speedup vs bit-packed random access: " << 
                              (time_to_decompress / time_pre_unpacked_ra) << "x" << std::endl;
                }
                
                // Verify correctness
                bool pre_unpacked_correct = true;
                int pre_unpacked_mismatches = 0;
                for (size_t i = 0; i < random_positions.size(); i++) {
                    if (pre_unpacked_ra_output[i] != data_to_test[random_positions[i]]) {
                        pre_unpacked_correct = false;
                        pre_unpacked_mismatches++;
                        if (pre_unpacked_mismatches <= 5) {
                            std::cerr << "Pre-unpacked mismatch at index " << random_positions[i] 
                                     << ": Expected " << data_to_test[random_positions[i]] 
                                     << ", Got " << pre_unpacked_ra_output[i] << "\n";
                        }
                    }
                }
                
                std::cout << "Pre-unpacked random access correctness: " << 
                          (pre_unpacked_correct ? "PASSED" : "FAILED");
                if (pre_unpacked_mismatches > 0) {
                    std::cout << " (" << pre_unpacked_mismatches << " mismatches out of " 
                             << random_positions.size() << " queries)";
                }
                std::cout << std::endl;
                
                // Memory overhead analysis
                long long bitpacked_delta_size = (current_compressed_size - 
                    (current_compressed_ptr->num_partitions * sizeof(PartitionInfo)));
                long long preunpacked_delta_size = 
                    (long long)pre_unpacked_data->total_values * sizeof(long long);
                
                std::cout << "Memory overhead for pre-unpacking: " << 
                          ((double)preunpacked_delta_size / bitpacked_delta_size) << "x" << std::endl;
                
                leco_instance.cleanup(pre_unpacked_data);
            } else {
                std::cout << "Failed to create pre-unpacked data structure." << std::endl;
            }
            
            delete temp_serialized;
        }
        // ========== END: Pre-Unpacked Random Access Test ==========

        // Full-file decompression test
        if (!current_use_variable_part) {
            // Fixed-length case: perform comprehensive testing
            std::cout << "\n--- Full-File Decompression (Fixed Partitions) ---" << std::endl;

            // Test with original compressed data (bit-packed deltas)
            std::cout << "  [Bit-Packed Mode] Testing with original compressed data..." << std::endl;
            std::vector<T> full_decompressed_bitpacked;
            auto time_bitpacked = benchmark([&]() {
                leco_instance.decompressFullFile(current_compressed_ptr, full_decompressed_bitpacked);
            }, 10);
            double throughput_bitpacked = (data_to_test.size() * sizeof(T)) / (time_bitpacked / 1000.0) / (1024.0 * 1024.0);
            std::cout << "    -> Time: " << time_bitpacked << " ms, Throughput: " << throughput_bitpacked << " MB/s" << std::endl;
            
            bool correct_bitpacked = (full_decompressed_bitpacked.size() == data_to_test.size() && 
                                    std::equal(full_decompressed_bitpacked.begin(), full_decompressed_bitpacked.end(), data_to_test.begin()));
            std::cout << "    -> Correctness: " << (correct_bitpacked ? "PASSED" : "FAILED") << std::endl;

            // Test the optimized fixed-partition kernel (no binary search)
            std::cout << "  [Fixed-Optimized] Testing decompressFullFile_Fix (no binary search)..." << std::endl;
            std::vector<T> full_decompressed_fix;
            auto time_fix = benchmark([&]() {
                leco_instance.decompressFullFile_Fix(current_compressed_ptr, full_decompressed_fix, TILE_SIZE);
            }, 10);
            double throughput_fix = (data_to_test.size() * sizeof(T)) / (time_fix / 1000.0) / (1024.0 * 1024.0);
            std::cout << "    -> Time: " << time_fix << " ms, Throughput: " << throughput_fix << " MB/s" << std::endl;
            
            bool correct_fix = (full_decompressed_fix.size() == data_to_test.size() &&
                              std::equal(full_decompressed_fix.begin(), full_decompressed_fix.end(), data_to_test.begin()));
            std::cout << "    -> Correctness: " << (correct_fix ? "PASSED" : "FAILED") << std::endl;

            std::cout << "\n  [Fixed-Partition Analysis]" << std::endl;
            if (time_fix > 0 && time_bitpacked > 0) {
                double speedup = time_bitpacked / time_fix;
                std::cout << "    -> Speedup of 'Fix' over standard: " << speedup << "x" << std::endl;
            }
        } else {
            // Variable-length case
            std::cout << "\n--- Full-File Decompression (" << config_name << ") ---" << std::endl;
            std::vector<T> full_decompressed_data;
            auto time_full_decompress = benchmark([&]() {
                leco_instance.decompressFullFile(current_compressed_ptr, full_decompressed_data);
            }, 10);
            double throughput = (data_to_test.size() * sizeof(T)) / (time_full_decompress / 1000.0) / (1024.0 * 1024.0);
            std::cout << "  -> Time: " << time_full_decompress << " ms, Throughput: " << throughput << " MB/s" << std::endl;
            
            bool correct = (full_decompressed_data.size() == data_to_test.size() &&
                            std::equal(full_decompressed_data.begin(), full_decompressed_data.end(), data_to_test.begin()));
            std::cout << "  -> Correctness: " << (correct ? "PASSED" : "FAILED") << std::endl;
        }
        
        // Serialization/Deserialization
        std::cout << "\n--- Serialization/Deserialization ---" << std::endl;
        SerializedData* current_serialized_obj = nullptr;
        auto time_to_serialize = benchmark([&]() { 
            if(current_serialized_obj) delete current_serialized_obj; 
            current_serialized_obj = leco_instance.serialize(current_compressed_ptr); 
        }, 10);
        std::cout << "CPU Serialization time: " << time_to_serialize << " ms\n";
        if (!current_serialized_obj || !current_serialized_obj->data || current_serialized_obj->size == 0) {
            leco_instance.cleanup(current_compressed_ptr); 
            if(current_serialized_obj) delete current_serialized_obj; 
            std::cerr << "Serialization FAILED." << std::endl; 
            continue;
        }
        std::cout << "Serialized size: " << current_serialized_obj->size << " bytes\n";
        
        // GPU Serialization test
        SerializedData* gpu_serialized_obj = nullptr;
        auto time_to_serialize_gpu = benchmark([&]() { 
            if(gpu_serialized_obj) delete gpu_serialized_obj; 
            gpu_serialized_obj = leco_instance.serializeGPU(current_compressed_ptr); 
        }, 10);
        std::cout << "GPU Serialization time: " << time_to_serialize_gpu << " ms\n";
        
        if (!gpu_serialized_obj || !gpu_serialized_obj->data || gpu_serialized_obj->size == 0) {
            std::cerr << "GPU Serialization FAILED." << std::endl; 
        } else {
            std::cout << "GPU Serialized size: " << gpu_serialized_obj->size << " bytes\n";
            std::cout << "GPU Serialization speedup over CPU: " << (time_to_serialize / time_to_serialize_gpu) << "x\n";
            
            // Use GPU serialized data for further tests
            delete current_serialized_obj;
            current_serialized_obj = gpu_serialized_obj;
            gpu_serialized_obj = nullptr;
        }
        
        leco_instance.cleanup(current_compressed_ptr); 
        current_compressed_ptr = nullptr; 
        
        // CPU Deserialization
        CompressedData<T>* current_deserialized_data = nullptr;
        auto time_to_deserialize = benchmark([&]() { 
            if(current_deserialized_data) leco_instance.cleanup(current_deserialized_data); 
            current_deserialized_data = leco_instance.deserialize(current_serialized_obj); 
        }, 10);
        std::cout << "CPU Deserialization time: " << time_to_deserialize << " ms\n";
        
        // GPU Deserialization
        CompressedData<T>* gpu_deserialized_data = nullptr;
        auto time_to_deserialize_gpu = benchmark([&]() { 
            if(gpu_deserialized_data) leco_instance.cleanup(gpu_deserialized_data); 
            gpu_deserialized_data = leco_instance.deserializeGPU(current_serialized_obj, false); 
        }, 10);
        std::cout << "GPU Deserialization time: " << time_to_deserialize_gpu << " ms\n";
        std::cout << "GPU Deserialization speedup over CPU: " << (time_to_deserialize / time_to_deserialize_gpu) << "x\n";
        
        if (!current_deserialized_data || !gpu_deserialized_data) { 
            if (current_deserialized_data) leco_instance.cleanup(current_deserialized_data);
            if (gpu_deserialized_data) leco_instance.cleanup(gpu_deserialized_data);
            delete current_serialized_obj; 
            std::cerr << "Deserialization FAILED." << std::endl; 
            continue; 
        }
        
        // Use GPU deserialized data for consistency
        leco_instance.cleanup(current_deserialized_data);
        current_deserialized_data = gpu_deserialized_data;
        gpu_deserialized_data = nullptr;

        // Comprehensive Decompression Mode Testing with Work-Stealing
        std::cout << "\n--- Comprehensive Decompression Mode Testing (Including Work-Stealing) ---" << std::endl;

        if (current_serialized_obj) {
            std::cout << "Testing different decompression strategies:" << std::endl;
            
            // 1. Deserialize in standard mode (bit-packed)
            CompressedData<T>* standard_mode_data = nullptr;
            auto time_deser_standard = benchmark([&]() {
                if(standard_mode_data) leco_instance.cleanup(standard_mode_data);
                standard_mode_data = leco_instance.deserializeGPU(current_serialized_obj, false);
            }, 5);
            
            // 2. Deserialize in high-throughput mode (pre-unpacked)
            CompressedData<T>* high_throughput_data = nullptr;
            auto time_deser_high_throughput = benchmark([&]() {
                if(high_throughput_data) leco_instance.cleanup(high_throughput_data);
                high_throughput_data = leco_instance.deserializeGPU(current_serialized_obj, true);
            }, 5);
            
            std::cout << "\n1. Deserialization Performance:" << std::endl;
            std::cout << "   Standard mode (bit-packed): " << time_deser_standard << " ms" << std::endl;
            std::cout << "   High-throughput mode (pre-unpacked): " << time_deser_high_throughput << " ms" << std::endl;
            std::cout << "   Pre-unpacking overhead: " << 
                    (time_deser_high_throughput / time_deser_standard) << "x" << std::endl;
            
            if (standard_mode_data && high_throughput_data) {
                std::cout << "\n2. Full-File Decompression Performance (Including Work-Stealing):" << std::endl;
                
                // Test all decompression methods
                std::vector<T> decomp_bitpacked;
                std::vector<T> decomp_preunpacked;
                std::vector<T> decomp_auto_standard;
                std::vector<T> decomp_auto_ht;
                std::vector<T> decomp_optimized_v2;
                std::vector<T> decomp_work_stealing;
                std::vector<T> decomp_work_stealing_advanced;
                
                // 2.1 Bit-packed kernel with standard data
                auto time_bitpacked = benchmark([&]() {
                    leco_instance.decompressFullFile_BitPacked(standard_mode_data, decomp_bitpacked);
                }, 10);
                double throughput_bitpacked = (data_to_test.size() * sizeof(T)) / 
                                            (time_bitpacked / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.1 Bit-packed kernel (standard data): " 
                        << time_bitpacked << " ms (" << throughput_bitpacked << " MB/s)" << std::endl;
                
                // 2.2 Pre-unpacked kernel with high-throughput data
                auto time_preunpacked = benchmark([&]() {
                    leco_instance.decompressFullFile_PreUnpacked(high_throughput_data, decomp_preunpacked);
                }, 10);
                double throughput_preunpacked = (data_to_test.size() * sizeof(T)) / 
                                            (time_preunpacked / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.2 Pre-unpacked kernel (HT data): " 
                        << time_preunpacked << " ms (" << throughput_preunpacked << " MB/s)" << std::endl;
                
                // 2.3 Auto-selection with standard data
                auto time_auto_standard = benchmark([&]() {
                    leco_instance.decompressFullFile(standard_mode_data, decomp_auto_standard);
                }, 10);
                double throughput_auto_standard = (data_to_test.size() * sizeof(T)) / 
                                                (time_auto_standard / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.3 Auto-selection (standard data): " 
                        << time_auto_standard << " ms (" << throughput_auto_standard << " MB/s)" << std::endl;
                
                // 2.4 Auto-selection with HT data
                auto time_auto_ht = benchmark([&]() {
                    leco_instance.decompressFullFile(high_throughput_data, decomp_auto_ht);
                }, 10);
                double throughput_auto_ht = (data_to_test.size() * sizeof(T)) / 
                                        (time_auto_ht / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.4 Auto-selection (HT data): " 
                        << time_auto_ht << " ms (" << throughput_auto_ht << " MB/s)" << std::endl;
                
                // 2.5 Optimized V2 kernel
                auto time_optimized_v2 = benchmark([&]() {
                    leco_instance.decompressFullFile_OnTheFly_Optimized_V2(standard_mode_data, decomp_optimized_v2);
                }, 10);
                double throughput_optimized_v2 = (data_to_test.size() * sizeof(T)) / 
                                                (time_optimized_v2 / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.5 Optimized V2 kernel (standard data): " 
                        << time_optimized_v2 << " ms (" << throughput_optimized_v2 << " MB/s)" << std::endl;
                
                // 2.6 Work-stealing kernel (standard data)
                auto time_work_stealing = benchmark([&]() {
                    leco_instance.decompressFullFile_WorkStealing(standard_mode_data, decomp_work_stealing);
                }, 10);
                double throughput_work_stealing = (data_to_test.size() * sizeof(T)) / 
                                                (time_work_stealing / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.6 Work-stealing kernel (standard data): " 
                        << time_work_stealing << " ms (" << throughput_work_stealing << " MB/s)" << std::endl;
                
                // 2.7 Work-stealing advanced kernel (HT data)
                auto time_work_stealing_advanced = benchmark([&]() {
                    leco_instance.decompressFullFile_WorkStealingAdvanced(high_throughput_data, decomp_work_stealing_advanced);
                }, 10);
                double throughput_work_stealing_advanced = (data_to_test.size() * sizeof(T)) / 
                                                        (time_work_stealing_advanced / 1000.0) / (1024.0 * 1024.0);
                std::cout << "   2.7 Work-stealing advanced (HT data): " 
                        << time_work_stealing_advanced << " ms (" << throughput_work_stealing_advanced << " MB/s)" << std::endl;
                
                // 2.8 Fixed partition optimization (if applicable)
                if (!current_use_variable_part) {
                    std::vector<T> decomp_fix_standard;
                    std::vector<T> decomp_fix_ht;
                    
                    auto time_fix_standard = benchmark([&]() {
                        leco_instance.decompressFullFile_Fix(standard_mode_data, decomp_fix_standard, TILE_SIZE);
                    }, 10);
                    double throughput_fix_standard = (data_to_test.size() * sizeof(T)) / 
                                                (time_fix_standard / 1000.0) / (1024.0 * 1024.0);
                    std::cout << "   2.8 Fixed optimization (standard data): " 
                            << time_fix_standard << " ms (" << throughput_fix_standard << " MB/s)" << std::endl;
                    
                    auto time_fix_ht = benchmark([&]() {
                        leco_instance.decompressFullFile_Fix(high_throughput_data, decomp_fix_ht, TILE_SIZE);
                    }, 10);
                    double throughput_fix_ht = (data_to_test.size() * sizeof(T)) / 
                                            (time_fix_ht / 1000.0) / (1024.0 * 1024.0);
                    std::cout << "   2.9 Fixed optimization (HT data): " 
                            << time_fix_ht << " ms (" << throughput_fix_ht << " MB/s)" << std::endl;
                }
                
                // 3. Performance Analysis
                std::cout << "\n3. Performance Analysis:" << std::endl;
                if (time_bitpacked > 0) {
                    std::cout << "   Pre-unpacking speedup: " << 
                            (time_bitpacked / time_preunpacked) << "x" << std::endl;
                    std::cout << "   Work-stealing speedup over standard: " << 
                            (time_bitpacked / time_work_stealing) << "x" << std::endl;
                    std::cout << "   Work-stealing advanced speedup: " << 
                            (time_bitpacked / time_work_stealing_advanced) << "x" << std::endl;
                    
                    // Find best performance
                    double best_time = std::min({time_preunpacked, time_optimized_v2, 
                                               time_work_stealing, time_work_stealing_advanced});
                    std::cout << "   Best overall speedup: " << 
                            (time_bitpacked / best_time) << "x" << std::endl;
                }
                
                // 4. Work-Stealing Efficiency Analysis
                std::cout << "\n4. Work-Stealing Efficiency Analysis:" << std::endl;
                if (standard_mode_data->num_partitions > 0) {
                    std::cout << "   Number of partitions: " << standard_mode_data->num_partitions << std::endl;
                    std::cout << "   Average partition size: " << 
                            (data_to_test.size() / standard_mode_data->num_partitions) << " elements" << std::endl;
                    
                    // Calculate load imbalance metric
                    std::vector<int> partition_sizes;
                    std::vector<int32_t> h_starts(standard_mode_data->num_partitions);
                    std::vector<int32_t> h_ends(standard_mode_data->num_partitions);
                    CUDA_CHECK(cudaMemcpy(h_starts.data(), standard_mode_data->d_start_indices,
                                        standard_mode_data->num_partitions * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost));
                    CUDA_CHECK(cudaMemcpy(h_ends.data(), standard_mode_data->d_end_indices,
                                        standard_mode_data->num_partitions * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost));
                    
                    int min_size = INT_MAX, max_size = 0;
                    for (int i = 0; i < standard_mode_data->num_partitions; i++) {
                        int size = h_ends[i] - h_starts[i];
                        min_size = std::min(min_size, size);
                        max_size = std::max(max_size, size);
                    }
                    
                    std::cout << "   Min partition size: " << min_size << " elements" << std::endl;
                    std::cout << "   Max partition size: " << max_size << " elements" << std::endl;
                    std::cout << "   Load imbalance ratio: " << 
                            (max_size > 0 ? (double)max_size / min_size : 0) << std::endl;
                    
                    // Work-stealing efficiency
                    if (time_optimized_v2 > 0 && time_work_stealing > 0) {
                        double efficiency = (time_optimized_v2 - time_work_stealing) / time_optimized_v2 * 100;
                        std::cout << "   Work-stealing efficiency gain: " << efficiency << "%" << std::endl;
                    }
                }
                
                // 5. Correctness Verification
                std::cout << "\n5. Correctness Verification:" << std::endl;
                auto verify_correctness = [&](const std::vector<T>& result, const std::string& method) {
                    bool correct = (result.size() == data_to_test.size()) &&
                                std::equal(result.begin(), result.end(), data_to_test.begin());
                    std::cout << "   " << method << ": " << (correct ? "PASSED" : "FAILED");
                    
                    if (!correct && result.size() == data_to_test.size()) {
                        int mismatches = 0;
                        for (size_t i = 0; i < result.size(); ++i) {
                            if (result[i] != data_to_test[i]) {
                                mismatches++;
                                if (mismatches <= 3) {
                                    std::cout << "\n     Mismatch at " << i << ": expected " 
                                            << data_to_test[i] << ", got " << result[i];
                                }
                            }
                        }
                        std::cout << "\n     Total mismatches: " << mismatches;
                    }
                    std::cout << std::endl;
                };
                
                verify_correctness(decomp_bitpacked, "Bit-packed kernel");
                verify_correctness(decomp_preunpacked, "Pre-unpacked kernel");
                verify_correctness(decomp_auto_standard, "Auto-selection (standard)");
                verify_correctness(decomp_auto_ht, "Auto-selection (HT)");
                verify_correctness(decomp_optimized_v2, "Optimized V2 kernel");
                verify_correctness(decomp_work_stealing, "Work-stealing kernel");
                verify_correctness(decomp_work_stealing_advanced, "Work-stealing advanced");
                
                // 6. Memory Analysis
                std::cout << "\n6. Memory Usage Analysis:" << std::endl;
                long long bitpacked_delta_size = (current_compressed_size - 
                    (standard_mode_data->num_partitions * sizeof(PartitionInfo)));
                long long preunpacked_delta_size = 
                    (long long)high_throughput_data->total_values * sizeof(long long);
                
                std::cout << "   Bit-packed delta storage: ~" << bitpacked_delta_size << " bytes" << std::endl;
                std::cout << "   Pre-unpacked delta storage: " << preunpacked_delta_size << " bytes" << std::endl;
                std::cout << "   Memory overhead factor: " << 
                        ((double)preunpacked_delta_size / bitpacked_delta_size) << "x" << std::endl;
            }
            
            // Cleanup
            if (standard_mode_data) leco_instance.cleanup(standard_mode_data);
            if (high_throughput_data) leco_instance.cleanup(high_throughput_data);
        }

        // Direct Random Access tests
        std::cout << "\n--- Direct Random Access ---" << std::endl;
        current_da_handle = leco_instance.createDirectAccessHandle(
            current_serialized_obj->data, current_serialized_obj->size, true);
            
        if (current_da_handle && !random_positions.empty()) {
            std::vector<T> gpu_direct_output;
            time_gpu_da = benchmark([&]() { 
                leco_instance.directRandomAccessGPU(current_da_handle, random_positions, gpu_direct_output); 
            }, 100);
            std::cout << "GPU Direct Random Access: " << time_gpu_da << " ms, Throughput: " 
                     << (time_gpu_da > 0 ? (random_positions.size() / time_gpu_da * 1000) : 0) 
                     << " queries/sec\n";
            bool gpu_da_correct = true; 
            int gpu_da_mismatches = 0;
            for(size_t i=0; i<random_positions.size(); ++i) {
                if(gpu_direct_output[i] != data_to_test[random_positions[i]]) {
                    gpu_da_correct=false; 
                    gpu_da_mismatches++;
                }
            }
            std::cout << "GPU Direct Access Correctness: " << (gpu_da_correct ? "PASSED" : "FAILED") 
                     << (gpu_da_mismatches > 0 ? " (" + std::to_string(gpu_da_mismatches) + " mismatches)" : "") 
                     << "\n";

            // CPU Direct Random Access
            std::vector<T> cpu_direct_output;
            auto time_cpu_da = benchmark([&]() { 
                leco_instance.directRandomAccessCPU(current_da_handle, random_positions, 
                                                   cpu_direct_output, false); 
            }, 10);
            std::cout << "CPU Direct Random Access: " << time_cpu_da << " ms, Throughput: " 
                     << (time_cpu_da > 0 ? (random_positions.size() / time_cpu_da * 1000) : 0) 
                     << " queries/sec\n";
            if (time_gpu_da > 0 && time_cpu_da > 0) 
                std::cout << "GPU Speedup over CPU: " << (time_cpu_da / time_gpu_da) << "x\n";
            
            bool cpu_da_correct = true; 
            int cpu_da_mismatches = 0;
            for(size_t k=0; k<random_positions.size(); ++k) {
                if(cpu_direct_output[k] != data_to_test[random_positions[k]]) {
                    cpu_da_correct=false; 
                    cpu_da_mismatches++;
                }
            }
            std::cout << "CPU Direct Access Correctness: " << (cpu_da_correct ? "PASSED" : "FAILED") 
                     << (cpu_da_mismatches > 0 ? " (" + std::to_string(cpu_da_mismatches) + " mismatches)" : "") 
                     << "\n";
            
            leco_instance.destroyDirectAccessHandle(current_da_handle);
        }
        
        // File I/O tests
        std::cout << "\n--- File I/O ---" << std::endl;
        std::string output_file_name = "test_" + data_type_string_name + "_" + config_name + ".leco";
        // Replace spaces with underscores in filename
        std::replace(output_file_name.begin(), output_file_name.end(), ' ', '_');
        std::replace(output_file_name.begin(), output_file_name.end(), '(', '_');
        std::replace(output_file_name.begin(), output_file_name.end(), ')', '_');
        std::replace(output_file_name.begin(), output_file_name.end(), '-', '_');
        
        if (leco_instance.saveToFile(current_serialized_obj->data, current_serialized_obj->size, 
                                    output_file_name)) {
            std::cout << "File save/load test: SUCCESS\n";
            std::remove(output_file_name.c_str());
        }
        
        leco_instance.cleanup(current_deserialized_data); 
        delete current_serialized_obj;
    }
}

// Main Function
int main(int argc, char* argv[]) {
    srand(static_cast<unsigned int>(time(NULL)));
    std::string data_type_arg_str = "int";
    std::string file_type_flag_str = "";
    std::string input_filename_str = "";

    if (argc >= 2) data_type_arg_str = argv[1];
    if (argc == 3) {
        input_filename_str = argv[2];
        if (input_filename_str == "--text" || input_filename_str == "--binary" || 
            input_filename_str == "--help" || input_filename_str == "-h") {
            std::cerr << "Error: Flag '" << input_filename_str 
                     << "' needs a filename, or invalid help arguments." << std::endl;
            input_filename_str = ""; 
            file_type_flag_str = "";
        } else { 
            file_type_flag_str = "--text"; // Default to text
        }
    } else if (argc >= 4) {
        file_type_flag_str = argv[2]; 
        input_filename_str = argv[3];
        if (file_type_flag_str != "--text" && file_type_flag_str != "--binary") {
            std::cerr << "Error: Invalid file type specifier '" << file_type_flag_str 
                     << "'. Use --text or --binary." << std::endl;
            input_filename_str = ""; 
            file_type_flag_str = "";
        }
    }

    std::cout << "LeCo GPU Compression Test - Integer Focus" << std::endl;
    std::cout << "Selected Data Type: " << data_type_arg_str << std::endl;
    if (!input_filename_str.empty() && !file_type_flag_str.empty()) 
        std::cout << "Input File: " << input_filename_str << " (Type: " << file_type_flag_str << ")" << std::endl;
    else { 
        std::cout << "Using synthetic data." << std::endl; 
        input_filename_str = ""; 
        file_type_flag_str = ""; 
    }
    
    int cuda_device_count; 
    CUDA_CHECK(cudaGetDeviceCount(&cuda_device_count));
    if (cuda_device_count == 0) { 
        std::cerr << "No CUDA devices found." << std::endl; 
        return 1; 
    }
    cudaDeviceProp cuda_dev_props; 
    CUDA_CHECK(cudaGetDeviceProperties(&cuda_dev_props, 0));
    std::cout << "Using GPU: " << cuda_dev_props.name << " (Compute: " << cuda_dev_props.major 
             << "." << cuda_dev_props.minor << ")" << std::endl;
    
    bool data_loaded_ok = false;

    std::string dataset_name;
    if (!input_filename_str.empty()) {
        // 去除路径，只保留文件名
        size_t last_slash = input_filename_str.find_last_of("/\\");
        dataset_name = (last_slash != std::string::npos) ? 
                       input_filename_str.substr(last_slash + 1) : input_filename_str;
        
        // 去除扩展名
        size_t last_dot = dataset_name.find_last_of(".");
        if (last_dot != std::string::npos) {
            dataset_name = dataset_name.substr(0, last_dot);
        }
    } else {
        dataset_name = "synthetic_" + data_type_arg_str;
    }

    // printf("file name = %s\n", dataset_name);
    std::cout << "file name = " << dataset_name << std::endl;
    

    if (data_type_arg_str == "int") {
        std::vector<int> int_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") 
                data_loaded_ok = read_text_file(input_filename_str, int_data_vec);
            else if (file_type_flag_str == "--binary") 
                data_loaded_ok = read_binary_file(input_filename_str, int_data_vec);
        } else { 
            int_data_vec.resize(1000000); 
            for(size_t i=0; i<int_data_vec.size(); ++i) 
                int_data_vec[i] = 1000 + static_cast<int>(i) * 5 + (rand() % 20 - 10); 
            data_loaded_ok=true; 
            std::cout << "Generated synthetic int data: " << int_data_vec.size() << std::endl;
        }
        if (data_loaded_ok && !int_data_vec.empty()) 
            run_compression_test(int_data_vec, "int", dataset_name);
        else if(data_loaded_ok) 
            std::cout << "Data empty for int." << std::endl;
        else if(!input_filename_str.empty()) 
            std::cout << "Failed to load int data." << std::endl;

    } else if (data_type_arg_str == "long") {
        std::vector<long> long_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") 
                data_loaded_ok = read_text_file(input_filename_str, long_data_vec);
            else if (file_type_flag_str == "--binary") 
                data_loaded_ok = read_binary_file(input_filename_str, long_data_vec);
        } else { 
            long_data_vec.resize(1000000); 
            for(size_t i=0; i<long_data_vec.size(); ++i) 
                long_data_vec[i] = 100000L + static_cast<long>(i) * 50L + (rand() % 100 - 50);
            data_loaded_ok=true; 
            std::cout << "Generated synthetic long data: " << long_data_vec.size() << std::endl;
        }
        if (data_loaded_ok && !long_data_vec.empty()) 
            run_compression_test(long_data_vec, "long", dataset_name);
        else if(data_loaded_ok) 
            std::cout << "Data empty for long." << std::endl;
        else if(!input_filename_str.empty()) 
            std::cout << "Failed to load long data." << std::endl;

    } else if (data_type_arg_str == "long_long" || data_type_arg_str == "longlong") {
        std::vector<long long> ll_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") 
                data_loaded_ok = read_text_file(input_filename_str, ll_data_vec);
            else if (file_type_flag_str == "--binary") 
                data_loaded_ok = read_binary_file(input_filename_str, ll_data_vec);
        } else { 
            std::cout << "Generating synthetic long long data..." << std::endl;
            ll_data_vec.resize(1000000);
            for (size_t i = 0; i < ll_data_vec.size(); i++) { 
                 ll_data_vec[i] = 1000000000LL + static_cast<long long>(i) * 
                                 (500LL + (rand()%200 - 100)) + (rand() % 1000 - 500);
            }
            std::cout << "Synthetic data generated: " << ll_data_vec.size() << " long longs." << std::endl;
            data_loaded_ok = true;
        }
        if (data_loaded_ok && !ll_data_vec.empty()) 
            run_compression_test(ll_data_vec, "long_long", dataset_name);
        else if(data_loaded_ok) 
            std::cout << "Data empty for long_long." << std::endl;
        else if(!input_filename_str.empty()) 
            std::cout << "Failed to load long_long data." << std::endl;

    } else if (data_type_arg_str == "unsigned_long_long" || data_type_arg_str == "ull") {
        std::vector<unsigned long long> ull_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") 
                data_loaded_ok = read_text_file(input_filename_str, ull_data_vec);
            else if (file_type_flag_str == "--binary") 
                data_loaded_ok = read_binary_file(input_filename_str, ull_data_vec);
        } else { 
            std::cout << "Generating synthetic unsigned long long data..." << std::endl;
            ull_data_vec.resize(1000000);
            // Use smaller base value to avoid overflow issues
            unsigned long long base_val = 1000000000000ULL; // 1 trillion instead of ULLONG_MAX/2
            for (size_t i = 0; i < ull_data_vec.size(); i++) {
                ull_data_vec[i] = base_val + static_cast<unsigned long long>(i) * 
                                 (500ULL + (rand()%200 - 100)) + (rand() % 1000 - 500);
            }
            std::cout << "Synthetic data generated: " << ull_data_vec.size() 
                     << " unsigned long longs." << std::endl;
            data_loaded_ok = true;
        }
        if (data_loaded_ok && !ull_data_vec.empty()) 
            run_compression_test(ull_data_vec, "unsigned_long_long", dataset_name);
        else if(data_loaded_ok) 
            std::cout << "Data empty for unsigned_long_long." << std::endl;
        else if(!input_filename_str.empty()) 
            std::cout << "Failed to load unsigned_long_long data." << std::endl;
        
    } else if (data_type_arg_str == "uint" || data_type_arg_str == "uint32" || 
               data_type_arg_str == "unsigned_int") {
        std::vector<unsigned int> uint_data_vec;
        if (!input_filename_str.empty()) {
            if (file_type_flag_str == "--text") 
                data_loaded_ok = read_text_file(input_filename_str, uint_data_vec);
            else if (file_type_flag_str == "--binary") 
                data_loaded_ok = read_binary_file(input_filename_str, uint_data_vec);
        } else { 
            std::cout << "Generating synthetic unsigned int data..." << std::endl;
            uint_data_vec.resize(1000000);
            for (size_t i = 0; i < uint_data_vec.size(); i++) {
                uint_data_vec[i] = 1000000U + static_cast<unsigned int>(i) * 50U + (rand() % 100);
            }
            std::cout << "Synthetic data generated: " << uint_data_vec.size() << " unsigned ints." << std::endl;
            data_loaded_ok = true;
        }
        if (data_loaded_ok && !uint_data_vec.empty()) 
            run_compression_test(uint_data_vec, "unsigned_int", dataset_name);
        
    } else {
        std::cerr << "\nUsage: " << argv[0] << " <data_type> [FILENAME | --text FILENAME | --binary FILENAME]" 
                  << std::endl;
        std::cerr << "Supported integer data types: int, long, long_long (or longlong), "
                  << "unsigned_long_long (or ull), uint (or uint32, unsigned_int)" << std::endl;
        std::cerr << "If FILENAME is provided without --text/--binary, --text is assumed for it." << std::endl;
        std::cerr << "If no filename arguments, synthetic data is used." << std::endl;
        return 1;
    }
    return 0;
}