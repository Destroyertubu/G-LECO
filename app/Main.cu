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