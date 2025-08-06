#include <cuda_runtime.h>       // for cudaDeviceSynchronize
#include <chrono>               // for timing
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>



template<typename Func>
double benchmark(Func func_to_run, int num_iterations = 100) {
    if (num_iterations > 0) 
        for (int i = 0; i < std::min(5, num_iterations); i++) 
            func_to_run();
    auto timer_start = std::chrono::high_resolution_clock::now();
    if (num_iterations > 0) 
        for (int i = 0; i < num_iterations; i++) 
            func_to_run();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto timer_end = std::chrono::high_resolution_clock::now();
    if (num_iterations <= 0) return 0.0;
    return std::chrono::duration<double, std::milli>(timer_end - timer_start).count() / num_iterations;
}

template<typename T>
bool read_text_file(const std::string& in_filename, std::vector<T>& out_data_vec) {
    std::ifstream file_stream(in_filename);
    if (!file_stream.is_open()) { 
        std::cerr << "Error: Could not open text file " << in_filename << std::endl; 
        return false; 
    }
    out_data_vec.clear(); 
    std::string line_str;
    while (std::getline(file_stream, line_str)) {
        try {
            line_str.erase(0, line_str.find_first_not_of(" \t\n\r\f\v"));
            line_str.erase(line_str.find_last_not_of(" \t\n\r\f\v") + 1);
            if (line_str.empty()) continue;
            if (std::is_same<T, int>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoi(line_str)));
            else if (std::is_same<T, long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stol(line_str)));
            else if (std::is_same<T, long long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoll(line_str)));
            else if (std::is_same<T, unsigned int>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoul(line_str)));
            else if (std::is_same<T, unsigned long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoul(line_str)));
            else if (std::is_same<T, unsigned long long>::value) 
                out_data_vec.push_back(static_cast<T>(std::stoull(line_str)));
            else { 
                std::cerr << "Warning: Read_text_file unsupported integer type." << std::endl; 
            }
        } catch (const std::exception& e) { 
            std::cerr << "Warning: Parsing line '"<< line_str << "': " << e.what() << std::endl;
        }
    }
    file_stream.close();
    std::cout << "Successfully read " << out_data_vec.size() << " values from text file: " 
             << in_filename << std::endl;
    return true;
}

template<typename T>
bool read_binary_file(const std::string& in_filename, std::vector<T>& out_data_vec) {
    std::ifstream file_stream(in_filename, std::ios::binary | std::ios::ate);
    if (!file_stream.is_open()) { 
        std::cerr << "Error: Could not open binary file " << in_filename << std::endl; 
        return false; 
    }
    std::streampos stream_file_size = file_stream.tellg();
    if (stream_file_size < 0 || stream_file_size % sizeof(T) != 0) { 
        std::cerr << "Error: Binary file " << in_filename << " has invalid size." << std::endl; 
        file_stream.close(); 
        return false;
    }
    if (stream_file_size == 0) {
        out_data_vec.clear(); 
        file_stream.close(); 
        std::cout << "Read 0 values from empty binary file: " << in_filename << std::endl; 
        return true;
    }
    file_stream.seekg(0, std::ios::beg);
    size_t num_file_elements = static_cast<size_t>(stream_file_size) / sizeof(T);
    try {
        out_data_vec.resize(num_file_elements);
    } catch(const std::bad_alloc&){ 
        std::cerr << "Error: Malloc failed for binary data." << std::endl; 
        file_stream.close();
        return false;
    }
    file_stream.read(reinterpret_cast<char*>(out_data_vec.data()), num_file_elements * sizeof(T));
    bool read_success = file_stream.good() && 
                       (static_cast<size_t>(file_stream.gcount()) == num_file_elements * sizeof(T));
    file_stream.close();
    if(!read_success) {
        out_data_vec.clear(); 
        std::cerr << "Error reading binary file: " << in_filename << std::endl; 
        return false;
    }
    std::cout << "Successfully read " << out_data_vec.size() << " values from binary file: " 
             << in_filename << std::endl;
    return true;
}
