#include <cstdint>      // for uint32_t, int64_t, etc.
#include <cstddef>      // for size_t

enum ModelType {
    MODEL_CONSTANT = 0,
    MODEL_LINEAR = 1,
    MODEL_POLYNOMIAL2 = 2,
    MODEL_POLYNOMIAL3 = 3,
    MODEL_DIRECT_COPY = 4   // New model type for direct copy when overflow detected
};


// Template compressed data structure - SoA LAYOUT
template<typename T>
struct CompressedData {
    // --- SoA Data Pointers (all are device pointers) ---
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    int32_t* d_model_types;
    double* d_model_params; // Note: This will store params for all partitions contiguously.
                            // For a linear model, layout will be [p0_t0, p0_t1, p1_t0, p1_t1, ...].
    int32_t* d_delta_bits;
    int64_t* d_delta_array_bit_offsets;
    long long* d_error_bounds;

    uint32_t* delta_array;          // This remains the same.

// 383 -------------------------------------------------------
    long long* d_plain_deltas;
// 383 -------------------------------------------------------

    // --- Host-side metadata ---
    int num_partitions;
    int total_values;

    // --- Device-side self pointer ---
    CompressedData<T>* d_self;
};

// Serialized data container (for host-side blob)
struct SerializedData {
    uint8_t* data;
    size_t size;

    SerializedData() : data(nullptr), size(0) {}
    ~SerializedData() {
        if (data) {
            delete[] data;
            data = nullptr;
        }
    }
    SerializedData(const SerializedData&) = delete;
    SerializedData& operator=(const SerializedData&) = delete;
    SerializedData(SerializedData&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    SerializedData& operator=(SerializedData&& other) noexcept {
        if (this != &other) {
            if (data) delete[] data;
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
};


template<typename T>
struct alignas(256) DirectAccessHandle {  // Increased alignment to 256 bytes
    const uint8_t* data_blob_host;
    const SerializedHeader* header_host;
    
    // Host-side SoA pointers
    const int32_t* start_indices_host;
    const int32_t* end_indices_host;
    const int32_t* model_types_host;
    const double* model_params_host;
    const int32_t* delta_bits_host;
    const int64_t* delta_array_bit_offsets_host;
    const long long* error_bounds_host;
    const uint32_t* delta_array_host;
    
    size_t data_blob_size;

    uint8_t* d_data_blob_device;
    SerializedHeader* d_header_device;
    
    // Device-side SoA pointers
    int32_t* d_start_indices_device;
    int32_t* d_end_indices_device;
    int32_t* d_model_types_device;
    double* d_model_params_device;
    int32_t* d_delta_bits_device;
    int64_t* d_delta_array_bit_offsets_device;
    long long* d_error_bounds_device;
    uint32_t* d_delta_array_device;
    
    // Padding to ensure size is multiple of alignment
    char padding[256 - (sizeof(void*) * 20 + sizeof(size_t)) % 256];
};
