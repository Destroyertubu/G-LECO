#include <cstdint>
#include "core/CudaUtils.cuh"   // 依赖其中定义的常量

// Enhanced partition metadata structure - ONLY FOR HOST USE AND SERIALIZATION
struct PartitionInfo {
    int32_t start_idx;
    int32_t end_idx;
    int32_t model_type;
    double model_params[4];
    int32_t delta_bits;
    int64_t delta_array_bit_offset;
    long long error_bound;
    int32_t reserved[1];
};

struct PartitionMetaOpt {
    int32_t start_idx;
    int32_t model_type;
    int32_t delta_bits;
    int32_t partition_len;
    double theta0;
    double theta1;
    int64_t bit_offset_base;
};

// Partition metadata structure for shared memory caching
struct PartitionMeta {
    int32_t start_idx;
    int32_t end_idx;
    int32_t model_type;
    double theta0;
    double theta1;
};

// Binary format header for serialized data - UPDATED FOR SoA
struct SerializedHeader {
    uint32_t magic;
    uint32_t version; // Increment this to reflect the new format (4)
    uint32_t total_values;
    uint32_t num_partitions;

    // --- New SoA Table Offsets and Sizes ---
    // All offsets are relative to the beginning of the data blob.
    uint64_t start_indices_offset;
    uint64_t end_indices_offset;
    uint64_t model_types_offset;
    uint64_t model_params_offset;
    uint64_t delta_bits_offset;
    uint64_t delta_array_bit_offsets_offset;
    uint64_t error_bounds_offset;
    uint64_t delta_array_offset; // Offset for the main delta bitstream

    // Field for the size in bytes of the model_params array, as it contains doubles.
    uint64_t model_params_size_bytes;
    uint64_t delta_array_size_bytes; // This remains.

    uint32_t data_type_size;
    uint32_t header_checksum;
    uint32_t reserved[3];
};

// Structure to pass partition candidates to kernel
struct PartitionCandidateGPU {
    int start_idx;
    int end_idx;
    double theta0;
    double theta1;
    long long max_error;
    int delta_bits;
    double total_cost;
};


// Work-stealing queue structure for GPU
struct WorkStealingQueue {
    int* tasks;          // Array of task indices
    int* head;           // Per-thread queue heads
    int* tail;           // Per-thread queue tails
    int* global_head;    // Global queue head for stealing
    int* global_tail;    // Global queue tail
    int max_tasks;
    int num_threads;
};