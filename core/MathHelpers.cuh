#ifndef CORE_MATH_HELPERS_CUH
#define CORE_MATH_HELPERS_CUH

#include <cuda_runtime.h>       // for __device__, __host__, intrinsics
#include <type_traits>          // for std::is_signed
#include <cmath>                // for fmax, fmin, round
#include <limits>               // for LLONG_MAX, LLONG_MIN
#include <climits>              // for INT_MAX

template<typename T>
__device__ __host__ inline bool mightOverflowDoublePrecision(T value) {
    if (std::is_signed<T>::value) {
        return false;  // Signed types within long long range are OK
    } else {
        // For unsigned types, check if value exceeds double precision (2^53)
        const uint64_t DOUBLE_PRECISION_LIMIT = (1ULL << 53);
        return static_cast<uint64_t>(value) > DOUBLE_PRECISION_LIMIT;
    }
}

// Helper template for safe delta calculation
template<typename T>
__device__ __host__ inline long long calculateDelta(T actual, T predicted) {
    if (std::is_signed<T>::value) {
        return static_cast<long long>(actual) - static_cast<long long>(predicted);
    } else {
        // For unsigned types
        if (sizeof(T) == 8) {
            // For 64-bit unsigned types (unsigned long long)
            unsigned long long actual_ull = static_cast<unsigned long long>(actual);
            unsigned long long pred_ull = static_cast<unsigned long long>(predicted);
            
            if (actual_ull >= pred_ull) {
                unsigned long long diff = actual_ull - pred_ull;
                if (diff <= static_cast<unsigned long long>(LLONG_MAX)) {
                    return static_cast<long long>(diff);
                } else {
                    return LLONG_MAX;
                }
            } else {
                unsigned long long diff = pred_ull - actual_ull;
                if (diff <= static_cast<unsigned long long>(LLONG_MAX)) {
                    return -static_cast<long long>(diff);
                } else {
                    return LLONG_MIN;
                }
            }
        } else {
            // For smaller unsigned types, direct conversion is safe
            return static_cast<long long>(actual) - static_cast<long long>(predicted);
        }
    }
}

template<typename T>
__device__ __host__ inline T applyDelta(T predicted, long long delta) {
    if (std::is_signed<T>::value) {
        // For signed types, simple addition
        return predicted + static_cast<T>(delta);
    } else {
        // For unsigned types, use unsigned arithmetic to handle wraparound correctly
        if (sizeof(T) == 8) {
            // For 64-bit unsigned types
            unsigned long long pred_ull = static_cast<unsigned long long>(predicted);
            unsigned long long delta_ull = static_cast<unsigned long long>(delta);
            return static_cast<T>(pred_ull + delta_ull);
        } else if (sizeof(T) == 4) {
            // For 32-bit unsigned types
            unsigned long pred_ul = static_cast<unsigned long>(predicted);
            unsigned long delta_ul = static_cast<unsigned long>(static_cast<long>(delta));
            return static_cast<T>(pred_ul + delta_ul);
        } else if (sizeof(T) == 2) {
            // For 16-bit unsigned types
            unsigned pred_u = static_cast<unsigned>(predicted);
            unsigned delta_u = static_cast<unsigned>(static_cast<int>(delta));
            return static_cast<T>(pred_u + delta_u);
        } else {
            // For 8-bit unsigned types
            unsigned pred_u = static_cast<unsigned>(predicted);
            unsigned delta_u = static_cast<unsigned>(static_cast<int>(delta));
            return static_cast<T>(pred_u + delta_u);
        }
    }
}

inline uint64_t alignOffset(uint64_t offset, uint64_t alignment) {
    return ((offset + alignment - 1) / alignment) * alignment;
}

// Helper function for warp reduction - sum
__device__ __forceinline__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Helper function for warp reduction - max
__device__ __forceinline__ long long warpReduceMax(long long val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Helper for block reduction - sum
__device__ __forceinline__ double blockReduceSum(double val) {
    __shared__ double shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Helper for block reduction - max
__device__ __forceinline__ long long blockReduceMax(long long val) {
    __shared__ long long shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    
    val = warpReduceMax(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceMax(val);
    
    return val;
}

#endif // CORE_MATH_HELPERS_CUH