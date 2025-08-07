#ifndef CORE_BIT_MANIPULATION_CUH
#define CORE_BIT_MANIPULATION_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>            // for min

// Helper function for delta extraction
template<typename T>
__device__ inline long long extractDelta(const uint32_t* delta_array, 
                                        int64_t bit_offset, 
                                        int delta_bits) {
    if (delta_bits <= 0) return 0;
    
    if (delta_bits <= 32) {
        int word_idx = bit_offset / 32;
        int bit_offset_in_word = bit_offset % 32;
        uint32_t extracted_bits;
        
        if (bit_offset_in_word + delta_bits <= 32) {
            extracted_bits = (delta_array[word_idx] >> bit_offset_in_word) & 
                            ((1U << delta_bits) - 1U);
        } else {
            uint32_t w1 = delta_array[word_idx];
            uint32_t w2 = delta_array[word_idx + 1]; 
            extracted_bits = (w1 >> bit_offset_in_word) | (w2 << (32 - bit_offset_in_word));
            extracted_bits &= ((1U << delta_bits) - 1U);
        }

        // Sign extension
        if (delta_bits < 32) {
            uint32_t sign_bit = 1U << (delta_bits - 1);
            if (extracted_bits & sign_bit) {
                uint32_t sign_extend_mask = ~((1U << delta_bits) - 1U);
                return static_cast<long long>(static_cast<int32_t>(extracted_bits | sign_extend_mask));
            } else {
                return static_cast<long long>(extracted_bits);
            }
        } else {
            return static_cast<long long>(static_cast<int32_t>(extracted_bits));
        }
    } else {
        // Handle > 32 bit deltas
        int start_word_idx = bit_offset / 32;
        int offset_in_word = bit_offset % 32;
        int bits_remaining = delta_bits;
        uint64_t extracted_val_64 = 0;
        int shift = 0;
        int word_idx = start_word_idx;
        
        while (bits_remaining > 0 && shift < 64) {
            int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
            uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
            uint32_t word_val = (delta_array[word_idx] >> offset_in_word) & mask;
            extracted_val_64 |= (static_cast<uint64_t>(word_val) << shift);
            
            shift += bits_in_this_word;
            bits_remaining -= bits_in_this_word;
            word_idx++;
            offset_in_word = 0;
        }
        
        // Sign extension for 64-bit deltas
        if (delta_bits < 64) {
            uint64_t sign_mask_64 = 1ULL << (delta_bits - 1);
            if (extracted_val_64 & sign_mask_64) {
                uint64_t sign_ext_mask_64 = ~((1ULL << delta_bits) - 1ULL);
                return static_cast<long long>(extracted_val_64 | sign_ext_mask_64);
            } else {
                return static_cast<long long>(extracted_val_64);
            }
        } else {
            return static_cast<long long>(extracted_val_64);
        }
    }
}

// Optimized delta extraction function that eliminates branching
// Optimized delta extraction function with reduced branching
template<typename T>
__device__ inline long long extractDelta_Optimized(const uint32_t* __restrict__ delta_array, 
                                                   int64_t bit_offset, 
                                                   int delta_bits) {
    if (delta_bits <= 0) return 0;
    
    // Use bit operations instead of division/modulo
    int word_idx = bit_offset >> 5;  // bit_offset / 32
    int bit_offset_in_word = bit_offset & 31;  // bit_offset % 32
    
    if (delta_bits <= 32) {
        // Always read two words to avoid branching
        uint32_t w1 = __ldg(&delta_array[word_idx]);
        uint32_t w2 = __ldg(&delta_array[word_idx + 1]);
        
        // Combine words into 64-bit value for branchless extraction
        uint64_t combined = (static_cast<uint64_t>(w2) << 32) | static_cast<uint64_t>(w1);
        uint32_t extracted_bits = (combined >> bit_offset_in_word) & ((1U << delta_bits) - 1U);
        
        // Branchless sign extension
        if (delta_bits < 32) {
            uint32_t sign_bit = extracted_bits >> (delta_bits - 1);
            uint32_t sign_mask = -sign_bit;  // All 1s if sign bit set, 0 otherwise
            uint32_t extend_mask = ~((1U << delta_bits) - 1U);
            extracted_bits |= (sign_mask & extend_mask);
        }
        
        return static_cast<long long>(static_cast<int32_t>(extracted_bits));
    } else {
        // Handle > 32 bit deltas with optimized loop
        uint64_t extracted_val_64 = 0;
        
        // First word
        uint32_t first_word = __ldg(&delta_array[word_idx]);
        int bits_from_first = 32 - bit_offset_in_word;
        extracted_val_64 = (first_word >> bit_offset_in_word);
        
        // Middle words (if any)
        int bits_remaining = delta_bits - bits_from_first;
        int shift = bits_from_first;
        word_idx++;
        
        // Unroll for common case of 64-bit values
        if (bits_remaining > 0) {
            uint32_t word = __ldg(&delta_array[word_idx]);
            if (bits_remaining >= 32) {
                extracted_val_64 |= (static_cast<uint64_t>(word) << shift);
                shift += 32;
                bits_remaining -= 32;
                word_idx++;
                
                if (bits_remaining > 0) {
                    word = __ldg(&delta_array[word_idx]);
                    uint32_t mask = (bits_remaining == 32) ? ~0U : ((1U << bits_remaining) - 1U);
                    extracted_val_64 |= (static_cast<uint64_t>(word & mask) << shift);
                }
            } else {
                uint32_t mask = (1U << bits_remaining) - 1U;
                extracted_val_64 |= (static_cast<uint64_t>(word & mask) << shift);
            }
        }
        
        // Branchless sign extension for 64-bit
        if (delta_bits < 64) {
            uint64_t sign_bit = extracted_val_64 >> (delta_bits - 1);
            uint64_t sign_mask = -(int64_t)sign_bit;
            uint64_t extend_mask = ~((1ULL << delta_bits) - 1ULL);
            extracted_val_64 |= (sign_mask & extend_mask);
        }
        
        return static_cast<long long>(extracted_val_64);
    }
}

// Extract value for direct copy model
template<typename T>
__device__ inline T extractDirectValue(const uint32_t* delta_array, 
                                      int64_t bit_offset, 
                                      int value_bits) {
    if (value_bits <= 0) return static_cast<T>(0);
    
    if (value_bits <= 32) {
        int word_idx = bit_offset / 32;
        int bit_offset_in_word = bit_offset % 32;
        uint32_t extracted_bits;
        
        if (bit_offset_in_word + value_bits <= 32) {
            extracted_bits = (delta_array[word_idx] >> bit_offset_in_word) & 
                            ((1U << value_bits) - 1U);
        } else {
            uint32_t w1 = delta_array[word_idx];
            uint32_t w2 = delta_array[word_idx + 1]; 
            extracted_bits = (w1 >> bit_offset_in_word) | (w2 << (32 - bit_offset_in_word));
            extracted_bits &= ((1U << value_bits) - 1U);
        }
        
        return static_cast<T>(extracted_bits);
    } else {
        // Handle > 32 bit values
        int start_word_idx = bit_offset / 32;
        int offset_in_word = bit_offset % 32;
        int bits_remaining = value_bits;
        uint64_t extracted_val_64 = 0;
        int shift = 0;
        int word_idx = start_word_idx;
        
        while (bits_remaining > 0 && shift < 64) {
            int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
            uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
            uint32_t word_val = (delta_array[word_idx] >> offset_in_word) & mask;
            extracted_val_64 |= (static_cast<uint64_t>(word_val) << shift);
            
            shift += bits_in_this_word;
            bits_remaining -= bits_in_this_word;
            word_idx++;
            offset_in_word = 0;
        }
        
        return static_cast<T>(extracted_val_64);
    }
}

#endif // CORE_BIT_MANIPULATION_CUH