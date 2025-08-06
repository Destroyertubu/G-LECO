#include <cstdint>
#include <immintrin.h>          // for _mm_crc32_u8


uint32_t calculateChecksum(const void* data_to_check, size_t size_of_data) {
    const uint8_t* byte_ptr = static_cast<const uint8_t*>(data_to_check);
    uint32_t csum = 0;
    #ifdef __SSE4_2__
    for (size_t i = 0; i < size_of_data; i++) 
        csum = _mm_crc32_u8(csum, byte_ptr[i]);
    #else
    for (size_t i = 0; i < size_of_data; i++) { 
        csum += byte_ptr[i]; 
        csum = (csum << 1) | (csum >> 31); 
    }
    #endif
    return csum;
}