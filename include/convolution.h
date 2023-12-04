#include <cstdint>
#include <vector>

// call this before using conv
void conv_init(const uint32_t T_ceil);

// assumes v's length is a power of 2
std::vector<uint32_t> conv(std::vector<uint32_t> a, std::vector<uint32_t> b);
