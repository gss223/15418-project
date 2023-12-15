#include <cstdint>
#include <vector>

std::vector<uint64_t> solve_naive(const std::vector<uint32_t>& w, const uint32_t T, const uint32_t l, const uint32_t r);

bool solve_bits(const std::vector<uint32_t>& w, const uint32_t T);

void set_bit(std::vector<uint64_t>& d, int index);

int test_bit(const std::vector<uint64_t>& d, int index);
