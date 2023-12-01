#include <cstdint>
#include <vector>

constexpr int RECURSION_LIMIT = 12500;
constexpr int USE_NAIVE = 12500;
constexpr int USE_EXPONENTIAL = 17;

bool solve_parallel(const std::vector<uint32_t>& w, const uint32_t T);
