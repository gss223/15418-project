#include <vector>
#include <cstdint> // For uint64_t
#include <cmath>
#include "naive_bits.h"
#include "timer.h"
#include "utils.h"

std::vector<uint64_t> solve_naive(const std::vector<uint32_t>& w, uint32_t T, const uint32_t l, const uint32_t r) {
    T = bit_ceil(T);
    int num_ints; //number of words needed
    num_ints = static_cast<int>(std::ceil(static_cast<double>(T+1) / 64.0));
    std::vector<uint64_t> dp(num_ints);
    setBitInArray(dp,0);

    for (uint32_t i = l; i < r; i++) {
        const uint32_t x = w[i];

        for (uint32_t j = T; j >= x; j--) {
            if (tstBitInArray(dp,j) || tstBitInArray(dp, j-x)){
                setBitInArray(dp,j);
            }
            //dp[j] = dp[j] || dp[j - x];
        }
    }

    return dp;
}

bool solve_bits(const std::vector<uint32_t>& w, const uint32_t T) {
    auto dp = solve_naive(w, T, 0, std::size(w));

    return tstBitInArray(dp,T);
}

void setBitInArray(std::vector<uint64_t>& d, int index) {
    int word = index / 64;
    int bit = index % 64;
    uint64_t t = 1ULL << bit;
    d[word] |= t;
}

int tstBitInArray(const std::vector<uint64_t>& d, int index) {
    int word = index / 64;
    int bit = index % 64;
    uint64_t t = 1ULL << bit;
    
    return (d[word] & t) ? 1 : 0;
}