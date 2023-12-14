#include "naive.h"
#include "timer.h"
#include "utils.h"

std::vector<uint32_t> solve_naive(const std::vector<uint32_t>& w, const uint32_t T, const uint32_t l, const uint32_t r) {
    std::vector<uint32_t> dp(T + 1);
    dp[0] = true;

    for (uint32_t i = l; i < r; i++) {
        const uint32_t x = w[i];

        for (uint32_t j = T; j >= x; j--) {
            dp[j] = dp[j] || dp[j - x];
        }
    }

    return dp;
}

bool solve_naive(const std::vector<uint32_t>& w, const uint32_t T) {
    auto dp = solve_naive(w, T, 0, std::size(w));

    return dp[T];
}
