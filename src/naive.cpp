#include "naive.h"
#include "timer.h"

std::vector<int> solve_naive(const std::vector<int>& w, const int T, const int l, const int r, bool& is_possible) {
    std::vector<int> dp(T + 1);
    dp[0] = true;

    for (int i = l; i < r; i++) {
        const int x = w[i];

        for (int j = T; j >= x; j--) {
            dp[j] = dp[j] || dp[j - x];
        }
    }

    is_possible = is_possible || dp[T];
    return dp;
}

bool solve_naive(const std::vector<int>& w, const int T) {
    bool is_possible = false;

    solve_naive(w, T, 0, std::size(w), is_possible);

    return is_possible;
}
