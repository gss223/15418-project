#include "parallel_cpu.h"
#include "convolution.h"

std::vector<uint32_t> solve_naive(const std::vector<uint32_t>& w, const uint32_t T, const uint32_t l, const uint32_t r) {
    std::vector<uint32_t> dp(T + 1);
    dp[0] = 1;

    for (uint32_t i = l; i < r; i++) {
        const uint32_t x = w[i];

        for (uint32_t j = T; j >= x; j--) {
            dp[j] = dp[j] || dp[j - x];
        }
    }

    return dp;
}

void solve_iterative(const std::vector<uint32_t>& w, const uint32_t T, bool& is_possible) {
    const int n = std::size(w);
    const uint32_t num_blocks = (n + NAIVE_SIZE - 1) / NAIVE_SIZE;
    const int num_iterations = std::__lg(num_blocks);

    std::vector blocks(num_iterations + 1, std::vector<std::vector<uint32_t>>(num_blocks));

#pragma omp parallel for
    for (uint32_t i = 0; i < num_blocks; i++) {
        const int l = NAIVE_SIZE * i, r = std::min(l + NAIVE_SIZE, n);
        blocks[0][i] = solve_naive(w, T, l, r);
    }

    for (int iter = 0, iter_num_blocks = num_blocks; iter < num_iterations; iter++, iter_num_blocks = (iter_num_blocks + 1) / 2) {
        const int next_iter_num_blocks = (iter_num_blocks + 1) / 2;

        for (int i = 0; i < next_iter_num_blocks; i++) {
            if (2 * i + 1 < iter_num_blocks) {
#pragma omp task shared(blocks)
                blocks[iter + 1][i] = conv(std::move(blocks[iter][2 * i]), std::move(blocks[iter][2 * i + 1]));
            } else {
                blocks[iter + 1][i] = std::move(blocks[iter][2 * i]);
            }
        }

#pragma omp taskwait
    }

    is_possible = blocks[num_iterations][0][T];
}

bool solve_parallel(const std::vector<uint32_t>& w, const uint32_t T) {
    bool is_possible = false;

    conv_init(T);
    solve_iterative(w, T, is_possible);

    return is_possible;
}
