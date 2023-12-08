#include <bit>

#include "parallel_cpu.h"
#include "convolution.h"
#include "naive.h"

void solve_iterative(const std::vector<uint32_t>& w, const uint32_t T, const uint32_t T_ceil, bool& is_possible) {
    const int n = std::size(w);
    const uint32_t num_blocks = (n + NAIVE_SIZE - 1) / NAIVE_SIZE;

    std::vector<std::vector<uint32_t>> blocks(num_blocks);

#pragma omp parallel for
    for (uint32_t i = 0; i < num_blocks; i++) {
        const int l = NAIVE_SIZE * i, r = std::min(l + NAIVE_SIZE, n);
        blocks[i] = solve_naive(w, T, T_ceil, l, r, is_possible);

        is_possible = is_possible || blocks[i][T];
    }

    const int num_iterations = std::bit_width(num_blocks) - 1;
    for (int iter = 0; iter < num_iterations; iter++) {
        const uint32_t blocks_next_size = (std::size(blocks) + 1) / 2;
        std::vector<std::vector<uint32_t>> blocks_next(blocks_next_size);

#pragma omp parallel for
        for (uint32_t i = 0; i < blocks_next_size; i++) {
            if (2 * i + 1 < std::size(blocks)) {
                blocks_next[i] = conv(std::move(blocks[2 * i]), std::move(blocks[2 * i + 1]));
            } else {
                blocks_next[i] = std::move(blocks[2 * i]);
            }

            is_possible = is_possible || blocks_next[i][T];
        }

        blocks = std::move(blocks_next);
    }
}

bool solve_parallel(const std::vector<uint32_t>& w, const uint32_t T) {
    bool is_possible = false;
    const uint32_t T_ceil = std::bit_ceil(T + 1);

    conv_init(T_ceil);
    solve_iterative(w, T, T_ceil, is_possible);

    return is_possible;
}
