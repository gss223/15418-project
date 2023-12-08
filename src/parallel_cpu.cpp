#include <bit>

#include "parallel_cpu.h"
#include "convolution.h"
#include "naive.h"

void solve_iterative(const std::vector<uint32_t>& w, const uint32_t T, bool& is_possible) {
    const int n = std::size(w);
    const uint32_t num_blocks = (n + NAIVE_SIZE - 1) / NAIVE_SIZE;
    const int num_iterations = std::bit_width(num_blocks) - 1;

    std::vector blocks(num_iterations + 1, std::vector<std::vector<uint32_t>>(num_blocks));

#pragma omp parallel for
    for (uint32_t i = 0; i < num_blocks; i++) {
        const int l = NAIVE_SIZE * i, r = std::min(l + NAIVE_SIZE, n);
        blocks[0][i] = solve_naive(w, T, l, r, is_possible);
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
