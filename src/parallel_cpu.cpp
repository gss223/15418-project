#include <bit>

#include "parallel_cpu.h"
#include "convolution.h"
#include "naive.h"

/*
std::vector<uint32_t> solve_sequential(const std::vector<uint32_t>& w, const uint32_t T, const uint32_t T_ceil, const uint32_t l, const uint32_t r, bool& is_possible) {
    const uint32_t len = r - l;

    if (len <= USE_NAIVE) {
        return solve_naive(w, T, T_ceil, l, r, is_possible);
    }

    const uint32_t m = l + len / 2;

    std::vector<uint32_t> left, right;

    left = solve_sequential(w, T, T_ceil, l, m, is_possible);

    if (is_possible) {
        // return {};
    }

    right = solve_sequential(w, T, T_ceil, m, r, is_possible);

    if (is_possible) {
        // return {};
    }

    auto convolution = conv(left, right);

    if (convolution[T]) {
        is_possible = true;

        // return {};
    }

    return convolution;
}

std::vector<uint32_t> solve(const std::vector<uint32_t>& w, const uint32_t T, const uint32_t T_ceil, const uint32_t l, const uint32_t r, bool& is_possible) {
    const uint32_t len = r - l;

    if (len <= RECURSION_LIMIT) {
        return solve_sequential(w, T, T_ceil, l, r, is_possible);
    }

    if (len <= USE_NAIVE) {
        return solve_naive(w, T, T_ceil, l, r, is_possible);
    }

    const uint32_t m = l + len / 2;

    std::vector<uint32_t> left, right;

#pragma omp task shared(w, T, l, r, is_possible, left)
    left = solve(w, T, T_ceil, l, m, is_possible);

    if (is_possible) {
        // return {};
    }

#pragma omp task shared(w, T, l, r, is_possible, right)
    right = solve(w, T, T_ceil, m, r, is_possible);

#pragma omp taskwait

    if (is_possible) {
        // return {};
    }

    auto convolution = conv(left, right);

    if (convolution[T]) {
        is_possible = true;

        // return {};
    }

    return convolution;
}
*/

void solve_iterative(const std::vector<uint32_t>& w, const uint32_t T, const uint32_t T_ceil, bool& is_possible) {
    const int n = std::size(w);
    const uint32_t num_blocks = (n + USE_NAIVE - 1) / USE_NAIVE;

    std::vector<std::vector<uint32_t>> blocks(num_blocks);

#pragma omp parallel for
    for (uint32_t i = 0; i < num_blocks; i++) {
        const int l = USE_NAIVE * i, r = std::min(l + USE_NAIVE, n);
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
