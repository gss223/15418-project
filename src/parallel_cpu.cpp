#include <algorithm>

#include "get_subset_sums.h"
#include "parallel_cpu.h"
#include "convolution.h"
#include "naive.h"

// TODO: replace atcoder's fft and parallelize it

std::vector<uint32_t> solve_exponential(const std::vector<uint32_t>& w, const uint32_t T, const uint32_t l, const uint32_t r, bool& is_possible) {
    const uint32_t len = r - l;

    std::vector<uint32_t> sums(1 << len);
    ispc::get_subset_sums(w.data(), T, l, r, sums.data());

    const uint32_t total_sum = sums.back();
    std::vector<uint32_t> possible(std::min(T, total_sum) + 1);

    for (uint32_t x : sums) {
        if (x <= T) {
            possible[x] = true;
        }
    }

    is_possible = is_possible || (static_cast<uint32_t>(std::size(possible)) >= T + 1 && possible[T]);
    return (is_possible ? std::vector<uint32_t>() : possible);
}

std::vector<uint32_t> solve_sequential(const std::vector<uint32_t>& w, const uint32_t T, const uint32_t l, const uint32_t r, bool& is_possible) {
    const uint32_t len = r - l;

    if (len <= USE_NAIVE) {
        return solve_naive(w, T, l, r, is_possible);
    }

    /*
     * Probably never worth to use this
        if (len <= USE_EXPONENTIAL) {
            return solve_exponential(w, T, l, r, is_possible);
        }
    */

    const uint32_t m = l + len / 2;

    std::vector<uint32_t> left, right;

    left = solve_sequential(w, T, l, m, is_possible);

    if (is_possible) {
        return {};
    }

    right = solve_sequential(w, T, m, r, is_possible);

    if (is_possible) {
        return {};
    }

    auto convolution = atcoder::convolution(left, right);

    if (static_cast<uint32_t>(std::size(convolution)) > T + 1) {
        convolution.resize(T + 1);

        if (convolution[T]) {
            is_possible = true;

            return {};
        }
    }

    for (uint32_t& x : convolution) {
        x = std::min(1U, x);
    }

    return convolution;
}

// maybe switch to naive if sum is small enough
std::vector<uint32_t> solve(const std::vector<uint32_t>& w, const uint32_t T, const uint32_t l, const uint32_t r, bool& is_possible) {
    const uint32_t len = r - l;

    if (len <= RECURSION_LIMIT) {
        return solve_sequential(w, T, l, r, is_possible);
    }

    if (len <= USE_NAIVE) {
        return solve_naive(w, T, l, r, is_possible);
    }

    /*
     * Probably never worth to use this
        if (len <= USE_EXPONENTIAL) {
            return solve_exponential(w, T, l, r, is_possible);
        }
    */

    const uint32_t m = l + len / 2;

    std::vector<uint32_t> left, right;

#pragma omp task shared(w, T, l, r, is_possible, left)
    left = solve(w, T, l, m, is_possible);

    if (is_possible) {
        return {};
    }

#pragma omp task shared(w, T, l, r, is_possible, right)
    right = solve(w, T, m, r, is_possible);

#pragma omp taskwait

    if (is_possible) {
        return {};
    }

    auto convolution = atcoder::convolution(left, right);

    if (static_cast<uint32_t>(std::size(convolution)) > T + 1) {
        convolution.resize(T + 1);

        if (convolution[T]) {
            is_possible = true;

            return {};
        }
    }

    for (uint32_t& x : convolution) {
        x = std::min(1U, x);
    }

    return convolution;
}

bool solve_parallel(const std::vector<uint32_t>& w, const uint32_t T) {
    bool is_possible = false;

    solve(w, T, 0, std::size(w), is_possible);

    return is_possible;
}
