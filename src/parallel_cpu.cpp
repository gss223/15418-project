#include <algorithm>

#include "parallel_cpu.h"
#include "get_subset_sums.h"

// TODO: write fft/ntt and try to parallelize (maybe)?

std::vector<int> solve_exponential(const std::vector<int>& w, const int T, const int l, const int r, bool& is_possible) {
    const int len = r - l;

    std::vector<int> sums(1 << len);
    ispc::get_subset_sums(w.data(), T, l, r, sums.data());

    const int total_sum = sums.back();
    std::vector<int> possible(std::min(T, total_sum) + 1);

    for (int x : sums) {
        if (x <= T) {
            possible[x] = true;
        }
    }

    is_possible = is_possible || (static_cast<int>(std::size(possible)) >= T + 1 && possible[T]);
    return (is_possible ? std::vector<int>() : possible);
}

std::vector<int> solve_sequential(const std::vector<int>& w, const int T, const int l, const int r, bool& is_possible) {
    const int len = r - l;

    if (len <= USE_EXPONENTIIAL) {
        return solve_exponential(w, T, l, r, is_possible);
    }

    const int m = l + len / 2;

    std::vector<int> left, right;

    left = solve_sequential(w, T, l, m, is_possible);

    if (is_possible) {
        return {};
    }

    right = solve_sequential(w, T, m, r, is_possible);

    if (is_possible) {
        return {};
    }

    auto convolution = conv(left, right);

    if (static_cast<int>(std::size(convolution)) > T + 1) {
        convolution.resize(T + 1);

        if (convolution[T]) {
            is_possible = true;

            return {};
        }
    }

    for (int& x : convolution) {
        x = std::min(1, x);
    }

    return convolution;
}

// maybe switch to naive if sum is small enough
std::vector<int> solve(const std::vector<int>& w, const int T, const int l, const int r, bool& is_possible) {
    const int len = r - l;

    if (len <= RECURSION_LIMIT) {
        return solve_sequential(w, T, l, r, is_possible);
    }

    if (len <= USE_EXPONENTIIAL) {
        return solve_exponential(w, T, l, r, is_possible);
    }

    const int m = l + len / 2;

    std::vector<int> left, right;

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

    auto convolution = conv(left, right);

    if (static_cast<int>(std::size(convolution)) > T + 1) {
        convolution.resize(T + 1);

        if (convolution[T]) {
            is_possible = true;

            return {};
        }
    }

    for (int& x : convolution) {
        x = std::min(1, x);
    }

    return convolution;
}

bool solve_parallel(const std::vector<int>& w, const int T) {
    bool is_possible = false;

    solve(w, T, 0, std::size(w), is_possible);

    return is_possible;
}
