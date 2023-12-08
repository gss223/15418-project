#include <array>
#include <complex>
#include <vector>
#include <bit>

#include "convolution.h"
#include "utils.h"

std::vector<std::complex<double>> roots, roots_inv;
std::vector<uint32_t> reversed;
std::vector<std::array<uint32_t, 2>> swaps;
uint32_t max_n, half;

// T_ceil is the target rounded up to nearest power of 2
void conv_init(const uint32_t T_ceil) {
    max_n = T_ceil;
    half = max_n >> 1;

    roots.resize(half);
    roots_inv.resize(half);
    reversed.resize(max_n);

    const int bit_count = std::bit_width(max_n) - 1;

#pragma omp parallel for
    for (uint32_t i = 0; i < max_n; i++) {
        reversed[i] = reverse_bits(i, bit_count);
    }

    for (uint32_t i = 0; i < max_n; i++) {
        if (i < reversed[i]) {
            swaps.push_back({i, reversed[i]});
        }
    }
    
    const double theta = 2 * M_PI / T_ceil;

#pragma omp parallel for
    for (uint32_t i = 0; i < half; i++) {
        roots[i] = std::polar(1.0, theta * i);
        roots_inv[i] = -roots[i];
    }
}

template<bool inverse>
void fft_iterative(std::vector<std::complex<double>>& v) {
    for (const auto& [i, j] : swaps) {
        swap(v[i], v[j]);
    }

    for (uint32_t len = 2, shift = half, half_len = len >> 1; len <= max_n; len *= 2, shift >>= 1, half_len *= 2) {
#pragma omp parallel for
        for (uint32_t chunk_start = 0; chunk_start < max_n; chunk_start += len) {
            for (uint32_t i = 0; i < half_len; i++) {
                std::complex<double> w;

                if constexpr (inverse) {
                    w = roots_inv[shift * i];
                } else {
                    w = roots[shift * i];
                }

                std::complex<double> y0 = v[chunk_start + i], y1 = w * v[chunk_start + i + half_len];

                v[chunk_start + i] = y0 + y1;
                v[chunk_start + i + half_len] = y0 - y1;
            }
        }
    }
}

std::vector<uint32_t> conv(std::vector<uint32_t> a, std::vector<uint32_t> b) {
    std::vector<std::complex<double>> ca(std::begin(a), std::end(a)), cb(std::begin(b), std::end(b));

    fft_iterative<false>(ca);
    fft_iterative<false>(cb);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < max_n; i++) {
        ca[i] *= cb[i];
    }

    fft_iterative<true>(ca);

    std::vector<uint32_t> res(max_n);
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < max_n; i++) {
        res[i] = (ca[i].real() >= half);
    }

    return res;
}

