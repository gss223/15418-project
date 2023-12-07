#include <complex>
#include <vector>
#include <bit>

#include "convolution.h"
#include "utils.h"

std::vector<std::complex<double>> roots, roots_inv;
std::vector<uint32_t> reversed;
uint32_t max_n, half;

// T_ceil is the target rounded up to nearest power of 2
void conv_init(const uint32_t T_ceil) {
    max_n = T_ceil;
    half = max_n / 2;

    roots.resize(half);
    roots_inv.resize(half);
    reversed.resize(max_n);

    const int bit_count = std::bit_width(max_n) - 1;

#pragma omp parallel for
    for (uint32_t i = 0; i < max_n; i++) {
        reversed[i] = reverse_bits(i, bit_count);
    }
    
    const double theta = 2 * M_PI / T_ceil, theta_inv = theta * -1;

#pragma omp parallel for
    for (uint32_t i = 0; i < half; i++) {
        roots[i] = std::polar(1.0, theta * i);
        roots_inv[i] = std::polar(1.0, theta_inv * i);
    }
}

template<bool inverse>
void fft(std::vector<std::complex<double>>& v) {
    const int n = std::size(v), half = n / 2;

    if (n == 1) {
        return;
    }

    std::vector<std::complex<double>> even(n / 2), odd(n / 2);
    for (int i = 0; i < half; i++) {
        even[i] = v[2 * i];
        odd[i] = v[2 * i + 1];
    }

    fft<inverse>(even);
    fft<inverse>(odd);

    const int roots_shift = max_n / n;

    for (int i = 0; i < half; i++) {
        if constexpr (inverse) {
            v[i] = even[i] + roots_inv[i * roots_shift] * odd[i];
            v[i + half] = even[i] - roots_inv[i * roots_shift] * odd[i];

            v[i] /= 2;
            v[i + half] /= 2;
        } else {
            v[i] = even[i] + roots[i * roots_shift] * odd[i];
            v[i + half] = even[i] - roots[i * roots_shift] * odd[i];
        }
    }
}

template<bool inverse>
void fft_iterative(std::vector<std::complex<double>>& v) {
#pragma omp parallel for schedule(dynamic)
    for (uint32_t i = 0; i < max_n; i++) {
        if (i < reversed[i]) {
            swap(v[i], v[reversed[i]]);
        }
    }

    for (uint32_t len = 2; len <= max_n; len *= 2) {
#pragma omp parallel for
        for (uint32_t chunk_start = 0; chunk_start < max_n; chunk_start += len) {
            for (uint32_t i = 0; i < len / 2; i++) {
                std::complex<double> w;

                if constexpr (inverse) {
                    w = roots_inv[max_n / len * i];
                } else {
                    w = roots[max_n / len * i];
                }

                std::complex<double> y0 = v[chunk_start + i], y1 = w * v[chunk_start + i + len / 2];

                v[chunk_start + i] = y0 + y1;
                v[chunk_start + i + len / 2] = y0 - y1;
            }
        }
    }

    if constexpr (inverse) {
#pragma omp parallel for
        for (auto& x : v) {
            x /= max_n;
        }
    }
}

std::vector<uint32_t> conv(std::vector<uint32_t> a, std::vector<uint32_t> b) {
    std::vector<std::complex<double>> ca(std::begin(a), std::end(a)), cb(std::begin(b), std::end(b));

    fft_iterative<false>(ca);
    fft_iterative<false>(cb);

    for (uint32_t i = 0; i < max_n; i++) {
        ca[i] *= cb[i];
    }

    fft_iterative<true>(cb);

    std::vector<uint32_t> res(max_n);
    for (uint32_t i = 0; i < max_n; i++) {
        res[i] = (ca[i].real() > 0);
    }

    return res;
}

