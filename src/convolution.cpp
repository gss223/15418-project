#include <complex>
#include <vector>
#include <cmath>

#include "convolution.h"

std::vector<std::complex<double>> roots, roots_inv;
uint32_t n, half;

// T_ceil is the target rounded up to nearest power of 2
void conv_init(const uint32_t T_ceil) {
    n = T_ceil;
    half = n / 2;

    roots.resize(half);
    roots_inv.resize(half);
    
    const double theta = 2 * M_PI / T_ceil, theta_inv = theta * -1;
    const std::complex<double> w(std::cos(theta), std::sin(theta)), w_inv(std::cos(theta_inv), std::sin(theta_inv));
    roots[0] = roots_inv[0] = std::complex<double>(1);

    for (uint32_t i = 1; i < half; i++) {
        roots[i] = roots[i - 1] * w;
        roots_inv[i] = roots_inv[i - 1] * w_inv;
    }
}

// assumes v has length n
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

    for (int i = 0; i < half; i++) {
        if constexpr (inverse) {
            v[i] = even[i] + roots_inv[i] * odd[i];
            v[i + half] = even[i] - roots_inv[i] * odd[i];

            v[i] /= n;
            v[i + half] /= n;
        } else {
            v[i] = even[i] + roots[i] * odd[i];
            v[i + half] = even[i] - roots[i] * odd[i];
        }
    }
}

std::vector<uint32_t> conv(std::vector<uint32_t> a, std::vector<uint32_t> b) {
    std::vector<std::complex<double>> ca(std::begin(a), std::end(a)), cb(std::begin(b), std::end(b));

    fft<false>(ca);
    fft<false>(cb);

    for (uint32_t i = 0; i < n; i++) {
        ca[i] *= cb[i];
    }

    fft<true>(ca);

    std::vector<uint32_t> res(n);
    for (uint32_t i = 0; i < n; i++) {
        res[i] = (ca[i].real() > 0);
    }

    return res;
}

