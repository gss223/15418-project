#include <array>
#include <complex>
#include <vector>

#include "convolution.h"
#include "butterfly.h"
#include "utils.h"

std::vector<std::vector<float>> w_re, w_im, w_inv_re, w_inv_im;
std::vector<uint32_t> reversed;
std::vector<std::array<uint32_t, 2>> swaps;
uint32_t result_len, n, half;

void conv_init(const uint32_t T) {
    result_len = T + 1;
    n = bit_ceil(2 * T);
    half = n >> 1;

    const int bit_count = std::__lg(n) + 1;

    w_re.resize(bit_count);
    w_im.resize(bit_count);
    w_inv_re.resize(bit_count);
    w_inv_im.resize(bit_count);

    w_re.back().resize(n);
    w_im.back().resize(n);
    w_inv_re.back().resize(n);
    w_inv_im.back().resize(n);

    reversed.resize(n);


#pragma omp parallel for
    for (uint32_t i = 0; i < n; i++) {
        reversed[i] = reverse_bits(i, bit_count - 1);
    }

    for (uint32_t i = 0; i < n; i++) {
        if (i < reversed[i]) {
            swaps.push_back({i, reversed[i]});
        }
    }
    
    const double theta = 2 * M_PI / n;

#pragma omp parallel for
    for (uint32_t i = 0; i < n; i++) {
        auto pt = std::polar<float>(1, theta * i);

        w_re.back()[i] = pt.real();
        w_im.back()[i] = pt.imag();

        w_inv_re.back()[i] = w_re.back()[i];
        w_inv_im.back()[i] = -w_im.back()[i];
    }
    
    for (int i = bit_count - 2, cnt = half; i >= 1; i--, cnt >>= 1) {
        w_re[i].resize(cnt);
        w_im[i].resize(cnt);
        w_inv_re[i].resize(cnt);
        w_inv_im[i].resize(cnt);

#pragma omp parallel for
        for (int j = 0; j < cnt; j++) {
            w_re[i][j] = w_re[i + 1][2 * j];
            w_im[i][j] = w_im[i + 1][2 * j];
            w_inv_re[i][j] = w_inv_re[i + 1][2 * j];
            w_inv_im[i][j] = w_inv_im[i + 1][2 * j];
        }
    }
}

template<bool inverse>
void fft_iterative(std::vector<float>& re, std::vector<float>& im) {
#pragma omp parallel for
    for (const auto& [i, j] : swaps) {
        std::swap(re[i], re[j]);
        std::swap(im[i], im[j]);
    }

    for (uint32_t chunk_len = 2, half_len = 1, layer = 1; chunk_len <= n; chunk_len *= 2, half_len *= 2, layer++) {
#pragma omp parallel for
        for (uint32_t chunk_start = 0; chunk_start < n; chunk_start += chunk_len) {
            if constexpr (inverse) {
                ispc::butterfly(re.data() + chunk_start, im.data() + chunk_start, w_inv_re[layer].data(), w_inv_im[layer].data(), half_len);
            } else {
                ispc::butterfly(re.data() + chunk_start, im.data() + chunk_start, w_re[layer].data(), w_im[layer].data(), half_len);
            }
        }
    }
}

std::vector<uint32_t> conv(std::vector<uint32_t> p, std::vector<uint32_t> q) {
    if (std::empty(q)) {
        return p;
    }

    std::vector<float> p_re(n), p_im(n), q_re(n), q_im(n);
    std::copy(std::begin(p), std::end(p), std::begin(p_re));
    std::copy(std::begin(q), std::end(q), std::begin(q_re));

    fft_iterative<false>(p_re, p_im);
    fft_iterative<false>(q_re, q_im);

    ispc::multiply(p_re.data(), p_im.data(), q_re.data(), q_im.data(), n);

    fft_iterative<true>(p_re, p_im);

    std::vector<uint32_t> res(result_len);
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < result_len; i++) {
        res[i] = (p_re[i] >= half);
    }

    return res;
}

