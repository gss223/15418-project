#include <array>
#include <complex>
#include <vector>

#include "immintrin.h"

#include "convolution.h"
#include "utils.h"

std::vector<float*> w_re, w_im, w_inv_re, w_inv_im;
std::vector<uint32_t> reversed;
std::vector<std::array<uint32_t, 2>> swaps;
uint32_t result_len, n, half;

alignas(32) float buf[1 << 28];
float* ptr = buf;

float* alloc(uint32_t sz) {
    float* res = ptr;
    ptr += sz;

    return res;
}

void conv_init(const uint32_t T) {
    result_len = T + 1;
    n = bit_ceil(2 * T);
    half = n >> 1;

    const int bit_count = std::__lg(n) + 1;

    w_re.resize(bit_count);
    w_im.resize(bit_count);
    w_inv_re.resize(bit_count);
    w_inv_im.resize(bit_count);

    w_re.back() = alloc(n);
    w_im.back() = alloc(n);
    w_inv_re.back() = alloc(n);
    w_inv_im.back() = alloc(n);

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
        w_re[i] = alloc(cnt);
        w_im[i] = alloc(cnt);
        w_inv_re[i] = alloc(cnt);
        w_inv_im[i] = alloc(cnt);

#pragma omp parallel for
        for (int j = 0; j < cnt; j++) {
            w_re[i][j] = w_re[i + 1][2 * j];
            w_im[i][j] = w_im[i + 1][2 * j];
            w_inv_re[i][j] = w_inv_re[i + 1][2 * j];
            w_inv_im[i][j] = w_inv_im[i + 1][2 * j];
        }
    }
}

// Butterfly on scalars
void butterfly(float* re, float* im, const float* w_re, const float* w_im, const uint32_t half_len) {
    float u_re = *re;
    float u_im = *im;

    float v_re = re[half_len] * *w_re - im[half_len] * *w_im;
    float v_im = re[half_len] * *w_im + im[half_len] * *w_re;

    *re = u_re + v_re;
    *im = u_im + v_im;

    re[half_len] = u_re - v_re;
    im[half_len] = u_im - v_im;
}

// Butterfly on 8 values
void butterfly8(float* re, float* im, const float* w_re, const float* w_im) {
    __m128 u_re = _mm_load_ps(re);
    __m128 u_im = _mm_load_ps(im);
    __m128 c = _mm_load_ps(re + 4);
    __m128 d = _mm_load_ps(im + 4);
    __m128 w_re_vec = _mm_load_ps(w_re);
    __m128 w_im_vec = _mm_load_ps(w_im);

    __m128 v_re = _mm_sub_ps(_mm_mul_ps(c, w_re_vec), _mm_mul_ps(d, w_im_vec));
    __m128 v_im = _mm_add_ps(_mm_mul_ps(c, w_im_vec), _mm_mul_ps(d, w_re_vec));

    _mm_store_ps(re, _mm_add_ps(u_re, v_re));
    _mm_store_ps(im, _mm_add_ps(u_im, v_im));
    _mm_store_ps(re + 4, _mm_sub_ps(u_re, v_re));
    _mm_store_ps(im + 4, _mm_sub_ps(u_im, v_im));
}

// Butterfly on 16 values
void butterfly16(float* re, float* im, const float* w_re, const float* w_im, const uint32_t half_len) {
    __m256 u_re = _mm256_load_ps(re);
    __m256 u_im = _mm256_load_ps(im);
    __m256 c = _mm256_load_ps(re + half_len);
    __m256 d = _mm256_load_ps(im + half_len);
    __m256 w_re_vec = _mm256_load_ps(w_re);
    __m256 w_im_vec = _mm256_load_ps(w_im);

    __m256 v_re = _mm256_sub_ps(_mm256_mul_ps(c, w_re_vec), _mm256_mul_ps(d, w_im_vec));
    __m256 v_im = _mm256_add_ps(_mm256_mul_ps(c, w_im_vec), _mm256_mul_ps(d, w_re_vec));

    _mm256_store_ps(re, _mm256_add_ps(u_re, v_re));
    _mm256_store_ps(im, _mm256_add_ps(u_im, v_im));
    _mm256_store_ps(re + half_len, _mm256_sub_ps(u_re, v_re));
    _mm256_store_ps(im + half_len, _mm256_sub_ps(u_im, v_im));
}

template<bool inverse>
void fft_iterative(float* re, float* im) {
#pragma omp parallel for
    for (const auto& [i, j] : swaps) {
        std::swap(re[i], re[j]);
        std::swap(im[i], im[j]);
    }

    for (uint32_t chunk_len = 2, half_len = 1, layer = 1; chunk_len <= 4; chunk_len *= 2, half_len *= 2, layer++) {
#pragma omp parallel for
        for (uint32_t chunk_start = 0; chunk_start < n; chunk_start += chunk_len) {
            for (uint32_t i = 0; i < half_len; i++) {
                if constexpr (inverse) {
                    butterfly(re + chunk_start + i, im + chunk_start + i, w_inv_re[layer] + i, w_inv_im[layer] + i, half_len);
                } else {
                    butterfly(re + chunk_start + i, im + chunk_start + i, w_re[layer] + i, w_im[layer] + i, half_len);
                }
            }
        }
    }

#pragma omp parallel for
    for (uint32_t chunk_start = 0; chunk_start < n; chunk_start += 8) {
        if constexpr (inverse) {
            butterfly8(re, im, w_inv_re[3], w_inv_im[3]);
        } else {
            butterfly8(re, im, w_re[3], w_im[3]);
        }
    }

    for (uint32_t chunk_len = 16, half_len = 8, layer = 4; chunk_len <= n; chunk_len *= 2, half_len *= 2, layer++) {
#pragma omp parallel for
        for (uint32_t chunk_start = 0; chunk_start < n; chunk_start += chunk_len) {
            for (uint32_t i = 0; i < half_len; i += 8) {
                if constexpr (inverse) {
                    butterfly16(re + chunk_start + i, im + chunk_start + i, w_inv_re[layer] + i, w_inv_im[layer] + i, half_len);
                } else {
                    butterfly16(re + chunk_start + i, im + chunk_start + i, w_re[layer] + i, w_im[layer] + i, half_len);
                }
            }
        }
    }
}

void multiply(float* p_re, float* p_im, const float* q_re, const float* q_im) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < result_len; i += 8) {
        __m256 a = _mm256_load_ps(p_re + i);
        __m256 b = _mm256_load_ps(p_im + i);

        __m256 c = _mm256_load_ps(q_re + i);
        __m256 d = _mm256_load_ps(q_im + i);

        // multiply the complex numbers (a + bi) and (c + di)

        __m256 ac = _mm256_mul_ps(a, c);
        __m256 bd = _mm256_mul_ps(b, d);
        __m256 ad = _mm256_mul_ps(a, d);
        __m256 bc = _mm256_mul_ps(b, c);

        _mm256_store_ps(p_re + i, _mm256_sub_ps(ac, bd));
        _mm256_store_ps(p_im + i, _mm256_add_ps(ad, bc));
    }
}

std::vector<uint32_t> conv(std::vector<uint32_t> p, std::vector<uint32_t> q) {
    if (std::empty(q)) {
        return p;
    }

    float* p_re = alloc(n);
    float* p_im = alloc(n);
    float* q_re = alloc(n);
    float* q_im = alloc(n);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < result_len; i++) {
        p_re[i] = p[i];
        q_re[i] = q[i];
    }

    fft_iterative<false>(p_re, p_im);
    fft_iterative<false>(q_re, q_im);

    multiply(p_re, p_im, q_re, q_im);

    fft_iterative<true>(p_re, p_im);

    std::vector<uint32_t> res(result_len);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < result_len; i++) {
        res[i] = (p_re[i] >= half);
    }

    return res;
}

