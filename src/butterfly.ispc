export void butterfly(uniform float re[], uniform float im[], uniform float w_re[], uniform float w_im[], uniform const uint32 half_len) {
    foreach (i = 0 ... half_len) {
        float a_re = re[i];
        float a_im = im[i];

        float b_re = re[i + half_len] * w_re[i] - im[i + half_len] * w_im[i];
        float b_im = re[i + half_len] * w_im[i] + im[i + half_len] * w_re[i];

        re[i] = a_re + b_re;
        im[i] = a_im + b_im;

        re[i + half_len] = a_re - b_re;
        im[i + half_len] = a_im - b_im;
    }
}

export void multiply(uniform float a_re[], uniform float a_im[], uniform const float b_re[], uniform const float b_im[], uniform const uint32 len) {
    foreach (i = 0 ... len) {
        float new_re = a_re[i] * b_re[i] - a_im[i] * b_im[i];
        float new_im = a_re[i] * b_im[i] + a_im[i] * b_re[i];

        a_re[i] = new_re;
        a_im[i] = new_im;
    }
}
