export void get_subset_sums(uniform const int w[], uniform const int T, uniform const int l, uniform const int r, uniform int sums[]) {
    uniform const int len = r - l;
    uniform const int num_masks = 1 << len;

    foreach (mask = 0 ... num_masks) {
        for (int i = 0; i < len; i++) {
            if (mask & (1 << i)) {
                sums[mask] += w[l + i];
            }
        }
    }
}
