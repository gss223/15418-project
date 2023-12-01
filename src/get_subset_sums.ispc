export void get_subset_sums(uniform const uint32 w[], uniform const uint32 T, uniform const uint32 l, uniform const uint32 r, uniform uint32 sums[]) {
    uniform const uint32 len = r - l;
    uniform const uint32 num_masks = 1 << len;

    foreach (mask = 0 ... num_masks) {
        for (uint32 i = 0; i < len; i++) {
            if (mask & (1 << i)) {
                sums[mask] += w[l + i];
            }
        }
    }
}
