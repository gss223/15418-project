#include <iostream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include "timer.h"

bool solve_fft(const std::vector<uint32_t>&w, uint32_t T);
//void printCudaInfo();

int main() {
    std::ios_base::sync_with_stdio(false);

    int N, T;
    std::cin >> N >> T;

    std::vector<uint32_t> w(N);
    for (uint32_t& x : w) {
        std::cin >> x;
    }

    Timer timer;
    timer.start();

    bool is_possible = solve_fft(w, T);

    timer.end();

    std::cout << is_possible << '\n' << std::fixed << std::setprecision(10) << (timer.get_duration<std::chrono::microseconds>() / 1e6) << '\n';
}
