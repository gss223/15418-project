#include <iostream>
#include <iomanip>

#include "naive.h"
#include "timer.h"

int main() {
    int N, T;
    std::cin >> N >> T;

    std::vector<int> w(N);
    for (int& x : w) {
        std::cin >> x;
    }

    Timer timer;
    timer.start();

    bool is_possible = solve_naive(w, T);

    timer.end();

    std::cout << is_possible << '\n' << std::fixed << std::setprecision(10) << (timer.get_duration<std::chrono::microseconds>() / 1e6) << '\n';
}
