#include <iostream>
#include <fstream>
#include <random>

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cout << "Usage: " << argv[0] << " <number of elements> <element value upper bound> <all even> <target sum> <output file path> \n";

        return 0;
    }

    std::mt19937 rng(2023);
    const int N = atoi(argv[1]), C = atoi(argv[2]), all_even = atoi(argv[3]), T = atoi(argv[4]);

    std::ofstream fout(argv[5]);

    fout << N << ' ' << T << '\n';
    for (int i = 0; i < N; i++) {
        if (all_even) {
            fout << std::uniform_int_distribution<int>(1, C / 2)(rng) * 2 << ' ';
        } else {
            fout << std::uniform_int_distribution<int>(1, C)(rng) << ' ';
        }
    }
    fout << '\n';
}
