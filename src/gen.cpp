#include <iostream>
#include <fstream>
#include <random>

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " <number of elements> <element value upper bound> <target sum> <output file path> \n";

        return 0;
    }

    std::mt19937 rng(2023);
    const int N = atoi(argv[1]), C = atoi(argv[2]), T = atoi(argv[3]);

    std::ofstream fout(argv[4]);

    fout << N << ' ' << T << '\n';
    for (int i = 0; i < N; i++) {
        fout << std::uniform_int_distribution<int>(0, C)(rng) << ' ';
    }
    fout << '\n';
}
