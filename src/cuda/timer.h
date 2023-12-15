#include <chrono>

class Timer {
    public:
        void start() {
            start_timestamp = std::chrono::steady_clock::now();
        }

        void end() {
            end_timestamp = std::chrono::steady_clock::now();
        }

        template<class ToDuration> double get_duration() {
            return std::chrono::duration_cast<ToDuration>(end_timestamp - start_timestamp).count();
        }

    private:
        std::chrono::time_point<std::chrono::steady_clock> start_timestamp;
        std::chrono::time_point<std::chrono::steady_clock> end_timestamp;
};
