#pragma once

#include <chrono>

using TimePoint = decltype(std::chrono::high_resolution_clock::now());

inline auto Tick() { return std::chrono::high_resolution_clock::now(); }

// seconds, nanosecond resolution
inline double Elapsed(const TimePoint& start, const TimePoint& end) {
    return std::chrono::duration<double>(end - start).count();
}

struct To {
    static constexpr auto us(double s) { return s  * 10E3; }
    static constexpr auto ms(double s) { return s  * 10E6; }
    static constexpr auto s(double ts) { return ts * 10E9; }
};