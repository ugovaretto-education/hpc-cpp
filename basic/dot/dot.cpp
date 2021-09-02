#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
#include <cassert>

#ifdef NUM_THREADS
#if NUM_THREADS > 1
#include <future>
#endif
#endif

#ifdef AVX
#include <x86intrin.h>
#endif

#include "../time/timer.h"
#ifdef ALIGNED
#include "allocators/aligned_allocators.h"
#endif

using namespace std;

#ifndef SIZE
#define SIZE 0x10000000
#endif

#ifndef TYPE
#define TYPE double
#endif

#ifdef NUM_THREADS
#if NUM_THREADS < 1
#error NUM_THREAD < 1!
#endif
#else
#define NUM_THREADS 1
#endif

#ifndef NTRIES
#define NTRIES 10
#endif

#ifdef AVX
float DotProd256FMA(const float* __restrict v1, const float* __restrict v2,
                    size_t len) {
    // assert(len % 8 == 0 && len > 0);
    __m256 s = _mm256_setzero_ps();
    thread_local static float buf[8];
    const size_t upperBound = len - 7;
#ifdef ALIGNED
    for (size_t i = 0; i < upperBound; i += 8) {
        __m256 x1 = _mm256_load_ps(v1 + i);
        __m256 x2 = _mm256_load_ps(v2 + i);
        s = _mm256_fmadd_ps(x1, x2, s);
    }
    _mm256_store_ps(buf, s);
#else
    for (size_t i = 0; i < upperBound; i += 8) {
        __m256 x1 = _mm256_loadu_ps(v1 + i);
        __m256 x2 = _mm256_loadu_ps(v2 + i);
        s = _mm256_fmadd_ps(x1, x2, s);
    }
    _mm256_storeu_ps(buf, s);
#endif
    return buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] +
           buf[7];
}

double DotProd256FMA(const double* __restrict v1, const double* __restrict v2,
                     size_t len) {
    // assert(len % 4 == 0 && len > 0);
    __m256d s = _mm256_setzero_pd();
    thread_local static double buf[4];
    const size_t upperBound = len - 3;
#ifdef ALIGNED
    for (size_t i = 0; i < upperBound; i += 4) {
        __m256d x1 = _mm256_load_pd(v1 + i);
        __m256d x2 = _mm256_load_pd(v2 + i);
        s = _mm256_fmadd_pd(x1, x2, s);
    }
    _mm256_store_pd(buf, s);

#else
    for (size_t i = 0; i < upperBound; i += 4) {
        __m256d x1 = _mm256_loadu_pd(v1 + i);
        __m256d x2 = _mm256_loadu_pd(v2 + i);
        s = _mm256_fmadd_pd(x1, x2, s);
    }
    _mm256_storeu_pd(buf, s);

#endif
    return buf[0] + buf[1] + buf[2] + buf[3];
}
#endif

template <typename T>
struct Result {
    T result;
    double elapsed;
};

template <typename T>
T Dot(const T* __restrict v1, const T* __restrict v2, size_t len) {
#ifndef AVX
    const T result = inner_product(v1, v1 + len, v2, T(0));
#else
    const T result = DotProd256FMA(v1, v2, len);
#endif
    return result;
}

template <typename T>
Result<T> DotProd(const T* __restrict v1, const T* __restrict v2, size_t len) {
    const auto start = Tick();
#if NUM_THREADS == 1
    const T res = Dot(v1, v2, len);
    return {res, Elapsed(start, Tick())};
#else
    const size_t sz = len / NUM_THREADS;
    vector<future<T> > results;
    for (int i = 0; i != NUM_THREADS; ++i) {
        results.push_back(
            async([&, i]() { return Dot(v1 + sz * i, v2 + sz * i, sz); }));
    }
    T res = T(0);
    for_each(begin(results), end(results),
             [&res](future<T>& f) { res += f.get(); });

#endif
    return {res, Elapsed(start, Tick())};
}

int main(int argc, char const* argv[]) {
    using Type = TYPE;
#ifndef ALIGNED
    vector<Type> v1(SIZE, Type(1));
    vector<Type> v2(SIZE, Type(1));
#else
    using Allocator = AlignedAllocator<Type>;
    const size_t alignment = 32;
    vector<Type, Allocator> v1(SIZE, Type(1), Allocator(alignment));
    vector<Type, Allocator> v2(SIZE, Type(1), Allocator(alignment));
#endif
    vector<double> results;
    for (int i = 0; i != NTRIES; ++i) {
        const auto res = DotProd(v1.data(), v2.data(), v1.size());
        assert(res.result == Type(v1.size()));
        results.push_back(res.elapsed);
    }
    const double avg =
        accumulate(++begin(results), end(results), 0.) / (results.size()-1);
    auto mm = minmax_element(begin(results), end(results));
    const int ops = SIZE + SIZE - 1;
#ifdef AVX
    const bool avx = true;
#else
    const bool avx = false;
#endif
#ifdef ALIGNED
    const bool aligned = true;
#else
    const bool aligned = false;
#endif
    cout << "Array length (MiB): " << SIZE / 0x100000 << endl
         << "Precision (bits):   " << sizeof(Type) * 8 << endl
         << "Number of threads:  " << NUM_THREADS << endl
         << "Min (s):            " << *mm.first << endl
         << "Max (s):            " << *mm.second << endl
         << "Average (s):        " << avg << endl
         << "Peak GFLOPS:        " << 1E-9 * double(ops) / *mm.first << endl
         << "Bytes/flop:         " << double(SIZE * sizeof(Type)) / ops << endl
         << "AVX 256:            " << boolalpha << avx << endl
         << "Aligned (32 bytes): " << aligned << endl;
    return 0;
}
