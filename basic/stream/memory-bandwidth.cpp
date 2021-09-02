//Copyright © 2013-2021, Ugo Varetto
//All rights reserved.
//
//This source code is licensed under the BSD-style license found in the
//LICENSE file in the root directory of this source tree. 

// Stream-like benchmark, support for page-locked and aligned memory, MiB/s or
// MB/s output, c++11 threads.
// C++17 required.
//
// Author: Ugo Varetto
//
// Note #1:
// Similar to the standard stream benchmark, sepending on the size of the 
// static C arrays, the following array might be reported:
//   relocation truncated to fit: R_X86_64_PC32 against `.bss'
// which is fixed by setting -mcmodel=large when -DSTATIC_ARRAYS is specified
// Note #2: removed all the custom copying code (AVX+unroll+prefetch etc.) since
// no noticeable performance benefit displayed compared to std::copy
//
// Configuration parameters:
//
// -D STREAM_SIZE= number of array elements.
//
// -D STREAM_TYPE= array element type, default is double.
//
// -D STATIC_ARRAYS use static C arrays (no noticeable difference from
// std::vectors).
//
// -D NTIMES= number or runs (first run not benchmarked).
//
// -D ALIGNED allocate aligned memory (default 64 bytes).
//
// -D ALIGN= specify alignment.
//
// -D NUM_THREADS= number of concurrent threads.
//
// -D PINNED use page locked memory from storage.
//
// -D CONSUME_DATA consume generated data by writing to file in case optimizer
// detects that data is not used   

#include <algorithm>
#include <cassert>
#include <cstring>
#ifdef CONSUME_DATA
#include <fstream>
#endif
#if NUM_THREADS > 1
#include <future>
#endif
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#if PINNED || ALIGNED
#include "../allocators/aligned_allocators.h"
#endif
#include "../timer.h"
using namespace std;

//------------------------------------------------------------------------------
// configure everything at compile time

// array size
#ifndef STREAM_SIZE
#define STREAM_SIZE 0x1000000 //128MB
#endif

// number of runs
#ifndef NTIME
#define NTIMES 10
#endif

// memory alignment
#ifdef ALIGNED
#ifndef ALIGN
#define ALIGN 64
#endif
#endif

// data type
#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif

#ifdef CONSUME_DATA
// file to write to, to consume data, in case compiler optimises away the code
#ifndef NULL_FILE
#define NULL_FILE "/dev/null"
#endif
#endif

// number of threads
#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif

// divide bytes by this number (default 1 MiB)
#ifndef DIVIDER
#define DIVIDER 0x100000
#endif

//------------------------------------------------------------------------------
template <typename T>
double Copy(const T* __restrict src, T* __restrict dest, size_t size,
            int numThreads) {
    assert(numThreads >= 1);
    if (numThreads == 1) {
        const auto start = Tick();
        // will automatically call __memcpy_avx_unaligned and/or vectorise as
        // required
        copy(src, src + size, dest);
        return Elapsed(start, Tick()) / 2;
    } else {
#if NUM_THREADS > 1
        vector<future<void>> f;
        f.reserve(numThreads);
        const auto start = Tick();
        const size_t sz = size / numThreads;
        for (int t = 0; t != numThreads; ++t) {
            f.push_back(async([&, t]() {
                copy(src + t * sz, src + t * sz + sz, dest + t * sz);
            }));
        }
        for_each(begin(f), end(f), [](future<void>& v) { v.wait(); });
        return Elapsed(start, Tick()) / 2;
#endif
    }
    return 0.;
}

template <typename T>
double Scale(const T* __restrict src, T* __restrict dest, size_t size, T s,
             int numThreads) {
    assert(numThreads >= 1);
    if (numThreads == 1) {
        const auto start = Tick();
        const double* b = src;
        for (; b != src + size; ++b, ++dest) {
            *dest = *b * s;
        }
        return Elapsed(start, Tick()) / 2;
    } else {
#if NUM_THREADS > 1
        vector<future<void>> f;
        f.reserve(numThreads);
        const auto start = Tick();
        const size_t sz = size / numThreads;
        for (int t = 0; t != numThreads; ++t) {
            f.push_back(async([&, t]() {
                const double* b = src + t * sz;
                double* d = dest + t * sz;
                for (; b != src + t * sz + sz; ++b, ++d) {
                    *d = *b * s;
                }
            }));
        }
        for_each(begin(f), end(f), [](future<void>& v) { v.wait(); });
        return Elapsed(start, Tick()) / 2;
#endif
    }
    return 0.;
}

template <typename T>
double Add(const T* __restrict v1, const T* __restrict v2, T* v3, size_t size,
           int numThreads = 1) {
    if (numThreads == 1) {
        const auto start = Tick();
        const T* a = v1;
        const T* b = v2;
        T* c = v3;
        for (; a != v1 + size; ++a, ++b, ++c) {
            *c = *a + *b;
        }
        return Elapsed(start, Tick()) / 3;
    } else {
#if NUM_THREADS > 1
        vector<future<void>> f;
        f.reserve(numThreads);
        const auto start = Tick();
        const size_t sz = size / numThreads;
        for (int t = 0; t != numThreads; ++t) {
            f.push_back(async([&, t]() {
                const T* a = v1 + t * sz;
                const T* b = v2 + t * sz;
                T* c = v3 + t * sz;
                for (; a != v1 + t * sz + sz; ++a, ++b, ++c) {
                    *c = *a + *b;
                }
            }));
        }
        for_each(begin(f), end(f), [](future<void>& v) { v.wait(); });
        return Elapsed(start, Tick()) / 3;
#endif
    }
    return 0.;
}

template <typename T>
double Triad(const T* __restrict v1, const T* __restrict v2, T* v3, size_t size,
             T s, int numThreads) {
    assert(numThreads >= 1);
    if (numThreads == 1) {
        const auto start = Tick();
        const T* a = v1;
        const T* b = v2;
        T* c = v3;
        for (; a != v1 + size; ++a, ++b, ++c) {
            *c = *a + *b * s;
        }
        return Elapsed(start, Tick()) / 3;
    } else {
#if NUM_THREADS > 1
        vector<future<void>> f;
        f.reserve(numThreads);
        const auto start = Tick();
        const size_t sz = size / numThreads;
        for (int t = 0; t != numThreads; ++t) {
            f.push_back(async([&, t]() {
                const T* a = v1 + t * sz;
                const T* b = v2 + t * sz;
                T* c = v3 + t * sz;
                for (; a != v1 + t * sz + sz; ++a, ++b, ++c) {
                    *c = *a + *b * s;
                }
            }));
        }
        for_each(begin(f), end(f), [](future<void>& v) { v.wait(); });
        return Elapsed(start, Tick()) / 3;
#endif
    }
    return 0.;
}

/// Bandwidth in MB/s or MiB/s
constexpr double BWMB(double t, size_t bytes, size_t divider) {
    assert(t > 0.);
    return (double(bytes) / divider) / t;
}

//------------------------------------------------------------------------------
struct Result {
    double min;  // min bandwidth
    double max;  // max bandwidth
    double avg;  // average bandwidth
};

using Results = map<string, Result>;

//------------------------------------------------------------------------------
Results GenResults(const map<string, vector<double>>& bws) {
    Results res;
    // discard firs result
    for (const auto& [name, v] : bws) {
        auto mm = minmax_element(++begin(v), end(v));
        auto avg = accumulate(++begin(v), end(v), 0.) / (v.size() - 1);
        res[name] = {*mm.first, *mm.second, avg};
    }
    return res;
}

//------------------------------------------------------------------------------
string Unit(size_t divider) {
    if (divider % size_t(1E6) == 0)
        return "MB";
    else if (divider % 0x100000 == 0)
        return "MiB";
    else if (divider % 0x40000000 == 0)
        return "GiB";
    else if (divider % size_t(0x1E9) == 0)
        return "GB";
    else
        return "";
}

//------------------------------------------------------------------------------
template <typename OsT>
OsT& Print(const Results& results, size_t size, int numThreads, size_t align,
           bool pinned, size_t divider, OsT& os) {
    os << "Bandwidth test (" << Unit(divider) << "/s)" << endl;
    os << "Function\t"
       << "Max\t"
       << "Avg\t"
       << "Min\t" << endl
       << endl;
    for_each(begin(results), end(results),
             [&os](const std::pair<std::string, Result>& e) {
                 os << e.first << "\t\t" << e.second.max << '\t' << e.second.avg
                    << '\t' << e.second.min << endl;
             });
    os << endl;
    os << "Size (MiB):\t" << size / 0x100000 << endl
       << "Num threads:\t" << numThreads << endl
       << "Alignment:\t" << align << endl
       << "Page locked:\t" << boolalpha << pinned << endl;
    return os;
}

//------------------------------------------------------------------------------
template <typename OsT>
OsT& PrintCSV(const Results& results, size_t size, int numThreads, size_t align,
              bool pinned, size_t divider, OsT& os) {
    constexpr auto sep = ',';
    for_each(begin(results), end(results),
             [&os, divider](const std::pair<std::string, Result>& e) {
                 os << e.first << sep << e.second.max << sep << e.second.avg
                    << sep << e.second.min << sep << divider << endl;
             });
    os << size << sep << numThreads << sep << align << sep << pinned << endl;
    return os;
}

//------------------------------------------------------------------------------
int main(int argc, char const* argv[]) {
    // if a command line parameter passed CSV output is enabled
    const bool CSV = argc > 1;
    const double scalar = 3;
    using Type = STREAM_TYPE;
    const size_t divider = size_t(DIVIDER);

#ifdef PINNED
#ifdef STATIC_ARRAYS
#warning PINNED selected, STATIC_ARRAYS disabled
#undef STATIC_ARRAYS
#endif
#ifdef ALIGNED
#warning PINNED selected, ALIGNED disabled
#undef ALIGNED
#endif
    const bool pinned = true;
#else
    const bool pinned = false;
#endif

#ifdef ALIGNED
#ifdef STATIC_ARRAYS
#warning ALIGNED selected, STATIC_ARRAYS disabled
#undef STATIC_ARRAYS
#endif
    const size_t align = ALIGN;
#else
#ifndef PINNED
    const size_t align = alignment_of<Type>::value;
#else
    const size_t align = getpagesize();
#endif
#endif

#if NUM_THREADS == 1
    const size_t SIZE = STREAM_SIZE;
    const int numThreads = 1;
#else
#ifndef STATIC_ARRAYS
    const int numThreads =
        NUM_THREADS ? NUM_THREADS : std::thread::hardware_concurrency();
#else
    constexpr int numThreads = NUM_THREADS ? NUM_THREADS : 8;
#endif
    // resize to make buffer divisible by number of threads:
    // first number <= SIZE divisible by num_threads
    const size_t SIZE = (STREAM_SIZE / numThreads) * numThreads;
#endif
#ifdef ALIGNED
    vector<Type, AlignedAllocator<Type>> b(SIZE, AlignedAllocator<Type>(ALIGN));
    vector<Type, AlignedAllocator<Type>> a(SIZE, AlignedAllocator<Type>(ALIGN));
    vector<Type, AlignedAllocator<Type>> c(SIZE, AlignedAllocator<Type>(ALIGN));
#elif PINNED
    vector<Type, PinnedAllocator<Type>> a(SIZE);
    vector<Type, PinnedAllocator<Type>> b(SIZE);
    vector<Type, PinnedAllocator<Type>> c(SIZE);
#else
#ifdef STATIC_ARRAYS
    static Type a[SIZE];
    static Type b[SIZE];
    static Type c[SIZE];
    fill(begin(a), end(a), 1.0);
    fill(begin(b), end(b), 2.0);
    fill(begin(c), end(c), 0.0);
#else
    vector<Type> a(SIZE, 1.0);
    vector<Type> b(SIZE, 2.0);
    vector<Type> c(SIZE);
#endif

#endif
    const size_t byteSize = SIZE * sizeof(Type);
    map<string, vector<double>> bws;
    for (size_t i = 0; i != NTIMES; ++i) {
        bws["Copy"].push_back(
            BWMB(Copy(&a[0], &c[0], SIZE, numThreads), byteSize, divider));
        bws["Scale"].push_back(BWMB(
            Scale(&a[0], &c[0], SIZE, scalar, numThreads), byteSize, divider));
        bws["Add"].push_back(BWMB(Add(&a[0], &b[0], &c[0], SIZE, numThreads),
                                  byteSize, divider));
        bws["Triad"].push_back(
            BWMB(Triad(&a[0], &b[0], &c[0], SIZE, scalar, numThreads), byteSize,
                 divider));
    }
    auto results = GenResults(bws);

    if (CSV)
        PrintCSV(results, byteSize, numThreads, align, pinned, divider, cout);
    else
        Print(results, byteSize, numThreads, align, pinned, divider, cout);

    // prevent optimisation if needed by consuming data: write to file
#ifdef CONSUME_DATA
    ofstream os(NULL_FILE);
    os << b[SIZE-1] << c[0];
#endif
    return 0;
}
