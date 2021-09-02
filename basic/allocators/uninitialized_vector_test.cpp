// https://quick-bench.com/q/j1bvsXHGw1mRsqIRRJ-H2YqYXgc
// Standard std::vector initialization is slow because of the requirement
// of default initializing elements; using calloc greatly improves performance
//
// what slows done evreything is this:
// explicit
//      vector(size_type __n, const allocator_type& __a = allocator_type())
//      : _Base(_S_check_init_len(__n, __a), __a)
//      { /*_M_default_initialize(__n);*/ } // <-- !!!
// inside
// /usr/include/c++/10/bits/stl_vector.h
// commenting it gives orders of magnitude improvement in performance, similar
// to a C array; calloc default initializes data with performance similar to raw 
//               malloc
// Options to make it faster:
// 1) do not use std::vector
// 2) wrap type into type with empty constructor - issue:
//             std::vector<Uninitialized<T>> not convertible to
//             std::vector<T>, references and pointers can however be converted
//             through reinterpret_cast given the same memory layout of T and
//             Uninitialized<T>
// 3) [preferred] implement custom allocator which uses malloc/calloc
//    internally; using calloc achieves performance similar to malloc with
//    default initialized elements - no issues
//
// performance depends on execution order! Best way to test is to test
// each allocation method in isolation

#include <malloc.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "raw_allocators.h"
#include "timer.h"

using namespace std;

//------------------------------------------------------------------------------
// Full wrappper for numerical types with empty constructor
// 0x100000 x floats:
// malloc/new = 86 NOP
// Unininitialzed = 95 x NOP
// std::vector 5400 x NOP!
template <typename T>
struct Uninitialized {
    T data;  //& address of is same as data;
    Uninitialized() {}
    // Uninitialized(T) {}
    T* operator&() { return &data; }
    const T* operator&() const { return &data; }
    T& operator*() { return data; }
    const T& operator*() const { return data; }
    auto operator>(T other) const { return data > other; }
    auto operator>=(T other) const { return data >= other; }
    auto operator<(T other) const { return data > other; }
    auto operator<=(T other) const { return data >= other; }
    auto operator!=(T other) const { return data != other; }
    auto operator!() const { return !data; }
    auto operator+(T other) const { return data + other; }
    auto operator+=(T other) { return data += other; }
    auto operator-(T other) const { return data - other; }
    auto operator-=(T other) { return data -= other; }
    auto operator*(T other) const { return data * other; }
    auto operator*=(T other) { return data *= other; }
    auto operator/(T other) const { return data / other; }
    auto operator/=(T other) { return data /= other; }
    auto operator&(T other) const { return data & other; }
    auto operator&=(T other) { return data &= other; }
    auto operator|(T other) const { return data | other; }
    auto operator|=(T other) { return data |= other; }
    auto operator^(T other) const { return data ^ other; }
    auto operator^=(T other) { return data ^= other; }
    auto operator>>(int i) const { return data >> i; }
    auto operator>>=(int i) { return data >>= i; }
    auto operator<<(int i) const { return data << i; }
    auto operator<<=(int i) { return data <<= i; }
    auto operator++() { return data++; }
    auto operator++(int) { return ++data; }
    auto operator--() { return data--; }
    auto operator--(int) { return --data; }
    auto operator=(const T& d) {
        data = d;
        return data;
    }
    operator T() const { return data; }
    operator T&() { return data; }
    // operator const T&() const { return data; }
    template <typename U>
    operator U() {
        return U(data);
    }
};

template <typename T>
std::istream& operator>>(std::istream& is, Uninitialized<T>& d) {
    T i;
    is >> i;
    d.data = i;
    return is;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, Uninitialized<T> d) {
    os << d.data;
    return os;
}
//------------------------------------------------------------------------------

struct Large {
    int data[128];
    Large() {
        for (int i = 0; i != 128; ++i) data[i] = i;
    }
};

// Type wrapper: store object into internal byte array
template <typename T>
struct Wrapper {
    uint8_t data[sizeof(T)];
    Wrapper() {}
    Wrapper(const T& v) { new (data) T(v); }
    void Init() { new (this) T(); }
    template <typename... ArgsT>
    void Init(ArgsT&&... args) {
        new (this) T(std::forward<args>...);
    }
    operator T() const { *((const T*)data); }
    operator const T&() const { *((const T*)data); }
    operator T&() { *((const T*)data); }
};

template <typename T>
std::vector<T> uvec_to_vec(const std::vector<Uninitialized<T>>& v) {
    return std::vector<T>(begin(v), end(v));
}

//------------------------------------------------------------------------------
int main(int argc, char const* argv[]) {
    
    constexpr size_t SIZE = 0x1000000;  // 16 Mx
    cout << "Size: " << (SIZE * sizeof(int)) / 0x100000 << " MB" << endl;

    //--------------------------------------------------------------------------

    // malloc
    auto start3 = Tick();
    int* vi3 = (int*)malloc(SIZE * sizeof(int));
    vi3[SIZE - 1] = 1;
    vi3[0] = 1;
    const int vi3Out = vi3[SIZE - 1] - vi3[0];  // touch memory
    cout << "malloc (s): " << Elapsed(start3, Tick()) << endl;
    // free
    auto start4 = Tick();
    free(vi3);
    cout << "free (s): " << Elapsed(start4, Tick()) << endl;

    //--------------------------------------------------------------------------

    // std::vector
    auto start = Tick();
    vector<int> vi(SIZE);
    vi[SIZE - 1] = 0;
    const int viOut = vi.back();
    cout << "vector<int> (s): " << Elapsed(start, Tick()) << endl;
    // std::vector resize + shrink_to_fit
    auto start6 = Tick();
    vi.resize(0);
    vi.shrink_to_fit();
    cout << "vector<int>.resize(0) (s): " << Elapsed(start6, Tick()) << endl;

    //--------------------------------------------------------------------------

    // std::vector custom allocator - default initialized
    auto startci = Tick();
    vector<int, RawAllocator<int>> vici(SIZE, RawAllocator<int>(true));
    // vic[SIZE-1] = 0;
    const int viciOut = vici[0];
    cout << "vector<int> custom allocator default initialized (s): "
         << Elapsed(startci, Tick()) << endl;
    // std::vector resize + shrink_to_fit custom allocator - default initialized
    auto startci2 = Tick();
    vici.resize(0);
    vici.shrink_to_fit();
    cout << "vector<int>.resize(0) custom allocator default initialized (s): "
         << Elapsed(startci2, Tick()) << endl;

    //--------------------------------------------------------------------------

    // std::vector custom allocator
    auto startc = Tick();
    vector<int, RawAllocator<int>> vic(SIZE, RawAllocator<int>(false, 0, true));
    // vic[SIZE-1] = 0;
    const int vicOut = vic[0];
    cout << "vector<int> custom allocator (s): " << Elapsed(startc, Tick())
         << endl;
    // std::vector resize + shrink_to_fit custom allocator
    auto startc2 = Tick();
    vic.resize(0);
    vic.shrink_to_fit();
    cout << "vector<int>.resize(0) custom allocator (s): "
         << Elapsed(startc2, Tick()) << endl;

    //--------------------------------------------------------------------------

    // std::vector<Uninitialized>
    auto start2 = Tick();
    vector<Uninitialized<int>> vi2(SIZE);
    vi2[SIZE - 1] = 0;
    const int vi2Out = vi2.back();
    cout << "vector<Uninitialized<int>> (s): " << Elapsed(start2, Tick())
         << endl;
    // std::vector<Uninitialized> resize + shrink_to_fit
    auto start5 = Tick();
    vi2.resize(0);
    vi2.shrink_to_fit();
    cout << "vector<Uninitialized<int>>.resize(0) (s): "
         << Elapsed(start5, Tick()) << endl;

    //--------------------------------------------------------------------------

    // std::vector (large object)
    auto start7 = Tick();
    vector<Large> vh(SIZE);
    vh[SIZE - 1] = Large();
    const int vhOut = vh.back().data[0];
    cout << "vector<Large> (s): " << Elapsed(start7, Tick()) << endl;
    // std::vector<Uninitialzed> large object
    auto start8 = Tick();
    vector<Wrapper<Large>> vh1(SIZE);
    vh1[SIZE - 1] = Large();
    const int vh1Out = vh1.back().data[0];
    cout << "vector<Wrapper<Large>> (s): " << Elapsed(start8, Tick()) << endl;
    // std::vector (large object) custom allocator
    auto starthc = Tick();
    vector<Large, LazyAllocator<Large>> vhc(SIZE);
    vhc[SIZE - 1] = Large();
    const int vhcOut = vhc.back().data[0];
    cout << "vector<Large> custom allocator (s): " << Elapsed(starthc, Tick())
         << endl;

    //--------------------------------------------------------------------------
    // consume data
    cout << viOut << ' ' << vicOut << ' ' << vi2Out << ' ' << vi3Out << ' '
         << vhOut << ' ' << vh1Out << ' ' << vhcOut << endl;
    return 0;
}

// Hardware:
//  model name      : Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
//  6 physical cores, hyperthreading enabled
//  64 GB memory: Configured Memory Speed: 2667 MT/s (2x32 GB banks)
// Software:
//  PopOS 21.04
//  clang version 12.0.0-3ubuntu1~21.04.1, GNU STL
//  Target: x86_64-pc-linux-gnu
//  Thread model: posix
//  clang++ -O2 uninitialized_vector_test.cpp
//
// ./a.out
//
// SIZE: 64 MB
// malloc (s): 1.4054e-05
// free (s): 1.07e-07
// vector<int> (s): 0.0229717
// vector<int>.resize(0) (s): 0.00352966
// vector<int> custom allocator default initialized (s): 2.5944e-05
// vector<int>.resize(0) custom allocator default initialized (s): 2.381e-05
// vector<int> custom allocator (s): 2.16e-06
// vector<int>.resize(0) custom allocator (s): 3.248e-06
// vector<Uninitialized<int>> (s): 1.9735e-05
// vector<Uninitialized<int>>.resize(0) (s): 3.418e-06
// vector<Large> (s): 2.4341
// vector<Wrapper<Large>> (s): 9.158e-06
// vector<Large> custom allocator (s): 5.711e-06
//
// Page locked:
// vector<int> custom allocator (s): 0.00990538
// vector<int>.resize(0) custom allocator (s): 0.00427079
