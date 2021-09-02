/// @author Ugo Varetto
/// Block matrix-matrix multiply with non square matrices and blocks of varying
/// size;
/// all dimensions and offsets are encoded through dim2 structs
/// where x: column index or number of columns and y = row index
/// or number of rows
/// Huge difference in single threaded mode between using blocks and not using
/// blocks:
/// No blocking:
/// user    0m47.232s
/// Blocking (caching effects?):
/// user    0m13.883s !
/// Blocking + threads:
/// real    0m3.306s !
/// user    0m34.688s (64 threads on a 6 core w/ hyperthreading CPU)
//
// ./a.out 10000 10000 100
// Size (columns, rows):       (10000 10000)
// Block size:                 100
// matrix(0, 0):               10000
// matrix(rows - 1, cols - 1): 10000
// time:                       121.464 (s)
// ./a.out 10000 10000 1000
// Size (columns, rows):       (10000 10000)
// Block size:                 1000
// matrix(0, 0):               10000
// matrix(rows - 1, cols - 1): 10000
// time:                       782.025 (s)

#include <x86intrin.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <future>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "aligned_allocators.h"
#include "raw_allocators.h"
#include "timer.h"

/// quick print
template <typename... ArgsT>
void p_(const ArgsT... args) {
    (std::cout << ... << args) << std::endl;
}

#ifdef PARALLEL
#define parallel_init std::vector<std::future<void>> futures__;
#define parallel_begin \
    futures__.push_back((std::async([=]() {
#define parallel_end \
    })));
#define parallel_sync \
    std::for_each(begin(futures__), end(futures__), [](auto& f) { f.wait(); });
#else
#define parallel_init
#define parallel_begin
#define parallel_end
#define parallel_sync
#endif

using namespace std;

#ifdef DOUBLE
using Float = double;
#else
using Float = float;
#endif

#ifdef ALIGNED
using Vector = vector<Float, AlignedAllocator<Float>>;
#else
using Vector = vector<Float>;
#endif

/// Dimensions (x = columns, y = rows) and offsets.
struct dim2 {
    size_t x = 0;
    size_t y = 0;
};

/// print @c dim2 class
std::ostream& operator<<(ostream& os, const dim2& d) {
    return os << '(' << d.x << ' ' << d.y << ')';
}

/// Swap x and y.
dim2 Swap(const dim2& d) { return {d.y, d.x}; }

/// Compute dimension of matrix resulting from multiplying two matrices
/// of sizes b1 and b1.
dim2 DimMul(const dim2& b1, const dim2& b2) { return {b1.y, b2.x}; }

/// Print matrix
void Print(const dim2& size, const Float* m) {
    for (size_t row = 0; row != size.y; ++row) {
        for (size_t col = 0; col != size.x; ++col)
            cout << m[row * size.x + col] << " ";
        cout << endl;
    }
}

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

template <typename T>
T Dot(const T* __restrict v1, const T* __restrict v2, size_t len) {
#ifndef AVX
    return std::inner_product(v1, v1 + len, v2, T(0));
#else
    return DotProd256FMA(v1, v2, len);
#endif
}

/// Multiply one sub-matrix from matrix A, one sub-matrix from matrix B
/// and store the result as a sub-matrix of C. Note that A, B, C can be
/// matrices of arbitrary size with no requirement of having rows(C) == rows(A)
/// and columns(C) == columns(B).
///
/// @param a input matrix A
/// @param b input matrix B
/// @param c output matrix C
/// @param blockDimA dimensions of A sub-matrix
/// @param blockDimB dimensions of B sub-batrix
/// @param dimA dimensions of matrix A
/// @param dimB dimensions of matrix B
/// @param offsetA location of input sub-matrix inside A
/// @param offsetB location of input sub-batrix inside B
void BlockMul(const Float* __restrict a, const Float* __restrict b,
              Float* __restrict c, const dim2& blockDimA, const dim2& blockDimB,
              const dim2& dimA, const dim2& dimB, const dim2& offsetA,
              const dim2& offsetB) {
    const dim2 blockDimC = {blockDimB.x, blockDimA.y};
    thread_local static std::vector<Float, RawAllocator<Float>> column(
#ifdef ALIGNED
        blockDimC.y, RawAllocator<Float>(false, 32));
#else
        blockDimC.y, RawAllocator<Float>(false));
#endif
    for (size_t col = 0; col != blockDimC.x; ++col) {
        // extract column into array:
        const size_t offB = offsetB.x + col;
        for (size_t r = 0; r != blockDimB.y; ++r) {
            column[r] = b[(offsetB.y + r) * dimB.x + offB];
        }
        for (size_t row = 0; row != blockDimC.y; ++row) {
            c[row * blockDimC.x + col] =
                Dot(column.data(), &a[(row + offsetA.y) * dimA.x], blockDimC.y);
        }
    }
}

/// Add sub-matrix to sub-matrix in place.
///
/// @param a input matrix A
/// @param c output matrix C
/// @param dimA dimensions of A
/// @param nCols number of columns in output matrix C
/// @param dim dimensions of input matrix A
/// @param offset in output matrix C
void InplaceMatAdd(const Float* __restrict a, Float* __restrict c, size_t nCols,
                   const dim2& dim, const dim2& offset) {
    for (size_t row = 0; row != dim.y; ++row) {
        const size_t rowOffset = offset.y + row;
        for (size_t col = 0; col != dim.x; ++col) {
            c[rowOffset * nCols + offset.x + col] += a[row * dim.y + col];
        }
    }
}

/// Generic matrix-matrix multiply algorithm, use blockDim = {numCols, numRows}
/// if no blocking required.
///
/// @param a input matrix A
/// @param b input matrix B
/// @param c output matrix C
/// @param block cache to store intermediate multiply results
/// @param blockSize size of output block
/// @param dimA matrix A dimensions
/// @param dimB matrix B dimensions
void MatMul(const Float* __restrict a, const Float* __restrict b,
            Float* __restrict c, size_t blockSize, dim2 dimA, dim2 dimB) {
    assert(a);
    assert(b);
    assert(c);

    parallel_init;
    const dim2 dimC = {dimB.x, dimA.y};
    // iterate over C
    const size_t cBlockRows = dimC.y / blockSize;
    const size_t cBlockCols = dimC.x / blockSize;
    dim2 blockDim = {blockSize, blockSize};
    for (size_t row = 0; row != cBlockRows; ++row) {
        for (size_t col = 0; col != cBlockCols; ++col) {
            // if las block in row or column need to
            // adjust size in case number of row/column
            // not divisible by block size

            const size_t numBlocks = dimA.x / blockDim.x;
            // - iterate over column blocks in A and over row blocks in B
            // - compute block in C by summing up block A x block B
            //
            // The following loop is the equivalent of a CUDA/OpenCL/HIP kernel:
            //  pointer and array indices are passed over to function which
            //  computes local values
            parallel_begin;
            Vector tmpBlock;
            for (size_t cc = 0; cc != numBlocks; ++cc) {
                // C[row][col] = A[row][c] x B[c][col];
                // A: lock row, iterate over columns
                const dim2 offsetA = {cc * blockDim.x, row * blockDim.y};
                // B: lock column, iterate over rows
                const dim2 offsetB = {col * blockDim.x, cc * blockDim.y};
                dim2 bC = {blockDim.x, blockDim.y};
                if (cc == numBlocks - 1) {
                    bC.x += dimA.x % blockDim.x;
                }
                if (row == cBlockRows - 1) {
                    bC.y += dimA.y % blockDim.y;
                }
                tmpBlock.resize(bC.y * bC.y);
                BlockMul(a, b, tmpBlock.data(), bC, Swap(bC), dimA, dimB,
                         offsetA, offsetB);
                InplaceMatAdd(tmpBlock.data(), c, dimB.x, {bC.y, bC.y},
                              {offsetB.x, offsetA.y});
            }
            parallel_end;
        }
    }
    parallel_sync;
}

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

/// Generate matrices, multiply and print.
void Test(const dim2& size, size_t blockSize, bool csv) {
    const dim2 bsize = Swap(size);
    Vector a(size.x * size.y, Float(1.f));
    Vector b(size.x * size.y, Float(1.f));
    Vector c(size.y * bsize.x);
    const auto begin = Tick();
    MatMul(a.data(), b.data(), c.data(), blockSize, size, bsize);
    const int precision = sizeof(Float) * 8;
    auto end = Elapsed(begin, Tick());

#ifdef PARALLEL
    int numThreads = (size.x / blockSize) * (size.y / blockSize);
#else
    int numThreads = 1;
#endif
    if (csv) {
        cout << size.y << ',' << size.x << ',' << blockSize << ',' << precision
             << ',' << numThreads << ',' << aligned << ',' << avx << ',' << end
             << endl;
    } else {
        // Print(size, c.data());
        cout << "Size (columns, rows):       " << size << endl
             << "Block size:                 " << blockSize << endl
             << "matrix(0, 0):               " << c.front() << endl
             << "matrix(rows - 1, cols - 1): " << c.back() << endl
             << "Precision:                  " << sizeof(Float) * 8 << " bit"
             << endl
             << "Number of threads:          " << numThreads << endl
             << "Aligned:                    " << aligned << endl
             << "AVX:                        " << avx << endl
             << "time:                       " << end << " (s)" << endl;
    }
}

/// Invoke matrix multiply test
int main(int argc, char** argv) {
    if (argc >= 4) {
        const dim2 size = {stoul(argv[1]), stoul(argv[2])};
        const size_t blockSize = stoul(argv[3]);
        const bool csv = argc == 5;
        if (avx && (size.x % 8 || size.y % 8)) {
            // requirement for float, double can be multiple of 4
            cerr << "Error: when AVX enabled size of matrices must be a "
                    "multiple of 8"
                 << endl;
        }
        Test(size, blockSize, csv);
    } else {
        const dim2 size = {0x1000 - 1, 0x100 - 3};
        const size_t blockSize = 17;
        Test(size, blockSize, false);
    }
    return 0;
}
