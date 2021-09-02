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
#include <string>
#include <vector>

#include "timer.h"

#define __device__
#define __global__
#define __host__


/// quick print
template <typename... ArgsT>
void p_(const ArgsT... args) {
    (std::cout << ... << args) << std::endl;
}

using Stream = std::future<void>;
    

using namespace std;

using Float = float;  // double;
using Vector = vector<Float>;

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
__device__ __host__
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

/// Globals
dim2 blockDimG;
dim2 blockIdxG;
dim2 gridDimG;
static thread_local dim2 blockDim;
static thread_local dim2 blockIdx;
static thread_local dim2 gridDim;

std::vector<Stream> streams;

template <typename F, typename...ArgsT>
void launchKernel(F f, ArgsT...args) {
    // make local copy
    dim2 bd = blockDimG;
    dim2 bi = blockIdxG;
    dim2 gd = gridDimG;
    // pass local copy to thread by value
    streams.push_back(
        std::async(
            std::launch::async,
            [f,bd,bi,gd](ArgsT...args) {
                //std::forward<Args>(args)... not required (?)
                //set global thread local variables
                blockDim = bd;
                blockIdx = bi;
                gridDim  = gd;
                //invoke function
                f(args...);
            },args...));
}

void deviceSynchronize() {
    std::for_each(begin(streams), end(streams), [](auto& f) { f.wait(); });
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
__device__
void BlockMul(const Float* a, const Float* b, Float* c, const dim2& blockDimA,
              const dim2& blockDimB, const dim2& dimA, const dim2& dimB, 
              const dim2& offsetA, const dim2& offsetB) {
    const dim2 blockDimC = {blockDimB.x, blockDimA.y};
    for (size_t row = 0; row != blockDimC.y; ++row) {
        const size_t rowOffset = (row + offsetA.y);
        for (size_t col = 0; col != blockDimC.x; ++col) {
            Float v = Float(0);
            for (size_t x = 0; x != blockDimA.x; ++x) {
                const size_t rA = rowOffset * dimA.x;
                const size_t cA = offsetA.x + x;
                const size_t rB = (x + offsetB.y) * dimB.x;
                const size_t cB = offsetB.x + col;
                v += a[rA + cA] * b[rB + cB];
            }
            c[row * blockDimC.x + col] = v;
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
__device__
void InplaceMatAdd(const Float* a, Float* c, size_t nCols, const dim2& dim,
                   const dim2& offset) {
    for (size_t row = 0; row != dim.y; ++row) {
        const size_t rowOffset = offset.y + row;
        for (size_t col = 0; col != dim.x; ++col) {
            c[rowOffset * nCols + offset.x + col] += a[row * dim.y + col];
        }
    }
}

/// Compute kernel: compute one block in the output matrix
///
/// @param a input matrix
/// @param b input matrix
/// @param c output matrix
/// @param dimA dimensions of matrix a
/// @param dimB dimensions of matrix b
__global__
void BlockMatMul(const Float* a, const Float* b, Float* c, const dim2& dimA,
                 const dim2& dimB) {
    Vector tmpBlock;
    const size_t numBlocks = dimA.x / blockDim.x;
    const size_t col = blockIdx.y;
    const size_t row = blockIdx.x;
    const size_t cBlockRows = gridDim.y;
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
        BlockMul(a, b, tmpBlock.data(), bC, Swap(bC), dimA, dimB, offsetA,
                 offsetB);
        InplaceMatAdd(tmpBlock.data(), c, dimB.x, {bC.y, bC.y},
                      {offsetB.x, offsetA.y});
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
void MatMul(const Float* a, const Float* b, Float* c, size_t blockSize,
            dim2 dimA, dim2 dimB) {
    assert(a);
    assert(b);
    assert(c);
    
    const dim2 dimC = {dimB.x, dimA.y};
    // iterate over C
    const size_t cBlockRows = dimC.y / blockSize;
    const size_t cBlockCols = dimC.x / blockSize;
    blockDimG = {blockSize, blockSize};
    gridDimG.y = cBlockRows;
    gridDimG.x = cBlockCols;
    // launch grid NxM blocks
    for (size_t row = 0; row != cBlockRows; ++row) {
        for (size_t col = 0; col != cBlockCols; ++col) {
            // if las block in row or column need to
            // adjust size in case number of row/column
            // not divisible by block size

            // - iterate over column blocks in A and over row blocks in B
            // - compute block in C by summing up block A x block B
            // index of current block in output matrix
            blockIdxG.x = col;
            blockIdxG.y = row;
            //launch kernel
            launchKernel(BlockMatMul, a, b, c, dimA, dimB);

        }
    }
    deviceSynchronize();
}

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
    int numThreads = (size.x * size.y) / (blockSize * blockSize);
    if (csv) {
        cout << size.y << ',' << size.x << ',' << blockSize << ',' << precision
             << ',' << numThreads << ',' << end << endl;
    } else {
        // Print(size, c.data());
        cout << "Size (columns, rows):       " << size << endl
             << "Block size:                 " << blockSize << endl
             << "matrix(0, 0):               " << c.front() << endl
             << "matrix(rows - 1, cols - 1): " << c.back() << endl
             << "Precision:                  " << sizeof(Float) * 8 << " bit"
             << endl
             << "Number of threads:          " << numThreads << endl
             << "time:                       " << end << " (s)" << endl;
    }
}

/// Invoke matrix multiply test
int main(int argc, char** argv) {
    if (argc >= 4) {
        const dim2 size = {stoul(argv[1]), stoul(argv[2])};
        const size_t blockSize = stoul(argv[3]);
        const bool csv = argc == 5;
        Test(size, blockSize, csv);
    } else {
        const dim2 size = {0x1000 - 1, 0x100 - 3};
        const size_t blockSize = 17;
        Test(size, blockSize, false);
    }
    return 0;
}
