// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_HH
#define HCORE_HH

#include <cstdint>

#include "blas.hh"

#include "hcore/compressed_tile.hh"
#include "hcore/options.hh"
#include "hcore/tile.hh"

//------------------------------------------------------------------------------
/// @namespace hcore
/// HCORE's top-level namespace.
///
namespace hcore {

const char* version();
const char* id();

//------------------------------------------------------------------------------
// Level 3 BLAS

// General matrix-matrix multiplication (gemm)
template <typename T>
void gemm(T alpha, Tile<T> const& A,
                   Tile<T> const& B,
          T beta,  Tile<T>&       C);
// converts rvalue references to lvalue references
template <typename T>
void gemm(T alpha, Tile<T> const&& A,
                   Tile<T> const&& B,
          T beta,  Tile<T>&&       C)
{
    hcore::gemm(alpha, A, B, beta, C); // forward
}
template <typename T>
void gemm(T alpha, CompressedTile<T> const& A,
                             Tile<T> const& B,
          T beta,            Tile<T>&       C);
// converts rvalue references to lvalue references
template <typename T>
void gemm(T alpha, CompressedTile<T> const&& A,
                             Tile<T> const&& B,
          T beta,            Tile<T>&&       C)
{
    hcore::gemm(alpha, A, B, beta, C); // forward
}
template <typename T>
void gemm(T alpha,           Tile<T> const& A,
                   CompressedTile<T> const& B,
          T beta,            Tile<T>&       C);
// converts rvalue references to lvalue references
template <typename T>
void gemm(T alpha,           Tile<T> const&& A,
                   CompressedTile<T> const&& B,
          T beta,            Tile<T>&&       C)
{
    hcore::gemm(alpha, A, B, beta, C); // forward
}
template <typename T>
void gemm(T alpha, CompressedTile<T> const& A,
                   CompressedTile<T> const& B,
          T beta,            Tile<T>&       C);
// converts rvalue references to lvalue references
template <typename T>
void gemm(T alpha, CompressedTile<T> const&& A,
                   CompressedTile<T> const&& B,
          T beta,            Tile<T>&&       C)
{
    hcore::gemm(alpha, A, B, beta, C); // forward
}
template <typename T>
void gemm(T alpha,           Tile<T> const& A,
                             Tile<T> const& B,
          T beta,  CompressedTile<T>&       C);
// converts rvalue references to lvalue references
template <typename T>
void gemm(T alpha,           Tile<T> const&& A,
                             Tile<T> const&& B,
          T beta,  CompressedTile<T>&&       C)
{
    hcore::gemm(alpha, A, B, beta, C); // forward
}
template <typename T>
void gemm(T alpha,           Tile<T> const& A,
                   CompressedTile<T> const& B,
          T beta,  CompressedTile<T>&       C,
          Options const& opts = Options());
// converts rvalue references to lvalue references
template <typename T>
void gemm(T alpha,           Tile<T> const&& A,
                   CompressedTile<T> const&& B,
          T beta,  CompressedTile<T>&&       C,
          Options const& opts = Options())
{
    hcore::gemm(alpha, A, B, beta, C, opts); // forward
}
template <typename T>
void gemm(T alpha, CompressedTile<T> const& A,
                             Tile<T> const& B,
          T beta,  CompressedTile<T>&       C,
          Options const& opts = Options());
// converts rvalue references to lvalue references
template <typename T>
void gemm(T alpha, CompressedTile<T> const&& A,
                             Tile<T> const&& B,
          T beta,  CompressedTile<T>&&       C,
          Options const& opts = Options())
{
    hcore::gemm(alpha, A, B, beta, C, opts); // forward
}
template <typename T>
void gemm(T alpha, CompressedTile<T> const& A,
                   CompressedTile<T> const& B,
          T beta,  CompressedTile<T>&       C,
          Options const& opts = Options());
template <typename T>
void gemm(T alpha, CompressedTile<T> const&& A,
                   CompressedTile<T> const&& B,
          T beta,  CompressedTile<T>&&       C,
          Options const& opts = Options())
{
    hcore::gemm(alpha, A, B, beta, C, opts); // forward
}

// Symmetric rank-k update (syrk)
template <typename T>
void syrk(T alpha, Tile<T> const& A,
          T beta,  Tile<T>&       C);
// converts rvalue references to lvalue references
template <typename T>
void syrk(T alpha, Tile<T> const&& A,
          T beta,  Tile<T>&&       C)
{
    hcore::syrk(alpha, A, beta, C);  // forward
}
template <typename T>
void syrk(T alpha, CompressedTile<T> const& A,
          T beta,            Tile<T>&       C);
// converts rvalue references to lvalue references
template <typename T>
void syrk(T alpha, CompressedTile<T> const&& A,
          T beta,            Tile<T>&&       C)
{
    hcore::syrk(alpha, A, beta, C); // forward
}

template <typename T>
void syrk(T alpha,           Tile<T> const& A,
          T beta,  CompressedTile<T>&       C);
// converts rvalue references to lvalue references
template <typename T>
void syrk(T alpha,           Tile<T> const&& A,
          T beta,  CompressedTile<T>&&       C)
{
    hcore::syrk(alpha, A, beta, C); // forward
}

template <typename T>
void syrk(T alpha, CompressedTile<T> const& A,
          T beta,  CompressedTile<T>&       C);
// converts rvalue references to lvalue references
template <typename T>
void syrk(T alpha, CompressedTile<T> const&& A,
          T beta,  CompressedTile<T>&&       C)
{
    hcore::syrk(alpha, A, beta, C); // forward
}

template <typename T>
void trsm(blas::Side side, blas::Diag diag,
          T alpha, Tile<T> const& A,
                   Tile<T>&       B);
// converts rvalue references to lvalue references
template <typename T>
void trsm(blas::Side side, blas::Diag diag,
          T alpha, Tile<T> const&& A,
                   Tile<T>&&       B)
{
    hcore::trsm(side, diag, alpha, A, B); // forward
}

template <typename T>
void trsm(blas::Side side, blas::Diag diag,
          T alpha, CompressedTile<T> const& A,
                             Tile<T>&       B);
// converts rvalue references to lvalue references
template <typename T>
void trsm(blas::Side side, blas::Diag diag,
          T alpha, CompressedTile<T> const&& A,
                             Tile<T>&&       B)
{
    hcore::trsm(side, diag, alpha, A, B); // forward
}

template <typename T>
void trsm(blas::Side side, blas::Diag diag,
          T alpha,           Tile<T> const& A,
                   CompressedTile<T>&       B);
// converts rvalue references to lvalue references
template <typename T>
void trsm(blas::Side side, blas::Diag diag,
          T alpha,           Tile<T> const&& A,
                   CompressedTile<T>&&       B)
{
    hcore::trsm(side, diag, alpha, A, B); // forward
}

template <typename T>
void trsm(blas::Side side, blas::Diag diag,
          T alpha, CompressedTile<T> const& A,
                   CompressedTile<T>&       B);
// converts rvalue references to lvalue references
template <typename T>
void trsm(blas::Side side, blas::Diag diag,
          T alpha, CompressedTile<T> const&& A,
                   CompressedTile<T>&&       B)
{
    hcore::trsm(side, diag, alpha, A, B); // forward
}

template <typename T>
int64_t potrf(Tile<T>& A);
// converts rvalue references to lvalue references
template <typename T>
int64_t potrf(Tile<T>&& A)
{
    return hcore::potrf(A); // forward
}

template <typename T>
int64_t potrf(CompressedTile<T>& A);
// converts rvalue references to lvalue references
template <typename T>
int64_t potrf(CompressedTile<T>&& A)
{
    return hcore::potrf(A); // forward
}

} // namespace hcore

#endif // HCORE_HH