// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_HH
#define HCORE_HH

#include "hcore/tile/dense.hh"
#include "hcore/tile/compressed.hh"

#include "blas.hh"

#include <cstdint>

namespace hcore {

const char* version();
const char* id();

template <typename T>
void gemm(
    T alpha, DenseTile<T> const& A,
             DenseTile<T> const& B,
    T beta,  DenseTile<T>      & C);
// converts rvalue references to lvalue references
template <typename T>
void gemm(
    T alpha, DenseTile<T> const&& A,
             DenseTile<T> const&& B,
    T beta,  DenseTile<T>      && C)
{
    // forward
    hcore::gemm(alpha, A, B, beta, C);
}

template <typename T>
void gemm(
    T alpha,      DenseTile<T> const& A,
                  DenseTile<T> const& B,
    T beta,  CompressedTile<T>      & C);
// converts rvalue references to lvalue references
template <typename T>
void gemm(
    T alpha,      DenseTile<T> const&& A,
                  DenseTile<T> const&& B,
    T beta,  CompressedTile<T>      && C)
{
    // forward
    hcore::gemm(alpha, A, B, beta, C);
}

template <typename T>
void gemm(
    T alpha,      DenseTile<T> const& A,
             CompressedTile<T> const& B,
    T beta,       DenseTile<T>      & C);
// converts rvalue references to lvalue references
template <typename T>
void gemm(
    T alpha,      DenseTile<T> const&& A,
             CompressedTile<T> const&& B,
    T beta,       DenseTile<T>      && C)
{
    // forward
    hcore::gemm(alpha, A, B, beta, C);
}

template <typename T>
void gemm(
    T alpha,      DenseTile<T> const& A,
             CompressedTile<T> const& B,
    T beta,  CompressedTile<T>      & C,
    bool use_trmm=false, bool use_ungqr=true, bool truncation_with_tol=false,
    int64_t rk=0);
// converts rvalue references to lvalue references
template <typename T>
void gemm(
    T alpha,      DenseTile<T> const&& A,
             CompressedTile<T> const&& B,
    T beta,  CompressedTile<T>      && C,
    bool use_trmm=false, bool use_ungqr=true, bool truncation_with_tol=false,
    int64_t rk=0)
{
    // forward
    hcore::gemm(alpha, A, B, beta, C,
        use_trmm, use_ungqr, truncation_with_tol, rk);
}

template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
                  DenseTile<T> const& B,
    T beta,       DenseTile<T>      & C);
// converts rvalue references to lvalue references
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const&& A,
                  DenseTile<T> const&& B,
    T beta,       DenseTile<T>      && C)
{
    // forward
    hcore::gemm(alpha, A, B, beta, C);
}

template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
                  DenseTile<T> const& B,
    T beta,  CompressedTile<T>      & C,
    bool use_trmm=false, bool use_ungqr=true, bool truncation_with_tol=false,
    int64_t rk=0);
// converts rvalue references to lvalue references
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const&& A,
                  DenseTile<T> const&& B,
    T beta,  CompressedTile<T>      && C,
    bool use_trmm=false, bool use_ungqr=true, bool truncation_with_tol=false,
    int64_t rk=0)
{
    // forward
    hcore::gemm(alpha, A, B, beta, C,
        use_trmm, use_ungqr, truncation_with_tol, rk);
}

template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
             CompressedTile<T> const& B,
    T beta,       DenseTile<T>      & C);
// converts rvalue references to lvalue references
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const&& A,
             CompressedTile<T> const&& B,
    T beta,       DenseTile<T>      && C)
{
    // forward
    hcore::gemm(alpha, A, B, beta, C);
}

template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
             CompressedTile<T> const& B,
    T beta,  CompressedTile<T>      & C,
    bool use_trmm=false, bool use_ungqr=true, bool truncation_with_tol=false,
    int64_t rk=0);
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const&& A,
             CompressedTile<T> const&& B,
    T beta,  CompressedTile<T>      && C,
    bool use_trmm=false, bool use_ungqr=true, bool truncation_with_tol=false,
    int64_t rk=0)
{
    // forward
    hcore::gemm(alpha, A, B, beta, C,
        use_trmm, use_ungqr, truncation_with_tol, rk);
}

template <typename T>
void syrk(
    T alpha, DenseTile<T> const& A,
    T beta,  DenseTile<T>      & C);
// converts rvalue references to lvalue references
template <typename T>
void syrk(
    T alpha, DenseTile<T> const&& A,
    T beta,  DenseTile<T>      && C)
{
    // forward
    hcore::syrk(alpha, A, beta, C);
}

template <typename T>
void syrk(
    T alpha, CompressedTile<T> const& A,
    T beta,      DenseTile<T>       & C);
// converts rvalue references to lvalue references
template <typename T>
void syrk(
    T alpha, CompressedTile<T> const&& A,
    T beta,      DenseTile<T>       && C)
{
    // forward
    hcore::syrk(alpha, A, beta, C);
}

template <typename T>
void syrk(
    T alpha,      DenseTile<T> const& A,
    T beta,  CompressedTile<T>      & C);
// converts rvalue references to lvalue references
template <typename T>
void syrk(
    T alpha,      DenseTile<T> const&& A,
    T beta,  CompressedTile<T>      && C)
{
    // forward
    hcore::syrk(alpha, A, beta, C);
}

template <typename T>
void syrk(
    T alpha, CompressedTile<T> const& A,
    T beta,  CompressedTile<T>      & C);
// converts rvalue references to lvalue references
template <typename T>
void syrk(
    T alpha, CompressedTile<T> const&& A,
    T beta,  CompressedTile<T>      && C)
{
    // forward
    hcore::syrk(alpha, A, beta, C);
}

template <typename T>
void trsm(
    blas::Side side, blas::Diag diag,
    T alpha, DenseTile<T> const& A,
             DenseTile<T>      & B);
// converts rvalue references to lvalue references
template <typename T>
void trsm(
    blas::Side side, blas::Diag diag,
    T alpha, DenseTile<T> const&& A,
             DenseTile<T>      && B)
{
    // forward
    hcore::trsm(side, diag, alpha, A, B);
}

} // namespace hcore

#endif // HCORE_HH
