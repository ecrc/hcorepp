// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "hcore.hh"
#include "internal/check.hh"
#include "hcore/tile/dense.hh"
#include "hcore/tile/compressed.hh"

#include "blas.hh"

#include <vector>
#include <complex>
#include <cassert>
#include <stdexcept>

namespace hcore {

// =============================================================================
//
/// Triangular matrix-matrix multiplication that solves one of the matrix
/// equations: A * X = alpha * B or X * A = alpha * B.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] side
///     Whether A appears on the left or on the right of X:
///     - Side::Left: solve A * X = alpha * B.
///     - Side::Right: solve X * A = alpha * B.
/// @param[in] diag
///     Whether A is a unit or non-unit upper or lower triangular matrix.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     - Side::Left: the m-by-m triangular dense tile.
///     - Side::Right: the n-by-n triangular dense tile.
/// @param[in,out] B
///     On entry, the m-by-n dense tile.
///     On exit, overwritten by the result X.
template <typename T>
void trsm(
    blas::Side side, blas::Diag diag,
    T alpha, DenseTile<T> const& A,
             DenseTile<T>      & B)
{
    assert(B.layout() == blas::Layout::ColMajor); // todo

    internal::check_trsm(side, A, B);

    if (B.op() == blas::Op::NoTrans) {
        blas::trsm(
            blas::Layout::ColMajor, side, A.uplo_physical(), A.op(), diag,
            B.m(), B.n(),
            alpha, A.data(), A.ld(),
                   B.data(), B.ld());
    }
    else {
        if (blas::is_complex<T>::value  &&
            A.op() != blas::Op::NoTrans &&
            A.op() != B.op()) {
                throw std::invalid_argument(
                "B is complex, transB != Op::NoTrans, and transA != transB.");
        }

        blas::Side side_ =
            side == blas::Side::Left ? blas::Side::Right : blas::Side::Left;
        blas::Op opA;
        if (A.op() == blas::Op::NoTrans)
            opA = B.op();
        else if (A.op() == B.op() || (!blas::is_complex<T>::value)) {
            opA = blas::Op::NoTrans;
        }
        else
            assert(false);

        using blas::conj;

        if (B.op() == blas::Op::ConjTrans) {
            alpha = conj(alpha);
        }

        blas::trsm(
            blas::Layout::ColMajor, side_, A.uplo_physical(), opA, diag,
            B.n(), B.m(),
            alpha, A.data(), A.ld(),
                   B.data(), B.ld());
    }
}

template
void trsm(
    blas::Side side, blas::Diag diag,
    float alpha, DenseTile<float> const& A,
                 DenseTile<float>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    double alpha, DenseTile<double> const& A,
                  DenseTile<double>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, DenseTile<std::complex<float>> const& A,
                               DenseTile<std::complex<float>>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, DenseTile<std::complex<double>> const& A,
                                DenseTile<std::complex<double>>      & B);

// =============================================================================
//
/// Triangular matrix-matrix multiplication that solves one of the matrix
/// equations: A * X = alpha * B or X * A = alpha * B.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] side
///     Whether A appears on the left or on the right of X:
///     - Side::Left: solve A * X = alpha * B.
///     - Side::Right: solve X * A = alpha * B.
/// @param[in] diag
///     Whether A is a unit or non-unit upper or lower triangular matrix.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     - Side::Left: the m-by-m triangular dense tile.
///     - Side::Right: the n-by-n triangular dense tile.
/// @param[in,out] B
///     On entry, the m-by-n compressed tile (U: m-by-Brk; V: Brk-by-n).
///     On exit, overwritten by the result X.
template <typename T>
void trsm(
    blas::Side side, blas::Diag diag,
    T alpha,      DenseTile<T> const& A,
             CompressedTile<T>      & B)
{
    // todo
    assert(false);
}

template
void trsm(
    blas::Side side, blas::Diag diag,
    float alpha,      DenseTile<float> const& A,
                 CompressedTile<float>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    double alpha,      DenseTile<double> const& A,
                  CompressedTile<double>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha,      DenseTile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha,      DenseTile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>>      & B);

// =============================================================================
//
/// Triangular matrix-matrix multiplication that solves one of the matrix
/// equations: A * X = alpha * B or X * A = alpha * B.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] side
///     Whether A appears on the left or on the right of X:
///     - Side::Left: solve A * X = alpha * B.
///     - Side::Right: solve X * A = alpha * B.
/// @param[in] diag
///     Whether A is a unit or non-unit upper or lower triangular matrix.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     - Side::Left: the m-by-m triangular compressed tile
///                   (U: m-by-Ark; V: Ark-by-m).
///     - Side::Right: the n-by-n triangular compressed tile
///                    (U: n-by-Ark; V: Ark-by-n).
/// @param[in,out] B
///     On entry, the m-by-n dense tile.
///     On exit, overwritten by the result X.
template <typename T>
void trsm(
    blas::Side side, blas::Diag diag,
    T alpha, CompressedTile<T> const& A,
                  DenseTile<T>      & B)
{
    // todo
    assert(false);
}

template
void trsm(
    blas::Side side, blas::Diag diag,
    float alpha, CompressedTile<float> const& A,
                      DenseTile<float>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    double alpha, CompressedTile<double> const& A,
                       DenseTile<double>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                                    DenseTile<std::complex<float>>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                     DenseTile<std::complex<double>>      & B);

// =============================================================================
//
/// Triangular matrix-matrix multiplication that solves one of the matrix
/// equations: A * X = alpha * B or X * A = alpha * B.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] side
///     Whether A appears on the left or on the right of X:
///     - Side::Left: solve A * X = alpha * B.
///     - Side::Right: solve X * A = alpha * B.
/// @param[in] diag
///     Whether A is a unit or non-unit upper or lower triangular matrix.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     - Side::Left: the m-by-m triangular compressed tile
///                   (U: m-by-Ark; V: Ark-by-m).
///     - Side::Right: the n-by-n triangular compressed tile
///                    (U: n-by-Ark; V: Ark-by-n).
/// @param[in,out] B
///     On entry, the m-by-n compressed tile (U: m-by-Brk; V: Brk-by-n).
///     On exit, overwritten by the result X.
template <typename T>
void trsm(
    blas::Side side, blas::Diag diag,
    T alpha, CompressedTile<T> const& A,
             CompressedTile<T>      & B)
{
    // todo
    assert(false);
}

template
void trsm(
    blas::Side side, blas::Diag diag,
    float alpha, CompressedTile<float> const& A,
                 CompressedTile<float>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    double alpha, CompressedTile<double> const& A,
                  CompressedTile<double>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>>      & B);
template
void trsm(
    blas::Side side, blas::Diag diag,
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>>      & B);

} // namespace hcore
