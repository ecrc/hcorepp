// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "hcore.hh"
#include "hcore/exception.hh"
#include "hcore/tile/tile.hh"
#include "hcore/tile/dense.hh"
#include "hcore/tile/compressed.hh"

#include "blas.hh"

#include <vector>
#include <complex>
#include <cassert>

namespace hcore {
namespace internal {
namespace check {

template <typename T>
void syrk(Tile<T> const& A, Tile<T> const& C)
{
    hcore_error_if(C.m() != C.n());
    hcore_error_if(C.m() != A.m());
    hcore_error_if(A.layout() != C.layout());
    hcore_error_if(A.uplo_physical() != blas::Uplo::General);
    hcore_error_if(C.uplo_physical() == blas::Uplo::General);
}

} // namespace check
} // namespace internal

// =============================================================================
//
/// Symmetric rank-k update:
/// C = alpha * A * A.' + beta * C, or
/// C = alpha * A.' * A + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The n-by-k dense tile.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the n-by-n symmetric dense tile.
///     On exit, overwritten by the result:
///              alpha * A * A.' + beta * C, or
///              alpha * A.' * A + beta * C.
template <typename T>
void syrk(
    T alpha, DenseTile<T> const& A,
    T beta,  DenseTile<T>      & C)
{
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check::syrk(A, C);

    if (blas::is_complex<T>::value) {
        if (A.op() == blas::Op::ConjTrans) {
            throw hcore::Error("C is complex and transA == Op::ConjTrans.");
        }
        else if (C.op() == blas::Op::ConjTrans) {
            throw hcore::Error("C is complex and transC == Op::ConjTrans.");
        }
    }

    blas::syrk(
        blas::Layout::ColMajor, C.uplo_physical(), A.op(),
        C.n(), A.n(),
        alpha, A.data(), A.ld(),
        beta,  C.data(), C.ld());
}

template
void syrk(
    float alpha, DenseTile<float> const& A,
    float beta,  DenseTile<float>      & C);
template
void syrk(
    double alpha, DenseTile<double> const& A,
    double beta,  DenseTile<double>      & C);
template
void syrk(
    std::complex<float> alpha, DenseTile<std::complex<float>> const& A,
    std::complex<float> beta,  DenseTile<std::complex<float>>      & C);
template
void syrk(
    std::complex<double> alpha, DenseTile<std::complex<double>> const& A,
    std::complex<double> beta,  DenseTile<std::complex<double>>      & C);

// =============================================================================
//
/// Symmetric rank-k update:
/// C = alpha * A * A.' + beta * C, or
/// C = alpha * A.' * A + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The n-by-k dense tile.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the n-by-n symmetric compressed tile
///               (U: n-by-Ark; V: Ark-by-n).
///     On exit, overwritten by the result:
///              alpha * A * A.' + beta * C, or
///              alpha * A.' * A + beta * C.
template <typename T>
void syrk(
    T alpha,      DenseTile<T> const& A,
    T beta,  CompressedTile<T>      & C)
{
    assert(false); // todo
}

template
void syrk(
    float alpha,     DenseTile<float> const& A,
    float beta, CompressedTile<float>      & C);
template
void syrk(
    double alpha,     DenseTile<double> const& A,
    double beta, CompressedTile<double>      & C);
template
void syrk(
    std::complex<float> alpha,     DenseTile<std::complex<float>> const& A,
    std::complex<float> beta, CompressedTile<std::complex<float>>      & C);
template
void syrk(
    std::complex<double> alpha,     DenseTile<std::complex<double>> const& A,
    std::complex<double> beta, CompressedTile<std::complex<double>>      & C);

// =============================================================================
//
/// Symmetric rank-k update:
/// C = alpha * A * A.' + beta * C, or
/// C = alpha * A.' * A + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The n-by-k compressed tile (U: n-by-Ark; V: Ark-by-k).
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the n-by-n symmetric dense tile.
///     On exit, overwritten by the result:
///              alpha * A * A.' + beta * C, or
///              alpha * A.' * A + beta * C.
template <typename T>
void syrk(
    T alpha, CompressedTile<T> const& A,
    T beta,      DenseTile<T>       & C)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(C.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check::syrk(A, C);

    // C = beta * C + ((AU * (alpha * AV * AV.')) * AU.')

    std::vector<T> W0(A.rk() * A.rk());

    // W0 = alpha * AV * AV.'
    blas::syrk(
        blas::Layout::ColMajor, C.uplo(), blas::Op::NoTrans,
        A.rk(), A.n(),
        alpha, A.Vdata(), A.ldv(),
        0.0,   &W0[0],    A.rk());

    std::vector<T> W1(A.m() * A.rk());

    // W1 = AU * W0
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        A.m(), A.rk(), A.rk(),
        1.0, A.Udata(), A.ldu(),
             &W0[0],    A.rk(),
        0.0, &W1[0],    A.m());

    // C = W1 * AU.' + beta * C
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, 
        A.m(), A.m(), A.rk(),
        1.0,  &W1[0],    A.m(),
              A.Udata(), A.ldu(),
        beta, C.data(),  C.ld());
}

template
void syrk(
    float alpha, CompressedTile<float> const& A,
    float beta,       DenseTile<float>      & C);
template
void syrk(
    double alpha, CompressedTile<double> const& A,
    double beta,       DenseTile<double>      & C);
template
void syrk(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
    std::complex<float> beta,       DenseTile<std::complex<float>>      & C);
template
void syrk(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
    std::complex<double> beta,       DenseTile<std::complex<double>>      & C);

// =============================================================================
//
/// Symmetric rank-k update:
/// C = alpha * A * A.' + beta * C, or
/// C = alpha * A.' * A + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The n-by-k compressed tile (U: n-by-Ark; V: Ark-by-k).
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the n-by-n symmetric compressed tile
///               (U: n-by-Ark; V: Ark-by-n).
///     On exit, overwritten by the result:
///              alpha * A * A.' + beta * C, or
///              alpha * A.' * A + beta * C.
template <typename T>
void syrk(
    T alpha, CompressedTile<T> const& A,
    T beta,  CompressedTile<T>      & C)
{
    assert(false); // todo
}

template
void syrk(
    float alpha, CompressedTile<float> const& A,
    float beta,  CompressedTile<float>      & C);
template
void syrk(
    double alpha, CompressedTile<double> const& A,
    double beta,  CompressedTile<double>      & C);
template
void syrk(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
    std::complex<float> beta,  CompressedTile<std::complex<float>>      & C);
template
void syrk(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
    std::complex<double> beta,  CompressedTile<std::complex<double>>      & C);

} // namespace hcore
