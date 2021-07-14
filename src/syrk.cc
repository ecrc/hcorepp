// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include <complex>
#include <vector>

#include "blas.hh"

#include "hcore/compressed_tile.hh"
#include "hcore/exception.hh"
#include "hcore/check.hh"
#include "hcore/tile.hh"
#include "hcore.hh"

namespace hcore {

/// Symmetric rank-k update:
/// C = alpha * A * A.' + beta * C, or
/// C = alpha * A.' * A + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The n-by-k matrix.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the n-by-n symmetric matrix.
///     On exit, overwritten by the result:
///              alpha * A * A.' + beta * C, or
///              alpha * A.' * A + beta * C.
template <typename T>
void syrk(T alpha, Tile<T> const& A,
          T beta,  Tile<T>      & C) {
    internal::check::syrk(A, C);

    hcore_error_if(C.is_complex &&
                   (A.op() == blas::Op::ConjTrans ||
                    C.op() == blas::Op::ConjTrans));

    blas::syrk(C.layout(), C.uplo_physical(), A.op(),
               C.n(), A.n(), alpha, A.data(), A.ld(),
                             beta,  C.data(), C.ld());
}

template
void syrk(float alpha, Tile<float> const& A,
          float beta,  Tile<float>      & C);
template
void syrk(double alpha, Tile<double> const& A,
          double beta,  Tile<double>      & C);
template
void syrk(std::complex<float> alpha, Tile<std::complex<float>> const& A,
          std::complex<float> beta,  Tile<std::complex<float>>      & C);
template
void syrk(std::complex<double> alpha, Tile<std::complex<double>> const& A,
          std::complex<double> beta,  Tile<std::complex<double>>      & C);

/// Symmetric rank-k update:
/// C = alpha * A * A.' + beta * C, or
/// C = alpha * A.' * A + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The n-by-k matrix.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the n-by-n symmetric compressed matrix (C=UV),
///               U: n-by-Crk; V: Crk-by-n.
///     On exit, overwritten by the result:
///              alpha * A * A.' + beta * C, or
///              alpha * A.' * A + beta * C.
template <typename T>
void syrk(T alpha,           Tile<T> const& A,
          T beta,  CompressedTile<T>      & C) {
    throw hcore::Error("Not supported.");
}

template
void syrk(float alpha,          Tile<float> const& A,
          float beta, CompressedTile<float>      & C);
template
void syrk(double alpha,          Tile<double> const& A,
          double beta, CompressedTile<double>      & C);
template
void syrk(
    std::complex<float> alpha,          Tile<std::complex<float>> const& A,
    std::complex<float> beta, CompressedTile<std::complex<float>>      & C);
template
void syrk(
    std::complex<double> alpha,           Tile<std::complex<double>> const& A,
    std::complex<double> beta, CompressedTile<std::complex<double>>      & C);

/// Symmetric rank-k update:
/// C = alpha * A * A.' + beta * C, or
/// C = alpha * A.' * A + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The n-by-k compressed matrix (A=UV), U: n-by-Ark; V: Ark-by-k.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the n-by-n symmetric matrix.
///     On exit, overwritten by the result:
///              alpha * A * A.' + beta * C, or
///              alpha * A.' * A + beta * C.
template <typename T>
void syrk(T alpha, CompressedTile<T> const& A,
          T beta,            Tile<T>      & C) {
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(C.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check::syrk(A, C);

    T zero = 0.0;
    T one  = 1.0;

    // C = beta * C + ((AU * (alpha * AV * AV.')) * AU.')

    std::vector<T> W0(A.rk() * A.rk());

    // W0 = alpha * AV * AV.'
    blas::syrk(
        blas::Layout::ColMajor, C.uplo(), blas::Op::NoTrans,
        A.rk(), A.n(),
        alpha, A.Vdata(), A.ldv(),
        zero,  &W0[0],    A.rk());

    std::vector<T> W1(A.m() * A.rk());

    // W1 = AU * W0
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        A.m(), A.rk(), A.rk(),
        one,  A.Udata(), A.ldu(),
              &W0[0],    A.rk(),
        zero, &W1[0],    A.m());

    // C = W1 * AU.' + beta * C
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
        A.m(), A.m(), A.rk(),
        one,  &W1[0],    A.m(),
              A.Udata(), A.ldu(),
        beta, C.data(),  C.ld());
}

template
void syrk(float alpha, CompressedTile<float> const& A,
          float beta,            Tile<float>      & C);
template
void syrk(double alpha, CompressedTile<double> const& A,
          double beta,            Tile<double>      & C);
template
void syrk(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
    std::complex<float> beta,            Tile<std::complex<float>>      & C);
template
void syrk(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
    std::complex<double> beta,            Tile<std::complex<double>>      & C);

/// Symmetric rank-k update:
/// C = alpha * A * A.' + beta * C, or
/// C = alpha * A.' * A + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The n-by-k compressed matrix (A=UV), U: n-by-Ark; V: Ark-by-k.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the n-by-n symmetric compressed matrix (C=UV),
///               U: n-by-Crk; V: Crk-by-n.
///     On exit, overwritten by the result:
///              alpha * A * A.' + beta * C, or
///              alpha * A.' * A + beta * C.
template <typename T>
void syrk(T alpha, CompressedTile<T> const& A,
          T beta,  CompressedTile<T>      & C) {
    throw hcore::Error("Not supported.");
}

template
void syrk(float alpha, CompressedTile<float> const& A,
          float beta,  CompressedTile<float>      & C);
template
void syrk(double alpha, CompressedTile<double> const& A,
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
