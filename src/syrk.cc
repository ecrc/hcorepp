// Copyright (c) 2017,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "hcore.hh"

#include <assert.h>
#include <complex>

namespace hcore {

// =============================================================================
//
/// Symmetric rank-k update, C = alpha A * A^T + beta * C.
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
///     On exit, overwritten by the result: alpha A * A^T + beta * C.
template <typename T>
void syrk(
    T alpha, DenseTile<T> const& A,
    T beta,  DenseTile<T>       & C)
{
    // todo
    assert(false);
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
/// Symmetric rank-k update, C = alpha A * A^T + beta * C.
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
///     On exit, overwritten by the result: alpha A * A^T + beta * C.
template <typename T>
void syrk(
    T alpha,      DenseTile<T> const& A,
    T beta,  CompressedTile<T>      & C)
{
    // todo
    assert(false);
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
/// Symmetric rank-k update, C = alpha A * A^T + beta * C.
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
///     On exit, overwritten by the result: alpha A * A^T + beta * C.
template <typename T>
void syrk(
    T alpha, CompressedTile<T> const& A,
    T beta,      DenseTile<T>       & C)
{
    // todo
    assert(false);
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
/// Symmetric rank-k update, C = alpha A * A^T + beta * C.
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
///     On exit, overwritten by the result: alpha A * A^T + beta * C.
template <typename T>
void syrk(
    T alpha, CompressedTile<T> const& A,
    T beta,  CompressedTile<T>      & C)
{
    // todo
    assert(false);
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
