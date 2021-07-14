// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include <cstdint>
#include <complex>

#include "lapack.hh"

#include "hcore/compressed_tile.hh"
#include "hcore/tile.hh"
#include "hcore.hh"

namespace hcore {

/// Cholesky factorization of a Hermitian positive definite matrix:
/// A = L * L^H or A = U^H * U.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] A
///     On entry, the n-by-n Hermitian positive definite matrix.
///     On exit, the factor U or L from the Cholesky factorization.
template <typename T>
int64_t potrf(Tile<T>& A) {
    return lapack::potrf(A.uplo_physical(), A.n(), A.data(), A.ld());
}

template
int64_t potrf(Tile<float>& A);
template
int64_t potrf(Tile<double>& A);
template
int64_t potrf(Tile<std::complex<float>>& A);
template
int64_t potrf(Tile<std::complex<double>>& A);

/// Cholesky factorization of a Hermitian positive definite matrix:
/// A = L * L^H or A = U^H * U.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] A
///     On entry, the n-by-n Hermitian positive definite compressed matrix
///                (A=UV), U: n-by-Ark; V: Ark-by-n.
///     On exit, the factor U or L from the Cholesky factorization.
template <typename T>
int64_t potrf(CompressedTile<T>& A) {
    throw hcore::Error("Not supported.");
}

template
int64_t potrf(CompressedTile<float>& A);
template
int64_t potrf(CompressedTile<double>& A);
template
int64_t potrf(CompressedTile<std::complex<float>>& A);
template
int64_t potrf(CompressedTile<std::complex<double>>& A);

} // namespace hcore
