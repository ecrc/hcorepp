// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "hcore.hh"
#include "hcore/check.hh"
#include "hcore/exception.hh"
#include "hcore/dense_tile.hh"
#include "hcore/compressed_tile.hh"

#include "lapack.hh"

#include <complex>
#include <cassert>
#include <cstdint>

namespace hcore {

// =============================================================================
//
/// Cholesky factorization of a Hermitian positive definite tile:
/// A = L * L^H or A = U^H * U.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] A
///     On entry, the n-by-n Hermitian positive definite dense tile.
///     On exit, the factor U or L from the Cholesky factorization.
template <typename T>
void potrf(DenseTile<T>& A)
{
    int64_t info = lapack::potrf(A.uplo_physical(), A.n(), A.data(), A.ld());

    hcore_error_if_msg(
        info != 0, "lapack::potrf returned error %lld", (long long)info);
}

template
void potrf(DenseTile<float>& A);
template
void potrf(DenseTile<double>& A);
template
void potrf(DenseTile<std::complex<float>>& A);
template
void potrf(DenseTile<std::complex<double>>& A);

// =============================================================================
//
/// Cholesky factorization of a Hermitian positive definite tile:
/// A = L * L^H or A = U^H * U.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] A
///     On entry, the n-by-n Hermitian positive definite compressed tile:
///               (U: n-by-Ark; V: Ark-by-n).
///     On exit, the factor U or L from the Cholesky factorization.
template <typename T>
void potrf(CompressedTile<T>& A)
{
    // todo
    assert(false);
}

template
void potrf(CompressedTile<float>& A);
template
void potrf(CompressedTile<double>& A);
template
void potrf(CompressedTile<std::complex<float>>& A);
template
void potrf(CompressedTile<std::complex<double>>& A);

} // namespace hcore
