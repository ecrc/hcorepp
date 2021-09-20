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

//------------------------------------------------------------------------------
/// Cholesky factorization. Performs the Cholesky factorization of a Hermitian
/// positive definite matrix $A$. The factorization has the form $A = L L^H$,
/// if $A$ is stored lower, where $L$ is a lower triangular matrix, or
/// $A = U^H U$, if $A$ is stored upper, where $U$ is an upper triangular
/// matrix.
///
/// @tparam T
///     One of float, double, std::complex<float>, std::complex<double>.
/// @param[in,out] A
///     On entry, the nb-by-nb Hermitian positive definite matrix $A$.
///     On exit, if return value = 0, the factor $U$ or $L$ from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$.
/// @retval 0 successful exit
/// @retval >0 for return value = $i$, the leading minor of order $i$ of $A$ is
///         not positive definite, so the factorization could not be completed,
///         and the solution has not been computed.
///
/// @ingroup cholesky
template <typename T>
int64_t potrf(Tile<T>& A)
{
    return lapack::potrf(A.uploPhysical(), A.nb(), A.data(), A.stride());
}

template
int64_t potrf(Tile<float>& A);
template
int64_t potrf(Tile<double>& A);
template
int64_t potrf(Tile<std::complex<float>>& A);
template
int64_t potrf(Tile<std::complex<double>>& A);

//------------------------------------------------------------------------------
/// Cholesky factorization. Performs the Cholesky factorization of a Hermitian
/// positive definite matrix $A$. The factorization has the form $A = L L^H$,
/// if $A$ is stored lower, where $L$ is a lower triangular matrix, or
/// $A = U^H U$, if $A$ is stored upper, where $U$ is an upper triangular
/// matrix.
///
/// @tparam T
///     One of float, double, std::complex<float>, std::complex<double>.
/// @param[in,out] A
///     On entry, the nb-by-nb Hermitian positive definite compressed matrix
///               $A = U V$.
///     On exit, if return value = 0, the factor $U$ or $L$ from the Cholesky
///     factorization $A = U^H U$ or $A = L L^H$.
///
/// @ingroup cholesky
template <typename T>
int64_t potrf(CompressedTile<T>& A) { throw Error("Not supported."); } // todo

template
int64_t potrf(CompressedTile<float>& A);
template
int64_t potrf(CompressedTile<double>& A);
template
int64_t potrf(CompressedTile<std::complex<float>>& A);
template
int64_t potrf(CompressedTile<std::complex<double>>& A);

} // namespace hcore
