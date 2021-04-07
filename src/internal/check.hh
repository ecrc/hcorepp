// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_INTERNAL_CHECK_HH
#define HCORE_INTERNAL_CHECK_HH

#include "hcore/exception.hh"
#include "hcore/tile/tile.hh"

#include "blas.hh"

namespace hcore {
namespace internal {
namespace check {

template <typename T>
void gemm(Tile<T> const& A, Tile<T> const& B, Tile<T> const& C)
{
    hcore_error_if(A.layout() != B.layout());
    hcore_error_if(B.layout() != C.layout());
    hcore_error_if(A.m() != C.m());
    hcore_error_if(B.n() != C.n());
    hcore_error_if(A.n() != B.m());
    hcore_error_if(A.uplo_physical() != blas::Uplo::General);
    hcore_error_if(B.uplo_physical() != blas::Uplo::General);
    hcore_error_if(C.uplo_physical() != blas::Uplo::General);
}

template <typename T>
void syrk(Tile<T> const& A, Tile<T> const& C)
{
    hcore_error_if(A.layout() != C.layout());
    hcore_error_if(C.m() != C.n());
    hcore_error_if(C.m() != A.m());
    hcore_error_if(A.uplo_physical() != blas::Uplo::General);
    hcore_error_if(C.uplo_physical() == blas::Uplo::General);
}

template <typename T>
void trsm(blas::Side side, Tile<T> const& A, Tile<T> const& B)
{
    hcore_error_if(A.layout() != B.layout());
    hcore_error_if(A.m() != A.n());
    hcore_error_if(side == blas::Side::Left ? A.m() != B.m() : A.m() != B.n());
    hcore_error_if(B.uplo_physical() != blas::Uplo::General);
}

} // namespace check
} // namespace internal
} // namespace hcore

#endif // HCORE_INTERNAL_CHECK_HH
