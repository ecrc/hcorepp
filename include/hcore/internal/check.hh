// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_INTERNAL_CHECK_HH
#define HCORE_INTERNAL_CHECK_HH

#include "blas.hh"

#include "hcore/base_tile.hh"
#include "hcore/exception.hh"

namespace hcore {
namespace internal {

template <typename T>
void check_gemm(BaseTile<T> const& A,
                BaseTile<T> const& B,
                BaseTile<T> const& C)
{
    hcore_assert(A.mb() == C.mb());
    hcore_assert(B.nb() == C.nb());
    hcore_assert(A.nb() == B.mb());
    hcore_assert(A.layout() == B.layout());
    hcore_assert(B.layout() == C.layout());
    hcore_assert(A.uploPhysical() == blas::Uplo::General);
    hcore_assert(B.uploPhysical() == blas::Uplo::General);
    hcore_assert(C.uploPhysical() == blas::Uplo::General);
}

template <typename T>
void check_syrk(BaseTile<T> const& A,
                BaseTile<T> const& C)
{
    hcore_assert(C.mb() == C.nb());
    hcore_assert(C.mb() == A.mb());
    hcore_assert(A.layout() == C.layout());
    hcore_assert(A.uploPhysical() == blas::Uplo::General);
}

template <typename T>
void check_trsm(blas::Side side,
                BaseTile<T> const& A,
                BaseTile<T> const& B)
{
    hcore_assert(A.mb() == A.nb());
    hcore_assert(A.layout() == B.layout());
    hcore_assert(B.uploPhysical() == blas::Uplo::General);
    hcore_assert(side == blas::Side::Left
                 ? A.mb() == B.mb() : A.mb() == B.nb());
}

} // namespace internal
} // hcore

#endif  // HCORE_INTERNAL_CHECK_HH
