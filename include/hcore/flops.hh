// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_FLOPS_HH
#define HCORE_FLOPS_HH

#include "lapack/flops.hh"
#include "blas/flops.hh"

#include "hcore/compressed_tile.hh"
#include "hcore/internal/check.hh"
#include "hcore/tile.hh"

namespace hcore {
namespace internal {

inline double fmuls_gesvd(double m, double n) { return 14*m*n*n; }
inline double fadds_gesvd(double m, double n) { return  8*n*n*n; }

template <typename T>
class Gflop : public blas::Gflop<T>
{
public:
    using blas::Gflop<T>::mul_ops;
    using blas::Gflop<T>::add_ops;

    // The SVD is necessarily an iterative process, therefore no exact flop
    // formula is possible. It also depends on which singular vectors are
    // required. Some approximations given in Matrix Computations, 3rd ed.,
    // Golub & Van Loan, page 254 are:
    //     - singular values only, 4mn^2 - 4n^3/3, or
    //     - singular values and some singular vectors U (m x n) and V (n x n),
    //       14mn^2 + 8n^3
    static double lapack_gesvd(double m, double n)
    {
        return 1e-9*(mul_ops*fmuls_gesvd(m, n)+add_ops*fadds_gesvd(m, n));
    }

    // randomized svd
    static double rsvd(double Ark, CompressedTile<T>& C)
    {
        return   lapack::Gflop<T>::geqrf(C.mb(), Ark + C.rk())
               + lapack::Gflop<T>::geqrf(C.nb(), Ark + C.rk())
               + blas::Gflop<T>::gemm(blas::min(C.mb(), Ark + C.rk()),
                                      blas::min(C.nb(), Ark + C.rk()),
                                      Ark + C.rk())
               + lapack_gesvd(blas::min(C.mb(), Ark + C.rk()),
                              blas::min(C.nb(), Ark + C.rk()))
               + lapack::Gflop<T>::ungqr(C.mb(),
                                         blas::min(C.mb(), Ark + C.rk()),
                                         blas::min(C.mb(), Ark + C.rk()))
               + blas::Gflop<T>::gemm(C.mb(),
                                      C.rk(),
                                      blas::min(C.mb(), Ark + C.rk()))
               + blas::Gflop<T>::scal(blas::min(C.nb(), Ark + C.rk()))
               + lapack::Gflop<T>::ungqr(C.nb(),
                                         blas::min(C.nb(), Ark + C.rk()),
                                         blas::min(C.nb(), Ark + C.rk()))
               + blas::Gflop<T>::gemm(C.nb(),
                                      C.rk(),
                                      blas::min(C.nb(), Ark + C.rk()));
    }
}; // class Gflop
} // namespace internal

template <typename T>
class Gflop
{
public:
    static double gemm(Tile<T> A,
                       Tile<T> B,
                       Tile<T> C)
    {
        internal::check_gemm(A, B, C);

        return blas::Gflop<T>::gemm(C.mb(), C.nb(), A.nb());
    }

    static double gemm(CompressedTile<T> A,
                                 Tile<T> B,
                                 Tile<T> C)
    {
        internal::check_gemm(A, B, C);

        return   blas::Gflop<T>::gemm(A.rk(), C.nb(), A.nb())
               + blas::Gflop<T>::gemm(C.mb(), C.nb(), A.rk());
    }

    static double gemm(          Tile<T> A,
                       CompressedTile<T> B,
                                 Tile<T> C)
    {
        internal::check_gemm(A, B, C);

        return   blas::Gflop<T>::gemm(C.mb(), B.rk(), A.nb())
               + blas::Gflop<T>::gemm(C.mb(), C.nb(), B.rk());
    }

    static double gemm(CompressedTile<T> A,
                       CompressedTile<T> B,
                                 Tile<T> C)
    {
        internal::check_gemm(A, B, C);

        return   blas::Gflop<T>::gemm(A.rk(), B.rk(), A.nb())
               + (A.rk() <= B.rk()
                  ? blas::Gflop<T>::gemm(A.rk(), C.nb(), B.rk())
                    + blas::Gflop<T>::gemm(C.mb(), C.nb(),  A.rk())
                  : blas::Gflop<T>::gemm(C.mb(), B.rk(), A.rk())
                    + blas::Gflop<T>::gemm(C.mb(), C.nb(),  B.rk()));
    }

    static double gemm(          Tile<T> A,
                                 Tile<T> B,
                       CompressedTile<T> C)
    {
        internal::check_gemm(A, B, C);

        return   blas::Gflop<T>::gemm(C.mb(), C.nb(), A.nb())
               + blas::Gflop<T>::gemm(C.mb(), C.nb(), C.rk());
    }

    static double gemm(          Tile<T> A,
                       CompressedTile<T> B,
                       CompressedTile<T> C)
    {
        internal::check_gemm(A, B, C);

        return   blas::Gflop<T>::gemm(C.mb(), B.rk(), A.nb())
               + internal::Gflop<T>::rsvd(B.rk(), C);
    }

    static double gemm(CompressedTile<T> A,
                                 Tile<T> B,
                       CompressedTile<T> C)
    {
        internal::check_gemm(A, B, C);

        return   blas::Gflop<T>::gemm(A.rk(), C.nb(), A.nb())
               + internal::Gflop<T>::rsvd(A.rk(), C);
    }

    static double gemm(CompressedTile<T> A,
                       CompressedTile<T> B,
                       CompressedTile<T> C)
    {
        internal::check_gemm(A, B, C);

        return   blas::Gflop<T>::gemm(A.rk(), B.rk(), A.nb())
               + (A.rk() <= B.rk()
                  ? blas::Gflop<T>::gemm(A.rk(), C.nb(), B.rk())
                    + internal::Gflop<T>::rsvd(A.rk(), C)
                  : blas::Gflop<T>::gemm(C.mb(), B.rk(), A.rk())
                    + internal::Gflop<T>::rsvd(B.rk(), C));
    }
}; // class Gflop
}  // namespace hcore

#endif  // HCORE_FLOPS_HH
