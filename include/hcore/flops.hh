// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_FLOPS_HH
#define HCORE_FLOPS_HH

#include "hcore/check.hh"
#include "hcore/tile.hh"
#include "hcore/compressed_tile.hh"

#include "blas/flops.hh"
#include "lapack/flops.hh"

namespace hcore {
namespace internal {

inline double fmuls_gesvd(double m, double n)
    { return 14*m*n*n; }

inline double fadds_gesvd(double m, double n)
    { return 8*n*n*n; }

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
        { return 1e-9*(mul_ops*fmuls_gesvd(m, n)+add_ops*fadds_gesvd(m, n)); }

    // randomized svd
    static double rsvd(int64_t Ark, int64_t Crk_org, CompressedTile<T>& C)
    {
        double m = double(C.m());
        double n = double(C.n());
        double rk = double(Ark + Crk_org);
        double min_m_rk = double(std::min(C.m(), Ark + Crk_org));
        double min_n_rk = double(std::min(C.n(), Ark + Crk_org));

        return lapack::Gflop<T>::geqrf(m, rk) +
               lapack::Gflop<T>::geqrf(n, rk) +
               blas::Gflop<T>::gemm(min_m_rk, min_n_rk, rk) +
               lapack_gesvd(min_m_rk, min_n_rk) +
               lapack::Gflop<T>::ungqr(m, min_m_rk, min_m_rk) +
               blas::Gflop<T>::gemm(m, C.rk(), min_m_rk) +
               blas::Gflop<T>::scal(min_n_rk) +
               lapack::Gflop<T>::ungqr(n, min_n_rk, min_n_rk) +
               blas::Gflop<T>::gemm(n, C.rk(), min_n_rk);
    }
}; // class Gflop

} // namespace internal

template <typename T>
class Gflop
{
public:
    static double gemm(
        Tile<T> A, Tile<T> B, Tile<T> C)
    {
        internal::check::gemm(A, B, C);

        return blas::Gflop<T>::gemm(C.m(), C.n(), A.n());
    }
    static double gemm(
        Tile<T> A, Tile<T> B, CompressedTile<T> C,
        int64_t Crk_org)
    {
        internal::check::gemm(A, B, C);

        return blas::Gflop<T>::gemm(C.m(), C.n(), A.n()) +
               blas::Gflop<T>::gemm(C.m(), C.n(), Crk_org);
    }
    static double gemm(
        Tile<T> A, CompressedTile<T> B, Tile<T> C)
    {
        internal::check::gemm(A, B, C);

        return blas::Gflop<T>::gemm(C.m(), B.rk(), A.n()) +
               blas::Gflop<T>::gemm(C.m(), C.n(), B.rk());
    }
    static double gemm(
        Tile<T> A, CompressedTile<T> B, CompressedTile<T> C,
        int64_t Crk_org)
    {
        internal::check::gemm(A, B, C);

        return blas::Gflop<T>::gemm(C.m(), B.rk(), A.n()) +
               internal::Gflop<T>::rsvd(B.rk(), Crk_org, C);

    }
    static double gemm(
        CompressedTile<T> A, Tile<T> B, Tile<T> C)
    {
        internal::check::gemm(A, B, C);

        return blas::Gflop<T>::gemm(A.rk(), C.n(), A.n()) +
               blas::Gflop<T>::gemm(C.m(), C.n(), A.rk());

    }
    static double gemm(
        CompressedTile<T> A, Tile<T> B, CompressedTile<T> C,
        int64_t Crk_org)
    {
        internal::check::gemm(A, B, C);

        return blas::Gflop<T>::gemm(A.rk(), C.n(), A.n()) +
               internal::Gflop<T>::rsvd(A.rk(), Crk_org, C);
    }
    static double gemm(
        CompressedTile<T> A, CompressedTile<T> B, Tile<T> C)
    {
        internal::check::gemm(A, B, C);

        return blas::Gflop<T>::gemm(A.rk(), B.rk(), A.n()) +
               (A.rk() <= B.rk() ? blas::Gflop<T>::gemm(A.rk(), C.n(), B.rk()) +
                                   blas::Gflop<T>::gemm(C.m(), C.n(),  A.rk())
                                 : blas::Gflop<T>::gemm(C.m(), B.rk(), A.rk()) +
                                   blas::Gflop<T>::gemm(C.m(), C.n(),  B.rk()));
    }
    static double gemm(
        CompressedTile<T> A, CompressedTile<T> B, CompressedTile<T> C,
        int64_t Crk_org)
    {
        internal::check::gemm(A, B, C);

        return blas::Gflop<T>::gemm(A.rk(), B.rk(), A.n()) +
              (A.rk() <= B.rk() ? blas::Gflop<T>::gemm(A.rk(), C.n(), B.rk()) +
                                  internal::Gflop<T>::rsvd(A.rk(), Crk_org, C)
                                : blas::Gflop<T>::gemm(C.m(), B.rk(), A.rk()) +
                                  internal::Gflop<T>::rsvd(B.rk(), Crk_org, C));
    }
}; // class Gflop
}  // namespace hcore

#endif  // HCORE_FLOPS_HH
