// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_MATRIX_GENERATOR_HH
#define HCORE_TEST_MATRIX_GENERATOR_HH

#include <algorithm>
#include <cstdint>
#include <vector>

#include "testsweeper.hh"
#include "lapack.hh"
#include "blas.hh"

#include "lapack_wrappers.hh"
#include "print_matrix.hh"

namespace hcore {
namespace test {

template <typename T>
void compress_matrix(int64_t m, int64_t n, std::vector<T> A, int64_t lda,
                     T** UV, int64_t& ldu, int64_t& ldv, int64_t& rk,
                     blas::real_type<T> tol, int64_t align, bool verbose=false)
{
    int16_t min_m_n = std::min(m, n);

    std::vector<blas::real_type<T>> Sigma(min_m_n);

    // A
    // m-by-n
    // m-by-min_m_n      (U)
    //      min_m_n-by-n (VT)
    //
    // ldu  = lda
    // ldvt = min_m_n
    //
    // std::swap(m, n)
    // op(A)
    // n-by-m
    // n-by-min_m_n      (op(U))
    //      min_m_n-by-m (op(VT))
    //
    // ldut  = lda
    // ldvtt = min_m_n

    ldu = lda;
    int64_t ldvt = min_m_n;

    std::vector<T> U(ldu * min_m_n);
    std::vector<T> VT(ldvt * n);

    lapack::gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec,
                  m, n, &A[0],  lda, &Sigma[0],
                        &U[0],  ldu,
                        &VT[0], ldvt);

    if (verbose) {
        print_matrix( m,       n,       &A[0],  lda,  "Asvd"  );
        print_matrix( m,       min_m_n, &U[0],  ldu,  "Usvd"  );
        print_matrix( min_m_n, n,       &VT[0], ldvt, "VTsvd" );
    }

    rk = 0;
    while (Sigma[rk] >= tol && rk < min_m_n)
        rk++;

    // todo: more conservative max rank assumption, e.g., min_m_n / 3.
    int64_t max_rk = min_m_n / 2;
    if (rk > max_rk)
        rk = max_rk;

    // A
    // mb-by-nb
    // mb-by-rk       (U)
    //       rk-by-nb (V)
    //
    // ldu = lda
    // ldv = rk
    //
    // U(:,1:rk) * V(1:rk,:)
    //
    // UV
    //
    // std::swap(mb, nb)
    // op(A)
    // nb-by-mb
    // nb-by-rk       (VT)
    //       rk-by-mb (U)
    //
    // ldv = lda
    // ldu = rk
    //
    // VT(:,1:rk) * U(1:rk,:)
    //
    // VU

    // VT eats Sigma.
    // todo: we may need to have uplo parameter:
    //       scale VT, if Lower, or scale U otherwise.
    for (int64_t i = 0; i < rk; ++i)
        blas::scal(n, Sigma[i], &VT[i], ldvt);

    ldv = testsweeper::roundup(rk, align);

    *UV = new T[ldu*rk + ldv*n];
    T* Unew = *UV;
    T* Vnew = *UV + ldu*rk;

    // copy first rk cols of U; UV = U(:,1:rk)
    std::copy(U.begin(), U.begin() + ldu*rk, Unew);

    // copy first rk rows of VT; UV = VT(1:rk,:)
    lapack::lacpy(lapack::MatrixType::General, rk, n, &VT[0], ldvt, Vnew, ldv);
}

template <typename T>
void generate_matrix(int64_t m, int64_t n, std::vector<T>& A, int64_t lda,
                     int64_t* iseed, int64_t mode, blas::real_type<T> cond,
                     blas::real_type<T> dmax)
{
    int16_t min_m_n = std::min(m, n);

    std::vector<blas::real_type<T>> D(min_m_n);

    for (int64_t i = 0; i < min_m_n; ++i)
        D[i] = std::pow(10, -1*i);

    lapack::latms(m, n, lapack::Dist::Uniform, iseed, lapack::Sym::Nonsymmetric,
                  &D[0], mode, cond, dmax, m-1, n-1, lapack::Pack::NoPacking,
                  &A[0], lda);
}

} // namespace test
} // namespace hcore

#endif // HCORE_TEST_MATRIX_GENERATOR_HH