// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_MATRIX_UTILS_HH
#define HCORE_TEST_MATRIX_UTILS_HH

#include "lapacke_wrappers.hh"

#include "blas.hh"
#include "lapack.hh"

#include <cmath>
#include <vector>
#include <cstdint>
#include <cassert>
#include <algorithm>

template <typename T>
void generate_dense_matrix(
    int64_t m, int64_t n, T* A, int64_t lda,
    int* iseed, int mode=0, blas::real_type<T> cond=1)
{
    int16_t min_m_n = std::min(m, n);

    std::vector<blas::real_type<T>> D(min_m_n);

    for (int64_t i = 0; i < min_m_n; ++i)
        D[i] = std::pow(10, -1*i);

    lapacke_latms(
        m, n, 'U', iseed, 'N', &D[0], mode, cond, -1.0, m-1, n-1, 'N', A, lda);
}

template <typename T>
void set_dense_uplo(
    blas::Uplo uplo, int64_t m, int64_t n, T* A, int64_t lda, T v = nan(""))
{
    if (uplo == blas::Uplo::Lower) {
        lapack::laset(
            lapack::MatrixType::Upper, m-1, n-1, v, v, &A[0+1*lda], lda);
    }
    else if (uplo == blas::Uplo::Upper) {
        lapack::laset(
            lapack::MatrixType::Lower, m-1, n-1, v, v, &A[1+0*lda], lda);
    }
    else {
        assert(false);
    }
}

template <typename T>
void compress_dense_matrix(
    int64_t m, int64_t n, std::vector<T> A, int64_t lda,
    std::vector<T>& UV, int64_t& rk, blas::real_type<T> accuracy)
{
    int16_t min_m_n = std::min(m, n);

    std::vector<blas::real_type<T>> Sigma(min_m_n);

    std::vector<T> U(lda * min_m_n);
    std::vector<T> VT(min_m_n * n);

    lapack::gesvd(
        lapack::Job::SomeVec, lapack::Job::SomeVec,
        m, n, &A[0], lda, &Sigma[0],
              &U[0], lda,
              &VT[0], min_m_n);

    rk = 0;
    while (Sigma[rk] >= accuracy && rk < min_m_n)
        rk++;

    // todo: more conservative max rank assumption, e.g., min_m_n / 3.
    int64_t max_rk = min_m_n / 2;
    if (rk > max_rk)
        rk = max_rk;

    // VT eats Sigma.
    // todo: we may need to have uplo parameter:
    //       scale VT, if Lower, or scale U otherwise.
    for (int64_t i = 0; i < rk; ++i)
        blas::scal(n, Sigma[i], &VT[i], min_m_n);

    UV.reserve((lda + n) * rk);

    // copy first rk columns of U; UV = U(:,1:rk)
    // todo: assume column-major, what about row-major?
    UV.insert(UV.end(), U.begin(), U.begin() + (lda * rk));

    // copy first rk rows of VT; UV = VT(1:rk,:)
    // todo: assume column-major, what about row-major?
    lapack::lacpy(
        lapack::MatrixType::General, rk, n, &VT[0], min_m_n, &UV[lda * rk], rk);
}

#endif // HCORE_TEST_MATRIX_UTILS_HH
