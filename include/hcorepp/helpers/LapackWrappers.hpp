/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCORE_HELPERS_LAPACK_WRAPPERS_HPP
#define HCORE_HELPERS_LAPACK_WRAPPERS_HPP

#include <vector>
#include <complex>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

// Include a CPU-based Lapack to be utilized in the wrappers
// This was not GPU offloaded since most of it is for the matrix generation though if
// such use case arises this part might be revisited.
// Check if using OpenBLAS, import its lapack, otherwise use default lapack directory.
#if __has_include("openblas/lapack.h")
#include <openblas/lapack.h>
#undef LAPACK_FORTRAN_STRLEN_END
#else

#include <lapack/fortran.h>

#endif

#include "hcorepp/common/TypeCheck.hpp"
#include <hcorepp/operators/helpers/SVDParameters.hpp>

// Define appropriate fortran ending arguments that are needed
// sometime by the lapack library in case of building for CPU support.
#define HCOREPP_LATMS_ENDING
#define HCOREPP_LACPY_ENDING
#define HCOREPP_GESVD_ENDING
#define HCOREPP_LANGE_ENDING
#ifdef LAPACK_FORTRAN_STRLEN_END
#ifndef USE_CUDA
#undef HCOREPP_LATMS_ENDING
#undef HCOREPP_LACPY_ENDING
#undef HCOREPP_GESVD_ENDING
#undef HCOREPP_LANGE_ENDING
#define HCOREPP_LATMS_ENDING     , 1, 1, 1
#define HCOREPP_LACPY_ENDING     , 1
#define HCOREPP_GESVD_ENDING     , 1, 1
#define HCOREPP_LANGE_ENDING     , 1
#endif
#endif

template<typename T, typename C>
void lapack_latms_core(int64_t m, int64_t n, char dist, int64_t *iseed, char sym,
                       T *d, int64_t mode, T cond, T dmax, int64_t kl, int64_t ku,
                       char pack, C *A, int64_t lda) {
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        assert(std::abs(m) <= std::numeric_limits<lapack_int>::max());
        assert(std::abs(n) <= std::numeric_limits<lapack_int>::max());
        assert(std::abs(kl) <= std::numeric_limits<lapack_int>::max());
        assert(std::abs(ku) <= std::numeric_limits<lapack_int>::max());
        assert(std::abs(lda) <= std::numeric_limits<lapack_int>::max());
        assert(std::abs(mode) <= std::numeric_limits<lapack_int>::max());
    }

    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int kl_ = (lapack_int) kl;
    lapack_int ku_ = (lapack_int) ku;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int mode_ = (lapack_int) mode;

    std::vector<C> work(3 * std::max(m, n));

#ifndef HCORE_WITH_LAPACK_ILP64
    // 32-bit copy
    std::vector<lapack_int> iseed_(&iseed[0], &iseed[(4)]);
    lapack_int *iseed_ptr = &iseed_[0];
#else
    lapack_int* iseed_ptr = iseed;
#endif

    lapack_int info_ = 0;

    if constexpr(is_complex<C>()) {
        if constexpr(is_double<T>()) {
            LAPACK_zlatms(
                    &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
                    &pack, (lapack_complex_double *) A, &lda_,
                    (lapack_complex_double *) &work[0], &info_ HCOREPP_LATMS_ENDING);
        } else {
            LAPACK_clatms(
                    &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
                    &pack, (lapack_complex_float *) A, &lda_,
                    (lapack_complex_float *) &work[0], &info_ HCOREPP_LATMS_ENDING);
        }
    } else {
        if constexpr (is_double<T>()) {
            LAPACK_dlatms(
                    &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
                    &pack, A, &lda_, &work[0], &info_ HCOREPP_LATMS_ENDING);
        } else {
            LAPACK_slatms(
                    &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
                    &pack, A, &lda_, &work[0], &info_ HCOREPP_LATMS_ENDING);
        }
    }
    if (info_ != 0) {
        throw std::runtime_error("lapack_latms error " + std::to_string(info_));
    }

#ifndef HCORE_WITH_LAPACK_ILP64
    std::copy(iseed_.begin(), iseed_.end(), iseed);
#endif
}

template<typename T>
void lapack_latms(int64_t m, int64_t n, char dist, int64_t *iseed, char sym,
                  T *d, int64_t mode, T cond, T dmax, int64_t kl, int64_t ku,
                  char pack, T *A, int64_t lda) {
    lapack_latms_core<T, T>(m, n, dist, iseed, sym, d, mode, cond, dmax, kl, ku, pack, A, lda);
}

template<typename T>
void lapack_latms(int64_t m, int64_t n, char dist, int64_t *iseed, char sym,
                  T *d, int64_t mode, T cond, T dmax, int64_t kl, int64_t ku,
                  char pack, std::complex<T> *A, int64_t lda) {
    lapack_latms_core<T, std::complex<T>>(m, n, dist, iseed, sym, d, mode, cond, dmax, kl, ku, pack, A, lda);
}

template<typename T, typename C>
void lapack_lacpy_core(hcorepp::common::MatrixType matrixtype, int64_t m, int64_t n,
                       C const *A, int64_t lda, C *B, int64_t ldb) {
    char matrix_type = (char) matrixtype;
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldb_ = (lapack_int) ldb;
    if constexpr(is_complex<C>()) {
        if constexpr(is_double<T>()) {
            LAPACK_zlacpy(&matrix_type, &m_, &n_, A, &lda_, B, &ldb_ HCOREPP_LACPY_ENDING);
        } else {
            LAPACK_clacpy(&matrix_type, &m_, &n_, A, &lda_, B, &ldb_ HCOREPP_LACPY_ENDING);
        }
    } else {
        if constexpr (is_double<T>()) {
            LAPACK_dlacpy(&matrix_type, &m_, &n_, A, &lda_, B, &ldb_ HCOREPP_LACPY_ENDING);
        } else {
            LAPACK_slacpy(&matrix_type, &m_, &n_, A, &lda_, B, &ldb_ HCOREPP_LACPY_ENDING);
        }
    }
}

template<typename T>
void lapack_lacpy(hcorepp::common::MatrixType matrixtype, int64_t m, int64_t n,
                  T const *A, int64_t lda, T *B, int64_t ldb) {
    lapack_lacpy_core<T, T>(matrixtype, m, n, A, lda, B, ldb);
}

template<typename T>
void lapack_lacpy(hcorepp::common::MatrixType matrixtype, int64_t m, int64_t n,
                  std::complex<T> const *A, int64_t lda, std::complex<T> *B, int64_t ldb) {
    lapack_lacpy_core<T, std::complex<T>>(matrixtype, m, n, A, lda, B, ldb);
}

template<typename T, typename C>
void lapack_gesvd_core(hcorepp::common::Job jobu, hcorepp::common::Job jobvt,
                       int64_t m, int64_t n, C *A, int64_t lda, T *S,
                       C *U, int64_t ldu, C *VT, int64_t ldvt) {
    char job_u = (char) jobu;
    char job_vt = (char) jobvt;
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int ldu_ = (lapack_int) ldu;
    lapack_int ldvt_ = (lapack_int) ldvt;
    lapack_int info_ = 0;

    // query for workspace size
    C qry_work[1];
    lapack_int ineg_one = -1;
    if constexpr(is_complex<C>()) {
        if constexpr(is_double<T>()) {
            LAPACK_zgesvd(&job_u, &job_vt, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, qry_work,
                          &ineg_one, &info_ HCOREPP_GESVD_ENDING);
        } else {
            LAPACK_cgesvd(&job_u, &job_vt, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, qry_work,
                          &ineg_one, &info_ HCOREPP_GESVD_ENDING);
        }
    } else {
        if constexpr (is_double<T>()) {
            LAPACK_dgesvd(&job_u, &job_vt, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, qry_work,
                          &ineg_one, &info_ HCOREPP_GESVD_ENDING);
        } else {
            LAPACK_sgesvd(&job_u, &job_vt, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, qry_work,
                          &ineg_one, &info_ HCOREPP_GESVD_ENDING);
        }
    }
    lapack_int lwork_ = qry_work[0];

    // allocate workspace
    std::vector<C> work(lwork_);

    if constexpr(is_complex<C>()) {
        if constexpr(is_double<T>()) {
            LAPACK_zgesvd(&job_u, &job_vt, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, &work[0],
                          &lwork_, &info_ HCOREPP_GESVD_ENDING);
        } else {
            LAPACK_cgesvd(&job_u, &job_vt, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, &work[0],
                          &lwork_, &info_ HCOREPP_GESVD_ENDING);
        }
    } else {
        if constexpr (is_double<T>()) {
            LAPACK_dgesvd(&job_u, &job_vt, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, &work[0],
                          &lwork_, &info_ HCOREPP_GESVD_ENDING);
        } else {
            LAPACK_sgesvd(&job_u, &job_vt, &m_, &n_, A, &lda_, S, U, &ldu_, VT, &ldvt_, &work[0],
                          &lwork_, &info_ HCOREPP_GESVD_ENDING);
        }
    }
}

template<typename T>
void lapack_gesvd(hcorepp::common::Job jobu, hcorepp::common::Job jobvt,
                  int64_t m, int64_t n, T *A, int64_t lda, T *S,
                  T *U, int64_t ldu, T *VT, int64_t ldvt) {
    lapack_gesvd_core<T, T>(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt);
}

template<typename T>
void lapack_gesvd(hcorepp::common::Job jobu, hcorepp::common::Job jobvt,
                  int64_t m, int64_t n, std::complex<T> *A, int64_t lda, T *S,
                  std::complex<T> *U, int64_t ldu, std::complex<T> *VT, int64_t ldvt) {
    lapack_gesvd_core<T, std::complex<T>>(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt);
}

template<typename T, typename C>
blas::real_type<T> lapack_lange_core(hcorepp::common::Norm aNorm, int64_t m, int64_t n,
                                     C const *A, int64_t lda) {
    char norm_type = (char) aNorm;
    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int lwork = std::max(1, m_);
    std::vector<blas::real_type<T>> t(lwork);
    if constexpr(is_complex<C>()) {
        if constexpr(is_double<T>()) {
            return LAPACK_zlange(&norm_type, &m_, &n_, A, &lda_, &t[0]HCOREPP_LANGE_ENDING);
        } else {
            return LAPACK_clange(&norm_type, &m_, &n_, A, &lda_, &t[0]HCOREPP_LANGE_ENDING);
        }
    } else {
        if constexpr (is_double<T>()) {
            return LAPACK_dlange(&norm_type, &m_, &n_, A, &lda_, &t[0]HCOREPP_LANGE_ENDING);
        } else {
            return LAPACK_slange(&norm_type, &m_, &n_, A, &lda_, &t[0]HCOREPP_LANGE_ENDING);
        }
    }
}

template<typename T>
blas::real_type<T> lapack_lange(hcorepp::common::Norm aNorm, int64_t m, int64_t n,
                                T const *A, int64_t lda) {
    return lapack_lange_core<T, T>(aNorm, m, n, A, lda);
}

template<typename T>
blas::real_type<T> lapack_lange(hcorepp::common::Norm aNorm, int64_t m, int64_t n,
                                std::complex<T> const *A, int64_t lda) {
    return lapack_lange_core<T, std::complex<T>>(aNorm, m, n, A, lda);
}

#endif //HCORE_HELPERS_LAPACK_WRAPPERS_HPP
