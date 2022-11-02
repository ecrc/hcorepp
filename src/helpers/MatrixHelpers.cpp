/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#include <cstring>
#include <hcorepp/helpers/MatrixHelpers.hpp>
#include <iostream>
#include "hcorepp/helpers//LapackWrappers.hpp"
#if __has_include("openblas/lapack.h")
#include <openblas/lapack.h>
#else
#include <lapack/fortran.h>
#endif

namespace hcorepp {
    namespace helpers {
        namespace matrixhelpers {
            template<typename T>
            void generate_dense_matrix(int64_t m, int64_t n, T *A, int64_t lda, int64_t *iseed, int64_t mode,
                                       blas::real_type<T> cond) {
                int64_t min_m_n = std::min(m, n);

                blas::real_type<T> *D = (blas::real_type<T> *) malloc(min_m_n * sizeof(blas::real_type<T>));

                for (int64_t i = 0; i < min_m_n; ++i){
                    D[i] = std::pow(10, -1 * i);
                }
                T dmax = -1.0;
                lapack_latms(
                        m, n, 'U', iseed, 'N', D, mode, cond, dmax, m - 1, n - 1, 'N', A, lda);

                free(D);
            }

            template
            void
            generate_dense_matrix(int64_t m, int64_t n, float *A, int64_t lda, int64_t *iseed, int64_t mode,
                                  blas::real_type<float> cond);

            template
            void
            generate_dense_matrix(int64_t m, int64_t n, double *A, int64_t lda, int64_t *iseed, int64_t mode,
                                  blas::real_type<double> cond);

            template<typename T>
            void compress_dense_matrix(int64_t m, int64_t n, const T *A, int64_t lda, T **UV, int64_t &rk,
                                       blas::real_type<T> accuracy) {
                int64_t min_m_n = std::min(m, n);

                blas::real_type<T> *Sigma = (blas::real_type<T> *) malloc(min_m_n * sizeof(blas::real_type<T>));
                T *U = (T *) malloc(lda * min_m_n * sizeof(T));
                T *VT = (T *) malloc(min_m_n * n * sizeof(T));

                T *a_temp = (T *) malloc(m * n * sizeof(T));
                memcpy((void *) a_temp, (void *) A, m * n * sizeof(T));
                lapack_gesvd(common::Job::SomeVec, common::Job::SomeVec, m, n, a_temp, lda, Sigma, U, lda, VT,
                              min_m_n);

                rk = 0;
                while (Sigma[rk] >= accuracy && rk < min_m_n) {
                    rk++;
                    if (rk < min_m_n) {
                        continue;
                    } else {
                        break;
                    }

                }

                // todo: more conservative max rank assumption, e.g., min_m_n / 3.
                int64_t max_rk = min_m_n / 2;
                if (rk > max_rk) {
                    rk = max_rk;
                }

                // VT eats Sigma.
                // todo: we may need to have uplo parameter:
                //       scale VT, if Lower, or scale U otherwise.
                for (int64_t i = 0; i < rk; ++i) {
                    for (int j = 0; j < n; j++) {
                        VT[i + j * min_m_n] *= Sigma[i];
                    }
                }

                *UV = (T *) malloc((lda + n) * rk * sizeof(T));

                memcpy((void *) (*UV), (void *) U, (lda * rk) * sizeof(T));

                // todo: assume column-major, what about row-major?
                lapack_lacpy(common::MatrixType::General, rk, n, VT, min_m_n, &(*UV)[lda * rk], rk);

                free(U);
                free(VT);
                free(Sigma);
                free(a_temp);
            }

            template
            void compress_dense_matrix(int64_t m, int64_t n, const float *A, int64_t lda, float **UV, int64_t &rk,
                                       blas::real_type<float> accuracy);

            template
            void compress_dense_matrix(int64_t m, int64_t n, const double *A, int64_t lda, double **UV, int64_t &rk,
                                       blas::real_type<double> accuracy);

            template<typename T>
            void diff(T *Aref, int64_t lda_ref, T const *A, int64_t m, int64_t n, int64_t lda) {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = 0; i < m; ++i) {
                        Aref[i + j * lda_ref] -= A[i + j * lda];
                    }
                }
            }

            template
            void diff(float *Aref, int64_t lda_ref, float const *A, int64_t m, int64_t n, int64_t lda);

            template
            void diff(double *Aref, int64_t lda_ref, double const *A, int64_t m, int64_t n, int64_t lda);

        }
    }
}