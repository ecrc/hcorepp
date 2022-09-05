#include <cstring>
#include <lapack/wrappers.hh>
#include <hcorepp/helpers/MatrixHelpers.hpp>
#include "hcorepp/helpers/lapack_wrappers.hpp"

namespace hcorepp {
    namespace helpers {
        namespace matrixhelpers {
            template<typename T>
            void generate_dense_matrix(int64_t m, int64_t n, T *A, int64_t lda, int64_t *iseed, int64_t mode,
                                       blas::real_type<T> cond) {
                int16_t min_m_n = std::min(m, n);

                blas::real_type<T> *D = (blas::real_type<T> *) malloc(min_m_n * sizeof(blas::real_type<T>));

                for (int64_t i = 0; i < min_m_n; ++i)
                    D[i] = std::pow(10, -1 * i);

                lapack_latms(
                        m, n, 'U', iseed, 'N', D, mode, cond, -1.0, m - 1, n - 1, 'N', A, lda);

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
                int16_t min_m_n = std::min(m, n);

                blas::real_type<T> *Sigma = (blas::real_type<T> *) malloc(min_m_n * sizeof(blas::real_type<T>));
                T *U = (T *) malloc(lda * min_m_n * sizeof(T));
                T *VT = (T *) malloc(min_m_n * n * sizeof(T));

                T *a_temp = (T *) malloc(m * n * sizeof(T));
                memcpy((void *) a_temp, (void *) A, m * n * sizeof(T));
                lapack::gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec, m, n, a_temp, lda, Sigma, U, lda, VT, min_m_n);

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
                    blas::scal(n, Sigma[i], &VT[i], min_m_n);
                }

                *UV = (T *) malloc((lda + n) * rk * sizeof(T));

                memcpy((void *) (*UV), (void *) U, (lda * rk) * sizeof(T));

                // copy first rk rows of VT; UV = VT(1:rk,:)
                // todo: assume column-major, what about row-major?
                lapack::lacpy(
                        lapack::MatrixType::General, rk, n, VT, min_m_n, &(*UV)[lda * rk], rk);

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