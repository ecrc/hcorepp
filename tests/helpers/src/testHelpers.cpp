#include <iostream>
//#include <blas/util.hh>
//#include <vector>
#include <lapack/wrappers.hh>
#include <hcorepp/test-helpers/testHelpers.hpp>
#include <hcorepp/test-helpers/lapack_wrappers.hpp>


namespace hcorepp {
    namespace test_helpers {

        template<typename T>
        void
        rowMajorToColumnMajor(T *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, T *pOutputArray) {
            int index = 0;
            for (int i = 0; i < aNumOfCols; i++) {
                for (int j = 0; j < aNumOfRows; j++) {
                    int in_index = j * aNumOfCols + i;
                    pOutputArray[index] = pInputArray[in_index];
//                    std::cout << " a[ " << index << "]" << " == " << pOutputArray[index] << "\n";
                    index++;
                }
            }

        }

        template
        void
        rowMajorToColumnMajor(float *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, float *pOutputArray);

        template<typename T>
        void
        columnMajorToRowMajor(T *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, T *pOutputArray) {

//            std::cout << " Starting Column major to Row major conversion \n";
            int index = 0;
            for (int i = 0; i < aNumOfCols; i++) {
                for (int j = 0; j < aNumOfRows; j++) {
                    int in_index = i + (j * aNumOfCols);
//                    std::cout << " INput index = " << in_index << "\n";
                    pOutputArray[in_index] = pInputArray[index];
//                    std::cout << " a[ " << index << "]" << " == " << pOutputArray[in_index] << "\n";
                    index++;
                }
            }
        }

        template
        void
        columnMajorToRowMajor(float *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, float *pOutputArray);

        template<typename T>
        void compress_dense_matrix(int64_t m, int64_t n, std::vector<T> A, int64_t lda, std::vector<T> &UV, int64_t &rk,
                                   blas::real_type<T> accuracy) {
            int16_t min_m_n = std::min(m, n);

            std::vector<blas::real_type<T>> Sigma(min_m_n);

            std::vector<T> U(lda * min_m_n);
            std::vector<T> VT(min_m_n * n);

            lapack::gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec,
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
            lapack::lacpy(lapack::MatrixType::General, rk, n, &VT[0], min_m_n, &UV[lda * rk], rk);
        }

        template
        void compress_dense_matrix(int64_t m, int64_t n, std::vector<float> A, int64_t lda, std::vector<float> &UV,
                                   int64_t &rk, blas::real_type<float> accuracy);

        template<typename T>
        void generate_dense_matrix(int64_t m, int64_t n, T *A, int64_t lda, int64_t *iseed, int64_t mode,
                                   blas::real_type<T> cond) {
            int16_t min_m_n = std::min(m, n);

            std::vector<blas::real_type<T>> D(min_m_n);

            for (int64_t i = 0; i < min_m_n; ++i)
                D[i] = std::pow(10, -1 * i);

            lapack_latms(m, n, 'U', iseed, 'N', &D[0], mode, cond, -1.0, m - 1, n - 1, 'N', A, lda);
        }

        template
        void generate_dense_matrix(int64_t m, int64_t n, float *A, int64_t lda, int64_t *iseed, int64_t mode,
                                   blas::real_type<float> cond);

    }
}
