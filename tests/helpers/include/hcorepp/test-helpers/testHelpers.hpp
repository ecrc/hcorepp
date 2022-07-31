#ifndef HCOREPP_TEST_HELPERS_HPP
#define HCOREPP_TEST_HELPERS_HPP

#include <cstdint>
#include <vector>
#include <blas/util.hh>

namespace hcorepp {
    namespace test_helpers {

        template<typename T>
        void
        rowMajorToColumnMajor(T *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, T *pOutputArray);

        template<typename T>
        void
        columnMajorToRowMajor(T *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, T *pOutputArray);

        template<typename T>
        void compress_dense_matrix(int64_t m, int64_t n, std::vector<T> A, int64_t lda, std::vector<T> &UV, int64_t &rk,
                                   blas::real_type<T> accuracy);

        template<typename T>
        void generate_dense_matrix(int64_t m, int64_t n, T *A, int64_t lda, int64_t *iseed, int64_t mode = 0,
                                   blas::real_type<T> cond = 1);
    }
}
#endif //HCOREPP_TEST_HELPERS_HPP

