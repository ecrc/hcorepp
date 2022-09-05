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
        void
        printMatrix(T *pInput, int64_t aNumOfRows, int64_t aNumOfCols);

        template<typename T>
        void
        validateOutput(T *pInput, int64_t aNumOfRows, int64_t aNumOfCols, T *pExpectedOutput);

        template<typename T>
        void
        validateCompressedOutput(T *pInputA, int64_t aNumOfRowsA, int64_t aNumOfColsA, T *pExpectedOutputA,
                                 T *pInputB, int64_t aNumOfRowsB, int64_t aNumOfColsB, T *pExpectedOutputB);

    }
}
#endif //HCOREPP_TEST_HELPERS_HPP

