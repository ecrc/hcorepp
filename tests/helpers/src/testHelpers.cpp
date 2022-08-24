#include <iostream>
//#include <blas/util.hh>
//#include <vector>
#include <lapack/wrappers.hh>
#include <hcorepp/test-helpers/testHelpers.hpp>
#include <hcorepp/test-helpers/lapack_wrappers.hpp>
#include <libraries/catch/catch.hpp>

namespace hcorepp {
    namespace test_helpers {

        template<typename T>
        void
        rowMajorToColumnMajor(T *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, T *pOutputArray) {
            int index = 0;

            for (int64_t i = 0; i < aNumOfRows; i++) {
                for (int64_t j = 0; j < aNumOfCols; j++) {
                    int64_t out_index = j * aNumOfRows + i;
                    pOutputArray[out_index] = pInputArray[index];
                    index++;
                }
            }
        }

        template
        void
        rowMajorToColumnMajor(float *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, float *pOutputArray);

        template
        void
        rowMajorToColumnMajor(double *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, double *pOutputArray);

        template<typename T>
        void
        columnMajorToRowMajor(T *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, T *pOutputArray) {

            int index = 0;
            for (int64_t i = 0; i < aNumOfCols; i++) {
                for (int64_t j = 0; j < aNumOfRows; j++) {
                    int64_t in_index = i + (j * aNumOfCols);
                    pOutputArray[in_index] = pInputArray[index];
                    index++;
                }
            }
        }

        template
        void
        columnMajorToRowMajor(float *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, float *pOutputArray);

        template
        void
        columnMajorToRowMajor(double *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, double *pOutputArray);

        template<typename T>
        void
        printMatrix(T *pInput, int64_t aNumOfRows, int64_t aNumOfCols) {

            int index = 0;

            for (int i = 0; i < aNumOfRows; i++) {
                std::cout << "{ ";
                for (int j = 0; j < aNumOfCols; j++) {
                    index = i * aNumOfCols + j;
                    std::cout << pInput[index] << ", ";
                }
                std::cout << "} \n";
            }
        }

        template
        void
        printMatrix(float *pInput, int64_t aNumOfRows, int64_t aNumOfCols);

        template
        void
        printMatrix(double *pInput, int64_t aNumOfRows, int64_t aNumOfCols);

        template<typename T>
        void
        validateOutput(T *pInput, int64_t aNumOfRows, int64_t aNumOfCols, T *pExpectedOutput) {

            int index = 0;
            for (int i = 0; i < aNumOfRows * aNumOfCols; i++) {
                REQUIRE(pInput[i] == Approx(pExpectedOutput[i]).epsilon(1e-2));

            }

        }

        template
        void
        validateOutput(float *pInput, int64_t aNumOfRows, int64_t aNumOfCols, float *pExpectedOutput);

        template
        void
        validateOutput(double *pInput, int64_t aNumOfRows, int64_t aNumOfCols, double *pExpectedOutput);

    }
}
