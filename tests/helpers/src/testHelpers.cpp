/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <iostream>
#include <libraries/catch/catch.hpp>
#include <cstring>

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


        template<typename T>
        void
        validateCompressedOutput(T *pInputA, int64_t aNumOfRowsA, int64_t aNumOfColsA, T *pExpectedOutputA,
                                 T *pInputB, int64_t aNumOfRowsB, int64_t aNumOfColsB, T *pExpectedOutputB) {
            bool sign_reversed = false;

            if (pInputA[0] == Approx(pExpectedOutputA[0]).epsilon(1e-2)) {
                sign_reversed = false;
            } else if (pInputA[0] == Approx(-pExpectedOutputA[0]).epsilon(1e-2)) {
                sign_reversed = true;
            }

            for (int i = 0; i < aNumOfRowsA * aNumOfColsA; i++) {
                if (sign_reversed) {
                    REQUIRE(pInputA[i] == Approx(-pExpectedOutputA[i]).epsilon(1e-2));
                } else {
                    REQUIRE(pInputA[i] == Approx(pExpectedOutputA[i]).epsilon(1e-2));
                }
            }

            for (int i = 0; i < aNumOfRowsB * aNumOfColsB; i++) {
                if (sign_reversed) {
                    REQUIRE(pInputB[i] == Approx(-pExpectedOutputB[i]).epsilon(1e-2));
                } else {
                    REQUIRE(pInputB[i] == Approx(pExpectedOutputB[i]).epsilon(1e-2));
                }
            }
        }

        template
        void
        validateCompressedOutput(float *pInputA, int64_t aNumOfRowsA, int64_t aNumOfColsA, float *pExpectedOutputA,
                                 float *pInputB, int64_t aNumOfRowsB, int64_t aNumOfColsB, float *pExpectedOutputB);

        template
        void
        validateCompressedOutput(double *pInputA, int64_t aNumOfRowsA, int64_t aNumOfColsA, double *pExpectedOutputA,
                                 double *pInputB, int64_t aNumOfRowsB, int64_t aNumOfColsB, double *pExpectedOutputB);
    }
}
