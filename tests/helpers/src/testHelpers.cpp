/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <iostream>
#include <catch2/catch_all.hpp>
#include <cstring>

namespace hcorepp {
    namespace test_helpers {

        template<typename T>
        void
        rowMajorToColumnMajor(T *pInputArray, size_t aNumOfCols, size_t aNumOfRows, T *pOutputArray) {
            size_t index = 0;

            for (size_t i = 0; i < aNumOfRows; i++) {
                for (size_t j = 0; j < aNumOfCols; j++) {
                    size_t out_index = j * aNumOfRows + i;
                    pOutputArray[out_index] = pInputArray[index];
                    index++;
                }
            }
        }

        template
        void
        rowMajorToColumnMajor(float *pInputArray, size_t aNumOfCols, size_t aNumOfRows, float *pOutputArray);

        template
        void
        rowMajorToColumnMajor(double *pInputArray, size_t aNumOfCols, size_t aNumOfRows, double *pOutputArray);

        template<typename T>
        void
        columnMajorToRowMajor(T *pInputArray, size_t aNumOfCols, size_t aNumOfRows, T *pOutputArray) {

            size_t index = 0;
            for (size_t i = 0; i < aNumOfCols; i++) {
                for (size_t j = 0; j < aNumOfRows; j++) {
                    size_t in_index = i + (j * aNumOfCols);
                    pOutputArray[in_index] = pInputArray[index];
                    index++;
                }
            }
        }

        template
        void
        columnMajorToRowMajor(float *pInputArray, size_t aNumOfCols, size_t aNumOfRows, float *pOutputArray);

        template
        void
        columnMajorToRowMajor(double *pInputArray, size_t aNumOfCols, size_t aNumOfRows, double *pOutputArray);

        template<typename T>
        void
        printMatrix(T *pInput, size_t aNumOfRows, size_t aNumOfCols) {

            size_t index = 0;

            for (size_t i = 0; i < aNumOfRows; i++) {
                std::cout << "{ ";
                for (size_t j = 0; j < aNumOfCols; j++) {
                    index = i * aNumOfCols + j;
                    std::cout << pInput[index] << ", ";
                }
                std::cout << "} \n";
            }
        }

        template
        void
        printMatrix(float *pInput, size_t aNumOfRows, size_t aNumOfCols);

        template
        void
        printMatrix(double *pInput, size_t aNumOfRows, size_t aNumOfCols);

        template<typename T>
        void
        validateOutput(T *pInput, size_t aNumOfRows, size_t aNumOfCols, T *pExpectedOutput) {

            for (size_t i = 0; i < aNumOfRows * aNumOfCols; i++) {
                REQUIRE(pInput[i] == Catch::Approx(pExpectedOutput[i]).epsilon(1e-2));
            }
        }

        template<typename T>
        void
        validateOutputLenient(T *pInput, size_t aNumOfRows, size_t aNumOfCols, T *pExpectedOutput) {

            for (size_t i = 0; i < aNumOfRows * aNumOfCols; i++) {
                REQUIRE(std::abs(pInput[i] - pExpectedOutput[i]) <= 1.5);
            }
        }

        template
        void
        validateOutput(float *pInput, size_t aNumOfRows, size_t aNumOfCols, float *pExpectedOutput);

        template
        void
        validateOutput(double *pInput, size_t aNumOfRows, size_t aNumOfCols, double *pExpectedOutput);

        template
        void
        validateOutputLenient(float *pInput, size_t aNumOfRows, size_t aNumOfCols, float *pExpectedOutput);

        template
        void
        validateOutputLenient(double *pInput, size_t aNumOfRows, size_t aNumOfCols, double *pExpectedOutput);


        template<typename T>
        void
        validateCompressedOutput(T *pInputA, size_t aNumOfRowsA, size_t aNumOfColsA, T *pExpectedOutputA,
                                 T *pInputB, size_t aNumOfRowsB, size_t aNumOfColsB, T *pExpectedOutputB) {
            bool sign_reversed = false;

            if (pInputA[0] == Catch::Approx(pExpectedOutputA[0]).epsilon(1e-2)) {
                sign_reversed = false;
            } else if (pInputA[0] == Catch::Approx(-pExpectedOutputA[0]).epsilon(1e-2)) {
                sign_reversed = true;
            }

            for (size_t i = 0; i < aNumOfRowsA * aNumOfColsA; i++) {
                if (sign_reversed) {
                    REQUIRE(pInputA[i] == Catch::Approx(-pExpectedOutputA[i]).epsilon(1e-2));
                } else {
                    REQUIRE(pInputA[i] == Catch::Approx(pExpectedOutputA[i]).epsilon(1e-2));
                }
            }

            for (size_t i = 0; i < aNumOfRowsB * aNumOfColsB; i++) {
                if (sign_reversed) {
                    REQUIRE(pInputB[i] == Catch::Approx(-pExpectedOutputB[i]).epsilon(1e-2));
                } else {
                    REQUIRE(pInputB[i] == Catch::Approx(pExpectedOutputB[i]).epsilon(1e-2));
                }
            }
        }

        template
        void
        validateCompressedOutput(float *pInputA, size_t aNumOfRowsA, size_t aNumOfColsA, float *pExpectedOutputA,
                                 float *pInputB, size_t aNumOfRowsB, size_t aNumOfColsB, float *pExpectedOutputB);

        template
        void
        validateCompressedOutput(double *pInputA, size_t aNumOfRowsA, size_t aNumOfColsA, double *pExpectedOutputA,
                                 double *pInputB, size_t aNumOfRowsB, size_t aNumOfColsB, double *pExpectedOutputB);
    }
}
