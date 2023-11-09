/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_TEST_HELPERS_HPP
#define HCOREPP_TEST_HELPERS_HPP

#include <cstdint>
#include <vector>
#include <blas/util.hh>
#include <hcorepp/kernels/memory.hpp>
#include <hcorepp/data-units/DataHolder.hpp>

namespace hcorepp {
    namespace test_helpers {


        template<typename T>
        void
        rowMajorToColumnMajor(T *pInputArray, size_t aNumOfCols, size_t aNumOfRows, T *pOutputArray);

        template<typename T>
        void
        columnMajorToRowMajor(T *pInputArray, size_t aNumOfCols, size_t aNumOfRows, T *pOutputArray);

        template<typename T>
        void
        printMatrix(T *pInput, size_t aNumOfRows, size_t aNumOfCols);

        template<typename T>
        T *copy_output(dataunits::DataHolder<T> &apData, hcorepp::kernels::RunContext &aContext) {
            size_t num_elements = apData.GetNumOfCols() * apData.GetNumOfRows();
            T *arr = new T[num_elements];
            hcorepp::memory::Memcpy<T>(arr, apData.GetData(), num_elements, aContext,
                                       memory::MemoryTransfer::DEVICE_TO_HOST);
            aContext.Sync();
            return arr;
        }

        template<typename T>
        T *copy_output(const T* apData, size_t aNumOfElements, hcorepp::kernels::RunContext &aContext) {
            T *arr = new T[aNumOfElements];
            hcorepp::memory::Memcpy<T>(arr, apData, aNumOfElements, aContext,
                                       memory::MemoryTransfer::DEVICE_TO_HOST);
            aContext.Sync();
            return arr;
        }

        template<typename T>
        void
        validateOutput(T *pInput, size_t aNumOfRows, size_t aNumOfCols, T *pExpectedOutput);

        template<typename T>
        void
        validateOutputLenient(T *pInput, size_t aNumOfRows, size_t aNumOfCols, T *pExpectedOutput);


        template<typename T>
        void
        validateCompressedOutput(T *pInputA, size_t aNumOfRowsA, size_t aNumOfColsA, T *pExpectedOutputA,
                                 T *pInputB, size_t aNumOfRowsB, size_t aNumOfColsB, T *pExpectedOutputB);

    }
}
#endif //HCOREPP_TEST_HELPERS_HPP

