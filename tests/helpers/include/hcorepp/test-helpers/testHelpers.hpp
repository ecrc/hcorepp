/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
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
        rowMajorToColumnMajor(T *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, T *pOutputArray);

        template<typename T>
        void
        columnMajorToRowMajor(T *pInputArray, int64_t aNumOfCols, int64_t aNumOfRows, T *pOutputArray);

        template<typename T>
        void
        printMatrix(T *pInput, int64_t aNumOfRows, int64_t aNumOfCols);

        template<typename T>
        T *copy_output(const dataunits::DataHolder<T>& apData) {
            size_t num_elements = apData.GetNumOfCols() * apData.GetNumOfRows();
            T * arr = new T[num_elements];
            hcorepp::memory::Memcpy<T>(arr, apData.GetData(), num_elements, memory::MemoryTransfer::DEVICE_TO_HOST);
            return arr;
        }

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

