/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */
#include <hcorepp/data-units/DataHolder.hpp>
#include <hcorepp/kernels/memory.hpp>

namespace hcorepp {
    namespace dataunits {

        template<typename T>
        DataHolder<T>::DataHolder(size_t aRows, size_t aCols, size_t aLeadingDim, T *apData) {
            mNumOfRows = aRows;
            mNumOfCols = aCols;
            mLeadingDimension = aLeadingDim;
            mpDataArray = hcorepp::memory::AllocateArray<T>(mNumOfRows * mNumOfCols);
            if (apData == nullptr) {
                hcorepp::memory::Memset<T>(mpDataArray, 0, mNumOfRows * mNumOfCols);
            } else {
                hcorepp::memory::Memcpy(mpDataArray, apData, mNumOfRows * mNumOfCols,
                                        memory::MemoryTransfer::AUTOMATIC);
            }
        }

        template<typename T>
        DataHolder<T>::~DataHolder() {
            hcorepp::memory::DestroyArray<T>(mpDataArray);
        }

        template<typename T>
        T *DataHolder<T>::GetData() {
            return mpDataArray;
        }

        template<typename T>
        const T *DataHolder<T>::GetData() const {
            return mpDataArray;
        }


        template<typename T>
        size_t DataHolder<T>::GetNumOfRows() const {
            return mNumOfRows;
        }

        template<typename T>
        size_t DataHolder<T>::GetNumOfCols() const {
            return mNumOfCols;
        }

        template<typename T>
        size_t DataHolder<T>::GetLeadingDim() const {
            return mLeadingDimension;
        }

        template<typename T>
        void DataHolder<T>::Resize(size_t aRows, size_t aCols, size_t aLeadingDim) {
            mNumOfRows = aRows;
            mNumOfCols = aCols;
            mLeadingDimension = aLeadingDim;
            hcorepp::memory::DestroyArray(mpDataArray);
            mpDataArray = hcorepp::memory::AllocateArray<T>(mNumOfRows * mNumOfCols);
            hcorepp::memory::Memset<T>(mpDataArray, 0, mNumOfRows * mNumOfCols);
        }

        template<typename T>
        void DataHolder<T>::Print(std::ostream &aOutStream) const {
            T *temp_array = new T[mNumOfCols * mNumOfRows];
            hcorepp::memory::Memcpy(temp_array, mpDataArray, mNumOfRows * mNumOfCols,
                                    memory::MemoryTransfer::DEVICE_TO_HOST);
            aOutStream << "Data : " << std::endl;
            for (int i = 0; i < mNumOfRows * mNumOfCols; i++) {
                aOutStream << temp_array[i] << ", ";
            }
            std::string limiter(20, '=');
            aOutStream << std::endl << limiter << std::endl;
            delete[] temp_array;
        }

        HCOREPP_INSTANTIATE_CLASS(DataHolder)

    }
}