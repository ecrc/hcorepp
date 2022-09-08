
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <hcorepp/data-units/DataHolder.hpp>
#include <hcorepp/kernels/kernels.hpp>

namespace hcorepp {
    namespace dataunits {

        template<typename T>
        DataHolder<T>::DataHolder(size_t aRows, size_t aCols, size_t aLeadingDim, T *apData) {
            mNumOfRows = aRows;
            mNumOfCols = aCols;
            mLeadingDimension = aLeadingDim;
#ifdef USE_CUDA
            //            cuda_malloc(&mDataArray, mNumofRows*mNumOfCols*sizeof(T));
            //            if(apData!= nullptr) {
            //                cuda_memcpy(mDataArray, apData, mNumOfRows * mNumOfCols * sizeof(T), cudaMemcpyHostToDevice);
            //            } else {
            //                // add cudamemset.
            //            }
#else

#endif
            hcorepp::kernels::AllocateArray<T>(&mDataArray, mNumOfRows, mNumOfCols, apData);
        }

        template<typename T>
        DataHolder<T>::~DataHolder() {
#ifdef USE_CUDA
            cuda_free(mDataArray);
#else
#endif
            hcorepp::kernels::DestroyArray<T>(mDataArray);
        }

        template<typename T>
        T *DataHolder<T>::GetData() {
            return mDataArray;
        }

        template<typename T>
        const T *DataHolder<T>::GetData() const {
            return mDataArray;
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
        void DataHolder<T>::CopyDataArray(size_t aStIdx, T *aSrcDataArray, size_t aNumOfElements) {
            if (aNumOfElements > mNumOfRows * mNumOfCols || aSrcDataArray == nullptr) {
                return;
            }
#ifdef USE_CUDA
            //            cuda_memcpy(&mDataArray[aStIdx], aSrcDataArray, aNumOfElements * sizeof(T), cudaMemcpyHostToDevice);
#else

#endif
            hcorepp::kernels::Memcpy<T>(&mDataArray[aStIdx], aSrcDataArray, aNumOfElements);

        }

        template<typename T>
        void DataHolder<T>::Resize(size_t aRows, size_t aCols, size_t aLeadingDim) {
            mNumOfRows = aRows;
            mNumOfCols = aCols;
            mLeadingDimension = aLeadingDim;
#ifdef USE_CUDA
            //            cuda_malloc(&mDataArray, mNumofRows*mNumOfCols*sizeof(T));
            //            if(apData!= nullptr) {
            //                cuda_memcpy(mDataArray, apData, mNumOfRows * mNumOfCols * sizeof(T), cudaMemcpyHostToDevice);
            //            } else {
            //                // add cudamemset.
            //            }
#else

#endif
            hcorepp::kernels::DestroyArray(mDataArray);

            hcorepp::kernels::AllocateArray(&mDataArray, mNumOfRows, mNumOfCols);
        }


    }
}