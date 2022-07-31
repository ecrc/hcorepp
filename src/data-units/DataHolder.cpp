
#include <hcorepp/data-units/DataHolder.hpp>
#include <cstdlib>
#include <cstring>

namespace hcorepp {
    namespace dataunits {

        template<typename T>
        DataHolder<T>::DataHolder(size_t aRows, size_t aCols, size_t aLeadingDim, T *apData) {
            mNumOfRows = aRows;
            mNumOfCols = aCols;
            mLeadingDimension = aLeadingDim;
#ifdef USE_CUDA
            cuda_malloc(&mDataArray, mNumofRows*mNumOfCols*sizeof(T));
            if(apData!= nullptr) {
                cuda_memcpy(mDataArray, apData, mNumOfRows * mNumOfCols * sizeof(T), cudaMemcpyHostToDevice);
            } else {
                // add cudamemset.
            }
#else
            mDataArray = (T *) malloc(mNumOfRows * mNumOfCols * sizeof(T));
            if (apData != nullptr) {
                memcpy(mDataArray, apData, mNumOfRows * mNumOfCols * sizeof(T));
            } else {
                memset((void *) mDataArray, 0, mNumOfRows * mNumOfCols * sizeof(T));
            }
#endif
        }

        template<typename T>
        DataHolder<T>::~DataHolder() {
#ifdef USE_CUDA
            cuda_free(mDataArray);
#else
            free(mDataArray);
#endif
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
            cuda_memcpy(&mDataArray[aStIdx], aSrcDataArray, aNumOfElements * sizeof(T), cudaMemcpyHostToDevice);
#else
            memcpy(&mDataArray[aStIdx], aSrcDataArray, aNumOfElements * sizeof(T));
#endif

        }

        template<typename T>
        void DataHolder<T>::Resize(size_t aRows, size_t aCols, size_t aLeadingDim) {
            mNumOfRows = aRows;
            mNumOfCols = aCols;
            mLeadingDimension = aLeadingDim;
#ifdef USE_CUDA
            cuda_malloc(&mDataArray, mNumofRows*mNumOfCols*sizeof(T));
            if(apData!= nullptr) {
                cuda_memcpy(mDataArray, apData, mNumOfRows * mNumOfCols * sizeof(T), cudaMemcpyHostToDevice);
            } else {
                // add cudamemset.
            }
#else
            free(mDataArray);

            mDataArray = (T *) malloc(mNumOfRows * mNumOfCols * sizeof(T));
            memset((void *) mDataArray, 0, mNumOfRows * mNumOfCols * sizeof(T));
#endif
        }


    }
}