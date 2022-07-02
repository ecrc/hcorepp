
#include <hcorePP/data-units/DataHolder.hpp>

namespace hcorepp {
    namespace dataunits {

        template<typename T>
        DataHolder<T>::DataHolder(size_t aRows, size_t aCols, size_t aLeadingDim, T *apData) {
            mNumOfRows = aRows;
            mNumOfCols = aCols;
            mLeadingDimension = aLeadingDim;
            mDataArray = apData;
#ifdef USE_CUDA
            cuda_malloc(&mDataArray, mNumofRows*mNumOfCols*sizeof(T));
#else
            mDataArray = (T *) malloc(mNumOfRows * mNumOfCols * sizeof(T));
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
        size_t DataHolder<T>::GetNumOfRows() {
            return mNumOfRows;
        }

        template<typename T>
        size_t DataHolder<T>::GetNumOfCols() {
            return mNumOfCols;
        }

        template<typename T>
        size_t DataHolder<T>::GetLeadingDim() {
            return mLeadingDimension;
        }
    }
}