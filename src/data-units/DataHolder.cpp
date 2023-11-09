/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */
#include <hcorepp/data-units/DataHolder.hpp>
#include <hcorepp/kernels/memory.hpp>

namespace hcorepp {
    namespace dataunits {

        template<typename T>
        DataHolder<T>::DataHolder(size_t aRows, size_t aCols, size_t aLeadingDim, T *apData,
                                  const hcorepp::kernels::RunContext &aContext, bool aMemoryOwnership) : mRunContext(
                aContext) {

            mNumOfRows = aRows;
            mNumOfCols = aCols;
            mLeadingDimension = aLeadingDim;
            mMemoryOwnership = aMemoryOwnership;
            if (!mMemoryOwnership && apData != nullptr) {
                mpDataArray = apData;
            } else {
                mpDataArray = hcorepp::memory::AllocateArray<T>(mNumOfRows * mNumOfCols, mRunContext);
                if (apData == nullptr) {
                    hcorepp::memory::Memset<T>(mpDataArray, 0, mNumOfRows * mNumOfCols, mRunContext);
                } else {
                    hcorepp::memory::Memcpy<T>(mpDataArray, apData, mNumOfRows * mNumOfCols,
                                               mRunContext, memory::MemoryTransfer::AUTOMATIC, false);
                }
            }
        }

        template<typename T>
        DataHolder<T>::~DataHolder() {
            if (mMemoryOwnership)
                hcorepp::memory::DestroyArray<T>(mpDataArray, mRunContext);
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
            if (mNumOfRows * mNumOfCols == aRows * aCols) {
                mNumOfRows = aRows;
                mNumOfCols = aCols;
                mLeadingDimension = aLeadingDim;

                return;
            }
            mNumOfRows = aRows;
            mNumOfCols = aCols;
            mLeadingDimension = aLeadingDim;
            if (!mMemoryOwnership) {
                return;
            }
            hcorepp::memory::DestroyArray(mpDataArray, mRunContext);
            mpDataArray = hcorepp::memory::AllocateArray<T>(mNumOfRows * mNumOfCols, mRunContext);
            hcorepp::memory::Memset<T>(mpDataArray, 0, mNumOfRows * mNumOfCols, mRunContext);
        }

        template<typename T>
        void DataHolder<T>::Print(std::ostream &aOutStream) {
            T *temp_array = new T[mNumOfCols * mNumOfRows];
            hcorepp::memory::Memcpy<T>(temp_array, mpDataArray, mNumOfRows * mNumOfCols,
                                       mRunContext,
                                       memory::MemoryTransfer::DEVICE_TO_HOST, true);
            aOutStream << "Data : " << std::endl;
            aOutStream << "Rows : " << mNumOfRows << " Cols : " << mNumOfCols << std::endl;
            for (size_t i = 0; i < mNumOfRows * mNumOfCols; i++) {
                char str[10240];

                sprintf(str, "%f, ", temp_array[i]);
                aOutStream << str;
            }
            std::string limiter(20, '=');
            aOutStream << std::endl << limiter << std::endl;
            delete[] temp_array;
        }

        HCOREPP_INSTANTIATE_CLASS(DataHolder)

    }
}