
#include <cstdlib>
#include <hcorepp/data-units/DataHolder.hpp>
#include <hcorepp/kernels/memory.hpp>

namespace hcorepp {
    namespace dataunits {

        template<typename T>
        DataHolder<T>::DataHolder(size_t aRows, size_t aCols, size_t aLeadingDim, T *apData) {
            mNumOfRows = aRows;
            mNumOfCols = aCols;
            mLeadingDimension = aLeadingDim;
            mDataArray = hcorepp::memory::AllocateArray<T>(mNumOfRows * mNumOfCols);
            if (apData == nullptr) {
                hcorepp::memory::Memset<T>(mDataArray, 0, mNumOfRows * mNumOfCols);
            } else {
                hcorepp::memory::Memcpy(mDataArray, apData, mNumOfRows * mNumOfCols,
                                        memory::MemoryTransfer::AUTOMATIC);
            }
        }

        template<typename T>
        DataHolder<T>::~DataHolder() {
            hcorepp::memory::DestroyArray<T>(mDataArray);
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
            hcorepp::memory::Memcpy<T>(&mDataArray[aStIdx], aSrcDataArray, aNumOfElements,
                                       memory::MemoryTransfer::DEVICE_TO_DEVICE);
        }

        template<typename T>
        void DataHolder<T>::Resize(size_t aRows, size_t aCols, size_t aLeadingDim) {
            mNumOfRows = aRows;
            mNumOfCols = aCols;
            mLeadingDimension = aLeadingDim;
            hcorepp::memory::DestroyArray(mDataArray);
            mDataArray = hcorepp::memory::AllocateArray<T>(mNumOfRows * mNumOfCols);
            hcorepp::memory::Memset<T>(mDataArray, 0, mNumOfRows * mNumOfCols);
        }

        template<typename T>
        void DataHolder<T>::Print(std::ostream &aOutStream) const {
            T *temp_array = new T[mNumOfCols * mNumOfRows];
            hcorepp::memory::Memcpy(temp_array, mDataArray, mNumOfRows * mNumOfCols,
                                    memory::MemoryTransfer::DEVICE_TO_HOST);
            aOutStream << "Data : " << std::endl;
            for (int i = 0; i < mNumOfRows * mNumOfCols; i++) {
                aOutStream << temp_array[i] << ", ";
            }
            std::string limiter(20, '=');
            aOutStream << std::endl << limiter << std::endl;
            delete[] temp_array;
        }

    }
}