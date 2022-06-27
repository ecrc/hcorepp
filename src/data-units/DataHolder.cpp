
#include <hcorePP/data-units/DataHolder.hpp>

namespace hcorepp {
    namespace dataunits {

        template<typename T>
        DataHolder<T>::DataHolder(size_t aRows, size_t aCols, size_t aLeadingDim, T *apData) {
        }

        template<typename T>
        DataHolder<T>::~DataHolder() {

        }

        template<typename T>
        T *DataHolder<T>::GetData() {
        }


        template<typename T>
        size_t DataHolder<T>::GetNumOfRows() {

        }

        template<typename T>
        size_t DataHolder<T>::GetNumOfCols() {

        }

        template<typename T>
        size_t DataHolder<T>::GetLeadingDim() {

        }
    }
}