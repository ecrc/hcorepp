
#ifndef HCOREPP_DATA_UNITS_DATA_HOLDER_HPP
#define HCOREPP_DATA_UNITS_DATA_HOLDER_HPP

#include <cstddef>

namespace hcorepp {
    namespace dataunits {

        template<typename T>
        class DataHolder {
        public:

            /**
             * @brief
             * Data Holder constructor.
             * @param[in] aRows
             * Number of Rows.
             * @param[in] aCols
             * Number of Cols.
             * @param[in] aLeadingDim
             * Which dimension is the leading one.
             * @param[in] apData
             * pointer to a data array that can be null pointer,
             * if a non null pointer was passed then data is copied to an allocated internal buffer
             * of size = aRows * aLeadingDim * sizeof<T> assuming Row major format.
             *
             */
            DataHolder(size_t aRows, size_t aCols, size_t aLeadingDim, T *apData = nullptr);

            /**
             * @brief
             * Destructor to allow correct destruction of instances created.
             */
            ~DataHolder();

            /**
             * @brief
             * The dataHolder/ matrix array getter.
             *
             * @return
             * Pointer to the data array.
             *
             */
            T *
            GetData();

            /**
             * @brief
             * The dataHolder/ matrix array getter.
             *
             * @return
             * Pointer to the data array.
             *
             */
            const T *
            GetData() const;

            /**
             * @brief
             * Number of rows getter.
             *
             * @return
             * Number of Rows.
             */
            size_t
            GetNumOfRows() const;

            /**
             * @brief
             * Number of columns getter.
             *
             * @return
             * Number of Columns.
             */
            size_t
            GetNumOfCols() const;

            /**
             * The Data holder Leading dimension.
             *
             * @return
             * Leading dimension.
             */
            size_t
            GetLeadingDim() const;

            void
            CopyDataArray(size_t aStIdx, T* aSrcDataArray, size_t aNumOfElements);

        private:
            /** pointer to data array */
            T *mDataArray;
            /** number of rows */
            size_t mNumOfRows;
            /** number of cols */
            size_t mNumOfCols;
            /** leading dimension */
            size_t mLeadingDimension;
        };
        template class DataHolder<int>;
        template class DataHolder<long>;
        template class DataHolder<float>;
        template class DataHolder<double>;

    }
}
#endif //HCOREPP_DATA_UNITS_DATA_HOLDER_HPP

