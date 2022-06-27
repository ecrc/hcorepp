
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
             * pointer to the data array.
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
             * Number of rows getter.
             *
             * @return
             * Number of Rows.
             */
            size_t
            GetNumOfRows();

            /**
             * @brief
             * Number of columns getter.
             *
             * @return
             * Number of Columns.
             */
            size_t
            GetNumOfCols();

            /**
             * The Data holder Leading dimension.
             *
             * @return
             * Leading dimension.
             */
            size_t
            GetLeadingDim();

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
    }
}
#endif //HCOREPP_DATA_UNITS_DATA_HOLDER_HPP

