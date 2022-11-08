/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCOREPP_DATA_UNITS_DATA_HOLDER_HPP
#define HCOREPP_DATA_UNITS_DATA_HOLDER_HPP

#include <iostream>
#include <cstddef>
#include <hcorepp/common/Definitions.hpp>

namespace hcorepp {
    namespace dataunits {

        /**
         * @brief
         * The base unit of HCore, the data holder represents a single segment of contained numbers.
         * Representing the data of a single component, whether that's a Dense Tile, or a component of the
         * compressed tile(Whether U or V), it simply holds the data.
         *
         * @tparam T
         * Data type held by the data container.
         */
        template<typename T>
        class DataHolder {
        public:

            /**
             * @brief
             * Data Holder constructor.
             *
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
             * @brief
             * The Data holder Leading dimension.
             *
             * @return
             * Leading dimension.
             */
            size_t
            GetLeadingDim() const;

            /**
             * @brief
             * Resize the data holder to use new number of rows, cols and a leading dimension.
             * Function destroys previously allocated buffer and re-initializes them again.
             *
             * @param aRows
             * New number of rows.
             * @param aCols
             * New number of cols.
             * @param aLeadingDim
             * New leading dimension.
             */
            void
            Resize(size_t aRows, size_t aCols, size_t aLeadingDim);

            /**
             * @brief
             * Prints the data holder to the console in a readable format, useful for debugging purposes.
             */
            void Print(std::ostream &aOutStream) const;

        private:
            /** pointer to data array */
            T *mpDataArray;
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

