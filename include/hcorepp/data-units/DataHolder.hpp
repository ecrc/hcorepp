/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_DATA_UNITS_DATA_HOLDER_HPP
#define HCOREPP_DATA_UNITS_DATA_HOLDER_HPP

#include <functional>
#include <iostream>
#include <cstddef>
#include <hcorepp/common/Definitions.hpp>
#include <hcorepp/kernels/RunContext.hpp>
#include <hcorepp/helpers/DebuggingTimer.hpp>

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
             * @param[in] aContext
             * The context used to manage the data holder.
             * @param[in] aMemoryOwnership
             * Avoid new allocation if apData != nullptr by setting this flag.
             * apData should be at least of size aRows * aCols * sizeof(T)
             */
            DataHolder(size_t aRows, size_t aCols, size_t aLeadingDim, T *apData,
                       const hcorepp::kernels::RunContext &aContext, bool aMemoryOwnership = true);

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
            void Print(std::ostream &aOutStream);

        private:
            /** pointer to data array */
            T *mpDataArray;
            /** number of rows */
            size_t mNumOfRows;
            /** number of cols */
            size_t mNumOfCols;
            /** leading dimension */
            size_t mLeadingDimension;
            /** Run context used for the data holder */
            const hcorepp::kernels::RunContext &mRunContext;
            /** Memory ownership flag */
            bool mMemoryOwnership;
        };
    }
}
#endif //HCOREPP_DATA_UNITS_DATA_HOLDER_HPP

