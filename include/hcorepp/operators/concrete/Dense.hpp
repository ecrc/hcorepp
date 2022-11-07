/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCOREPP_OPERATORS_CONCRETE_DENSE_HPP
#define HCOREPP_OPERATORS_CONCRETE_DENSE_HPP

#include <hcorepp/operators/interface/Tile.hpp>
#include <functional>

namespace hcorepp {
    namespace operators {

        /**
         * @brief
         * Class responsible for encapsulating a dense tile for all operations.
         * Dense tiles represent a sub-matrix from a parent one that is fully packed and is
         * not compressed at all.
         *
         * @tparam T
         * The type of the elements inside the tile
         */
        template<typename T>
        class DenseTile : public Tile<T> {

        public:

            /**
             * @brief
             * Dense Tile default constructor
             *
             */
            DenseTile();

            /**
             * @brief
             * Dense Tile parameterized constructor.
             *
             * @param aNumOfRows
             * NUm of tile rows, should be >=0
             * @param aNumOfCols
             * Num of tile cols, should be >=0
             * @param aPdata
             * Data array of size :
             *              @aLeadingDim * @aNumOfCols, if layout =blas::Layout::ColMajor, or
             *              @aNumOfRows * @aLeadingDim, if layout =blas::Layout::RowMajor
             * @param aLeadingDim
             * Data array leading dimension :
             *              @aLeadingDim >= @aNumOfRows, if layout = blas::Layout::ColMajor, or
             *              @aLeadingDim >= @aNumOfCols, if layout = blas::Layout::RowMajor.
             * @param aLayout
             * The physical ordering of matrix elements in the data array buffer:
             *              blas::Layout::ColMajor: column elements are 1-strided (default), or
             *              blas::Layout::RowMajor: row elements are 1-strided.
             */
            DenseTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                      blas::Layout aLayout = blas::Layout::ColMajor);

            /**
             * @brief
             * Dense Tile destructor.
             */
            ~DenseTile();

            /**
             * @brief
             * Get data holder object describing specific tile subMatrix.
             *
             * @param aIndex
             * Matrix index (can only be zero).
             *
             * @return
             * DataHolder object describing the Tile sub matrix.
             */
            std::reference_wrapper<dataunits::DataHolder < T>>
            GetTileSubMatrix(
            size_t aIndex
            )
            override;

            /**
             * @brief
             * Get data holder object describing specific tile subMatrix.
             *
             * @param aIndex
             * Matrix index (can only be zero).
             *
             * @return
             * const DataHolder object describing the Tile sub matrix.
             */
            const std::reference_wrapper<dataunits::DataHolder < T>>
            GetTileSubMatrix(
            size_t aIndex
            )
            const override;

            /**
             * @brief
             * Get number of matrices describing the tile.
             *
             * @return
             * Number of matrices describing the tile (should be one).
             */
            size_t
            GetNumberOfMatrices() const override;

            /**
             * @brief
             * Get stride of a specific tile data holder.
             *
             * @param aIndex
             * Index of the dataHolder.(should be only zero)
             *
             * @return
             * Tile stride.
             */
            int64_t
            GetTileStride(size_t aIndex) const override;

            /**
             * @brief
             * General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
             *
             * @param[in] aAlpha
             * The scalar alpha.
             * @param[in] aTileA
             * The m-by-k tile.
             * @param[in] aTileB
             * The k-by-n tile.
             * @param[in] aBeta
             * The scalar beta.
             * @param[in] aLdAu
             * Tile A leading dimension. (used only in compressed Gemm functionality.)
             * @param[in] aARank
             * tile rank. (used only in compressed Gemm functionality.)
             * @param[in] aHelpers
             * SVD helpers object (used only in compressed Gemm functionality.)
             *
             */
            void
            Gemm(T &aAlpha, dataunits::DataHolder <T> const &aTileA, blas::Op aTileAOp,
                 dataunits::DataHolder <T> const &aTileB, blas::Op aTileBOp, T &aBeta, int64_t aLdAu, int64_t aARank,
                 const CompressionParameters &aCompressionParameters) override;

            /**
             * @brief
             * Readjust tile dimension according to new rank.
             * (Not supported yet in Dense tiles).
             *
             * @param aNumOfRows
             * New number of rows to use.
             * @param aNumOfCols
             * New number of cols to use.
             * @param aPdata
             * pointer to new data array to set tile data holders.
             * @param aLeadingDim
             * Leading dimension
             * @param aRank
             * New Linear algebra rank of the tile. rk >= 0.
             *
             */
            void
            ReadjustTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                         int64_t aRank) override;


        private:
            /** vector of references to data arrays representing the Dense tile. */
            std::vector<dataunits::DataHolder < T> *>
            mDataArrays;
            /** number of rows */
            size_t mNumOfRows;
            /** number of cols */
            size_t mNumOfCols;

        };
    }

}

#endif //HCOREPP_OPERATORS_CONCRETE_DENSE_HPP
