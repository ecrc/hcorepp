/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP
#define HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP

#include <hcorepp/operators/interface/Tile.hpp>

namespace hcorepp {
    namespace operators {

        /**
         * @brief
         * Class responsible for encapsulating a compressed tile for all operations.
         * Compressed tiles represent a sub-matrix from a parent one that is
         * compressed and represented by two matrices U and V where if the
         * dense and original representation of the sub-matrix is A, the compressed tile
         * will contain a matrix U and a matrix V where A = U * V with some error rate.
         *
         * @tparam T
         * The type of the elements inside the tile
         */
        template<typename T>
        class CompressedTile : public Tile<T> {
        public:


            /**
             * @brief
             * Compressed Tile default constructor
             *
             */
            CompressedTile();

            /**
             * @brief
             * Compressed Tile parameterized constructor taking an already compressed UV array.
             *
             * @param aNumOfRows
             * NUm of tile rows, should be >=0
             * @param aNumOfCols
             * Num of tile cols, should be >=0
             * @param apData
             * Compressed data array of @aNumOfRows * @aNumOfCols matrix.
             * @param aLeadingDim
             * Data array leading dimension :
             *              @aLeadingDim >= @aNumOfRows, if layout = blas::Layout::ColMajor, or
             *              @aLeadingDim >= @aNumOfCols, if layout = blas::Layout::RowMajor.
             * @param aRank
             * Linear algebra rank of the tile. rk >= 0.
             * @param aAccuracy
             * Numerical error threshold. accuracy >= 0.
             * @param aLayout
             * The physical ordering of matrix elements in the data array buffer:
             *              blas::Layout::ColMajor: column elements are 1-strided (default), or
             *              blas::Layout::RowMajor: row elements are 1-strided.
             */
            CompressedTile(int64_t aNumOfRows, int64_t aNumOfCols, T *apData, int64_t aLeadingDim, int64_t aRank,
                           blas::Layout aLayout = blas::Layout::ColMajor);

            /**
             * @brief
             * Constructor taking a dense matrix, compressing it and creating a tile
             * resulting from its compression as guided by the SVD parameters.
             *
             * @param[in] aNumOfRows
             * The number of rows in the given dense array.
             *
             * @param[in] aNumOfCols
             * The number of columns in the given dense array.
             *
             * @param[in] apData
             * The data pointer for the dense data.
             *
             * @param[in] aLeadingDim
             * The leading dimension for the data.
             *
             * @param[in] aParameters
             * The SVD parameters used for compression.
             *
             * @param[in] aLayout
             * The matrix layout used.
             */
            CompressedTile(int64_t aNumOfRows, int64_t aNumOfCols, T *apData, int64_t aLeadingDim,
                           const CompressionParameters &aParameters, blas::Layout aLayout = blas::Layout::ColMajor);


            /**
             * @brief
             * Compressed Tile copy constructor.
             *
             * @param aTile
             * Input compressed tile to copy.
             */
            CompressedTile(CompressedTile<T> &aTile);

            /**
             * @brief
             * Compressed Tile destructor.
             */
            ~CompressedTile();

            /**
             * @brief
             * Get data holder object describing specific tile subMatrix.
             *
             * @param aIndex
             * Matrix index (0 or 1).
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
             * Matrix index (0 or 1).
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
             * Number of matrices describing the tile (should be two).
             */
            size_t
            GetNumberOfMatrices() const override;

            void
            Gemm(T &aAlpha, dataunits::DataHolder <T> const &aTileA, blas::Op aTileAOp,
                 dataunits::DataHolder <T> const &aTileB, blas::Op aTileBOp, T &aBeta, int64_t aLdAu, int64_t aARank,
                 const CompressionParameters &aCompressionParameters, kernels::RunContext &aContext) override;

            /**
             * @brief
             * Set Rank of the tile.
             *
             * @param[in] aMatrixRank
             * Tile rank.
             */
            void
            SetTileRank(int64_t &aMatrixRank);

            /**
             * @brief
             * Get the rank of tile.
             *
             * @return
             * Tile rank.
             */
            int64_t
            GetTileRank() const;

            /**
             * Get Compressed Tile num of rows.
             *
             * @return
             * Num of rows.
             */
            size_t
            GetNumOfRows() const;

            /**
             * Get compressed tile num of cols.
             *
             * @return
             * Num of cols.
             */
            size_t
            GetNumOfCols() const;

            /**
             * @brief
             * Get stride of a specific tile data holder.
             *
             * @param aIndex
             * Index of the dataHolder.(0 or 1)
             *
             * @return
             * Tile stride.
             */
            int64_t
            GetTileStride(size_t aIndex) const override;

            void
            ReadjustTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                         int64_t aRank, kernels::RunContext &aContext) override;

        private:
            /** Vector of data arrays */
            std::vector<dataunits::DataHolder < T> *>
            mDataArrays;
            /** Linear Algebra Matrix rank*/
            int64_t mMatrixRank;
            /** number of rows */
            size_t mNumOfRows;
            /** number of cols */
            size_t mNumOfCols;
            static const int64_t FULL_RANK_ = -1;
        };

    }
}

#endif //HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP
