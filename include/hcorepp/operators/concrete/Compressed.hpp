/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP
#define HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP

#include <hcorepp/operators/interface/Tile.hpp>

/** This definition specifies the maximum rank 'k-max' for all compressed tiles as a ratio of the tile's full_rank.
 * This ratio will be used to derive the size of the data buffer holding the UV matrix,
 * i,e Total size = (M * k-max) + (k-max * N) where k-max = max-rank-ratio * min(M,N) */
#define MAX_RANK_RATIO 3

namespace hcorepp::operators {

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
            explicit CompressedTile() = default;

            /**
             * @brief
             * Compressed Tile parameterized constructor taking an already compressed UV array.
             *
             * @param aNumOfRows
             * NUm of tile rows, should be >=0
             * @param aNumOfCols
             * Num of tile cols, should be >=0
             * @param apDataU
             * Array U
             * @param apDataV
             * Array V
             * @param aLeadingDim
             * Data array leading dimension :
             *              aLeadingDim >= aNumOfRows, if layout = blas::Layout::ColMajor, or
             *              aLeadingDim >= aNumOfCols, if layout = blas::Layout::RowMajor.
             * @param aRank
             * Linear algebra rank of the tile. rk >= 0.
             * @param aLayout
             * The physical ordering of matrix elements in the data array buffer:
             *              blas::Layout::ColMajor: column elements are 1-strided (default), or
             *              blas::Layout::RowMajor: row elements are 1-strided.
             *
             * @param[in] aContext
             * The context used to manage the data holder.
             */
            CompressedTile(size_t aNumOfRows, size_t aNumOfCols, T *apDataU, T *apDataV, size_t aLeadingDim,
                           size_t aRank, blas::Layout aLayout, const hcorepp::kernels::RunContext &aContext);

            /**
             * @brief
             * Compressed Tile parameterized constructor taking an already compressed UV array.
             *
             * @param aNumOfRows
             * NUm of tile rows, should be >=0
             * @param aNumOfCols
             * Num of tile cols, should be >=0
             * @param apData
             * Compressed data array of aNumOfRows * aNumOfCols matrix.
             * @param aLeadingDim
             * Data array leading dimension :
             *              aLeadingDim >= aNumOfRows, if layout = blas::Layout::ColMajor, or
             *              aLeadingDim >= aNumOfCols, if layout = blas::Layout::RowMajor.
             * @param aRank
             * Linear algebra rank of the tile. rk >= 0.
             * @param[in] aContext
             * The context used to manage the data holder.
             */
            CompressedTile(size_t aNumOfRows, size_t aNumOfCols, T *apData, size_t aLeadingDim, size_t aRank,
                           const hcorepp::kernels::RunContext &aContext) :
                    CompressedTile(aNumOfRows, aNumOfCols, apData, aLeadingDim, aRank,
                                   blas::Layout::ColMajor, aContext) {

            }

            /**
             * @brief
             * Compressed Tile parameterized constructor taking an already compressed UV array.
             *
             * @param aNumOfRows
             * NUm of tile rows, should be >=0
             * @param aNumOfCols
             * Num of tile cols, should be >=0
             * @param apData
             * Compressed data array of aNumOfRows * aNumOfCols matrix.
             * @param aLeadingDim
             * Data array leading dimension :
             *              aLeadingDim >= aNumOfRows, if layout = blas::Layout::ColMajor, or
             *              aLeadingDim >= aNumOfCols, if layout = blas::Layout::RowMajor.
             * @param aRank
             * Linear algebra rank of the tile. rk >= 0.
             * @param aLayout
             * The physical ordering of matrix elements in the data array buffer:
             *              blas::Layout::ColMajor: column elements are 1-strided (default), or
             *              blas::Layout::RowMajor: row elements are 1-strided.
             *
             * @param[in] aContext
             * The context used to manage the data holder.
             */
            CompressedTile(size_t aNumOfRows, size_t aNumOfCols, T *apData, size_t aLeadingDim, size_t aRank,
                           blas::Layout aLayout, const hcorepp::kernels::RunContext &aContext);

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
             *
             * @param[in] aContext
             * The context used to manage the data holder.
             */
            CompressedTile(size_t aNumOfRows, size_t aNumOfCols, T *apData, size_t aLeadingDim,
                           const CompressionParameters &aParameters, blas::Layout aLayout,
                           const hcorepp::kernels::RunContext &aContext);


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

            T*
            GetUMatrix() const;

            T*
            GetVMatrix() const;

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
            T*
            GetTileSubMatrix(size_t aIndex) const override;

            void
            Gemm(T &aAlpha, dataunits::DataHolder<T> const &aTileA, blas::Op aTileAOp,
                 dataunits::DataHolder<T> const &aTileB, blas::Op aTileBOp, T &aBeta, size_t aLdAu, size_t aARank,
                 const CompressionParameters &aCompressionParameters, const kernels::RunContext &aContext,
                 size_t &aFlops, dataunits::MemoryUnit<T>& aMemoryUnit, bool aCholesky = false) override;

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
            size_t
            GetTileStride(size_t aIndex) const override;

            void
            ReadjustTile(size_t aNumOfRows, size_t aNumOfCols, T *aPdata, size_t aLeadingDim,
                         size_t aRank, const kernels::RunContext &aContext) override;

            void
            DeleteData();

            std::pair<TileMetadata *, T*> UnPackTile(const kernels::RunContext &aContext) override;

            void
            PackTile(TileMetadata aMetadata, T* aDataArray,
                     const kernels::RunContext &aContext) override;

            TileType
            GetTileType() override {
                return COMPRESSED;
            }

            [[nodiscard]] bool
            isDense() const override {
                return false;
            }

            [[nodiscard]] bool
            isCompressed() const override {
                return true;
            };

            [[nodiscard]] int64_t
            GetNumOfSubMatrices() const override{
                return 2;
            }

            [[nodiscard]] size_t GetULeadingDim() const {
                return mULeadingDim;
            }

            [[nodiscard]] size_t GetVLeadingDim() const {
                return mVLeadingDim;
            }

            void
            ReadjustTileRank(size_t aRank, const kernels::RunContext &aContext) override;

            void UpdateMetadata(TileMetadata aMetadata) override;

        private:
            /** U Matrix Leading Dimension */
            size_t mULeadingDim;
            /** V Matrix Leading Dimension */
            size_t mVLeadingDim;
        };

}

#endif //HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP
