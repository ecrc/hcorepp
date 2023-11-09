/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
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
             *              aLeadingDim * aNumOfCols, if layout =blas::Layout::ColMajor, or
             *              aNumOfRows * aLeadingDim, if layout =blas::Layout::RowMajor
             * @param aLeadingDim
             * Data array leading dimension :
             *              aLeadingDim >= aNumOfRows, if layout = blas::Layout::ColMajor, or
             *              aLeadingDim >= aNumOfCols, if layout = blas::Layout::RowMajor.
             *
             * @param[in] aContext
             * The context used to manage the data holder.
             */
            DenseTile(size_t aNumOfRows, size_t aNumOfCols, T *aPdata, size_t aLeadingDim,
                      const hcorepp::kernels::RunContext &aContext) :
                    DenseTile(aNumOfRows, aNumOfCols, aPdata, aLeadingDim, blas::Layout::ColMajor,
                              aContext) {

            }

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
             *              aLeadingDim * aNumOfCols, if layout =blas::Layout::ColMajor, or
             *              aNumOfRows * aLeadingDim, if layout =blas::Layout::RowMajor
             * @param aLeadingDim
             * Data array leading dimension :
             *              aLeadingDim >= aNumOfRows, if layout = blas::Layout::ColMajor, or
             *              aLeadingDim >= aNumOfCols, if layout = blas::Layout::RowMajor.
             * @param aLayout
             * The physical ordering of matrix elements in the data array buffer:
             *              blas::Layout::ColMajor: column elements are 1-strided (default), or
             *              blas::Layout::RowMajor: row elements are 1-strided.
             *
             * @param[in] aContext
             * The context used to manage the data holder.
             * @param[in] aMemoryOwnership
             * Avoid new allocation if apData != nullptr by setting this flag.
             * apData should be at least of size aRows * aCols * sizeof(T)
             */
            DenseTile(size_t aNumOfRows, size_t aNumOfCols, T *aPdata, size_t aLeadingDim,
                      blas::Layout aLayout, const hcorepp::kernels::RunContext &aContext, bool aMemoryOwnership = true);

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
            T*
            GetTileSubMatrix(size_t aIndex) const override;

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
            size_t
            GetTileStride(size_t aIndex) const override;

            void
            Gemm(T &aAlpha, dataunits::DataHolder<T> const &aTileA, blas::Op aTileAOp,
                 dataunits::DataHolder<T> const &aTileB, blas::Op aTileBOp, T &aBeta, size_t aLdAu, size_t aARank,
                 const CompressionParameters &aCompressionParameters, const kernels::RunContext &aContext,
                 size_t &aFlops, dataunits::MemoryUnit<T>& aMemoryUnit, bool aCholesky = false) override;

            void
            ReadjustTile(size_t aNumOfRows, size_t aNumOfCols, T *aPdata, size_t aLeadingDim,
                         size_t aRank, const kernels::RunContext &aContext) override;


            std::pair<TileMetadata *, T*>
            UnPackTile(const kernels::RunContext &aContext) override;

            void
            PackTile(TileMetadata aMetadata, T* aDataArray, const kernels::RunContext &aContext) override;

            TileType
            GetTileType() override {
                return DENSE;
            }

            [[nodiscard]] bool
            isDense() const override {
                return true;
            }

            [[nodiscard]] bool
            isCompressed() const override {
                return false;
            };

            [[nodiscard]] int64_t
            GetNumOfSubMatrices() const override{
                return 1;
            }

            void
            ReadjustTileRank(size_t aRank, const kernels::RunContext &aContext) override;

            void UpdateMetadata(TileMetadata aMetadata) override;

        };
    }

}

#endif //HCOREPP_OPERATORS_CONCRETE_DENSE_HPP
