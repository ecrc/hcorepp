/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_OPERATORS_INTERFACE_TILE_HPP
#define HCOREPP_OPERATORS_INTERFACE_TILE_HPP

#include <vector>
#include <functional>
#include <cstdint>
#include "blas.hh"
#include <hcorepp/data-units/DataHolder.hpp>
#include <hcorepp/operators/helpers/CompressionParameters.hpp>
#include <hcorepp/kernels/RunContext.hpp>

namespace hcorepp {
    namespace operators {

        template<typename T>
        class Tile {
        public:

            /**
             * @brief
             * Virtual destructor to allow correct destruction of concrete Tile classes.
             */
            virtual ~Tile() = default;

            /**
             * @brief
             * Get the physical ordering of the matrix elements in the data array.
             *
             * @return
             * Physical ordering of elements.
             */
            blas::Layout GetLayout() const {
                return mLayout;
            }

            /**
             * @brief
             * Get data holder object describing specific tile subMatrix.
             *
             * @param aIndex
             * Matrix index :
             *              0 in case of Dense tile.
             *              0 or 1 in case of Compressed tiles.
             *
             * @return
             * DataHolder object describing the Tile sub matrix.
             */
            virtual std::reference_wrapper<dataunits::DataHolder<T>>
            GetTileSubMatrix(size_t aIndex) = 0;

            /**
             * @brief
             * Get data holder object describing specific tile subMatrix.
             *
             * Matrix index :
             *              0 in case of Dense tile.
             *              0 or 1 in case of Compressed tiles.
             *
             * @return
             * const DataHolder object describing the Tile sub matrix.
             */
            virtual const std::reference_wrapper<dataunits::DataHolder<T>>
            GetTileSubMatrix(size_t aIndex) const = 0;

            /**
             * @brief
             * Get number of matrices describing the tile.
             *
             * @return
             * Number of matrices describing the tile:
             *             1 in Dense tile cases.
             *             2 in Compressed tile cases.
             */
            virtual size_t
            GetNumberOfMatrices() const = 0;

            /**
             * @brief
             * General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
             *
             * @param[in] aAlpha
             * The scalar alpha.
             * @param[in] aTileA
             * The m-by-k tile.
             * @param[in] aTileAOp
             * The operation to apply on data holder A.
             * @param[in] aTileB
             * The k-by-n tile.
             * @param[in] aTileBOp
             * The operation to apply on data holder B.
             * @param[in] aBeta
             * The scalar beta.
             * @param[in] aLdAu
             * Tile A leading dimension. (used only in compressed Gemm functionality.)
             * @param[in] aARank
             * tile rank. (used only in compressed Gemm functionality.)
             * @param[in] aHelpers
             * SVD helpers object (used only in compressed Gemm functionality.)
             * @param[in] aContext
             * The runtime context to apply the operation on.
             */
            virtual void
            Gemm(T &aAlpha, dataunits::DataHolder<T> const &aTileA, blas::Op aTileAOp,
                 dataunits::DataHolder<T> const &aTileB, blas::Op aTileBOp, T &aBeta, int64_t aLdAu, int64_t aARank,
                 const CompressionParameters &aCompressionParameters, kernels::RunContext &aContext) = 0;

            /**
             * @brief
             * Get stride of a specific tile data holder.
             *
             * @param aIndex
             * Index of the dataHolder.
             *
             * @return
             * Tile stride.
             */
            virtual int64_t
            GetTileStride(size_t aIndex) const = 0;

            /**
             * @brief
             * Readjust tile dimension according to new rank. (currently implemented in the compressed Tile concnrete
             * implementation only.)
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
             * @param[in] aContext
             * The runtime context to apply the operation on.
             */
            virtual void
            ReadjustTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim, int64_t aRank,
                         kernels::RunContext &aContext) = 0;

        protected:
            // Matrix physical layout -> column major or row major.
            blas::Layout mLayout;
            // Leading dimension of the tile.
            int64_t mLeadingDim;
        };

    }
}

#endif //HCOREPP_OPERATORS_INTERFACE_TILE_HPP
