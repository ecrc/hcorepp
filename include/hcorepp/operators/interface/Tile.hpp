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
#include <hcorepp/data-units/memory-handlers/MemoryHandler.hpp>
#include <hcorepp/operators/helpers/CompressionParameters.hpp>
#include <hcorepp/kernels/RunContext.hpp>

namespace hcorepp {
    namespace operators {

        enum TileType {
            DENSE,
            COMPRESSED
        };

        /**
         * Tile metadata struct wrapping all tile related metadata to be used by any higher level library, mainly to
         * facilitate communication between processes through runtime libraries (starpu,...)
         */
        struct TileMetadata {
            /** number of rows */
            size_t mNumOfRows;
            /** number of cols */
            size_t mNumOfCols;
            /** matrix rank */
            size_t mMatrixRank;
            /** matrix rank */
            size_t mMaxRank;
            /** leading dimension */
            size_t mLeadingDimension;
            /** data layout */
            blas::Layout mLayout;
            /** enum for tile type */
            TileType mType;

            TileMetadata(size_t aRows, size_t aCols, size_t aMatrixRank, size_t aMaxRank, size_t aLeadingDim,
                         blas::Layout aLayout, TileType aType) : mNumOfRows(aRows), mNumOfCols(aCols),
                                                              mMatrixRank(aMatrixRank), mMaxRank(aMaxRank), mLeadingDimension(aLeadingDim),
                                                              mLayout(aLayout), mType(aType) {

            }
        };

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
             * Get Tile num of rows.
             *
             * @return
             * Num of rows.
             */
            [[nodiscard]] size_t
            GetNumOfRows() const {
                return mNumOfRows;
            }

            [[nodiscard]] size_t
            GetNumOfRows() {
                return mNumOfRows;
            }

            /**
             * Get tile num of cols.
             *
             * @return
             * Num of cols.
             */
            [[nodiscard]] size_t
            GetNumOfCols() const {
                return mNumOfCols;
            }

            [[nodiscard]] size_t
            GetNumOfCols() {
                return mNumOfCols;
            }

            std::reference_wrapper<dataunits::DataHolder<T>>
            GetDataHolder() const {
                return *mpDataArray;
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
            virtual T*
            GetTileSubMatrix(size_t aIndex) const = 0;

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
             * @param[in] aCompressionParameters
             * SVD helpers object (used only in compressed Gemm functionality.)
             * @param[in] aContext
             * The runtime context to apply the operation on.
             * @param[in] aFlops
             * Flops required by operation
             * @param[in] aMemoryUnit
             * Memory Unit Responsible for allocations
             * @param[in] aCholesky
             * Flag to determine whether gemm is performed as part of the Cholesky flow or not
             * @return
             *
             */
            virtual void
            Gemm(T &aAlpha, dataunits::DataHolder<T> const &aTileA, blas::Op aTileAOp,
                 dataunits::DataHolder<T> const &aTileB, blas::Op aTileBOp, T &aBeta, size_t aLdAu, size_t aARank,
                 const CompressionParameters &aCompressionParameters, const kernels::RunContext &aContext,
                 size_t &aFlops, dataunits::MemoryUnit<T> &aMemoryUnit, bool aCholesky = false) = 0;

            /**
             * @brief
             * Get stride of a specific tile submatrix.
             *
             * @param aIndex
             * Index of the submatrix.
             *
             * @return
             * Tile stride.
             */
            [[nodiscard]] virtual size_t
            GetTileStride(size_t aIndex) const = 0;

            /**
             * @brief
             * Readjust tile dimension according to new rank. (currently implemented in the compressed Tile concrete
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
             * @return
             */
            virtual void
            ReadjustTile(size_t aNumOfRows, size_t aNumOfCols, T *aPdata, size_t aLeadingDim, size_t aRank,
                         const kernels::RunContext &aContext) = 0;

            /**
             * @brief
             * Unpack tile function extracting the tile metadata and the actual data arrays.
             * The data arrays are allocated on heap and are the responsibility of the user to delete them.
             *
             * @param[in] aContext
             * The runtime context
             *
             * @return
             * pair of tile metadata and data array.
             */
            virtual std::pair<TileMetadata *, T *>
            UnPackTile(const kernels::RunContext &aContext) = 0;

            /**
             * @brief
             * Pack tile through constructing dense or compressed tile based on the data arrays and metadata sent.
             *
             * @param[in] aMetadata
             * Tile metadata
             * @param[in] aData
             * data arrays
             * @param aContext
             * The runtime context.
             *
             * @return
             * Tile created - The tile is allocated on heap so its the responsibility of the user to delete it.
             */
            virtual void
            PackTile(TileMetadata aMetadata, T* aData, const kernels::RunContext &aContext) = 0;

            virtual void UpdateMetadata(TileMetadata aMetadata) = 0;

            [[nodiscard]] size_t GetLeadingDim() const {
                return mLeadingDim;
            }

            [[nodiscard]] size_t GetTileRank() const {
                return mRank;
            }

            [[nodiscard]] virtual
            bool isDense() const = 0;

            [[nodiscard]] virtual
            bool isCompressed() const = 0;

            [[nodiscard]] virtual
            int64_t GetNumOfSubMatrices() const = 0;

            void Print(std::ostream &aOutStream) {
                if (mpDataArray != nullptr) {
                    mpDataArray->Print(aOutStream);
                }
                aOutStream << "mLayout : " << (char)mLayout << std::endl;
                aOutStream << "mLeadingDim : " << mLeadingDim << std::endl;
                aOutStream << "mNumOfRows : " << mNumOfRows << std::endl;
                aOutStream << "mNumOfCols : " << mNumOfCols << std::endl;
                aOutStream << "mRank : " << mRank << std::endl;
                aOutStream << "mMaxRank : " << mMaxRank << std::endl;
                std::string limiter(20, '=');
                aOutStream << limiter << std::endl;
            }

            /**
             * @brief
             * Getter for type of tile object
             *
             * @return
             * Type of tile object
             */
            virtual TileType
            GetTileType() = 0;

            virtual void
            ReadjustTileRank(size_t aRank, const kernels::RunContext &aContext) = 0;

        protected:
            // Matrix physical layout -> column major or row major.
            blas::Layout mLayout;
            // Leading dimension of the tile.
            size_t mLeadingDim;
            /** vector of references to data arrays representing the Dense tile. */
            dataunits::DataHolder<T>* mpDataArray = nullptr;
            /** number of rows */
            size_t mNumOfRows;
            /** number of cols */
            size_t mNumOfCols;
            /** Rank of Tile */
            size_t mRank;

            size_t mMaxRank;
        };

    }
}

#endif //HCOREPP_OPERATORS_INTERFACE_TILE_HPP
