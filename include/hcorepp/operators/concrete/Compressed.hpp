
#ifndef HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP
#define HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP

#include <hcorepp/operators/interface/Tile.hpp>

namespace hcorepp {
    namespace operators {

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
             * Dense Tile parameterized constructor.
             *
             * @param aNumOfRows
             * NUm of tile rows, should be >=0
             * @param aNumOfCols
             * Num of tile cols, should be >=0
             * @param aPdata
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
             * @param aOperation
             * transposition operation of the tile.
             * @param aUplo
             * logical packed storage type of the tile
             */
            CompressedTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim, int64_t aRank,
                           blas::real_type<T> aAccuracy, blas::Layout aLayout = blas::Layout::ColMajor,
                           blas::Op aOperation = blas::Op::NoTrans, blas::Uplo aUplo = blas::Uplo::General);


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
            std::reference_wrapper<dataunits::DataHolder <T>>
            GetTileSubMatrix(size_t aIndex) override;

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
            const std::reference_wrapper<dataunits::DataHolder <T>>
            GetTileSubMatrix(size_t aIndex) const override;

            /**
             * @brief
             * Get number of matrices describing the tile.
             *
             * @return
             * Number of matrices describing the tile (should be two).
             */
            size_t
            GetNumberOfMatrices() const override;

            /**
             * @brief
             * General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
             * Equivalent to Reduced SVD in old codebase.
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
                 const helpers::SvdHelpers &aHelpers) override;

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
             * @brief
             * Get matrix accuracy.
             *
             * @return
             * Tile accuracy.
             */
            blas::real_type<T>
            GetAccuracy();

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

            /**
             * @brief
             * Readjust tile dimension according to new rank.
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
            /** Vector of data arrays */
            std::vector<dataunits::DataHolder < T> *>
            mDataArrays;
            /** Linear Algebra Matrix rank*/
            int64_t mMatrixRank;
            /** Numerical error thershold */
            blas::real_type<T> mAccuracy;
            /** number of rows */
            size_t mNumOfRows;
            /** number of cols */
            size_t mNumOfCols;
            static const int64_t FULL_RANK_ = -1;
        };

        template
        class CompressedTile<float>;

        template
        class CompressedTile<double>;

    }
}

#endif //HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP
