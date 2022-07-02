//
// Created by mirna on 21/06/2022.
//

#ifndef HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP
#define HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP

#include <hcorePP/operators/interface/Tile.hpp>
#include <cstdint>

using namespace hcorepp::dataunits;
using namespace std;

namespace hcorepp {
    namespace operators {

        template<typename T>
        class CompressedTile : public Tile<T> {
        public:

            CompressedTile();

            /**
             * @brief
             * Compressed Tile constructor
             *
             * @param[in] aDataArrays
             * Vector of Data arrays representing the dense tile.
             */
            CompressedTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim, int64_t aRank,
                           real_t aAccuracy, blas::Layout aLayout = blas::Layout::ColMajor,
                           blas::Op aOperation = blas::Op::NoTrans, blas::Uplo aUplo = blas::Uplo::General);


            /**
             * @brief
             * Dense Tile destructor.
             */
            ~CompressedTile();

            DataHolder<T> &
            GetTileSubMatrix(size_t aIndex) override;

            DataHolder<T> const *
            GetTileSubMatrixConst(size_t aIndex) const override;

            size_t
            GetNumberOfMatrices() override;

            void
            Gemm(T &aAlpha, DataHolder<T> const &aTileA, blas::Op aTileAOp, DataHolder<T> const &aTileB,
                 blas::Op aTileBOp, T &aBeta) override;

            /**
             * @brief
             * Set Rank of the matrix.
             *
             * @param[in] aMatrixRank
             * Matrix rank.
             */
            void
            SetTileRank(int64_t &aMatrixRank) const;

            /**
             * @brief
             * Get the rank of matrix.
             *
             * @return
             * matrix rank.
             */
            int64_t
            GetTileRank() const;

//            /**
//             * @brief
//             * Get matrix accuracy.
//             *
//             * @return
//             * Matrix accuracy.
//             */
//            real_t
//            GetAccuracy();
//
//            /**
//             * @brief
//             * Set matrix accuracy.
//             *
//             * @param[in] aAccuracy
//             * matrix accuracy to set.
//             */
//            void
//            SetAccuracy(real_t aAccuracy);

            int64_t
            GetTileStride(size_t aIndex) const override;

//            bool
//            IsFullRank() const;

        private:
            /** Vector of data arrays */
            vector<DataHolder<T> &> mDataArrays;
            /** Linear Algebra Matrix rank*/
            int64_t mMatrixRank;
            /** Numerical error thershold */
            real_t mAccuracy;
            /** number of rows */
            size_t mNumOfRows;
            /** number of cols */
            size_t mNumOfCols;

        };
    }
}

#endif //HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP
