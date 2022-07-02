
#ifndef HCOREPP_OPERATORS_CONCRETE_DENSE_HPP
#define HCOREPP_OPERATORS_CONCRETE_DENSE_HPP

#include <hcorePP/operators/interface/Tile.hpp>

namespace hcorepp {
    namespace operators {

        template<typename T>
        class DenseTile : public Tile<T> {

        public:

            /**
             * @brief
             * Dense Tile constructor
             *
             * @param[in] aDataArrays
             * Vector of Data arrays representing the dense tile.
             */
            DenseTile();

            DenseTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                      blas::Layout aLayout = blas::Layout::ColMajor, blas::Op aOperation = blas::Op::NoTrans,
                      blas::Uplo aUplo = blas::Uplo::General);
            /**
             * @brief
             * Dense Tile destructor.
             */
            ~DenseTile();

            DataHolder<T> &
            GetTileSubMatrix(size_t aIndex) override;

            size_t
            GetNumberOfMatrices() override;

            void
            Gemm(T &aAlpha, DataHolder<T> const &aTileA, blas::Op aTileAOp, DataHolder<T> const &aTileB,
                 blas::Op aTileBOp, T &aBeta) override;


        private:
            /** vector of references to data arrays representing the Dense tile. */
            vector<DataHolder<T> &> mDataArrays;
            /** number of rows */
            size_t mNumOfRows;
            /** number of cols */
            size_t mNumOfCols;

        };
    }
}

#endif //HCOREPP_DENSE_HPP
