
#ifndef HCOREPP_OPERATORS_CONCRETE_DENSE_HPP
#define HCOREPP_OPERATORS_CONCRETE_DENSE_HPP

#include <hcorepp/operators/interface/Tile.hpp>

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

            std::reference_wrapper<dataunits::DataHolder<T>>
            GetTileSubMatrix(size_t aIndex) override;

            const std::reference_wrapper<dataunits::DataHolder<T>>
            GetTileSubMatrix(size_t aIndex) const override;

            size_t
            GetNumberOfMatrices() const override;

            int64_t
            GetTileStride(size_t aIndex) const override;

            void
            Gemm(T &aAlpha, dataunits::DataHolder<T> const &aTileA, blas::Op aTileAOp, dataunits::DataHolder<T> const &aTileB,
                 blas::Op aTileBOp, T &aBeta, int64_t ldau, int64_t Ark, const helpers::SvdHelpers &aHelpers) override;

            void
            ReadjustTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                         int64_t aRank) override;


        private:
            /** vector of references to data arrays representing the Dense tile. */
            std::vector<dataunits::DataHolder<T> *> mDataArrays;
            /** number of rows */
            size_t mNumOfRows;
            /** number of cols */
            size_t mNumOfCols;

        };
//        template class DenseTile<int>;
//        template class DenseTile<long>;
        template class DenseTile<float>;
        template class DenseTile<double>;

    }

}

#endif //HCOREPP_DENSE_HPP
