
#ifndef HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP
#define HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP

#include <hcorepp/operators/interface/Tile.hpp>

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
                           blas::real_type<T> aAccuracy, blas::Layout aLayout = blas::Layout::ColMajor,
                           blas::Op aOperation = blas::Op::NoTrans, blas::Uplo aUplo = blas::Uplo::General);


            CompressedTile(CompressedTile<T> &atile);

            /**
             * @brief
             * Compressed Tile destructor.
             */
            ~CompressedTile();

            std::reference_wrapper<dataunits::DataHolder < T>>
            GetTileSubMatrix(
            size_t aIndex
            )
            override;

            std::reference_wrapper<dataunits::DataHolder < T>> const
            GetTileSubMatrix(size_t
            aIndex)
            const override;

            size_t
            GetNumberOfMatrices() const override;

            void
            Gemm(T &aAlpha, dataunits::DataHolder <T> const &aTileA, blas::Op aTileAOp,
                 dataunits::DataHolder <T> const &aTileB,
                 blas::Op aTileBOp, T &aBeta, int64_t ldau, int64_t Ark, const helpers::SvdHelpers &aHelpers) override;

            /**
             * @brief
             * Set Rank of the matrix.
             *
             * @param[in] aMatrixRank
             * Matrix rank.
             */
            void
            SetTileRank(int64_t &aMatrixRank);

            /**
             * @brief
             * Get the rank of matrix.
             *
             * @return
             * matrix rank.
             */
            int64_t
            GetTileRank() const;

            /**
             * @brief
             * Get matrix accuracy.
             *
             * @return
             * Matrix accuracy.
             */
            blas::real_type<T>
            GetAccuracy();

            size_t
            GetNumOfRows() const;

            size_t
            GetNumOfCols() const;

            int64_t
            GetTileStride(size_t aIndex) const override;


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

//        template
//        class CompressedTile<int>;

//        template
//        class CompressedTile<long>;

        template
        class CompressedTile<float>;

        template
        class CompressedTile<double>;

    }
}

#endif //HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP
