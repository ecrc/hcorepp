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

            /**
             * @brief
             * Compressed Tile constructor
             *
             * @param[in] aDataArrays
             * Vector of Data arrays representing the dense tile.
             */
            CompressedTile(vector <DataHolder<T> &> aDataArrays);

            /**
             * @brief
             * Dense Tile destructor.
             */
            ~CompressedTile();

            DataHolder<T> &
            GetTileSubMatrix(size_t aIndex) override;

            size_t
            GetNumberOfMatrices() override;

            void
            Gemm(T &aAlpha, DataHolder <T> const &aTileA, DataHolder <T> const &aTileB, T &aBeta) override;

            /**
             * @brief
             * Set Rank of the matrix.
             *
             * @param[in] aMatrixRank
             * Matrix rank.
             */
            void
            SetMatrixRank(int64_t &aMatrixRank);

            /**
             * @brief
             * Get the rank of matrix.
             *
             * @return
             * matrix rank.
             */
            int64_t
            GetMatrixRank();

            /**
             * @brief
             * Get matrix accuracy.
             *
             * @return
             * Matrix accuracy.
             */
            real_t
            GetAccuracy();

            /**
             * @brief
             * Set matrix accuracy.
             *
             * @param[in] aAccuracy
             * matrix accuracy to set.
             */
            void
            SetAccuracy(real_t aAccuracy);

        private:
            /** Vector of data arrays */
            vector<DataHolder < T> &>mDataArrays;
            /** Linear Algebra Matrix rank*/
            int64_t mMatrixRank;
            /** Numerical error thershold */
            real_t mAccuracy;
        };
    }
}

#endif //HCOREPP_OPERATORS_CONCRETE_COMPRESSED_HPP
