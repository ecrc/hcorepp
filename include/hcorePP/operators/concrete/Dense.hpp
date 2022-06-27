
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
            DenseTile(vector <DataHolder<T> &> aDataArrays);

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
            Gemm(T &aAlpha, DataHolder <T> const &aTileA, DataHolder <T> const &aTileB, T &aBeta) override;

        private:
            /** vector of references to data arrays representing the Dense tile. */
            vector <DataHolder<T> &> mDataArrays;
        };
    }
}

#endif //HCOREPP_DENSE_HPP
