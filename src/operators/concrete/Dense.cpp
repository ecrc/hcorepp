
#include <hcorePP/operators/concrete/Dense.hpp>

namespace hcorepp {
    namespace operators {

        template<typename T>
        DenseTile<T>::DenseTile(vector<DataHolder<T> &> aDataArrays) {

        }

        template<typename T>
        DenseTile<T>::~DenseTile<T>() {

        }

        template<typename T>
        DataHolder<T> & DenseTile<T>::GetTileSubMatrix(size_t aIndex) {
        }


        template<typename T>
        void DenseTile<T>::Gemm(T &aAlpha, DataHolder<T> const &aTileA, DataHolder<T> const &aTileB, T &aBeta) {

        }

        template<typename T>
        size_t DenseTile<T>::GetNumberOfMatrices() {
            return 0;
        }
    }
}
