
#include <hcorePP/operators/concrete/Compressed.hpp>

namespace hcorepp {
    namespace operators {

        template<typename T>
        CompressedTile<T>::CompressedTile(vector<DataHolder<T> &> aDataArrays) {

        }

        template<typename T>
        CompressedTile<T>::~CompressedTile<T>() {

        }

        template<typename T>
        DataHolder<T> & CompressedTile<T>::GetTileSubMatrix(size_t aIndex) {
        }


        template<typename T>
        void CompressedTile<T>::Gemm(T &aAlpha, DataHolder<T> const &aTileA, DataHolder<T> const &aTileB, T &aBeta) {

        }

        template<typename T>
        size_t CompressedTile<T>::GetNumberOfMatrices() {
            return 0;
        }

        template<typename T>
        void CompressedTile<T>::SetMatrixRank(int64_t &aMatrixRank) {

        }

        template<typename T>
        int64_t CompressedTile<T>::GetMatrixRank() {
            return 0;
        }

        template<typename T>
        real_t CompressedTile<T>::GetAccuracy() {
        }

        template<typename T>
        void CompressedTile<T>::SetAccuracy(real_t aAccuracy) {

        }
    }

}