
#include <hcorePP/operators/concrete/Dense.hpp>

namespace hcorepp {
    namespace operators {

        template<typename T>
        DenseTile<T>::DenseTile() {

        }

        template<typename T>
        DenseTile<T>::DenseTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                                blas::Layout aLayout, blas::Op aOperation, blas::Uplo aUplo) {
            this->mOperation = aOperation;
            this->mLayout = aLayout;
            this->mUpLo = aUplo;
            this->mLeadingDim = aLeadingDim;
            this->mNumOfRows = aNumOfRows;
            this->mNumOfCols = aNumOfCols;
            this->mDataArrays.emplace_back(DataHolder<T>(aNumOfRows, aNumOfCols, aLeadingDim, aPdata));
        }

        template<typename T>
        DenseTile<T>::~DenseTile<T>() {
        }

        template<typename T>
        DataHolder<T> &DenseTile<T>::GetTileSubMatrix(size_t aIndex) {
            return mDataArrays[aIndex];
        }

        template<typename T>
        void DenseTile<T>::Gemm(T &aAlpha, DataHolder<T> const &aTileA, blas::Op aTileAOp, DataHolder<T> const &aTileB,
                                blas::Op aTileBOp, T &aBeta) {

            /**
             * Assuming that C operation is blas::Op::NoTrans
             * And C Layout is Column major.
             */
            blas::gemm(blas::Layout::ColMajor, aTileAOp, aTileBOp,
                       this->mNumOfRows, this->mNumOfCols, aTileA.mNumOfCols,
                       alpha, aTileA.GetData(), aTileA.GetLeadingDim(),
                       aTileB.GetData(), aTileB.GetLeadingDim(),
                       beta, GetTileSubMatrix(0), GetTileSubMatrix(0).GetLeadingDim());
        }

        template<typename T>
        size_t DenseTile<T>::GetNumberOfMatrices() {
            return mDataArrays.size();
        }
    }
}
