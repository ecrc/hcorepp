#include <functional>
#include <iostream>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/kernels/kernels.hpp>

using namespace hcorepp::dataunits;
using namespace hcorepp::helpers;

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
            this->mDataArrays.push_back(new DataHolder<T>(aNumOfRows, aNumOfCols, aLeadingDim, aPdata));
        }

        template<typename T>
        DenseTile<T>::~DenseTile<T>() {
            for (auto data_holder: this->mDataArrays) {
                delete data_holder;
            }
        }

        template<typename T>
        std::reference_wrapper<dataunits::DataHolder<T>> DenseTile<T>::GetTileSubMatrix(size_t aIndex) {
            if (aIndex != 0) {
                throw std::invalid_argument(
                        "GetTileSubMatrix ::Index out of range, should be 0 in case of dense tile.\n");
            }

            return *mDataArrays[aIndex];
        }

        template<typename T>
        std::reference_wrapper<dataunits::DataHolder<T>> const DenseTile<T>::GetTileSubMatrix(size_t aIndex) const {
            if (aIndex != 0) {
                throw std::invalid_argument(
                        "DenseTile::GetTileSubMatrix:: Index out of range, should be 0 in case of dense tile.\n");
            }

            return *mDataArrays[aIndex];
        }

        template<typename T>
        void DenseTile<T>::Gemm(T &aAlpha, DataHolder<T> const &aTileA, blas::Op aTileAOp, DataHolder<T> const &aTileB,
                                blas::Op aTileBOp, T &aBeta, int64_t aLdAu, int64_t aARank,
                                const SvdHelpers &aHelpers) {

            /**
             * Assuming that C operation is blas::Op::NoTrans
             * And C Layout is Column major.
             */

            hcorepp::kernels::Gemm<T>(this->layout(), aTileAOp, aTileBOp,
                          this->mNumOfRows, this->mNumOfCols, aTileA.GetNumOfCols(),
                          aAlpha, (const T *) aTileA.GetData(), aTileA.GetLeadingDim(),
                          (const T *) aTileB.GetData(), aTileB.GetLeadingDim(),
                          aBeta, this->GetTileSubMatrix(0).get().GetData(),
                          this->GetTileSubMatrix(0).get().GetLeadingDim());

        }

        template<typename T>
        size_t DenseTile<T>::GetNumberOfMatrices() const {
            return mDataArrays.size();
        }

        template<typename T>
        int64_t DenseTile<T>::GetTileStride(size_t aIndex) const {
            if (aIndex != 0) {
                throw std::invalid_argument(
                        "DenseTile::GetTileStride:: Index out of range, should be 0 in case of dense tile.\n");

                std::cerr << "Index out of range, should be 0 in case of dense tile. ";
            }

            return this->mDataArrays[aIndex]->GetLeadingDim();
        }

        template<typename T>
        void
        DenseTile<T>::ReadjustTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                                   int64_t aRank) {

        }

    }
}
