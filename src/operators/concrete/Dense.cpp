
#include <hcorepp/operators/concrete/Dense.hpp>
#include <functional>
#include <iostream>

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
            return *mDataArrays[aIndex];
        }

        template<typename T>
        std::reference_wrapper<dataunits::DataHolder<T>> const DenseTile<T>::GetTileSubMatrix(size_t aIndex) const {
            return *mDataArrays[aIndex];
        }

        template<typename T>
        void DenseTile<T>::Gemm(T &aAlpha, DataHolder<T> const &aTileA, blas::Op aTileAOp, DataHolder<T> const &aTileB,
                                blas::Op aTileBOp, T &aBeta, int64_t aLdAu, int64_t aARank, const SvdHelpers &aHelpers) {

            /**
             * Assuming that C operation is blas::Op::NoTrans
             * And C Layout is Column major.
             */
            std::cout << "A LEading dim " << aTileA.GetLeadingDim() << " B leading dim " << aTileB.GetLeadingDim()
                      << "\n";
            std::cout << "C LEading dim " << this->GetTileSubMatrix(0).get().GetLeadingDim() << "\n";

            std::cout << " GEMM INPUT A \n";
            for (int i = 0; i < aTileA.GetNumOfRows(); i++) {
                std::cout << "{ ";
                for (int j = 0; j < aTileA.GetNumOfCols(); j++) {
                    int index = i * aTileA.GetNumOfCols() + j;
                    std::cout << aTileA.GetData()[index] << ", ";
                }
                std::cout << "} \n";
            }
            std::cout << " GEMM INPUT B \n";
            for (int i = 0; i < aTileB.GetNumOfRows(); i++) {
                std::cout << "{ ";
                for (int j = 0; j < aTileB.GetNumOfCols(); j++) {
                    int index = i * aTileB.GetNumOfCols() + j;
                    std::cout << aTileB.GetData()[index] << ", ";
                }
                std::cout << "} \n";
            }

            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                       this->mNumOfRows, this->mNumOfCols, aTileA.GetNumOfCols(),
                       aAlpha, (const T *) aTileA.GetData(), aTileA.GetLeadingDim(),
                       (const T *) aTileB.GetData(), aTileB.GetLeadingDim(),
                       aBeta, this->GetTileSubMatrix(0).get().GetData(),
                       this->GetTileSubMatrix(0).get().GetLeadingDim());

            std::cout << " GEMM OUTPUT \n";
            for (int i = 0; i < this->GetTileSubMatrix(0).get().GetNumOfCols(); i++) {
                std::cout << "{ ";
                for (int j = 0; j < this->GetTileSubMatrix(0).get().GetNumOfRows(); j++) {
                    int index = i * this->GetTileSubMatrix(0).get().GetNumOfRows() + j;
                    std::cout << this->GetTileSubMatrix(0).get().GetData()[index] << ", ";
                }
                std::cout << "} \n";
            }
            std::cout << " ================================================ \n";

        }

        template<typename T>
        size_t DenseTile<T>::GetNumberOfMatrices() const {
            return mDataArrays.size();
        }

        template<typename T>
        int64_t DenseTile<T>::GetTileStride(size_t aIndex) const {
            return this->mDataArrays[aIndex]->GetLeadingDim();
        }

        template<typename T> void
        DenseTile<T>::ReadjustTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                     int64_t aRank) {

        }

    }
}
