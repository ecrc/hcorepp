/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/kernels/kernels.hpp>
#include <hcorepp/kernels/memory.hpp>

using namespace hcorepp::dataunits;
using namespace hcorepp::common;
using namespace hcorepp::kernels;

namespace hcorepp::operators {

    template<typename T>
    DenseTile<T>::DenseTile() = default;

    template<typename T>
    DenseTile<T>::DenseTile(size_t aNumOfRows, size_t aNumOfCols, T *aPdata, size_t aLeadingDim,
                            blas::Layout aLayout, const hcorepp::kernels::RunContext &aContext,
                            bool aMemoryOwnership) {
        this->mLayout = aLayout;
        this->mLeadingDim = aLeadingDim;
        this->mNumOfRows = aNumOfRows;
        this->mNumOfCols = aNumOfCols;
        this->mRank = 0;
        this->mpDataArray = new DataHolder<T>(aNumOfRows, aNumOfCols, aLeadingDim, aPdata, aContext, aMemoryOwnership);
    }

    template<typename T>
    DenseTile<T>::~DenseTile<T>() {
        delete this->mpDataArray;
    }

    template<typename T>
    T *DenseTile<T>::GetTileSubMatrix(size_t aIndex) const {
        if (aIndex != 0) {
            throw std::invalid_argument(
                    "GetTileSubMatrix ::Index out of range, should be 0 in case of dense tile.\n");
        }

        return this->mpDataArray->GetData();
    }

    template<typename T>
    void
    DenseTile<T>::Gemm(T &aAlpha, DataHolder <T> const &aTileA, blas::Op aTileAOp, DataHolder <T> const &aTileB,
                       blas::Op aTileBOp, T &aBeta, size_t aLdAu, size_t aARank,
                       const CompressionParameters &aHelpers, const RunContext &aContext,
                       size_t &aFlops, MemoryUnit <T> &aMemoryUnit, bool aCholesky) {
        auto m = this->mNumOfRows;
        auto n = this->mNumOfCols;

        auto k = std::min(aTileA.GetNumOfCols(), aTileB.GetNumOfRows()); //aARank;//aTileA.GetNumOfCols();
        if (aTileAOp == blas::Op::Trans) {
            k = std::min(aTileA.GetNumOfRows(), aTileB.GetNumOfRows());
            if (aTileBOp == blas::Op::Trans) {
                k = std::min(aTileA.GetNumOfRows(), aTileB.GetNumOfCols());
            }
        } else if (aTileBOp == blas::Op::Trans) {
            k = std::min(aTileA.GetNumOfCols(), aTileB.GetNumOfCols());
        }
        size_t flops = 2 * m * n * aTileA.GetNumOfCols();
        aFlops += flops;

        auto lda = aTileA.GetLeadingDim();
        if (aTileAOp == blas::Op::Trans) {
            if (this->GetLayout() == blas::Layout::RowMajor) {
                lda = (aTileA.GetNumOfRows() < m) ? m : aTileA.GetNumOfRows();
            } else if (this->GetLayout() == blas::Layout::ColMajor) {
                lda = (aTileA.GetNumOfCols() < k) ? k : aTileA.GetNumOfCols();
            }
        } else if (aTileAOp == blas::Op::NoTrans) {
            if (this->GetLayout() == blas::Layout::RowMajor) {
                lda = (aTileA.GetNumOfCols() < k) ? k : aTileA.GetNumOfCols();
            } else if (this->GetLayout() == blas::Layout::ColMajor) {
                lda = (aTileA.GetNumOfRows() < m) ? m : aTileA.GetNumOfRows();
            }
        }

        auto ldb = aTileB.GetLeadingDim();

        if (aTileBOp == blas::Op::Trans) {
            if (this->GetLayout() == blas::Layout::RowMajor) {
                ldb = (aTileB.GetNumOfRows() < k) ? k : aTileB.GetNumOfRows();
            } else if (this->GetLayout() == blas::Layout::ColMajor) {
                ldb = (aTileB.GetNumOfCols() < n) ? n : aTileB.GetNumOfCols();
            }
        } else if (aTileBOp == blas::Op::NoTrans) {
            if (this->GetLayout() == blas::Layout::RowMajor) {
                ldb = (aTileB.GetNumOfCols() < n) ? n : aTileB.GetNumOfCols();
            } else if (this->GetLayout() == blas::Layout::ColMajor) {
                ldb = (aTileB.GetNumOfRows() < k) ? k : aTileB.GetNumOfRows();
            }
        }

        /**
         * Assuming that C operation is blas::Op::NoTrans
         * And C Layout is Column major.
         */
        return hcorepp::kernels::HCoreKernels<T>::Gemm(this->GetLayout(), aTileAOp, aTileBOp,
                                                       m, n, k, aAlpha, (const T *) aTileA.GetData(), lda,
                                                       (const T *) aTileB.GetData(), ldb,
                                                       aBeta, this->GetTileSubMatrix(0),
                                                       this->GetDataHolder().get().GetLeadingDim(),
                                                       aContext);
    }

    template<typename T>
    size_t DenseTile<T>::GetTileStride(size_t aIndex) const {
        if (aIndex != 0) {
            throw std::invalid_argument(
                    "DenseTile::GetTileStride:: Index out of range, should be 0 in case of dense tile.\n");
        }

        return this->mpDataArray->GetLeadingDim();
    }

    template<typename T>
    void
    DenseTile<T>::ReadjustTile(size_t aNumOfRows, size_t aNumOfCols, T *aPdata, size_t aLeadingDim,
                               size_t aRank, const RunContext &aContext) {
        /* Unimplemented in Dense Case */
    }

    template<typename T>
    std::pair<TileMetadata *, T *> DenseTile<T>::UnPackTile(const RunContext &aContext) {
        auto *metadata = new TileMetadata(this->mNumOfRows, this->mNumOfCols, this->mRank, this->mRank,
                                          this->mLeadingDim,
                                          this->mLayout, DENSE);

        return {metadata, this->mpDataArray->GetData()};
    }

    template<typename T>
    void
    DenseTile<T>::PackTile(TileMetadata aMetadata, T *aDataArray, const RunContext &aContext) {
        this->UpdateMetadata(aMetadata);
        this->mpDataArray = new DataHolder<T>(this->mNumOfRows, this->mNumOfCols, this->mLeadingDim, aDataArray,
                                              aContext, false);
    }

    template<typename T>
    void DenseTile<T>::ReadjustTileRank(size_t aRank, const kernels::RunContext &aContext) {
        /* Unimplemented in Dense Case */
    }

    template<typename T>
    void DenseTile<T>::UpdateMetadata(TileMetadata aMetadata) {
        this->mLayout = aMetadata.mLayout;
        this->mLeadingDim = aMetadata.mLeadingDimension;
        this->mNumOfRows = aMetadata.mNumOfRows;
        this->mNumOfCols = aMetadata.mNumOfCols;
        this->mRank = aMetadata.mMatrixRank;
    }

    HCOREPP_INSTANTIATE_CLASS(DenseTile)

}
