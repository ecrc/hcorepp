/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_KERNELS_KERNELS_HPP
#define HCOREPP_KERNELS_KERNELS_HPP

#include "blas.hh"
#include "hcorepp/operators/helpers/CompressionParameters.hpp"
#include "RunContext.hpp"

namespace hcorepp {
    namespace kernels {
        /**
         * @brief
         * Class containing all the major kernels for the HCore++ operations, allowing
         * multi-technology support.
         * 
         * @tparam T
         * The type the kernels will operate on.
         */
        template<typename T>
        class HCoreKernels {
        public:

            static void Gemm(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, size_t aM,
                             size_t aN, size_t aK, T &aAlpha, T const *apA, size_t aLdA, T const *apB,
                             size_t aLdB, T &aBeta, T *apC, size_t aLdC, const RunContext &aContext);


            static void MultiplyByAlpha(T *apArray, size_t aRows, size_t aCols, size_t aM, size_t aRank,
                                        T &aAlpha, const RunContext &aContext);


            static void
            ProcessVpointer(size_t aN, size_t aCRank, bool aGetUngqr, size_t Vm, T &aBeta, T *apCV, size_t aLdcV,
                            T *V, size_t aArank, const T *apBdata, const RunContext &aContext, bool aCholesky = false);


            static void CalculateNewRank(size_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma,
                                         size_t sizeS, blas::real_type<T> accuracy, const RunContext &aContext);


            static void CalculateUVptr(size_t aRank, size_t aVm, T *UVptr, const T *Vnew,
                                       const RunContext &aContext);


            static void
            CalculateVTnew(size_t aRkNew, bool aUngqr, size_t aMinVmVn, blas::real_type<T> *apSigma, T *apVTnew,
                           size_t aSizeS, size_t aVm, const RunContext &aContext);


            static void CalculateUVptrConj(size_t aRank, size_t aVm, T *UVptr, const RunContext &aContext);


            static void FillIdentityMatrix(size_t aNumOfElements, T *apMatrix, const RunContext &aContext);


            static void
            LaCpy(common::MatrixType aType, size_t aM, size_t aRank, T *apCU, size_t aLD, T *apU, size_t aUm,
                  const RunContext &aContext);


            static void Geqrf(size_t aM, size_t aN, T *apA, size_t aLdA, T *apTau, T *aWorkspace,
                              size_t aWorkspace_size, size_t aHostSize, const RunContext &aContext);


            static void Laset(common::MatrixType aMatrixType, size_t aM, size_t aN, T aOffdiag, T aDiag,
                              T *apA, size_t aLdA, const RunContext &aContext);


            static void
            Trmm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag,
                 size_t aM, size_t aN, T aAlpha, T const *apA, size_t aLdA, T *apB, size_t aLdB,
                 const RunContext &aContext);


            static void
            SVD(common::Job aJobu, common::Job aJobvt, size_t aM, size_t aN, T *apA, size_t aLdA, T *apS, T *apU,
                size_t aLdU, T *apVT, size_t aLdVt, common::CompressionType aSVDType, T *aWorkspace,
                size_t aWorkspace_size,
                size_t aHostSize, const RunContext &aContext);


            static void
            Unmqr(common::SideMode aSide, common::BlasOperation aTrans, size_t aM, size_t aN, size_t aK,
                  T const *apA, size_t aLdA,
                  T const *apTau, T *apC, size_t aLdC, T *aWorkspace, size_t aWorkspace_size,
                  const RunContext &aContext);


            static void
            ungqr(size_t aM, size_t aN, size_t aK, T *apA, size_t aLdA, T *apTau, T *aWorkspace,
                  size_t aWorkspace_size, const RunContext &aContext);

            static size_t
            CalculateGemmWorkspaceSize(size_t aUm, size_t aUn, size_t aVm, size_t aVn, size_t aSizeS,
                                       const operators::CompressionParameters &aHelpers, size_t &aHostSize,
                                       const RunContext &aContext);

            /** Computes the Cholesky factorization of a symmetric (Hermitian) positive-definite matrix.*/
            static int potrf(blas::Uplo aUplo, T *aWorkspace, size_t aWorkspaceSize, size_t aHostSize,
                             size_t aMatrixOrder, T *apMatrix, size_t aLeadingDim, blas::Layout aLayout,
                             const kernels::RunContext &aContext);

            static void FillMatrixTriangle(blas::Uplo aUplo, size_t aRows, size_t aCols, T *apMatrix,
                                           blas::Layout aLayout, size_t aValue, const kernels::RunContext &aContext);

            static void trsm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans,
                             blas::Diag aDiag, size_t aRows, size_t aCols, T aAlpha, T const *apMatrixA,
                             size_t aLeadingDimA, T *apMatrixB, size_t aLeadingDimB,
                             const kernels::RunContext &aContext);

            static void
            syrk(blas::Layout aLayout, blas::Uplo aUplo, blas::Op aTrans, size_t aRows, size_t aCols, T aAlpha,
                 T const *apMatrixA, size_t aLeadingDimA, T aBeta, T *apMatrixB, size_t aLeadingDimB,
                 const kernels::RunContext &aContext);

            static void
            Symmetrize(blas::Layout aLayout, T *apMatrixA, size_t aRows, size_t aCols, blas::Uplo aUplo,
                       const RunContext &aContext);

            static void transpose(blas::Layout aLayout, size_t aRows, size_t aCols, const T *aA, size_t aLeadingDimA,
                                  T *aOut, size_t aLeadingDimOut, const kernels::RunContext &aContext);

            static size_t CalculatePotrfWorkspaceSize(T *apMatrix, blas::Uplo aUplo, size_t aMatrixOrder,
                                                      size_t aLeadingDim, size_t &aHostSize,
                                                      const RunContext &aContext);

        private:
            /**
             * @brief
             * Private constructor to prevent creating instances
             */
            HCoreKernels() = default;
        };
    }
}

#endif //HCOREPP_KERNELS_KERNELS_HPP
