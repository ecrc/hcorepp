/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_KERNELS_CUDA_KERNELS_HPP
#define HCOREPP_KERNELS_CUDA_KERNELS_HPP

#include <cusolver_common.h>
#include "blas/util.hh"
#include <hcorepp/common/Definitions.hpp>
#include <hcorepp/operators/helpers/CompressionParameters.hpp>
#include <hcorepp/kernels/RunContext.hpp>

namespace hcorepp {
    namespace cudakernels {

        template<typename T>
        struct traits;

        template<>
        struct traits<float> {
            // scalar type
            typedef float T;
            typedef T S;

            static constexpr cudaDataType cuda_data_type = CUDA_R_32F;
#if CUDART_VERSION >= 11000
            static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_32F;
#endif

        };

        template<>
        struct traits<double> {
            // scalar type
            typedef double T;
            typedef T S;

            static constexpr T zero = 0.;
            static constexpr cudaDataType cuda_data_type = CUDA_R_64F;
#if CUDART_VERSION >= 11000
            static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_64F;
#endif

        };

        template<typename T>
        class HCoreCudaKernels {
        public:
            static void GenerateIdentityMatrix(size_t aNumOfCols, T *apMatrix, const kernels::RunContext &aContext);


            static void MultiplyByAlpha(T *apArray, size_t aRows, size_t aCols, size_t aM, size_t aRank,
                                        T &aAlpha, const kernels::RunContext &aContext);


            static void
            Geqrf(size_t aM, size_t aN, T *apA, size_t aLdA, T *apTau, T *aWorkspace, size_t aWorkspace_size,
                  size_t aHostSize, const kernels::RunContext &aContext);


            static void
            ProcessVpointer(size_t aN, size_t aCRank, bool aGetUngqr, size_t Vm, T &aBeta, T *apCV, size_t aLdcV,
                            T *V, size_t aArank, const T *apBdata, const kernels::RunContext &aContext,
                            bool aCholesky = false);


            static void
            CalculateUVptrConj(size_t aRank, size_t aVm, T *UVptr, const kernels::RunContext &aContext);

            static void
            CalculateVTnew(size_t aRkNew, bool aUngqr, size_t aMinVmVn, blas::real_type<T> *apSigma, T *apVTnew,
                           size_t aSizeS, size_t aVm, const kernels::RunContext &aContext);


            static void CalculateUVptr(size_t aRank, size_t aVm, T *UVptr, const T *Vnew,
                                       const kernels::RunContext &aContext);

            static void
            CalculateNewRank(size_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma, size_t sizeS,
                             blas::real_type<T> accuracy, const kernels::RunContext &aContext);


            static void
            SVD(common::Job aJobu, common::Job aJobvt, size_t aM, size_t aN, T *apA, size_t aLdA, T *apS, T *apU,
                size_t aLdU, T *apVT, size_t aLdVt, common::CompressionType aSVDOperationType,
                T *aWorkspace, size_t aWorkspace_size, size_t aHostSize, const kernels::RunContext &aContext);


            static void
            Unmqr(common::SideMode aSide, common::BlasOperation aTrans, size_t aM, size_t aN, size_t aK,
                  T const *apA, size_t aLdA, T const *apTau, T *apC, size_t aLdC,
                  T *aWorkspace, size_t aWorkspace_size, const kernels::RunContext &aContext);

            static void Laset(common::MatrixType aMatrixType, size_t aM, size_t aN, T aOffdiag, T aDiag,
                              T *apA, size_t aLdA, const kernels::RunContext &aContext);


            static void
            LaCpy(common::MatrixType aType, size_t aM, size_t aRank, T *apCU, size_t aLD, T *apU, size_t aUm,
                  const kernels::RunContext &aContext);

            static void
            ungqr(size_t aM, size_t aN, size_t aK, T *apA, size_t aLdA, T *apTau,
                  T *aWorkspace, size_t aWorkspace_size, const kernels::RunContext &aContext);

            static void
            potrf(blas::Uplo aUplo, T *aWorkspace, size_t aWorkspaceSize, size_t aHostSize,
                  size_t aMatrixOrder, T *apMatrix, size_t aLeadingDim,
                  const kernels::RunContext &aContext);

            static void
            FillMatrixTriangle(blas::Uplo aUplo, size_t aRows, T *apMatrix, blas::Layout aLayout,
                               size_t aValue, const kernels::RunContext &aContext);

            static void
            SymmetrizeMatrix(blas::Uplo aUplo, size_t aRows, T *apMatrix, blas::Layout aLayout,
                             const kernels::RunContext &aContext);

            static void
            TransposeMatrix(size_t aOuterLoopRange, size_t aInnerLoopRange, const T *aA, size_t aLeadingDimA,
                            T *aOut, size_t aLeadingDimOut, const kernels::RunContext &aContext);

        private:
            /**
             * @brief
             * Default constructor private to prevent construction of instances of this class.
             */
            HCoreCudaKernels() = default;
        };
    }
}

#endif //HCOREPP_KERNELS_CUDA_KERNELS_HPP
