/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
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

            static void Gemm(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, int64_t aM,
                             int64_t aN, int64_t aK, T &aAlpha, T const *apA, int64_t aLdA, T const *apB,
                             int64_t aLdB, T &aBeta, T *apC, int64_t aLdC, RunContext &aContext);


            static void MultiplyByAlpha(T *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank,
                                        T &aAlpha, RunContext &aContext);


            static void
            ProcessVpointer(int64_t aN, int64_t aCRank, bool aGetUngqr, int64_t Vm, T &aBeta, T *apCV, int64_t aLdcV,
                            T *V, int64_t aArank, const T *apBdata, RunContext &aContext);


            static void CalculateNewRank(int64_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma,
                                         int64_t sizeS, blas::real_type<T> accuracy, RunContext &aContext);


            static void CalculateUVptr(int64_t aRank, int64_t aVm, T *UVptr, const T *Vnew,
                                       RunContext &aContext);


            static void
            CalculateVTnew(int64_t aRkNew, bool aUngqr, int64_t aMinVmVn, blas::real_type<T> *apSigma, T *apVTnew,
                           int64_t aSizeS, int64_t aVm, RunContext &aContext);


            static void CalculateUVptrConj(int64_t aRank, int64_t aVm, T *UVptr, RunContext &aContext);


            static void FillIdentityMatrix(int64_t aNumOfElements, T *apMatrix, RunContext &aContext);


            static void
            LaCpy(common::MatrixType aType, int64_t aM, int64_t aRank, T *apCU, int64_t aLD, T *apU, int64_t aUm,
                  RunContext &aContext);


            static void Geqrf(int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apTau, RunContext &aContext);


            static void Laset(common::MatrixType aMatrixType, int64_t aM, int64_t aN, T aOffdiag, T aDiag,
                              T *apA, int64_t aLdA, RunContext &aContext);


            static void
            Trmm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag,
                 int64_t aM, int64_t aN, T aAlpha, T const *apA, int64_t aLdA, T *apB, int64_t aLdB,
                 RunContext &aContext);


            static void
            SVD(common::Job aJobu, common::Job aJobvt, int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apS, T *apU,
                int64_t aLdU, T *apVT, int64_t aLdVt, common::CompressionType aSVDType,
                RunContext &aContext);


            static void
            Unmqr(common::SideMode aSide, common::BlasOperation aTrans, int64_t aM, int64_t aN, int64_t aK,
                  T const *apA, int64_t aLdA,
                  T const *apTau, T *apC, int64_t aLdC, RunContext &aContext);


            static blas::real_type<T> *AllocateSigma(int64_t aSizeS, RunContext &aContext);


            static void DestroySigma(blas::real_type<T> *apSigma, RunContext &aContext);


            static void
            ungqr(int64_t aM, int64_t aN, int64_t aK, T *apA, int64_t aLdA, T *apTau, RunContext &aContext);

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
