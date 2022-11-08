/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <hcorepp/kernels/kernels.hpp>
#include <cstring>
#include <cublas_v2.h>
#include "hcorepp/kernels/cuda/CudaKernels.hpp"
#include "hcorepp/kernels/cuda/error_checking.h"

namespace hcorepp {
    namespace kernels {

        template<typename T>
        void HCoreKernels<T>::Gemm(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, int64_t aM, int64_t aN,
                                   int64_t aK,
                                   T &aAlpha, T const *apA, int64_t aLdA, T const *apB, int64_t aLdB, T &aBeta, T *apC,
                                   int64_t aLdC) {
            GPU_ERROR_CHECK(cudaDeviceSynchronize());
            blas::Queue queue;
            blas::gemm(aLayout, aTransA, aTransB, aM, aN, aK, aAlpha, apA, aLdA, apB, aLdB, aBeta, apC, aLdC, queue);
            queue.sync();
        }

        template<typename T>
        void
        HCoreKernels<T>::MultiplyByAlpha(T *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank,
                                         T &aAlpha) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::MultiplyByAlpha(apArray, aRows, aCols, aM, aRank, aAlpha);
        }

        template<typename T>
        void
        HCoreKernels<T>::ProcessVpointer(int64_t aN, int64_t aCRank, bool aGetUngqr, int64_t Vm, T &aBeta, T *apCV,
                                         int64_t aLdcV, T *V,
                                         int64_t aArank, const T *apBdata) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::ProcessVpointer(aN, aCRank, aGetUngqr, Vm, aBeta, apCV, aLdcV, V, aArank, apBdata);
        }

        template<typename T>
        void HCoreKernels<T>::CalculateNewRank(int64_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma,
                                               int64_t sizeS,
                                               blas::real_type<T> accuracy) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::CalculateNewRank(aNewRank, aTruncatedSvd, apSigma, sizeS, accuracy);
        }

        template<typename T>
        void HCoreKernels<T>::CalculateUVptr(int64_t aRank, int64_t aVm, T *UVptr, const T *Vnew) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::CalculateUVptr(aRank, aVm, UVptr, Vnew);
        }

        template<typename T>
        void
        HCoreKernels<T>::CalculateVTnew(int64_t aRkNew, bool aUngqr, int64_t aMinVmVn, blas::real_type<T> *apSigma,
                                        T *apVTnew,
                                        int64_t aSizeS, int64_t aVm) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::CalculateVTnew(aRkNew, aUngqr, aMinVmVn, apSigma, apVTnew, aSizeS, aVm);
        }

        template<typename T>
        void
        HCoreKernels<T>::CalculateUVptrConj(int64_t aRank, int64_t aVm, T *UVptr) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::CalculateUVptrConj(aRank, aVm, UVptr);
        }

        template<typename T>
        void
        HCoreKernels<T>::FillIdentityMatrix(int64_t aNumOfElements, T *apMatrix) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::GenerateIdentityMatrix(aNumOfElements, apMatrix);
        }

        template<typename T>
        void
        HCoreKernels<T>::LaCpy(common::MatrixType aType, int64_t aM, int64_t aRank, T *apCU, int64_t aLD, T *apU,
                               int64_t aUm) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::LaCpy(aType, aM, aRank, apCU, aLD, apU, aUm);
        }

        template<typename T>
        void HCoreKernels<T>::Geqrf(int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apTau) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::Geqrf(aM, aN, apA, aLdA, apTau);
        }

        template<typename T>
        void HCoreKernels<T>::Laset(common::MatrixType aMatrixType, int64_t aM, int64_t aN, T aOffdiag, T aDiag,
                                    T *apA, int64_t aLdA) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::Laset(aMatrixType, aM, aN, aOffdiag, aDiag, apA, aLdA);
        }

        template<typename T>
        void HCoreKernels<T>::Trmm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans,
                                   blas::Diag aDiag,
                                   int64_t aM, int64_t aN, T aAlpha, T const *apA, int64_t aLdA, T *apB, int64_t aLdB) {
            GPU_ERROR_CHECK(cudaDeviceSynchronize());
            blas::Queue queue;
            blas::trmm(aLayout, aSide, aUplo, aTrans, aDiag, aM, aN, aAlpha, apA, aLdA, apB, aLdB, queue);
            queue.sync();
        }

        template<typename T>
        void
        HCoreKernels<T>::SVD(common::Job aJobu, common::Job aJobvt, int64_t aM, int64_t aN, T *apA, int64_t aLdA,
                             T *apS, T *apU,
                             int64_t aLdU, T *apVT, int64_t aLdVt, common::CompressionType aSVDOperation) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::SVD(aJobu, aJobvt, aM, aN, apA, aLdA, apS, apU, aLdU,
                                                           apVT, aLdVt, aSVDOperation);
        }

        template<typename T>
        void
        HCoreKernels<T>::Unmqr(common::SideMode aSide, common::BlasOperation aTrans, int64_t aM, int64_t aN, int64_t aK,
                               T const *apA,
                               int64_t aLdA, T const *apTau, T *apC, int64_t aLdC) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::Unmqr(aSide, aTrans, aM, aN, aK, apA, aLdA, apTau, apC, aLdC);
        }

        template<typename T>
        blas::real_type<T> *HCoreKernels<T>::AllocateSigma(int64_t aSizeS) {
            blas::real_type<T> *sigma;
            cudaMalloc((void **) &sigma, aSizeS * sizeof(blas::real_type<T>));

            return sigma;
        }

        template<typename T>
        void HCoreKernels<T>::DestroySigma(blas::real_type<T> *apSigma) {
            if (apSigma != nullptr) {
                cudaFree(apSigma);
            }
        }

        template<typename T>
        void
        HCoreKernels<T>::ungqr(int64_t aM, int64_t aN, int64_t aK, T *apA, int64_t aLdA, T *apTau) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::ungqr(aM, aN, aK, apA, aLdA, apTau);
        }

        HCOREPP_INSTANTIATE_CLASS(HCoreKernels)

    }
}
