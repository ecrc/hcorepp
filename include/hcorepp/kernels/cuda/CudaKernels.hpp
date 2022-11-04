/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCOREPP_KERNELS_CUDA_KERNELS_HPP
#define HCOREPP_KERNELS_CUDA_KERNELS_HPP

#include <cusolver_common.h>
#include "blas/util.hh"
#include <hcorepp/common/Definitions.hpp>
#include <hcorepp/operators/helpers/CompressionParameters.hpp>

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
            static void GenerateIdentityMatrix(int64_t aNumOfCols, T *apMatrix);


            static void MultiplyByAlpha(T *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank, T &aAlpha);


            static void Geqrf(int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apTau);


            static void
            ProcessVpointer(int64_t aN, int64_t aCRank, bool aGetUngqr, int64_t Vm, T &aBeta, T *apCV, int64_t aLdcV,
                            T *V,
                            int64_t aArank, const T *apBdata);


            static void
            CalculateUVptrConj(int64_t aRank, int64_t aVm, T *UVptr);


            static void CalculateVTnew(int64_t aRkNew, bool aUngqr, int64_t aMinVmVn, blas::real_type<T> *apSigma, T *apVTnew,
                                int64_t aSizeS, int64_t aVm);


            static void CalculateUVptr(int64_t aRank, int64_t aVm, T *UVptr, const T *Vnew);


            static void CalculateNewRank(int64_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma, int64_t sizeS,
                                  blas::real_type<T> accuracy);


            static void
            Gesvd(common::Job aJobu, common::Job aJobvt, int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apS, T *apU,
                  int64_t aLdU, T *apVT, int64_t aLdVt);


            static void
            Unmqr(common::SideMode aSide, common::BlasOperation aTrans, int64_t aM, int64_t aN, int64_t aK,
                  T const *apA, int64_t aLdA, T const *apTau, T *apC, int64_t aLdC);


            static void Laset(common::MatrixType aMatrixType, int64_t aM, int64_t aN, T aOffdiag, T aDiag,
                       T *apA, int64_t aLdA);


            static void
            LaCpy(common::MatrixType aType, int64_t aM, int64_t aRank, T *apCU, int64_t aLD, T *apU, int64_t aUm);


            static void
            ungqr(int64_t aM, int64_t aN, int64_t aK, T *apA, int64_t aLdA, T *apTau);

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
