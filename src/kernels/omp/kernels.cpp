/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#include <iostream>
#include <hcorepp/kernels/kernels.hpp>
#include <cstring>
#include <lapack.hh>

namespace hcorepp {
    namespace kernels {

        template<typename T>
        void Gemm(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, int64_t aM, int64_t aN, int64_t aK,
                  T &aAlpha, T const *apA, int64_t aLdA, T const *apB, int64_t aLdB, T &aBeta, T *apC, int64_t aLdC) {
            blas::gemm(aLayout, aTransA, aTransB, aM, aN, aK, aAlpha, apA, aLdA, apB, aLdB, aBeta, apC, aLdC);
        }

        template
        void Gemm(blas::Layout, blas::Op, blas::Op, int64_t, int64_t, int64_t,
                  float &, float const *, int64_t, float const *, int64_t, float &, float *, int64_t);

        template
        void Gemm(blas::Layout, blas::Op, blas::Op, int64_t, int64_t, int64_t,
                  double &, double const *, int64_t, double const *, int64_t, double &, double *, int64_t);

        template<typename T>
        void
        MultiplyByAlpha(T *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank, T &aAlpha) {
            for (int i = 0; i < aRows * aCols; i++) {
                apArray[aM * aRank + i] *= aAlpha;
            }
        }

        template
        void
        MultiplyByAlpha(float *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank, float &aAlpha);

        template
        void
        MultiplyByAlpha(double *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank, double &aAlpha);

        template<typename T>
        void
        ProcessVpointer(int64_t aN, int64_t aCRank, bool aGetUngqr, int64_t Vm, T &aBeta, T *apCV, int64_t aLdcV, T *V,
                        int64_t aArank, const T *apBdata) {
            for (int64_t j = 0; j < aN; ++j) {
                for (int64_t i = 0; i < aCRank; ++i) {
                    if (aGetUngqr) {
                        V[j + i * Vm] = blas::conj(aBeta * apCV[i + j * aLdcV]);
                    } else {
                        V[j + i * Vm] = aBeta * apCV[i + j * aLdcV];
                    }
                }
            }

            for (int64_t j = 0; j < aN; ++j) {
                T *Vptr = &V[aN * aCRank];
                for (int64_t i = 0; i < aArank; ++i) {
                    if (aGetUngqr) {
                        Vptr[j + i * Vm] = blas::conj(apBdata[i + j * aArank]);
                    } else {
                        Vptr[j + i * Vm] = apBdata[i + j * aArank];
                    }
                }
            }
        }

        template
        void
        ProcessVpointer(int64_t, int64_t, bool, int64_t, float &, float *, int64_t, float *, int64_t, const float *);

        template
        void
        ProcessVpointer(int64_t, int64_t, bool, int64_t, double &, double *, int64_t, double *, int64_t,
                        const double *);

        template<typename T>
        void CalculateNewRank(int64_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma, int64_t sizeS,
                              blas::real_type<T> &accuracy) {
            aNewRank = sizeS;
            if (aTruncatedSvd) {
                blas::real_type<T> Sigma_0 = apSigma[0];
                for (int64_t i = 1; i < sizeS; i++) {
                    if (apSigma[i] < accuracy * Sigma_0) {
                        Sigma_0 = apSigma[i];
                        aNewRank = i;
                        break;
                    }
                }
            } else {
                for (int64_t i = 1; i < sizeS; i++) {
                    if (apSigma[i] < accuracy) {
                        aNewRank = i;
                        break;
                    }
                }
            }
        }

        template
        void CalculateNewRank<float>(int64_t &, bool, blas::real_type<float> *, int64_t, blas::real_type<float> &);

        template
        void CalculateNewRank<double>(int64_t &, bool, blas::real_type<double> *, int64_t, blas::real_type<double> &);

        template<typename T>
        void CalculateUVptr(int64_t aRank, int64_t aVm, T *UVptr, const T *Vnew) {
            for (int64_t j = 0; j < aRank; ++j) {
                for (int64_t i = 0; i < aVm; ++i) {
                    UVptr[j + i * aRank] = blas::conj(Vnew[i + j * aVm]);
                }
            }
        }

        template
        void CalculateUVptr(int64_t, int64_t, float *, const float *);

        template
        void CalculateUVptr(int64_t, int64_t, double *, const double *);

        template<typename T>
        void
        CalculateVTnew(int64_t aRkNew, bool aUngqr, int64_t aMinVmVn, blas::real_type<T> *apSigma, T *apVTnew,
                       int64_t aSizeS, int64_t aVm) {
            for (int64_t i = 0; i < aRkNew; ++i) {
                if (aUngqr) {
                    blas::scal(aMinVmVn, apSigma[i], &apVTnew[i], aSizeS);
                } else {
                    blas::scal(aVm, apSigma[i], &apVTnew[i], aSizeS);
                    for (int64_t j = 0; j < aVm; ++j) {
                        apVTnew[i + j * aSizeS] = blas::conj(apVTnew[i + j * aSizeS]);
                    }
                }
            }
        }

        template
        void
        CalculateVTnew(int64_t, bool, int64_t, blas::real_type<float> *, float *, int64_t, int64_t);

        template
        void
        CalculateVTnew(int64_t, bool, int64_t, blas::real_type<double> *, double *, int64_t, int64_t);

        template<typename T>
        void
        CalculateUVptrConj(int64_t aRank, int64_t aVm, T *UVptr) {
            for (int64_t i = 0; i < aRank; ++i) {
                for (int64_t j = 0; j < aVm; ++j) {
                    UVptr[i + j * aRank] = blas::conj(UVptr[i + j * aRank]);
                }
            }
        }

        template
        void CalculateUVptrConj(int64_t, int64_t, float *);

        template
        void CalculateUVptrConj(int64_t, int64_t, double *);

        template<typename T>
        void
        FillIdentityMatrix(int64_t aNumOfElements, T *apMatrix) {
            for (int i = 0; i < aNumOfElements; i++) {
                int index = i * aNumOfElements + i;
                apMatrix[index] = 1;
            }
        }

        template
        void
        FillIdentityMatrix(int64_t aNumOfElements, float *apMatrix);

        template
        void
        FillIdentityMatrix(int64_t aNumOfElements, double *apMatrix);

        template<typename T>
        void
        LaCpy(common::MatrixType aType, int64_t aM, int64_t aRank, T *apCU, int64_t aLD, T *apU, int64_t aUm) {
            lapack::lacpy((lapack::MatrixType) aType, aM, aRank, apCU, aLD, apU, aUm);
        }

        template
        void
        LaCpy(common::MatrixType, int64_t, int64_t, float *, int64_t, float *, int64_t);

        template
        void
        LaCpy(common::MatrixType, int64_t, int64_t, double *, int64_t, double *, int64_t);

        template<typename T>
        void Geqrf(int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apTau) {
            lapack::geqrf(aM, aN, apA, aLdA, apTau);
        }

        template
        void Geqrf(int64_t, int64_t, float *, int64_t, float *);

        template
        void Geqrf(int64_t, int64_t, double *, int64_t, double *);

        template<typename T>
        void Laset(common::MatrixType aMatrixType, int64_t aM, int64_t aN, T aOffdiag, T aDiag,
                   T *apA, int64_t aLdA) {
            lapack::laset((lapack::MatrixType) aMatrixType, aM, aN, aOffdiag, aDiag, apA, aLdA);
        }

        template
        void Laset(common::MatrixType aMatrixType, int64_t aM, int64_t aN, float aOffdiag, float aDiag,
                   float *apA, int64_t aLdA);

        template
        void Laset(common::MatrixType aMatrixType, int64_t aM, int64_t aN, double aOffdiag, double aDiag,
                   double *apA, int64_t aLdA);

        template<typename T>
        void Trmm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans, blas::Diag aDiag,
                  int64_t aM, int64_t aN, T aAlpha, T const *apA, int64_t aLdA, T *apB, int64_t aLdB) {
            blas::trmm(aLayout, aSide, aUplo, aTrans, aDiag, aM, aN, aAlpha, apA, aLdA, apB, aLdB);
        }

        template
        void Trmm(blas::Layout, blas::Side, blas::Uplo, blas::Op, blas::Diag, int64_t, int64_t, float,
                  float const *, int64_t, float *, int64_t);

        template
        void Trmm(blas::Layout, blas::Side, blas::Uplo, blas::Op, blas::Diag, int64_t, int64_t, double,
                  double const *, int64_t, double *, int64_t);

        template<typename T>
        void
        Gesvd(common::Job aJobu, common::Job aJobvt, int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apS, T *apU,
              int64_t aLdU, T *apVT, int64_t aLdVt) {
            lapack::gesvd((lapack::Job) aJobu, (lapack::Job) aJobvt, aM, aN, apA, aLdA, apS, apU, aLdU, apVT, aLdVt);
        }

        template
        void
        Gesvd(common::Job, common::Job, int64_t, int64_t, float *, int64_t, float *, float *, int64_t, float *,
              int64_t);

        template
        void
        Gesvd(common::Job, common::Job, int64_t, int64_t, double *, int64_t, double *, double *, int64_t, double *,
              int64_t);

        template<typename T>
        void
        Unmqr(common::SideMode aSide, common::BlasOperation aTrans, int64_t aM, int64_t aN, int64_t aK,
              T const *apA, int64_t aLdA,
              T const *apTau, T *apC, int64_t aLdC) {
            lapack::unmqr((lapack::Side) aSide, (lapack::Op) aTrans, aM, aN, aK, apA, aLdA, apTau, apC, aLdC);
        }

        template
        void Unmqr(common::SideMode, common::BlasOperation, int64_t, int64_t, int64_t, float const *, int64_t,
                   float const *, float *, int64_t);

        template
        void
        Unmqr(common::SideMode, common::BlasOperation, int64_t, int64_t, int64_t, double const *, int64_t,
              double const *, double *, int64_t);

        template<typename T>
        blas::real_type<T> *AllocateSigma(int64_t aSizeS) {
            blas::real_type<T> *sigma;
            sigma = (blas::real_type<T> *) malloc(aSizeS * sizeof(blas::real_type<T>));
            return sigma;
        }

        template
        blas::real_type<float> *AllocateSigma<float>(int64_t);

        template
        blas::real_type<double> *AllocateSigma<double>(int64_t);

        template<typename T>
        void DestroySigma(blas::real_type<T> *apSigma) {
            free(apSigma);
        }

        template
        void DestroySigma<float>(blas::real_type<float> *);

        template
        void DestroySigma<double>(blas::real_type<double> *);

        template<typename T>
        void
        ungqr(int64_t aM, int64_t aN, int64_t aK, T *apA, int64_t aLdA, T *apTau) {
            lapack::ungqr(aM, aN, aK, apA, aLdA, apTau);
        }

        template
        void
        ungqr(int64_t, int64_t, int64_t, float *, int64_t, float *);

        template
        void
        ungqr(int64_t, int64_t, int64_t, double *, int64_t, double *);


    }
}
