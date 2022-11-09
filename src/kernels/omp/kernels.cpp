/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <iostream>
#include <hcorepp/kernels/kernels.hpp>
#include <lapack.hh>

namespace hcorepp {
    namespace kernels {


        template<typename T>
        void HCoreKernels<T>::Gemm(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, int64_t aM, int64_t aN,
                                   int64_t aK,
                                   T &aAlpha, T const *apA, int64_t aLdA, T const *apB, int64_t aLdB, T &aBeta, T *apC,
                                   int64_t aLdC, RunContext &aContext) {
            blas::gemm(aLayout, aTransA, aTransB, aM, aN, aK, aAlpha, apA, aLdA, apB, aLdB, aBeta, apC, aLdC);
        }

        template<typename T>
        void
        HCoreKernels<T>::MultiplyByAlpha(T *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank,
                                         T &aAlpha, RunContext &aContext) {
            for (int i = 0; i < aRows * aCols; i++) {
                apArray[aM * aRank + i] *= aAlpha;
            }
        }

        template<typename T>
        void
        HCoreKernels<T>::ProcessVpointer(int64_t aN, int64_t aCRank, bool aGetUngqr, int64_t Vm, T &aBeta, T *apCV,
                                         int64_t aLdcV, T *V,
                                         int64_t aArank, const T *apBdata, RunContext &aContext) {
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


        template<typename T>
        void HCoreKernels<T>::CalculateNewRank(int64_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma,
                                               int64_t sizeS, blas::real_type<T> accuracy,
                                               RunContext &aContext) {
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

        template<typename T>
        void HCoreKernels<T>::CalculateUVptr(int64_t aRank, int64_t aVm, T *UVptr, const T *Vnew,
                                             RunContext &aContext) {
            for (int64_t j = 0; j < aRank; ++j) {
                for (int64_t i = 0; i < aVm; ++i) {
                    UVptr[j + i * aRank] = blas::conj(Vnew[i + j * aVm]);
                }
            }
        }

        template<typename T>
        void
        HCoreKernels<T>::CalculateVTnew(int64_t aRkNew, bool aUngqr, int64_t aMinVmVn, blas::real_type<T> *apSigma,
                                        T *apVTnew, int64_t aSizeS, int64_t aVm, RunContext &aContext) {
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


        template<typename T>
        void
        HCoreKernels<T>::CalculateUVptrConj(int64_t aRank, int64_t aVm, T *UVptr, RunContext &aContext) {
            for (int64_t i = 0; i < aRank; ++i) {
                for (int64_t j = 0; j < aVm; ++j) {
                    UVptr[i + j * aRank] = blas::conj(UVptr[i + j * aRank]);
                }
            }
        }


        template<typename T>
        void
        HCoreKernels<T>::FillIdentityMatrix(int64_t aNumOfElements, T *apMatrix, RunContext &aContext) {
            for (int i = 0; i < aNumOfElements; i++) {
                int index = i * aNumOfElements + i;
                apMatrix[index] = 1;
            }
        }


        template<typename T>
        void
        HCoreKernels<T>::LaCpy(common::MatrixType aType, int64_t aM, int64_t aRank, T *apCU, int64_t aLD, T *apU,
                               int64_t aUm, RunContext &aContext) {
            lapack::lacpy((lapack::MatrixType) aType, aM, aRank, apCU, aLD, apU, aUm);
        }

        template<typename T>
        void HCoreKernels<T>::Geqrf(int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apTau,
                                    RunContext &aContext) {
            lapack::geqrf(aM, aN, apA, aLdA, apTau);
        }

        template<typename T>
        void HCoreKernels<T>::Laset(common::MatrixType aMatrixType, int64_t aM, int64_t aN, T aOffdiag, T aDiag,
                                    T *apA, int64_t aLdA, RunContext &aContext) {
            lapack::laset((lapack::MatrixType) aMatrixType, aM, aN, aOffdiag, aDiag, apA, aLdA);
        }

        template<typename T>
        void HCoreKernels<T>::Trmm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans,
                                   blas::Diag aDiag, int64_t aM, int64_t aN, T aAlpha, T const *apA, int64_t aLdA,
                                   T *apB, int64_t aLdB, RunContext &aContext) {
            blas::trmm(aLayout, aSide, aUplo, aTrans, aDiag, aM, aN, aAlpha, apA, aLdA, apB, aLdB);
        }

        template<typename T>
        void
        HCoreKernels<T>::SVD(common::Job aJobu, common::Job aJobvt, int64_t aM, int64_t aN, T *apA, int64_t aLdA,
                             T *apS, T *apU, int64_t aLdU, T *apVT, int64_t aLdVt, common::CompressionType aSVDType,
                             RunContext &aContext) {
            if (aSVDType == common::CompressionType::LAPACK_GESVD) {
                lapack::gesvd((lapack::Job) aJobu, (lapack::Job) aJobvt, aM, aN, apA, aLdA, apS, apU, aLdU, apVT,
                              aLdVt);
            } else {
                lapack::gesdd((lapack::Job) aJobu, aM, aN, apA, aLdA, apS, apU, aLdU, apVT, aLdVt);
            }
        }

        template<typename T>
        void
        HCoreKernels<T>::Unmqr(common::SideMode aSide, common::BlasOperation aTrans, int64_t aM, int64_t aN, int64_t aK,
                               T const *apA, int64_t aLdA, T const *apTau, T *apC, int64_t aLdC,
                               RunContext &aContext) {
            lapack::unmqr((lapack::Side) aSide, (lapack::Op) aTrans, aM, aN, aK, apA, aLdA, apTau, apC, aLdC);
        }

        template<typename T>
        blas::real_type<T> *HCoreKernels<T>::AllocateSigma(int64_t aSizeS, RunContext &aContext) {
            blas::real_type<T> *sigma;
            sigma = (blas::real_type<T> *) malloc(aSizeS * sizeof(blas::real_type<T>));
            return sigma;
        }

        template<typename T>
        void HCoreKernels<T>::DestroySigma(blas::real_type<T> *apSigma, RunContext &aContext) {
            free(apSigma);
        }

        template<typename T>
        void
        HCoreKernels<T>::ungqr(int64_t aM, int64_t aN, int64_t aK, T *apA, int64_t aLdA, T *apTau,
                               RunContext &aContext) {
            lapack::ungqr(aM, aN, aK, apA, aLdA, apTau);
        }

        HCOREPP_INSTANTIATE_CLASS(HCoreKernels)

    }
}
