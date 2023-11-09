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
        void HCoreKernels<T>::Gemm(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, size_t aM, size_t aN,
                                   size_t aK, T &aAlpha, T const *apA, size_t aLdA, T const *apB, size_t aLdB,
                                   T &aBeta, T *apC, size_t aLdC, const RunContext &aContext) {
            blas::gemm(aLayout, aTransA, aTransB, aM, aN, aK, aAlpha, apA, aLdA, apB, aLdB, aBeta, apC, aLdC);
        }

        template<typename T>
        void
        HCoreKernels<T>::MultiplyByAlpha(T *apArray, size_t aRows, size_t aCols, size_t aM, size_t aRank,
                                         T &aAlpha, const RunContext &aContext) {
            for (size_t i = 0; i < aRows * aCols; i++) {
                apArray[aM * aRank + i] *= aAlpha;
            }
        }

        template<typename T>
        void
        HCoreKernels<T>::ProcessVpointer(size_t aN, size_t aCRank, bool aGetUngqr, size_t Vm, T &aBeta, T *apCV,
                                         size_t aLdcV, T *V, size_t aArank, const T *apBdata,
                                         const RunContext &aContext, bool aCholesky) {
            if (aCholesky) {
                for (size_t j = 0; j < aN; ++j) {
                    for (size_t i = 0; i < aCRank; ++i) {
                        if (aGetUngqr) {
                            V[i + j * aLdcV] = blas::conj(aBeta * apCV[i + j * aLdcV]);
                        } else {
                            V[i + j * aLdcV] = aBeta * apCV[i + j * aLdcV];
                        }
                    }
                }

                T *Vptr = &V[aN * aCRank];
                for (size_t j = 0; j < aN; ++j) {
                    for (size_t i = 0; i < aArank; ++i) {
                        if (aGetUngqr) {
                            Vptr[i + j * aArank] = blas::conj(apBdata[i + j * aArank]);
                        } else {
                            Vptr[i + j * aArank] = apBdata[i + j * aArank];
                        }
                    }
                }

            } else {
                for (size_t j = 0; j < aN; ++j) {
                    for (size_t i = 0; i < aCRank; ++i) {
                        if (aGetUngqr) {
                            V[j + i * Vm] = blas::conj(aBeta * apCV[i + j * aLdcV]);
                        } else {
                            V[j + i * Vm] = aBeta * apCV[i + j * aLdcV];
                        }
                    }
                }

                T *Vptr = &V[aN * aCRank];
                for (size_t j = 0; j < aN; ++j) {
                    for (size_t i = 0; i < aArank; ++i) {
                        if (aGetUngqr) {
                            Vptr[j + i * Vm] = blas::conj(apBdata[i + j * aArank]);
                        } else {
                            Vptr[j + i * Vm] = apBdata[i + j * aArank];
                        }
                    }
                }
            }
        }


        template<typename T>
        void HCoreKernels<T>::CalculateNewRank(size_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma,
                                               size_t sizeS, blas::real_type<T> accuracy,
                                               const RunContext &aContext) {
            aNewRank = sizeS;
            if (aTruncatedSvd) {
                blas::real_type<T> Sigma_0 = apSigma[0];
                for (size_t i = 1; i < sizeS; i++) {
                    if (apSigma[i] < accuracy * Sigma_0) {
                        Sigma_0 = apSigma[i];
                        aNewRank = i;
                        break;
                    }
                }
            } else {
                for (size_t i = 1; i < sizeS; i++) {
                    if (apSigma[i] < accuracy) {
                        aNewRank = i;
                        break;
                    }
                }
            }
        }

        template<typename T>
        void HCoreKernels<T>::CalculateUVptr(size_t aRank, size_t aVm, T *UVptr, const T *Vnew,
                                             const RunContext &aContext) {
            for (size_t j = 0; j < aRank; ++j) {
                for (size_t i = 0; i < aVm; ++i) {
                    UVptr[j + i * aRank] = blas::conj(Vnew[i + j * aVm]);
                }
            }
        }

        template<typename T>
        void
        HCoreKernels<T>::CalculateVTnew(size_t aRkNew, bool aUngqr, size_t aMinVmVn, blas::real_type<T> *apSigma,
                                        T *apVTnew, size_t aSizeS, size_t aVm, const RunContext &aContext) {
            for (size_t i = 0; i < aRkNew; ++i) {
                if (aUngqr) {
                    blas::scal(aMinVmVn, apSigma[i], &apVTnew[i], aSizeS);
                } else {
                    blas::scal(aVm, apSigma[i], &apVTnew[i], aSizeS);
                    for (size_t j = 0; j < aVm; ++j) {
                        apVTnew[i + j * aSizeS] = blas::conj(apVTnew[i + j * aSizeS]);
                    }
                }
            }
        }


        template<typename T>
        void
        HCoreKernels<T>::CalculateUVptrConj(size_t aRank, size_t aVm, T *UVptr, const RunContext &aContext) {
            for (size_t i = 0; i < aRank; ++i) {
                for (size_t j = 0; j < aVm; ++j) {
                    UVptr[i + j * aRank] = blas::conj(UVptr[i + j * aRank]);
                }
            }
        }


        template<typename T>
        void
        HCoreKernels<T>::FillIdentityMatrix(size_t aNumOfElements, T *apMatrix, const RunContext &aContext) {
            for (size_t i = 0; i < aNumOfElements; i++) {
                size_t index = i * aNumOfElements + i;
                apMatrix[index] = 1;
            }
        }


        template<typename T>
        void
        HCoreKernels<T>::LaCpy(common::MatrixType aType, size_t aM, size_t aRank, T *apCU, size_t aLD, T *apU,
                               size_t aUm, const RunContext &aContext) {
            lapack::lacpy((lapack::MatrixType) aType, aM, aRank, apCU, aLD, apU, aUm);
        }

        template<typename T>
        void HCoreKernels<T>::Geqrf(size_t aM, size_t aN, T *apA, size_t aLdA, T *apTau, T *aWorkspace,
                                    size_t aWorkspaceSize,
                                    size_t aHostSize, const RunContext &aContext) {
            lapack::geqrf(aM, aN, apA, aLdA, apTau);
        }

        template<typename T>
        void HCoreKernels<T>::Laset(common::MatrixType aMatrixType, size_t aM, size_t aN, T aOffdiag, T aDiag,
                                    T *apA, size_t aLdA, const RunContext &aContext) {
            lapack::laset((lapack::MatrixType) aMatrixType, aM, aN, aOffdiag, aDiag, apA, aLdA);
        }

        template<typename T>
        void HCoreKernels<T>::Trmm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans,
                                   blas::Diag aDiag, size_t aM, size_t aN, T aAlpha, T const *apA, size_t aLdA,
                                   T *apB, size_t aLdB, const RunContext &aContext) {
            blas::trmm(aLayout, aSide, aUplo, aTrans, aDiag, aM, aN, aAlpha, apA, aLdA, apB, aLdB);
        }

        template<typename T>
        void
        HCoreKernels<T>::SVD(common::Job aJobu, common::Job aJobvt, size_t aM, size_t aN, T *apA, size_t aLdA,
                             T *apS, T *apU, size_t aLdU, T *apVT, size_t aLdVt, common::CompressionType aSVDType,
                             T *aWorkspace, size_t aWorkspaceSize, size_t aHostSize, const RunContext &aContext) {
            if (aSVDType == common::CompressionType::LAPACK_GESVD) {
                lapack::gesvd((lapack::Job) aJobu, (lapack::Job) aJobvt, aM, aN, apA, aLdA, apS, apU, aLdU, apVT,
                              aLdVt);
            } else {
                lapack::gesdd((lapack::Job) aJobu, aM, aN, apA, aLdA, apS, apU, aLdU, apVT, aLdVt);
            }
        }

        template<typename T>
        void
        HCoreKernels<T>::Unmqr(common::SideMode aSide, common::BlasOperation aTrans, size_t aM, size_t aN, size_t aK,
                               T const *apA, size_t aLdA, T const *apTau, T *apC, size_t aLdC,
                               T *aWorkspace, size_t aWorkspaceSize, const RunContext &aContext) {
            lapack::Side side;
            if (aSide == common::SIDE_LEFT) {
                side = lapack::Side::Left;
            } else if (aSide == common::SIDE_RIGHT) {
                side = lapack::Side::Right;
            }

            lapack::Op trans;
            if (aTrans == common::OP_TRANS) {
                trans = lapack::Op::Trans;
            } else if (aTrans == common::OP_CONJG) {
                trans = lapack::Op::ConjTrans;
            } else if (aTrans == common::OP_NoTRANS) {
                trans = lapack::Op::NoTrans;
            }

            lapack::unmqr(side, trans, aM, aN, aK, apA, aLdA, apTau, apC, aLdC);
        }

        template<typename T>
        void
        HCoreKernels<T>::ungqr(size_t aM, size_t aN, size_t aK, T *apA, size_t aLdA, T *apTau,
                               T *aWorkspace, size_t aWorkspaceSize, const RunContext &aContext) {
            lapack::ungqr(aM, aN, aK, apA, aLdA, apTau);
        }

        template<typename T>
        size_t
        HCoreKernels<T>::CalculateGemmWorkspaceSize(size_t aUm, size_t aUn, size_t aVm, size_t aVn, size_t aSizeS,
                                                    const operators::CompressionParameters &aHelpers,
                                                    size_t &aHostSize, const RunContext &aContext) {
            aHostSize = 0;
            return 0;
        }

        template<typename T>
        int
        HCoreKernels<T>::potrf(blas::Uplo aUplo, T *aWorkspace, size_t aWorkspaceSize, size_t aHostSize,
                               size_t aMatrixOrder, T *apMatrix, size_t aLeadingDim, blas::Layout aLayout,
                               const kernels::RunContext &aContext) {

            int err = lapack::potrf((lapack::Uplo) aUplo, aMatrixOrder, apMatrix, aLeadingDim);

            return err;
        }

        template<typename T>
        void HCoreKernels<T>::FillMatrixTriangle(blas::Uplo aUplo, size_t aRows, size_t aCols, T *apMatrix,
                                                 blas::Layout aLayout, size_t aValue,
                                                 const kernels::RunContext &aContext) {
            if (aRows != aCols) {
                return;
            }

            for (auto i = 0; i < aRows; i++) {
                for (auto j = i + 1; j < aRows; j++) {
                    auto index = 0;
                    if (aUplo == lapack::Uplo::Upper) {
                        index = (aLayout == blas::Layout::RowMajor) ? i * aRows + j : j * aRows + i;
                    } else if (aUplo == lapack::Uplo::Lower) {
                        index = (aLayout == blas::Layout::RowMajor) ? j * aRows + i : i * aRows + j;
                    }
                    apMatrix[index] = aValue;
                }
            }
        }

        template<typename T>
        void HCoreKernels<T>::trsm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans,
                                   blas::Diag aDiag, size_t aRows, size_t aCols, T aAlpha, const T *apMatrixA,
                                   size_t aLeadingDimA, T *apMatrixB, size_t aLeadingDimB,
                                   const kernels::RunContext &aContext) {
            blas::trsm(aLayout, aSide, aUplo, aTrans, aDiag, aRows, aCols, aAlpha, apMatrixA, aLeadingDimA, apMatrixB,
                       aLeadingDimB);
        }

        template<typename T>
        void
        HCoreKernels<T>::syrk(blas::Layout aLayout, blas::Uplo aUplo, blas::Op aTrans, size_t aRows, size_t aCols,
                              T aAlpha, const T *apMatrixA, size_t aLeadingDimA, T aBeta, T *apMatrixB,
                              size_t aLeadingDimB, const RunContext &aContext) {
            blas::syrk<T>(aLayout, aUplo, aTrans, aRows, aCols, aAlpha, apMatrixA, aLeadingDimA, aBeta, apMatrixB,
                          aLeadingDimB);
        }

        template<typename T>
        void HCoreKernels<T>::Symmetrize(blas::Layout aLayout, T *apMatrixA, size_t aRows, size_t aCols,
                                         blas::Uplo aUplo, const RunContext &aContext) {

            if (aRows != aCols) {
                return;
            }

            for (size_t i = 0; i < aRows; i++) {
                for (size_t j = i + 1; j < aRows; j++) {
                    if (aUplo == blas::Uplo::Upper) {
                        auto src_idx = (aLayout == blas::Layout::RowMajor) ? i * aRows + j : j * aRows + i;
                        auto dest_idx = (aLayout == blas::Layout::RowMajor) ? j * aRows + i : i * aRows + j;
                        apMatrixA[dest_idx] = apMatrixA[src_idx];
                    } else if (aUplo == blas::Uplo::Lower) {
                        auto src_idx = (aLayout == blas::Layout::RowMajor) ? j * aRows + i : i * aRows + j;
                        auto dest_idx = (aLayout == blas::Layout::RowMajor) ? i * aRows + j : j * aRows + i;
                        apMatrixA[dest_idx] = apMatrixA[src_idx];
                    }
                }
            }
        }

        template<typename T>
        void HCoreKernels<T>::transpose(blas::Layout aLayout, size_t aRows, size_t aCols, const T *aA,
                                        size_t aLeadingDimA, T *aOut, size_t aLeadingDimOut,
                                        const kernels::RunContext &aContext) {
            size_t i, j, x, y;

            if (aA == nullptr || aOut == nullptr) {
                return;
            }

            if (aLayout == blas::Layout::ColMajor) {
                x = aCols;
                y = aRows;
            } else if (aLayout == blas::Layout::RowMajor) {
                x = aRows;
                y = aCols;
            } else {
                /* Unknown input layout */
                return;
            }

            /* In case of incorrect m, n, ldin or ldout the function does nothing */
            for (i = 0; i < std::min(y, aLeadingDimA); i++) {
                for (j = 0; j < std::min(x, aLeadingDimOut); j++) {
                    aOut[(size_t) i * aLeadingDimOut + j] = aA[(size_t) j * aLeadingDimA + i];
                }
            }

        }

        template<typename T>
        size_t HCoreKernels<T>::CalculatePotrfWorkspaceSize(T *apMatrix, blas::Uplo aUplo, size_t aMatrixOrder,
                                                            size_t aLeadingDim, size_t &aHostSize,
                                                            const RunContext &aContext) {
            aHostSize = 0;
            return 0;
        }

        HCOREPP_INSTANTIATE_CLASS(HCoreKernels)

    }
}
