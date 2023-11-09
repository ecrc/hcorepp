/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <cuComplex.h>
#include <cusolverDn.h>
#include <hcorepp/kernels/kernels.hpp>
#include <cstring>
#include <cublas_v2.h>
#include <algorithm>
#include "hcorepp/kernels/cuda/CudaKernels.hpp"
#include "hcorepp/common/TypeCheck.hpp"

namespace hcorepp {
    namespace kernels {

        template<typename T>
        void HCoreKernels<T>::Gemm(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, size_t aM, size_t aN,
                                   size_t aK, T &aAlpha, T const *apA, size_t aLdA, T const *apB, size_t aLdB,
                                   T &aBeta, T *apC, size_t aLdC, const RunContext &aContext) {
            blas::gemm(aLayout, aTransA, aTransB, aM, aN, aK, aAlpha, apA, aLdA, apB, aLdB, aBeta, apC, aLdC,
                       aContext.GetBLASQueue());
        }

        template<typename T>
        void
        HCoreKernels<T>::MultiplyByAlpha(T *apArray, size_t aRows, size_t aCols, size_t aM, size_t aRank,
                                         T &aAlpha, const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::MultiplyByAlpha(apArray, aRows, aCols, aM, aRank, aAlpha,
                                                                       aContext);
        }

        template<typename T>
        void
        HCoreKernels<T>::ProcessVpointer(size_t aN, size_t aCRank, bool aGetUngqr, size_t Vm, T &aBeta, T *apCV,
                                         size_t aLdcV, T *V, size_t aArank, const T *apBdata,
                                         const RunContext &aContext, bool aCholesky) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::ProcessVpointer(aN, aCRank, aGetUngqr, Vm, aBeta, apCV,
                                                                       aLdcV, V, aArank, apBdata, aContext, aCholesky);
        }

        template<typename T>
        void HCoreKernels<T>::CalculateNewRank(size_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma,
                                               size_t sizeS, blas::real_type<T> accuracy, const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::CalculateNewRank(aNewRank, aTruncatedSvd, apSigma, sizeS,
                                                                        accuracy, aContext);
        }

        template<typename T>
        void
        HCoreKernels<T>::CalculateUVptr(size_t aRank, size_t aVm, T *UVptr, const T *Vnew,
                                        const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::CalculateUVptr(aRank, aVm, UVptr, Vnew, aContext);
        }

        template<typename T>
        void
        HCoreKernels<T>::CalculateVTnew(size_t aRkNew, bool aUngqr, size_t aMinVmVn, blas::real_type<T> *apSigma,
                                        T *apVTnew, size_t aSizeS, size_t aVm, const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::CalculateVTnew(aRkNew, aUngqr, aMinVmVn, apSigma,
                                                                      apVTnew, aSizeS, aVm, aContext);
        }

        template<typename T>
        void
        HCoreKernels<T>::CalculateUVptrConj(size_t aRank, size_t aVm, T *UVptr, const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::CalculateUVptrConj(aRank, aVm, UVptr, aContext);
        }

        template<typename T>
        void
        HCoreKernels<T>::FillIdentityMatrix(size_t aNumOfElements, T *apMatrix, const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::GenerateIdentityMatrix(aNumOfElements, apMatrix, aContext);
        }

        template<typename T>
        void
        HCoreKernels<T>::LaCpy(common::MatrixType aType, size_t aM, size_t aRank, T *apCU, size_t aLD, T *apU,
                               size_t aUm, const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::LaCpy(aType, aM, aRank, apCU, aLD, apU, aUm, aContext);
        }

        template<typename T>
        void HCoreKernels<T>::Geqrf(size_t aM, size_t aN, T *apA, size_t aLdA, T *apTau, T *aWorkspace,
                                    size_t aWorkspace_size,
                                    size_t aHostSize, const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::Geqrf(aM, aN, apA, aLdA, apTau, aWorkspace, aWorkspace_size,
                                                             aHostSize, aContext);
        }

        template<typename T>
        void HCoreKernels<T>::Laset(common::MatrixType aMatrixType, size_t aM, size_t aN, T aOffdiag, T aDiag,
                                    T *apA, size_t aLdA, const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::Laset(aMatrixType, aM, aN, aOffdiag, aDiag, apA, aLdA,
                                                             aContext);
        }

        template<typename T>
        void HCoreKernels<T>::Trmm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans,
                                   blas::Diag aDiag, size_t aM, size_t aN, T aAlpha, T const *apA, size_t aLdA,
                                   T *apB, size_t aLdB, const RunContext &aContext) {
            blas::trmm(aLayout, aSide, aUplo, aTrans, aDiag, aM, aN, aAlpha, apA, aLdA, apB, aLdB,
                       aContext.GetBLASQueue());
        }

        template<typename T>
        void
        HCoreKernels<T>::SVD(common::Job aJobu, common::Job aJobvt, size_t aM, size_t aN, T *apA, size_t aLdA,
                             T *apS, T *apU, size_t aLdU, T *apVT, size_t aLdVt,
                             common::CompressionType aSVDOperation, T *aWorkspace, size_t aWorkspace_size,
                             size_t aHostSize, const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::SVD(aJobu, aJobvt, aM, aN, apA, aLdA, apS, apU, aLdU,
                                                           apVT, aLdVt, aSVDOperation, aWorkspace, aWorkspace_size,
                                                           aHostSize, aContext);
        }

        template<typename T>
        void
        HCoreKernels<T>::Unmqr(common::SideMode aSide, common::BlasOperation aTrans, size_t aM, size_t aN, size_t aK,
                               T const *apA,
                               size_t aLdA, T const *apTau, T *apC, size_t aLdC, T *aWorkspace,
                               size_t aWorkspace_size, const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::Unmqr(aSide, aTrans, aM, aN, aK, apA, aLdA, apTau, apC, aLdC,
                                                             aWorkspace, aWorkspace_size,
                                                             aContext);
        }

        template<typename T>
        void
        HCoreKernels<T>::ungqr(size_t aM, size_t aN, size_t aK, T *apA, size_t aLdA, T *apTau,
                               T *aWorkspace, size_t aWorkspace_size, const RunContext &aContext) {
            hcorepp::cudakernels::HCoreCudaKernels<T>::ungqr(aM, aN, aK, apA, aLdA, apTau, aWorkspace, aWorkspace_size,
                                                             aContext);
        }

        template<typename T>
        size_t
        HCoreKernels<T>::CalculateGemmWorkspaceSize(size_t aUm, size_t aUn, size_t aVm, size_t aVn, size_t aSizeS,
                                                    const operators::CompressionParameters &aHelpers,
                                                    size_t &aHostSize,
                                                    const RunContext &aContext) {
            std::vector<size_t> scratchpad_sizes(5, 0);
            std::vector<size_t> host_sizes(3, 0);
            size_t min_Um_Un = std::min(aUm, aUn);
            size_t min_Vm_Vn = std::min(aVm, aVn);

            cusolverDnXgeqrf_bufferSize(aContext.GetCusolverDnHandle(), NULL, aUm, aUn,
                                        cudakernels::traits<T>::cuda_data_type,
                                        nullptr, aUm, cudakernels::traits<T>::cuda_data_type, nullptr,
                                        cudakernels::traits<T>::cuda_data_type, &scratchpad_sizes[0], &host_sizes[0]);

            cusolverDnXgeqrf_bufferSize(aContext.GetCusolverDnHandle(), NULL, aVm, aVn,
                                        cudakernels::traits<T>::cuda_data_type,
                                        nullptr, aVm, cudakernels::traits<T>::cuda_data_type, nullptr,
                                        cudakernels::traits<T>::cuda_data_type, &scratchpad_sizes[1], &host_sizes[1]);


            if (aHelpers.GetTrmm()) {
                if (aHelpers.GetUngqr()) {
                    cusolverDnXgesvd_bufferSize(aContext.GetCusolverDnHandle(), NULL,
                                                (signed char) common::Job::SomeVec,
                                                (signed char) common::Job::SomeVec, min_Um_Un, aUn,
                                                cudakernels::traits<T>::cuda_data_type, nullptr, min_Um_Un,
                                                cudakernels::traits<T>::cuda_data_type, nullptr,
                                                cudakernels::traits<T>::cuda_data_type, nullptr, min_Um_Un,
                                                cudakernels::traits<T>::cuda_data_type,
                                                nullptr, aSizeS,
                                                cudakernels::traits<T>::cuda_data_type, &scratchpad_sizes[2],
                                                &host_sizes[2]);
                } else {
                    cusolverDnXgesvd_bufferSize(aContext.GetCusolverDnHandle(), NULL,
                                                (signed char) common::Job::SomeVec,
                                                (signed char) common::Job::SomeVec, min_Um_Un, aUn,
                                                cudakernels::traits<T>::cuda_data_type, nullptr, min_Um_Un,
                                                cudakernels::traits<T>::cuda_data_type, nullptr,
                                                cudakernels::traits<T>::cuda_data_type, nullptr, aUm,
                                                cudakernels::traits<T>::cuda_data_type,
                                                nullptr, aSizeS,
                                                cudakernels::traits<T>::cuda_data_type, &scratchpad_sizes[2],
                                                &host_sizes[2]);
                }
            } else {
                if (aHelpers.GetUngqr()) {
                    cusolverDnXgesvd_bufferSize(aContext.GetCusolverDnHandle(), NULL,
                                                (signed char) common::Job::SomeVec,
                                                (signed char) common::Job::SomeVec, min_Um_Un, min_Vm_Vn,
                                                cudakernels::traits<T>::cuda_data_type, nullptr, min_Um_Un,
                                                cudakernels::traits<T>::cuda_data_type, nullptr,
                                                cudakernels::traits<T>::cuda_data_type, nullptr, min_Um_Un,
                                                cudakernels::traits<T>::cuda_data_type,
                                                nullptr, aSizeS,
                                                cudakernels::traits<T>::cuda_data_type, &scratchpad_sizes[2],
                                                &host_sizes[2]);
                } else {
                    cusolverDnXgesvd_bufferSize(aContext.GetCusolverDnHandle(), NULL,
                                                (signed char) common::Job::SomeVec,
                                                (signed char) common::Job::SomeVec, min_Um_Un, min_Vm_Vn,
                                                cudakernels::traits<T>::cuda_data_type, nullptr, min_Um_Un,
                                                cudakernels::traits<T>::cuda_data_type, nullptr,
                                                cudakernels::traits<T>::cuda_data_type, nullptr, aUm,
                                                cudakernels::traits<T>::cuda_data_type,
                                                nullptr, aSizeS,
                                                cudakernels::traits<T>::cuda_data_type, &scratchpad_sizes[2],
                                                &host_sizes[2]);
                }
            }

            if (aHelpers.GetUngqr()) {
                if constexpr (is_complex<T>()) {
                    if constexpr (is_complex_float<T>()) {
                        cusolverDnCorgqr_bufferSize(aContext.GetCusolverDnHandle(), aUm, min_Um_Un, min_Um_Un,
                                                    (T *) nullptr,
                                                    aUm, (T *) nullptr,
                                                    (int *) (&scratchpad_sizes[3]));
                        cusolverDnCorgqr_bufferSize(aContext.GetCusolverDnHandle(), aVm, min_Vm_Vn, min_Vm_Vn,
                                                    (T *) nullptr,
                                                    aVm, (T *) nullptr,
                                                    (int *) (&scratchpad_sizes[4]));
                    } else {
                        cusolverDnZorgqr_bufferSize(aContext.GetCusolverDnHandle(), aUm, min_Um_Un, min_Um_Un,
                                                    (T *) nullptr, aUm, (T *) nullptr,
                                                    (int *) (&scratchpad_sizes[3]));
                        cusolverDnZorgqr_bufferSize(aContext.GetCusolverDnHandle(), aVm, min_Vm_Vn, min_Vm_Vn,
                                                    (T *) nullptr, aVm, (T *) nullptr,
                                                    (int *) (&scratchpad_sizes[4]));
                    }
                } else {
                    if constexpr (is_double<T>()) {
                        cusolverDnDorgqr_bufferSize(aContext.GetCusolverDnHandle(), aUm, min_Um_Un, min_Um_Un, nullptr,
                                                    aUm, nullptr,
                                                    (int *) (&scratchpad_sizes[3]));
                        cusolverDnDorgqr_bufferSize(aContext.GetCusolverDnHandle(), aVm, min_Vm_Vn, min_Vm_Vn, nullptr,
                                                    aVm, nullptr,
                                                    (int *) (&scratchpad_sizes[4]));
                    } else {
                        cusolverDnSorgqr_bufferSize(aContext.GetCusolverDnHandle(), aUm, min_Um_Un, min_Um_Un, nullptr,
                                                    aUm, nullptr,
                                                    (int *) (&scratchpad_sizes[3]));
                        cusolverDnSorgqr_bufferSize(aContext.GetCusolverDnHandle(), aVm, min_Vm_Vn, min_Vm_Vn, nullptr,
                                                    aVm, nullptr,
                                                    (int *) (&scratchpad_sizes[4]));
                    }
                }
            } else {
                cusolverDnDormqr_bufferSize(aContext.GetCusolverDnHandle(),
                                            (cublasSideMode_t) common::SideMode::SIDE_LEFT,
                                            (cublasOperation_t) common::BlasOperation::OP_NoTRANS, aUm, aSizeS, aUm,
                                            (const double *) nullptr, aUm, (const double *) nullptr,
                                            (const double *) nullptr,
                                            aUm,
                                            (int *) &scratchpad_sizes[3]);
                cusolverDnDormqr_bufferSize(aContext.GetCusolverDnHandle(),
                                            (cublasSideMode_t) common::SideMode::SIDE_RIGHT,
                                            (cublasOperation_t) common::BlasOperation::OP_CONJG, aSizeS, aVm, aSizeS,
                                            (const double *) nullptr, aVm, (const double *) nullptr,
                                            (const double *) nullptr,
                                            aSizeS,
                                            (int *) &scratchpad_sizes[4]);
            }

            aHostSize = *std::max_element(host_sizes.begin(), host_sizes.end());

            return *std::max_element(scratchpad_sizes.begin(), scratchpad_sizes.end());
        }

        template<typename T>
        int
        HCoreKernels<T>::potrf(blas::Uplo aUplo, T *aWorkspace, size_t aWorkspaceSize, size_t aHostSize,
                               size_t aMatrixOrder, T *apMatrix, size_t aLeadingDim, blas::Layout aLayout,
                               const kernels::RunContext &aContext) {

            hcorepp::cudakernels::HCoreCudaKernels<T>::potrf(aUplo, aWorkspace, aWorkspaceSize, aHostSize, aMatrixOrder,
                                                             apMatrix, aLeadingDim, aContext);
            return 0;

        }

        template<typename T>
        void HCoreKernels<T>::FillMatrixTriangle(blas::Uplo aUplo, size_t aRows, size_t aCols, T *apMatrix,
                                                 blas::Layout aLayout, size_t aValue,
                                                 const kernels::RunContext &aContext) {
            if (aRows != aCols) {
                return;
            }
            hcorepp::cudakernels::HCoreCudaKernels<T>::FillMatrixTriangle(aUplo, aRows, apMatrix, aLayout,
                                                                          aValue, aContext);
        }

        template<typename T>
        void HCoreKernels<T>::trsm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans,
                                   blas::Diag aDiag, size_t aRows, size_t aCols, T aAlpha, const T *apMatrixA,
                                   size_t aLeadingDimA, T *apMatrixB, size_t aLeadingDimB,
                                   const kernels::RunContext &aContext) {
            blas::trsm(aLayout, aSide, aUplo, aTrans, aDiag, aRows, aCols, aAlpha, apMatrixA, aLeadingDimA, apMatrixB,
                       aLeadingDimB, aContext.GetBLASQueue());
        }

        template<typename T>
        void
        HCoreKernels<T>::syrk(blas::Layout aLayout, blas::Uplo aUplo, blas::Op aTrans, size_t aRows, size_t aCols,
                              T aAlpha, const T *apMatrixA, size_t aLeadingDimA, T aBeta, T *apMatrixB,
                              size_t aLeadingDimB, const RunContext &aContext) {
            blas::syrk(aLayout, aUplo, aTrans, aRows, aCols, aAlpha, apMatrixA, aLeadingDimA, aBeta, apMatrixB,
                       aLeadingDimB, aContext.GetBLASQueue());
        }

        template<typename T>
        void HCoreKernels<T>::Symmetrize(blas::Layout aLayout, T *apMatrixA, size_t aRows, size_t aCols,
                                         blas::Uplo aUplo, const RunContext &aContext) {

            if (aRows != aCols) {
                return;
            }

            hcorepp::cudakernels::HCoreCudaKernels<T>::SymmetrizeMatrix(aUplo, aRows, apMatrixA, aLayout,
                                                                        aContext);
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

            hcorepp::cudakernels::HCoreCudaKernels<T>::TransposeMatrix(std::min(y, aLeadingDimA),
                                                                       std::min(x, aLeadingDimOut), aA, aLeadingDimA,
                                                                       aOut, aLeadingDimOut, aContext);

        }

        template<typename T>
        size_t HCoreKernels<T>::CalculatePotrfWorkspaceSize(T *apMatrix, blas::Uplo aUplo, size_t aMatrixOrder,
                                                            size_t aLeadingDim, size_t &aHostSize,
                                                            const RunContext &aContext) {
            size_t d_lwork = 0;     /* size of workspace */
            size_t h_lwork = 0;     /* size of workspace */

            cublasFillMode_t upper_lower;
            if (aUplo == blas::Uplo::Upper) {
                upper_lower = cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
            } else {
                upper_lower = cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
            }

            cusolverDnXpotrf_bufferSize(aContext.GetCusolverDnHandle(), NULL, upper_lower, aMatrixOrder,
                                        cudakernels::traits<T>::cuda_data_type, (const void *) apMatrix, aLeadingDim,
                                        cudakernels::traits<T>::cuda_data_type, &d_lwork, &h_lwork);

            aHostSize = h_lwork;
            return d_lwork;
        }

        HCOREPP_INSTANTIATE_CLASS(HCoreKernels)

    }
}
