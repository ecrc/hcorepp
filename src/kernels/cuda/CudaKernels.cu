/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <hcorepp/kernels/cuda/CudaKernels.hpp>
#include <iostream>
#include <cuda/std/type_traits>
#include "blas/util.hh"
#include "blas/scal.hh"
#include <cuComplex.h>
#include <hcorepp/common/TypeCheck.hpp>
#include <hcorepp/kernels/memory.hpp>

#define THREADS 32
#define THREADS_1D 1024
const int max_blocks = 65535;

namespace hcorepp {
    namespace cudakernels {

        template<typename T>
        __global__ void GenerateIdentityMatrix_kernel(int64_t aNumOfCols, T *apMatrix) {
            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;

            if (x >= aNumOfCols) {
                return;
            }

            int64_t arr_index = x * aNumOfCols + x;
            apMatrix[arr_index] = 1;

        }

        template<typename T>
        __global__ void
        MultiplyByAlpha_kernel(T *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank, T aAlpha) {
            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;

            if (x >= aRows * aCols) {
                return;
            }
            apArray[aM * aRank + x] *= aAlpha;
        }

        template<typename T>
        __global__ void
        ProcessVPointer_kernel_with_Ungqr(int64_t aN, int64_t aCRank, int64_t Vm, T aBeta, T *apCV,
                                          int64_t aLdcV, T *V, int64_t aArank, const T *apBdata) {
            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

            int64_t index = y * Vm + x;
            int64_t apCV_index = x * aLdcV + y;

            if (x >= aN || y >= aCRank) {
                return;
            }

            if (::cuda::std::is_same<T, cuFloatComplex>::value) {
//                cuFloatComplex temp = cuCmulf(aBeta, apCV[apCV_index]);
//                V[index] = (float2) cuConjf((cuFloatComplex) (aBeta * apCV[apCV_index]));
            } else if (::cuda::std::is_same<T, cuDoubleComplex>::value) {

            } else {
                V[index] = aBeta * apCV[apCV_index];
            }
        }

        template<typename T>
        __global__ void
        ProcessVPointer_kernel_without_Ungqr(int64_t aN, int64_t aCRank, int64_t Vm, T aBeta, T *apCV,
                                             int64_t aLdcV, T *V, int64_t aArank, const T *apBdata) {
            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

            int64_t index = y * Vm + x;
            int64_t apCV_index = x * aLdcV + y;

            if (x >= aN || y >= aCRank) {
                return;
            }

            V[index] = aBeta * apCV[apCV_index];
        }

        template<typename T>
        __global__ void
        ProcessVPointer_kernel_with_Ungqr_part2(int64_t aN, int64_t aCRank, int64_t Vm, T aBeta, T *apCV,
                                                int64_t aLdcV, T *V, int64_t aArank, const T *apBdata) {
            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

            int64_t index = y * Vm + x;
            int64_t apB_index = x * aArank + y;

            if (x >= aN || y >= aArank) {
                return;
            }
            T *Vptr = &V[aN * aCRank];
            if (::cuda::std::is_same<T, cuFloatComplex>::value) {

            } else if (::cuda::std::is_same<T, cuDoubleComplex>::value) {

            } else {

                Vptr[index] = apBdata[apB_index];
            }
        }

        template<typename T>
        __global__ void
        ProcessVPointer_kernel_without_Ungqr_part2(int64_t aN, int64_t aCRank, int64_t Vm, T aBeta, T *apCV,
                                                   int64_t aLdcV, T *V, int64_t aArank, const T *apBdata) {
            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

            int64_t index = y * Vm + x;
            int64_t apB_index = x * aArank + y;

            if (x >= aN || y >= aArank) {
                return;
            }

            T *Vptr = &V[aN * aCRank];
            Vptr[index] = apBdata[apB_index];
        }

        template<typename T>
        __global__ void
        CalculateUVptrConj_kernel_(int64_t aRank, int64_t aVm, T *UVptr) {

            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

            int64_t index = y * aRank + x;

            if (x >= aRank || y >= aVm) {
                return;
            }

            if (::cuda::std::is_same<T, cuFloatComplex>::value) {

            } else if (::cuda::std::is_same<T, cuDoubleComplex>::value) {

            } else {
                UVptr[index] = UVptr[index];
            }
        }

        template<typename T>
        __global__ void
        CalculateVTnew_kernel_with_Ungqr(int64_t aRkNew, bool aUngqr, int64_t aMinVmVn, blas::real_type<T> *apSigma,
                                         T *apVTnew, int64_t aSizeS, int64_t aVm) {
            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= aRkNew || y >= aMinVmVn) {
                return;
            }

            int64_t index = y * aSizeS;

            T alpha = apSigma[x];
            T *vt = &apVTnew[x];
            vt[index] *= alpha;
        }

        template<typename T>
        __global__ void
        CalculateVTnew_kernel_without_Ungqr(int64_t aRkNew, bool aUngqr, int64_t aMinVmVn, blas::real_type<T> *apSigma,
                                            T *apVTnew, int64_t aSizeS, int64_t aVm) {
            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= aRkNew || y >= aVm) {
                return;
            }

            int64_t index = y * aSizeS;

            T alpha = apSigma[x];
            T *vt = &apVTnew[x];
            vt[index] *= alpha;

            if (::cuda::std::is_same<T, cuFloatComplex>::value) {

            } else if (::cuda::std::is_same<T, cuDoubleComplex>::value) {

            } else {
                apVTnew[index] = apVTnew[index];
            }
        }

        template<typename T>
        __global__ void
        CalculateUVptr_kernel(int64_t aRank, int64_t aVm, T *UVptr, const T *Vnew) {
            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
            int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= aRank || y >= aVm) {
                return;
            }

            int64_t uv_index = y * aRank + x;
            int64_t vnew_index = x * aVm + y;

            if (::cuda::std::is_same<T, cuFloatComplex>::value) {

            } else if (::cuda::std::is_same<T, cuDoubleComplex>::value) {

            } else {
                UVptr[uv_index] = Vnew[vnew_index];
            }

        }

        template<typename T>
        __global__ void
        CalculateNewRank_kernel_withSVD(int64_t aNewRank, blas::real_type<T> *apSigma, int64_t sizeS,
                                        blas::real_type<T> accuracy) {
            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;

            if (x >= sizeS) {
                return;
            }
            blas::real_type<T> Sigma_0 = apSigma[0];

            if (x > 0) {
                if (apSigma[x] < accuracy * Sigma_0) {
                    Sigma_0 = apSigma[x];
                    aNewRank = x;
                    /// expected to have break statement here.
                }
            }
        }

        template<typename T>
        __global__ void
        CalculateNewRank_kernel_withoutSVD(int64_t aNewRank, blas::real_type<T> *apSigma, int64_t sizeS,
                                           blas::real_type<T> accuracy) {
            int64_t x = blockIdx.x * blockDim.x + threadIdx.x;

            if (x >= sizeS) {
                return;
            }

            if (x > 0) {
                if (apSigma[x] < accuracy) {
                    aNewRank = x;
                    /// expected to have break statement here.
                }
            }
        }

        template<typename T>
        static __device__
        void zlaset_lower_device(int m, int n, T offdiag, T diag, T *A, int lda) {
            int ind = blockIdx.x * THREADS + threadIdx.x;
            int iby = blockIdx.y * THREADS;
            /* check if full block-column && (below diag) */
            bool full = (iby + THREADS <= n && (ind >= iby + THREADS));
            /* do only rows inside matrix, and blocks not above diag */
            if (ind < m && ind + THREADS > iby) {
                A += ind + iby * lda;
                if (full) {
                    // full block-column, off-diagonal block
#pragma unroll
                    for (int j = 0; j < THREADS; ++j) {
                        A[j * lda] = offdiag;
                    }
                } else {
                    // either partial block-column or diagonal block
                    for (int j = 0; j < THREADS && iby + j < n; ++j) {
                        if (iby + j == ind)
                            A[j * lda] = diag;
                        else if (ind > iby + j)
                            A[j * lda] = offdiag;
                    }
                }
            }
        }

        template<typename T>
        __global__ void zlaset_lower_kernel(int m, int n, T offdiag, T diag, T *dA, int ldda) {
            zlaset_lower_device(m, n, offdiag, diag, dA, ldda);
        }

        template<typename T>
        static __device__
        void zlaset_upper_device(int m, int n, T offdiag, T diag, T *A, int lda) {
            int ind = blockIdx.x * THREADS + threadIdx.x;
            int iby = blockIdx.y * THREADS;
            /* check if full block-column && (above diag) */
            bool full = (iby + THREADS <= n && (ind + THREADS <= iby));
            /* do only rows inside matrix, and blocks not below diag */
            if (ind < m && ind < iby + THREADS) {
                A += ind + iby * lda;
                if (full) {
                    // full block-column, off-diagonal block
#pragma unroll
                    for (int j = 0; j < THREADS; ++j) {
                        A[j * lda] = offdiag;
                    }
                } else {
                    // either partial block-column or diagonal block
                    for (int j = 0; j < THREADS && iby + j < n; ++j) {
                        if (iby + j == ind)
                            A[j * lda] = diag;
                        else if (ind < iby + j)
                            A[j * lda] = offdiag;
                    }
                }
            }
        }

        template<typename T>
        __global__ void zlaset_upper_kernel(int m, int n, T offdiag, T diag, T *dA, int ldda) {
            zlaset_upper_device(m, n, offdiag, diag, dA, ldda);
        }

        template<typename T>
        static __device__
        void zlaset_full_device(int m, int n, T offdiag, T diag, T *A, int lda) {
            int ind = blockIdx.x * THREADS + threadIdx.x;
            int iby = blockIdx.y * THREADS;
            /* check if full block-column && (below diag || above diag || offdiag == diag) */
            bool full = (iby + THREADS <= n &&
                         (ind >= iby + THREADS || ind + THREADS <= iby || (offdiag == diag)));
//                         MAGMA_Z_EQUAL(offdiag, diag)));
            /* do only rows inside matrix */
            if (ind < m) {
                A += ind + iby * lda;
                if (full) {
                    // full block-column, off-diagonal block or offdiag == diag
#pragma unroll
                    for (int j = 0; j < THREADS; ++j) {
                        A[j * lda] = offdiag;
                    }
                } else {
                    // either partial block-column or diagonal block
                    for (int j = 0; j < THREADS && iby + j < n; ++j) {
                        if (iby + j == ind)
                            A[j * lda] = diag;
                        else
                            A[j * lda] = offdiag;
                    }
                }
            }
        }

        template<typename T>
        __global__ void zlaset_full_kernel(int m, int n, T offdiag, T diag, T *dA, int ldda) {
            zlaset_full_device(m, n, offdiag, diag, dA, ldda);
        }

        template<typename T>
        static __device__
        void zlacpy_lower_device(int m, int n, const T *dA, int ldda, T *dB, int lddb) {
            int BLK_X = blockDim.x; // THREADS
            int BLK_Y = blockDim.y;
            int ind = blockIdx.x * BLK_X + threadIdx.x;
            int iby = blockIdx.y * BLK_Y;
            /* check if full block-column && (below diag) */
            bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
            /* do only rows inside matrix, and blocks not above diag */
            if (ind < m && ind + BLK_X > iby) {
                dA += ind + iby * ldda;
                dB += ind + iby * lddb;
                if (full) {
                    // full block-column, off-diagonal block
#pragma unroll
                    for (int j = 0; j < BLK_Y; ++j) {
                        dB[j * lddb] = dA[j * ldda];
                    }
                } else {
                    // either partial block-column or diagonal block
                    for (int j = 0; j < BLK_Y && iby + j < n && ind >= iby + j; ++j) {
                        dB[j * lddb] = dA[j * ldda];
                    }
                }
            }
        }

        template<typename T>
        __global__ void zlacpy_lower_kernel(int m, int n, const T *dA, int ldda, T *dB, int lddb) {
            zlacpy_lower_device(m, n, dA, ldda, dB, lddb);
        }

        template<typename T>
        __global__ void zlacpy_full_kernel(int m, int n, const T *dA, int ldda, T *dB, int lddb) {
            int BLK_X = blockDim.x;
            int BLK_Y = blockDim.y;
            int ind = blockIdx.x * BLK_X + threadIdx.x;
            int iby = blockIdx.y * BLK_Y + threadIdx.y;
            /* do only rows inside matrix */
            if (ind < m) {
                if (iby < n) {
                    dB[ind + iby * lddb] = dA[ind + iby * ldda];
                }
            }
        }

        template<typename T>
        __global__ void zlacpy_upper_kernel(int m, int n, const T *dA, int ldda, T *dB, int lddb) {
            int BLK_X = blockDim.x;
            int BLK_Y = blockDim.y;
            int ind = blockIdx.x * BLK_X + threadIdx.x;
            int iby = blockIdx.y * BLK_Y + threadIdx.y;
            /* check if full block-column && (above diag) */
            /* do only rows inside matrix, and blocks not below diag */
            if (ind < m && ind <= iby && iby < n) {
                dB[ind + iby * lddb] = dA[ind + iby * ldda];
            }
        }

        template<typename T>
        void HCoreCudaKernels<T>::GenerateIdentityMatrix(int64_t aNumOfCols, T *apMatrix,
                                                         kernels::RunContext &aContext) {
            dim3 dimBlock(THREADS_1D, 1);
            dim3 dimGrid((aNumOfCols + dimBlock.x - 1) / dimBlock.x);

            GenerateIdentityMatrix_kernel<<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aNumOfCols, apMatrix);
        }

        template<typename T>
        void HCoreCudaKernels<T>::MultiplyByAlpha(T *apArray, int64_t aRows, int64_t aCols, int64_t aM, int64_t aRank,
                                                  T &aAlpha, kernels::RunContext &aContext) {
            dim3 dimBlock(THREADS_1D, 1);
            dim3 dimGrid(((aRows * aCols) + dimBlock.x - 1) / dimBlock.x);

            MultiplyByAlpha_kernel<<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(apArray, aRows, aCols, aM, aRank,
                                                                                   aAlpha);
        }

        template<typename T>
        void HCoreCudaKernels<T>::Geqrf(int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apTau,
                                        kernels::RunContext &aContext) {
            /// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xgeqrf/cusolver_Xgeqrf_example.cu
            size_t d_lwork = 0;     /* size of workspace */
            size_t h_lwork = 0;     /* size of workspace */
            void *h_work = nullptr; /* host workspace */

            /* step 3: query working space of geqrf */
            cusolverDnXgeqrf_bufferSize(aContext.GetCusolverDnHandle(), NULL, aM, aN, traits<T>::cuda_data_type, apA,
                                        aLdA,
                                        traits<T>::cuda_data_type, apTau, traits<T>::cuda_data_type, &d_lwork,
                                        &h_lwork);
            auto d_work = aContext.RequestWorkBuffer(sizeof(T) * d_lwork);

            /* step 4: QR factorization */
            cusolverDnXgeqrf(aContext.GetCusolverDnHandle(), NULL, aM, aN, traits<T>::cuda_data_type, apA, aLdA,
                             traits<T>::cuda_data_type,
                             apTau, traits<T>::cuda_data_type, d_work, d_lwork, h_work, h_lwork,
                             aContext.GetInfoPointer());
        }

        template<typename T>
        void
        HCoreCudaKernels<T>::ProcessVpointer(int64_t aN, int64_t aCRank, bool aGetUngqr, int64_t Vm, T &aBeta, T *apCV,
                                             int64_t aLdcV, T *V,
                                             int64_t aArank, const T *apBdata, kernels::RunContext &aContext) {

            dim3 dimBlock(THREADS, THREADS);
            dim3 dimGrid((aN + dimBlock.x - 1) / dimBlock.x, (aCRank + dimBlock.y - 1) / dimBlock.y);

            if (aGetUngqr) {
                ProcessVPointer_kernel_with_Ungqr<T><<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aN,
                                                                                                     aCRank, Vm, aBeta,
                                                                                                     apCV, aLdcV,
                                                                                                     V, aArank,
                                                                                                     apBdata);
            } else {
                ProcessVPointer_kernel_without_Ungqr<T><<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aN,
                                                                                                        aCRank, Vm,
                                                                                                        aBeta, apCV,
                                                                                                        aLdcV, V,
                                                                                                        aArank,
                                                                                                        apBdata);
            }

            dim3 dimBlock_(THREADS, THREADS);
            dim3 dimGrid_((aN + dimBlock.x - 1) / dimBlock.x, (aArank + dimBlock.y - 1) / dimBlock.y);

            if (aGetUngqr) {
                ProcessVPointer_kernel_with_Ungqr_part2<T><<<dimGrid_, dimBlock_, 0, aContext.GetStream()>>>(aN, aCRank,
                                                                                                             Vm, aBeta,
                                                                                                             apCV,
                                                                                                             aLdcV,
                                                                                                             V, aArank,
                                                                                                             apBdata);
            } else {
                ProcessVPointer_kernel_without_Ungqr_part2<T><<<dimGrid_, dimBlock_, 0, aContext.GetStream()>>>(aN,
                                                                                                                aCRank,
                                                                                                                Vm,
                                                                                                                aBeta,
                                                                                                                apCV,
                                                                                                                aLdcV,
                                                                                                                V,
                                                                                                                aArank,
                                                                                                                apBdata);
            }

        }

        template<typename T>
        void
        HCoreCudaKernels<T>::CalculateUVptrConj(int64_t aRank, int64_t aVm, T *UVptr, kernels::RunContext &aContext) {
            dim3 dimBlock(THREADS, THREADS);
            dim3 dimGrid((aRank + dimBlock.x - 1) / dimBlock.x, (aVm + dimBlock.y - 1) / dimBlock.y);
            CalculateUVptrConj_kernel_<<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aRank, aVm, UVptr);
        }

        template<typename T>
        void
        HCoreCudaKernels<T>::CalculateVTnew(int64_t aRkNew, bool aUngqr, int64_t aMinVmVn, blas::real_type<T> *apSigma,
                                            T *apVTnew,
                                            int64_t aSizeS, int64_t aVm, kernels::RunContext &aContext) {
            dim3 dimBlock(THREADS, THREADS);

            if (aUngqr) {
                dim3 dimGrid((aRkNew + dimBlock.x - 1) / dimBlock.x, (aMinVmVn + dimBlock.y - 1) / dimBlock.y);
                CalculateVTnew_kernel_with_Ungqr<<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aRkNew, aUngqr,
                                                                                                 aMinVmVn, apSigma,
                                                                                                 apVTnew,
                                                                                                 aSizeS, aVm);
            } else {
                dim3 dimGrid((aRkNew + dimBlock.x - 1) / dimBlock.x, (aVm + dimBlock.y - 1) / dimBlock.y);
                CalculateVTnew_kernel_without_Ungqr<<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aRkNew, aUngqr,
                                                                                                    aMinVmVn, apSigma,
                                                                                                    apVTnew,
                                                                                                    aSizeS, aVm);
            }
        }

        template<typename T>
        void HCoreCudaKernels<T>::CalculateUVptr(int64_t aRank, int64_t aVm, T *UVptr, const T *Vnew,
                                                 kernels::RunContext &aContext) {
            dim3 dimBlock(THREADS, THREADS);

            dim3 dimGrid((aRank + dimBlock.x - 1) / dimBlock.x, (aVm + dimBlock.y - 1) / dimBlock.y);

            CalculateUVptr_kernel<<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aRank, aVm, UVptr, Vnew);
        }

        template<typename T>
        void HCoreCudaKernels<T>::CalculateNewRank(int64_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma,
                                                   int64_t sizeS, blas::real_type<T> accuracy,
                                                   kernels::RunContext &aContext) {
            auto host_sigma = new blas::real_type<T>[sizeS];
            hcorepp::memory::Memcpy<blas::real_type<T>>(host_sigma, apSigma, sizeS, aContext,
                                                        memory::MemoryTransfer::DEVICE_TO_HOST);
            aContext.Sync();
            //TODO do a proper reduction kernel and memcpy only rank
//            aNewRank = sizeS;
//
//            dim3 dimBlock(THREADS);
//
//            dim3 dimGrid((sizeS + dimBlock.x - 1) / dimBlock.x);
//
//            if (aTruncatedSvd) {
//                CalculateNewRank_kernel_withSVD<T><<<dimGrid, dimBlock>>>(aNewRank, apSigma, sizeS, accuracy);
//            } else {
//                CalculateNewRank_kernel_withoutSVD<T><<<dimGrid, dimBlock>>>(aNewRank, apSigma, sizeS, accuracy);
//            }
            aNewRank = sizeS;
            if (aTruncatedSvd) {
                blas::real_type<T> Sigma_0 = host_sigma[0];
                for (int64_t i = 1; i < sizeS; i++) {
                    if (host_sigma[i] < accuracy * Sigma_0) {
                        Sigma_0 = host_sigma[i];
                        aNewRank = i;
                        break;
                    }
                }
            } else {
                for (int64_t i = 1; i < sizeS; i++) {
                    if (host_sigma[i] < accuracy) {
                        aNewRank = i;
                        break;
                    }
                }
            }
            delete[] host_sigma;

        }

        template<typename T>
        void
        HCoreCudaKernels<T>::SVD(common::Job aJobu, common::Job aJobvt, int64_t aM, int64_t aN, T *apA, int64_t aLdA,
                                 T *apS, T *apU,
                                 int64_t aLdU, T *apVT, int64_t aLdVt, common::CompressionType aSVDOperationType,
                                 kernels::RunContext &aContext) {
            ///https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xgesvd/cusolver_Xgesvd_example.cu
            size_t d_lwork = 0;     /* size of workspace */
            size_t h_lwork = 0;     /* size of workspace */
            void *h_work = nullptr; /* host workspace */
            /* step 3: query working space of geqrf */
            cusolverDnXgesvd_bufferSize(aContext.GetCusolverDnHandle(), NULL, (signed char) aJobu, (signed char) aJobvt,
                                        aM, aN,
                                        traits<T>::cuda_data_type, apA, aLdA, traits<T>::cuda_data_type, apS,
                                        traits<T>::cuda_data_type, apU, aLdU, traits<T>::cuda_data_type, apVT, aLdVt,
                                        traits<T>::cuda_data_type, &d_lwork, &h_lwork);
            T *d_work = (T *) aContext.RequestWorkBuffer(sizeof(T) * d_lwork);

            /* step 4: compute SVD */
            cusolverDnXgesvd(aContext.GetCusolverDnHandle(), NULL, (signed char) aJobu, (signed char) aJobvt, aM, aN,
                             traits<T>::cuda_data_type, apA, aLdA, traits<T>::cuda_data_type, apS,
                             traits<T>::cuda_data_type, apU, aLdU, traits<T>::cuda_data_type, apVT, aLdVt,
                             traits<T>::cuda_data_type, d_work, d_lwork,
                             h_work, h_lwork, aContext.GetInfoPointer());
        }

        template<typename T>
        void
        HCoreCudaKernels<T>::Unmqr(common::SideMode aSide, common::BlasOperation aTrans, int64_t aM, int64_t aN,
                                   int64_t aK,
                                   T const *apA, int64_t aLdA, T const *apTau, T *apC, int64_t aLdC,
                                   kernels::RunContext &aContext) {
            ///https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/ormqr/cusolver_ormqr_example.cu
            size_t d_lwork = 0;     /* size of workspace */
            /* step 3: query working space of geqrf */

            cusolverDnDormqr_bufferSize(aContext.GetCusolverDnHandle(), (cublasSideMode_t) aSide,
                                        (cublasOperation_t) aTrans, aM, aN, aM,
                                        (const double *) apA, aLdA, (const double *) apTau, (const double *) apC, aLdC,
                                        (int *) &d_lwork);
            T *d_work = (T *) aContext.RequestWorkBuffer(sizeof(T) * d_lwork);
            cusolverDnDormqr(aContext.GetCusolverDnHandle(), (cublasSideMode_t) aSide, (cublasOperation_t) aTrans, aM,
                             aN, aM,
                             (const double *) apA, aLdA, (const double *) apTau, (double *) apC, aLdC,
                             (double *) d_work, d_lwork, aContext.GetInfoPointer());
        }

        template<typename T>
        void HCoreCudaKernels<T>::Laset(common::MatrixType aMatrixType, int64_t aM, int64_t aN, T aOffdiag, T aDiag,
                                        T *apA, int64_t aLdA, kernels::RunContext &aContext) {

#define dA(i_, j_) (dA + (i_) + (j_)*ldda)


            int info = 0;
            if (aMatrixType != common::MatrixType::Lower && aMatrixType != common::MatrixType::Upper &&
                aMatrixType != common::MatrixType::General)
                info = -1;
            else if (aM < 0)
                info = -2;
            else if (aN < 0)
                info = -3;
            else if (aLdA < std::max((int64_t) 1, aM))
                info = -7;

            if (info != 0) {
//                magma_xerbla(__func__, -(info));
                return;  //info;
            }

            if (aM == 0 || aN == 0) {
                return;
            }

            const int super_NB = max_blocks * THREADS;

            dim3 super_grid(ceil(aM / super_NB), ceil(aN / super_NB));

            dim3 threads(THREADS, 1);
            dim3 grid;

            int64_t mm, nn;
            if (aMatrixType == common::MatrixType::Lower) {
                for (unsigned int i = 0; i < super_grid.x; ++i) {
                    mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                    grid.x = ceil(mm / THREADS);
                    for (unsigned int j = 0; j < super_grid.y && j <= i; ++j) {  // from left to diagonal
                        nn = (j == super_grid.y - 1 ? aN % super_NB : super_NB);
                        grid.y = ceil(nn / THREADS);
                        if (i == j) {  // diagonal super block
                            zlaset_lower_kernel<<< grid, threads, 0, aContext.GetStream() >>>(mm, nn, aOffdiag, aDiag,
                                                                                              &apA[i * super_NB, j *
                                                                                                                 super_NB],
                                                                                              aLdA);
                        } else {           // off diagonal super block
                            zlaset_full_kernel<<< grid, threads, 0, aContext.GetStream()>>>
                                    (mm, nn, aOffdiag, aOffdiag, &apA[i * super_NB, j * super_NB], aLdA);
                        }
                    }
                }
            } else if (aMatrixType == common::MatrixType::Upper) {
                for (unsigned int i = 0; i < super_grid.x; ++i) {
                    mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                    grid.x = ceil(mm / THREADS);
                    for (unsigned int j = i; j < super_grid.y; ++j) {  // from diagonal to right
                        nn = (j == super_grid.y - 1 ? aN % super_NB : super_NB);
                        grid.y = ceil(nn / THREADS);
                        if (i == j) {  // diagonal super block
                            zlaset_upper_kernel<<< grid, threads, 0, aContext.GetStream() >>>(mm, nn, aOffdiag, aDiag,
                                                                                              &apA[i * super_NB, j *
                                                                                                                 super_NB],
                                                                                              aLdA);
                        } else {           // off diagonal super block
                            zlaset_full_kernel<<< grid, threads, 0, aContext.GetStream()>>>(mm, nn, aOffdiag, aOffdiag,
                                                                                            &apA[i * super_NB, j *
                                                                                                               super_NB],
                                                                                            aLdA);
                        }
                    }
                }
            } else {
                for (unsigned int i = 0; i < super_grid.x; ++i) {
                    mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                    grid.x = ceil(mm / THREADS);
                    for (unsigned int j = 0; j < super_grid.y; ++j) {  // full row
                        nn = (j == super_grid.y - 1 ? aN % super_NB : super_NB);
                        grid.y = ceil(nn / THREADS);
                        if (i == j) {  // diagonal super block
                            zlaset_full_kernel<<< grid, threads, 0, aContext.GetStream()>>>(mm, nn, aOffdiag, aDiag,
                                                                                            &apA[i * super_NB, j *
                                                                                                               super_NB],
                                                                                            aLdA);
                        } else {           // off diagonal super block
                            zlaset_full_kernel<<< grid, threads, 0, aContext.GetStream() >>>(mm, nn, aOffdiag, aOffdiag,
                                                                                             &apA[i * super_NB, j *
                                                                                                                super_NB],
                                                                                             aLdA);
                        }
                    }
                }
            }
        }

        template<typename T>
        void
        HCoreCudaKernels<T>::LaCpy(common::MatrixType aType, int64_t aM, int64_t aN, T *apA, int64_t aLdA, T *apB,
                                   int64_t aLdB, kernels::RunContext &aContext) {
#define dA(i_, j_) (dA + (i_) + (j_)*ldda)
#define dB(i_, j_) (dB + (i_) + (j_)*lddb)

            int info = 0;
            if (aType != common::MatrixType::Lower && aType != common::MatrixType::Upper &&
                aType != common::MatrixType::General)
                info = -1;
            else if (aM < 0)
                info = -2;
            else if (aN < 0)
                info = -3;
            else if (aLdA < std::max((int64_t) 1, aM))
                info = -5;
            else if (aLdB < std::max((int64_t) 1, aM))
                info = -7;

            if (info != 0) {
                return;
            }

            if (aM == 0 || aN == 0) {
                return;
            }

            const int64_t super_NB = max_blocks * THREADS;
            double divider = super_NB;
            dim3 super_grid(ceil(aM / divider), ceil(aN / divider));
            dim3 threads(THREADS, THREADS);
            dim3 grid;

            int64_t mm, nn;
            if (aType == common::MatrixType::Lower) {
                for (unsigned int i = 0; i < super_grid.x; ++i) {
                    mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                    grid.x = ceil(mm / THREADS);
                    for (unsigned int j = 0; j < super_grid.y && j <= i; ++j) {  // from left to diagonal
                        nn = (j == super_grid.y - 1 ? aN % super_NB : super_NB);
                        grid.y = ceil(nn / THREADS);
                        if (i == j) {  // diagonal super block
                            dim3 threads(THREADS, 1);
                            zlacpy_lower_kernel<<< grid, threads, 0, aContext.GetStream()>>>
                                    (mm, nn, &apA[i * super_NB, j * super_NB], aLdA, &apB[i * super_NB, j * super_NB],
                                     aLdB);
                        } else {           // off diagonal super block
                            zlacpy_full_kernel <<< grid, threads, 0, aContext.GetStream() >>>
                                    (mm, nn, &apA[i * super_NB, j * super_NB], aLdA, &apB[i * super_NB, j * super_NB],
                                     aLdB);
                        }
                    }
                }
            } else if (aType == common::MatrixType::Upper) {
                for (unsigned int i = 0; i < super_grid.x; ++i) {
                    mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                    grid.x = (mm + THREADS - 1) / THREADS;
                    for (unsigned int j = i; j < super_grid.y; ++j) {  // from diagonal to right
                        nn = (j == super_grid.y - 1 ? aN % super_NB : super_NB);
                        grid.y = (nn + THREADS - 1) / THREADS;
                        if (i == j) {  // diagonal super block
                            zlacpy_upper_kernel<<< grid, threads, 0, aContext.GetStream()>>>
                                    (mm, nn, &apA[i * super_NB, j * super_NB], aLdA, &apB[i * super_NB, j * super_NB],
                                     aLdB);
                        } else {           // off diagonal super block
                            zlacpy_full_kernel <<< grid, threads, 0, aContext.GetStream() >>>
                                    (mm, nn, &apA[i * super_NB, j * super_NB], aLdA, &apB[i * super_NB, j * super_NB],
                                     aLdB);
                        }
                    }
                }
            } else {
                if (aLdA == aLdB) {
                    cudaMemcpyAsync(apB, apA, aM * aN * sizeof(T), cudaMemcpyDeviceToDevice, aContext.GetStream());
                } else {
                    for (unsigned int i = 0; i < super_grid.x; ++i) {
                        mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                        grid.x = (mm + THREADS - 1) / THREADS;
                        for (unsigned int j = 0; j < super_grid.y; ++j) {  // full row
                            nn = (j == super_grid.y - 1 ? aN % super_NB : super_NB);
                            grid.y = (nn + THREADS - 1) / THREADS;
                            zlacpy_full_kernel <<< grid, threads, 0, aContext.GetStream() >>>
                                    (mm, nn, &apA[i * super_NB, j * super_NB], aLdA, &apB[i * super_NB, j * super_NB],
                                     aLdB);
                        }
                    }
                }
            }
        }

        template<typename T>
        void
        HCoreCudaKernels<T>::ungqr(int64_t aM, int64_t aN, int64_t aK, T *apA, int64_t aLdA, T *apTau,
                                   kernels::RunContext &aContext) {
            ///https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/orgqr/cusolver_orgqr_example.cu
            int lwork_orgqr = 0;
            int lwork = 0;

            /* step 3: query working space of geqrf and orgqr */
            if constexpr(is_complex<T>()) {
                if constexpr(is_complex_float<T>()) {
                    cusolverDnCorgqr_bufferSize(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau,
                                                (int *) (&lwork_orgqr));
                } else {
                    cusolverDnZorgqr_bufferSize(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau,
                                                (int *) (&lwork_orgqr));
                }
            } else {
                if constexpr(is_double<T>()) {
                    cusolverDnDorgqr_bufferSize(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau,
                                                (int *) (&lwork_orgqr));
                } else {
                    cusolverDnSorgqr_bufferSize(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau,
                                                (int *) (&lwork_orgqr));
                }
            }


            lwork = lwork_orgqr;

            T *d_work = (T *)aContext.RequestWorkBuffer(sizeof(T) * lwork);
            int *d_info = aContext.GetInfoPointer();
            /* step 5: compute Q */
            if constexpr(is_complex<T>()) {
                if constexpr(is_complex_float<T>()) {
                    cusolverDnCorgqr(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau, d_work, lwork,
                                     d_info);
                } else {
                    cusolverDnZorgqr(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau, d_work, lwork,
                                     d_info);
                }
            } else {
                if constexpr(is_double<T>()) {
                    cusolverDnDorgqr(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau, d_work, lwork,
                                     d_info);
                } else {
                    cusolverDnSorgqr(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau, d_work, lwork,
                                     d_info);
                }
            }
        }

        HCOREPP_INSTANTIATE_CLASS(HCoreCudaKernels)

    }
}