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
const size_t max_blocks = 65535;

namespace hcorepp {
    namespace cudakernels {

        template<typename T>
        __global__ void GenerateIdentityMatrix_kernel(size_t aNumOfCols, T *apMatrix) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;

            if (x >= aNumOfCols) {
                return;
            }

            size_t arr_index = x * aNumOfCols + x;
            apMatrix[arr_index] = 1;

        }

        template<typename T>
        __global__ void
        MultiplyByAlpha_kernel(T *apArray, size_t aRows, size_t aCols, size_t aM, size_t aRank, T aAlpha) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;

            if (x >= aRows * aCols) {
                return;
            }
            apArray[aM * aRank + x] *= aAlpha;
        }

        template<typename T>
        __global__ void
        ProcessVPointer_kernel_with_Ungqr(size_t aN, size_t aCRank, size_t Vm, T aBeta, T *apCV,
                                          size_t aLdcV, T *V, size_t aArank, const T *apBdata, bool aCholesky) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;

            size_t index = y * Vm + x;
            if (aCholesky) {
                index = y + x * aLdcV;
            }
            size_t apCV_index = x * aLdcV + y;

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
        ProcessVPointer_kernel_without_Ungqr(size_t aN, size_t aCRank, size_t Vm, T aBeta, T *apCV,
                                             size_t aLdcV, T *V, size_t aArank, const T *apBdata, bool aCholesky) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;

            size_t index = y * Vm + x;
            if (aCholesky) {
                index = y + x * aLdcV;
            }
            size_t apCV_index = x * aLdcV + y;

            if (x >= aN || y >= aCRank) {
                return;
            }

            V[index] = aBeta * apCV[apCV_index];
        }

        template<typename T>
        __global__ void
        ProcessVPointer_kernel_with_Ungqr_part2(size_t aN, size_t aCRank, size_t Vm, T aBeta, T *apCV,
                                                size_t aLdcV, T *V, size_t aArank, const T *apBdata, bool aCholesky) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;

            size_t index = y * Vm + x;
            if (aCholesky) {
                index = x * aArank + y;
            }
            size_t apB_index = x * aArank + y;

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
        ProcessVPointer_kernel_without_Ungqr_part2(size_t aN, size_t aCRank, size_t Vm, T aBeta, T *apCV,
                                                   size_t aLdcV, T *V, size_t aArank, const T *apBdata,
                                                   bool aCholesky) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;

            size_t index = y * Vm + x;
            if (aCholesky) {
                index = x * aArank + y;
            }
            size_t apB_index = x * aArank + y;

            if (x >= aN || y >= aArank) {
                return;
            }

            T *Vptr = &V[aN * aCRank];
            Vptr[index] = apBdata[apB_index];
        }

        template<typename T>
        __global__ void
        CalculateUVptrConj_kernel_(size_t aRank, size_t aVm, T *UVptr) {

            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;

            size_t index = y * aRank + x;

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
        CalculateVTnew_kernel_with_Ungqr(size_t aRkNew, bool aUngqr, size_t aMinVmVn, blas::real_type<T> *apSigma,
                                         T *apVTnew, size_t aSizeS, size_t aVm) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= aRkNew || y >= aMinVmVn) {
                return;
            }

            size_t index = y * aSizeS;

            T alpha = apSigma[x];
            T *vt = &apVTnew[x];
            vt[index] *= alpha;
        }

        template<typename T>
        __global__ void
        CalculateVTnew_kernel_without_Ungqr(size_t aRkNew, bool aUngqr, size_t aMinVmVn, blas::real_type<T> *apSigma,
                                            T *apVTnew, size_t aSizeS, size_t aVm) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= aRkNew || y >= aVm) {
                return;
            }

            size_t index = y * aSizeS;

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
        CalculateUVptr_kernel(size_t aRank, size_t aVm, T *UVptr, const T *Vnew) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= aRank || y >= aVm) {
                return;
            }

            size_t uv_index = y * aRank + x;
            size_t vnew_index = x * aVm + y;

            if (::cuda::std::is_same<T, cuFloatComplex>::value) {

            } else if (::cuda::std::is_same<T, cuDoubleComplex>::value) {

            } else {
                UVptr[uv_index] = Vnew[vnew_index];
            }

        }

        template<typename T>
        __global__ void
        CalculateNewRank_kernel_withSVD(size_t aNewRank, blas::real_type<T> *apSigma, size_t sizeS,
                                        blas::real_type<T> accuracy) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;

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
        CalculateNewRank_kernel_withoutSVD(size_t aNewRank, blas::real_type<T> *apSigma, size_t sizeS,
                                           blas::real_type<T> accuracy) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;

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
        void zlaset_lower_device(size_t m, size_t n, T offdiag, T diag, T *A, size_t lda) {
            size_t ind = blockIdx.x * THREADS + threadIdx.x;
            size_t iby = blockIdx.y * THREADS;
            /* check if full block-column && (below diag) */
            bool full = (iby + THREADS <= n && (ind >= iby + THREADS));
            /* do only rows inside matrix, and blocks not above diag */
            if (ind < m && ind + THREADS > iby) {
                A += ind + iby * lda;
                if (full) {
                    // full block-column, off-diagonal block
#pragma unroll
                    for (size_t j = 0; j < THREADS; ++j) {
                        A[j * lda] = offdiag;
                    }
                } else {
                    // either partial block-column or diagonal block
                    for (size_t j = 0; j < THREADS && iby + j < n; ++j) {
                        if (iby + j == ind)
                            A[j * lda] = diag;
                        else if (ind > iby + j)
                            A[j * lda] = offdiag;
                    }
                }
            }
        }

        template<typename T>
        __global__ void zlaset_lower_kernel(size_t m, size_t n, T offdiag, T diag, T *dA, size_t ldda) {
            zlaset_lower_device(m, n, offdiag, diag, dA, ldda);
        }

        template<typename T>
        static __device__
        void zlaset_upper_device(size_t m, size_t n, T offdiag, T diag, T *A, size_t lda) {
            size_t ind = blockIdx.x * THREADS + threadIdx.x;
            size_t iby = blockIdx.y * THREADS;
            /* check if full block-column && (above diag) */
            bool full = (iby + THREADS <= n && (ind + THREADS <= iby));
            /* do only rows inside matrix, and blocks not below diag */
            if (ind < m && ind < iby + THREADS) {
                A += ind + iby * lda;
                if (full) {
                    // full block-column, off-diagonal block
#pragma unroll
                    for (size_t j = 0; j < THREADS; ++j) {
                        A[j * lda] = offdiag;
                    }
                } else {
                    // either partial block-column or diagonal block
                    for (size_t j = 0; j < THREADS && iby + j < n; ++j) {
                        if (iby + j == ind)
                            A[j * lda] = diag;
                        else if (ind < iby + j)
                            A[j * lda] = offdiag;
                    }
                }
            }
        }

        template<typename T>
        __global__ void zlaset_upper_kernel(size_t m, size_t n, T offdiag, T diag, T *dA, size_t ldda) {
            zlaset_upper_device(m, n, offdiag, diag, dA, ldda);
        }

        template<typename T>
        static __device__
        void zlaset_full_device(size_t m, size_t n, T offdiag, T diag, T *A, size_t lda) {
            size_t ind = blockIdx.x * THREADS + threadIdx.x;
            size_t iby = blockIdx.y * THREADS;
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
                    for (size_t j = 0; j < THREADS; ++j) {
                        A[j * lda] = offdiag;
                    }
                } else {
                    // either partial block-column or diagonal block
                    for (size_t j = 0; j < THREADS && iby + j < n; ++j) {
                        if (iby + j == ind)
                            A[j * lda] = diag;
                        else
                            A[j * lda] = offdiag;
                    }
                }
            }
        }

        template<typename T>
        __global__ void zlaset_full_kernel(size_t m, size_t n, T offdiag, T diag, T *dA, size_t ldda) {
            zlaset_full_device(m, n, offdiag, diag, dA, ldda);
        }

        template<typename T>
        static __device__
        void zlacpy_lower_device(size_t m, size_t n, const T *dA, size_t ldda, T *dB, size_t lddb) {
            size_t BLK_X = blockDim.x; // THREADS
            size_t BLK_Y = blockDim.y;
            size_t ind = blockIdx.x * BLK_X + threadIdx.x;
            size_t iby = blockIdx.y * BLK_Y;
            /* check if full block-column && (below diag) */
            bool full = (iby + BLK_Y <= n && (ind >= iby + BLK_Y));
            /* do only rows inside matrix, and blocks not above diag */
            if (ind < m && ind + BLK_X > iby) {
                dA += ind + iby * ldda;
                dB += ind + iby * lddb;
                if (full) {
                    // full block-column, off-diagonal block
#pragma unroll
                    for (size_t j = 0; j < BLK_Y; ++j) {
                        dB[j * lddb] = dA[j * ldda];
                    }
                } else {
                    // either partial block-column or diagonal block
                    for (size_t j = 0; j < BLK_Y && iby + j < n && ind >= iby + j; ++j) {
                        dB[j * lddb] = dA[j * ldda];
                    }
                }
            }
        }

        template<typename T>
        __global__ void zlacpy_lower_kernel(size_t m, size_t n, const T *dA, size_t ldda, T *dB, size_t lddb) {
            zlacpy_lower_device(m, n, dA, ldda, dB, lddb);
        }

        template<typename T>
        __global__ void zlacpy_full_kernel(size_t m, size_t n, const T *dA, size_t ldda, T *dB, size_t lddb) {
            size_t BLK_X = blockDim.x;
            size_t BLK_Y = blockDim.y;
            size_t ind = blockIdx.x * BLK_X + threadIdx.x;
            size_t iby = blockIdx.y * BLK_Y + threadIdx.y;
            /* do only rows inside matrix */
            if (ind < m) {
                if (iby < n) {
                    dB[ind + iby * lddb] = dA[ind + iby * ldda];
                }
            }
        }

        template<typename T>
        __global__ void zlacpy_upper_kernel(size_t m, size_t n, const T *dA, size_t ldda, T *dB, size_t lddb) {
            size_t BLK_X = blockDim.x;
            size_t BLK_Y = blockDim.y;
            size_t ind = blockIdx.x * BLK_X + threadIdx.x;
            size_t iby = blockIdx.y * BLK_Y + threadIdx.y;
            /* check if full block-column && (above diag) */
            /* do only rows inside matrix, and blocks not below diag */
            if (ind < m && ind <= iby && iby < n) {
                dB[ind + iby * lddb] = dA[ind + iby * ldda];
            }
        }

        template<typename T>
        __global__ void
        fill_matrix_triangle_kernel(size_t aNumOfelements, T *apMatrix, blas::Layout aLayout, blas::Uplo aUplo,
                                    size_t aValue) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= aNumOfelements || y >= aNumOfelements) {
                return;
            }

            cublasFillMode_t upper_lower;
            if (aUplo == blas::Uplo::Upper) {
                upper_lower = cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
            } else {
                upper_lower = cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
            }

            if (y > x) {
                auto index = 0;
                if (upper_lower == cublasFillMode_t::CUBLAS_FILL_MODE_UPPER) {
                    index = (aLayout == blas::Layout::RowMajor) ? x * aNumOfelements + y : y * aNumOfelements + x;
                } else if (upper_lower == cublasFillMode_t::CUBLAS_FILL_MODE_LOWER) {
                    index = (aLayout == blas::Layout::RowMajor) ? y * aNumOfelements + x : x * aNumOfelements + y;
                }
                apMatrix[index] = aValue;
            }
        }

        template<typename T>
        __global__ void
        symmetrize_matrix_kernel(size_t aNumOfelements, T *apMatrix, blas::Uplo aUplo, blas::Layout aLayout) {

            /// Works only on squared matrices...

            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= aNumOfelements || y >= aNumOfelements) {
                return;
            }


            cublasFillMode_t upper_lower;
            if (aUplo == blas::Uplo::Upper) {
                upper_lower = cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
            } else {
                upper_lower = cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
            }

            if (y > x) {
                auto src_idx = 0;
                auto dest_idx = 0;
                if (upper_lower == cublasFillMode_t::CUBLAS_FILL_MODE_UPPER) {
                    src_idx = (aLayout == blas::Layout::RowMajor) ? x * aNumOfelements + y : y * aNumOfelements + x;
                    dest_idx = (aLayout == blas::Layout::RowMajor) ? y * aNumOfelements + x : x * aNumOfelements + y;
                } else if (upper_lower == cublasFillMode_t::CUBLAS_FILL_MODE_LOWER) {
                    src_idx = (aLayout == blas::Layout::RowMajor) ? y * aNumOfelements + x : x * aNumOfelements + y;
                    dest_idx = (aLayout == blas::Layout::RowMajor) ? x * aNumOfelements + y : y * aNumOfelements + x;
                }
                apMatrix[dest_idx] = apMatrix[src_idx];
            }
        }

        template<typename T>
        __global__ void
        transpose_matrix_kernel(size_t aOuterLoopRange, size_t aInnerLoopRange, const T *aA,
                                size_t aLeadingDimA, T *aOut, size_t aLeadingDimOut) {
            size_t x = blockIdx.x * blockDim.x + threadIdx.x;
            size_t y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= aOuterLoopRange || y >= aInnerLoopRange) {
                return;
            }

            if (::cuda::std::is_same<T, cuFloatComplex>::value) {

            } else if (::cuda::std::is_same<T, cuDoubleComplex>::value) {

            } else {
                aOut[x * aLeadingDimOut + y] = aA[y * aLeadingDimA + x];
            }

        }

        template<typename T>
        void HCoreCudaKernels<T>::GenerateIdentityMatrix(size_t aNumOfCols, T *apMatrix,
                                                         const kernels::RunContext &aContext) {
            dim3 dimBlock(THREADS_1D, 1);
            dim3 dimGrid((aNumOfCols + dimBlock.x - 1) / dimBlock.x);

            GenerateIdentityMatrix_kernel<<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aNumOfCols, apMatrix);
        }

        template<typename T>
        void HCoreCudaKernels<T>::MultiplyByAlpha(T *apArray, size_t aRows, size_t aCols, size_t aM, size_t aRank,
                                                  T &aAlpha, const kernels::RunContext &aContext) {
            dim3 dimBlock(THREADS_1D, 1);
            dim3 dimGrid(((aRows * aCols) + dimBlock.x - 1) / dimBlock.x);

            MultiplyByAlpha_kernel<<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(apArray, aRows, aCols, aM, aRank,
                                                                                   aAlpha);
        }

        template<typename T>
        void HCoreCudaKernels<T>::Geqrf(size_t aM, size_t aN, T *apA, size_t aLdA, T *apTau, T *aWorkspace,
                                        size_t aWorkspace_size, size_t aHostSize,
                                        const kernels::RunContext &aContext) {
            /// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xgeqrf/cusolver_Xgeqrf_example.cu
            size_t d_lwork = aWorkspace_size;     /* size of workspace */
            size_t h_lwork = aHostSize;     /* size of workspace */
            void *h_work = nullptr; /* host workspace */
            T *d_work = aWorkspace;

            if (aWorkspace == nullptr || aWorkspace_size == 0) {
                d_lwork = 0;
                h_lwork = 0;
                /* step 3: query working space of geqrf */
                cusolverDnXgeqrf_bufferSize(aContext.GetCusolverDnHandle(), NULL, aM, aN, traits<T>::cuda_data_type,
                                            apA,
                                            aLdA,
                                            traits<T>::cuda_data_type, apTau, traits<T>::cuda_data_type, &d_lwork,
                                            &h_lwork);
                d_work = (T *) aContext.RequestWorkBuffer(sizeof(T) * d_lwork);
            }

            /* step 4: QR factorization */
            cusolverDnXgeqrf(aContext.GetCusolverDnHandle(), NULL, aM, aN, traits<T>::cuda_data_type, apA, aLdA,
                             traits<T>::cuda_data_type,
                             apTau, traits<T>::cuda_data_type, d_work, d_lwork, h_work, h_lwork,
                             aContext.GetInfoPointer());
        }

        template<typename T>
        void
        HCoreCudaKernels<T>::ProcessVpointer(size_t aN, size_t aCRank, bool aGetUngqr, size_t Vm, T &aBeta, T *apCV,
                                             size_t aLdcV, T *V, size_t aArank, const T *apBdata,
                                             const kernels::RunContext &aContext, bool aCholesky) {

            dim3 dimBlock(THREADS, THREADS);
            dim3 dimGrid((aN + dimBlock.x - 1) / dimBlock.x, (aCRank + dimBlock.y - 1) / dimBlock.y);

            if (aGetUngqr) {
                ProcessVPointer_kernel_with_Ungqr<T><<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aN,
                                                                                                     aCRank, Vm, aBeta,
                                                                                                     apCV, aLdcV,
                                                                                                     V, aArank,
                                                                                                     apBdata,
                                                                                                     aCholesky);
            } else {
                ProcessVPointer_kernel_without_Ungqr<T><<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aN,
                                                                                                        aCRank, Vm,
                                                                                                        aBeta, apCV,
                                                                                                        aLdcV, V,
                                                                                                        aArank,
                                                                                                        apBdata,
                                                                                                        aCholesky);
            }

            dim3 dimBlock_(THREADS, THREADS);
            dim3 dimGrid_((aN + dimBlock.x - 1) / dimBlock.x, (aArank + dimBlock.y - 1) / dimBlock.y);

            if (aGetUngqr) {
                ProcessVPointer_kernel_with_Ungqr_part2<T><<<dimGrid_, dimBlock_, 0, aContext.GetStream()>>>(aN, aCRank,
                                                                                                             Vm, aBeta,
                                                                                                             apCV,
                                                                                                             aLdcV,
                                                                                                             V, aArank,
                                                                                                             apBdata,
                                                                                                             aCholesky);
            } else {
                ProcessVPointer_kernel_without_Ungqr_part2<T><<<dimGrid_, dimBlock_, 0, aContext.GetStream()>>>(aN,
                                                                                                                aCRank,
                                                                                                                Vm,
                                                                                                                aBeta,
                                                                                                                apCV,
                                                                                                                aLdcV,
                                                                                                                V,
                                                                                                                aArank,
                                                                                                                apBdata,
                                                                                                                aCholesky);
            }

        }

        template<typename T>
        void
        HCoreCudaKernels<T>::CalculateUVptrConj(size_t aRank, size_t aVm, T *UVptr,
                                                const kernels::RunContext &aContext) {
            dim3 dimBlock(THREADS, THREADS);
            dim3 dimGrid((aRank + dimBlock.x - 1) / dimBlock.x, (aVm + dimBlock.y - 1) / dimBlock.y);
            CalculateUVptrConj_kernel_<<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aRank, aVm, UVptr);
        }

        template<typename T>
        void
        HCoreCudaKernels<T>::CalculateVTnew(size_t aRkNew, bool aUngqr, size_t aMinVmVn, blas::real_type<T> *apSigma,
                                            T *apVTnew, size_t aSizeS, size_t aVm,
                                            const kernels::RunContext &aContext) {
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
        void HCoreCudaKernels<T>::CalculateUVptr(size_t aRank, size_t aVm, T *UVptr, const T *Vnew,
                                                 const kernels::RunContext &aContext) {
            dim3 dimBlock(THREADS, THREADS);

            dim3 dimGrid((aRank + dimBlock.x - 1) / dimBlock.x, (aVm + dimBlock.y - 1) / dimBlock.y);

            CalculateUVptr_kernel<<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aRank, aVm, UVptr, Vnew);
        }

        template<typename T>
        void HCoreCudaKernels<T>::CalculateNewRank(size_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma,
                                                   size_t sizeS, blas::real_type<T> accuracy,
                                                   const kernels::RunContext &aContext) {
            auto host_sigma = new blas::real_type<T>[sizeS];
            hcorepp::memory::Memcpy<blas::real_type<T>>
                    (host_sigma, apSigma, sizeS, aContext,
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
                for (size_t i = 1; i < sizeS; i++) {
                    if (host_sigma[i] < accuracy * Sigma_0) {
                        Sigma_0 = host_sigma[i];
                        aNewRank = i;
                        break;
                    }
                }
            } else {
                for (size_t i = 1; i < sizeS; i++) {
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
        HCoreCudaKernels<T>::SVD(common::Job aJobu, common::Job aJobvt, size_t aM, size_t aN, T *apA, size_t aLdA,
                                 T *apS, T *apU, size_t aLdU, T *apVT, size_t aLdVt,
                                 common::CompressionType aSVDOperationType, T *aWorkspace, size_t aWorkspace_size,
                                 size_t aHostSize, const kernels::RunContext &aContext) {
            ///https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xgesvd/cusolver_Xgesvd_example.cu
            size_t d_lwork = aWorkspace_size;     /* size of workspace */
            size_t h_lwork = aHostSize;     /* size of workspace */
            void *h_work = nullptr; /* host workspace */

            T *d_work = aWorkspace;
            if (aWorkspace == nullptr || aWorkspace_size == 0) {
                d_lwork = 0;
                h_lwork = 0;
                /* step 3: query working space of geqrf */
                cusolverDnXgesvd_bufferSize(aContext.GetCusolverDnHandle(), NULL, (signed char) aJobu,
                                            (signed char) aJobvt,
                                            aM, aN,
                                            traits<T>::cuda_data_type, apA, aLdA, traits<T>::cuda_data_type, apS,
                                            traits<T>::cuda_data_type, apU, aLdU, traits<T>::cuda_data_type, apVT,
                                            aLdVt,
                                            traits<T>::cuda_data_type, &d_lwork, &h_lwork);
                d_work = (T *) aContext.RequestWorkBuffer(sizeof(T) * d_lwork);
            }
            /* step 4: compute SVD */
            cusolverDnXgesvd(aContext.GetCusolverDnHandle(), NULL, (signed char) aJobu, (signed char) aJobvt, aM, aN,
                             traits<T>::cuda_data_type, apA, aLdA, traits<T>::cuda_data_type, apS,
                             traits<T>::cuda_data_type, apU, aLdU, traits<T>::cuda_data_type, apVT, aLdVt,
                             traits<T>::cuda_data_type, d_work, d_lwork,
                             h_work, h_lwork, aContext.GetInfoPointer());
        }

        template<typename T>
        void
        HCoreCudaKernels<T>::Unmqr(common::SideMode aSide, common::BlasOperation aTrans, size_t aM, size_t aN,
                                   size_t aK, T const *apA, size_t aLdA, T const *apTau, T *apC, size_t aLdC,
                                   T *aWorkspace, size_t aWorkspace_size, const kernels::RunContext &aContext) {
            ///https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/ormqr/cusolver_ormqr_example.cu
            size_t d_lwork = aWorkspace_size;     /* size of workspace */
            T *d_work = aWorkspace;
            /* step 3: query working space of geqrf */
            if (aWorkspace == nullptr || aWorkspace_size == 0) {
                cusolverDnDormqr_bufferSize(aContext.GetCusolverDnHandle(), (cublasSideMode_t) aSide,
                                            (cublasOperation_t) aTrans, aM, aN, aM,
                                            (const double *) apA, aLdA, (const double *) apTau, (const double *) apC,
                                            aLdC,
                                            (int *) &d_lwork);
                d_work = (T *) aContext.RequestWorkBuffer(sizeof(T) * d_lwork);
            }

            cublasSideMode_t side;
            if (aSide == common::SIDE_LEFT) {
                side = CUBLAS_SIDE_LEFT;
            } else if (aSide == common::SIDE_RIGHT) {
                side = CUBLAS_SIDE_RIGHT;
            }
            cublasOperation_t trans;
            if (aTrans == common::OP_TRANS) {
                trans = CUBLAS_OP_T;
            } else if (aTrans == common::OP_CONJG) {
                trans = CUBLAS_OP_C;
            } else if (aTrans == common::OP_NoTRANS) {
                trans = CUBLAS_OP_N;
            }

            cusolverDnDormqr(aContext.GetCusolverDnHandle(), side, trans, aM,
                             aN, aM, (const double *) apA, aLdA, (const double *) apTau, (double *) apC, aLdC,
                             (double *) d_work, d_lwork, aContext.GetInfoPointer());
        }

        template<typename T>
        void HCoreCudaKernels<T>::Laset(common::MatrixType aMatrixType, size_t aM, size_t aN, T aOffdiag, T aDiag,
                                        T *apA, size_t aLdA, const kernels::RunContext &aContext) {

#define dA(i_, j_) (dA + (i_) + (j_)*ldda)


            int info = 0;
            if (aMatrixType != common::MatrixType::Lower && aMatrixType != common::MatrixType::Upper &&
                aMatrixType != common::MatrixType::General)
                info = -1;
            else if (aM < 0)
                info = -2;
            else if (aN < 0)
                info = -3;
            else if (aLdA < std::max((size_t) 1, aM))
                info = -7;

            if (info != 0) {
//                magma_xerbla(__func__, -(info));
                return;  //info;
            }

            if (aM == 0 || aN == 0) {
                return;
            }

            const size_t super_NB = max_blocks * THREADS;

            double divider = super_NB;

            dim3 super_grid(ceil(aM / divider), ceil(aN / divider));

            dim3 threads(THREADS, 1);
            dim3 grid;

            size_t mm, nn;
            if (aMatrixType == common::MatrixType::Lower) {
                for (size_t i = 0; i < super_grid.x; ++i) {
                    mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                    grid.x = ceil((mm + THREADS - 1) / THREADS);
                    for (size_t j = 0; j < super_grid.y && j <= i; ++j) {  // from left to diagonal
                        nn = (j == super_grid.y - 1 ? aN % super_NB : super_NB);
                        grid.y = ceil((nn + THREADS - 1) / THREADS);
                        if (i == j) {  // diagonal super block
                            zlaset_lower_kernel<<< grid, threads, 0, aContext.GetStream() >>>(mm, nn, aOffdiag, aDiag,
                                                                                              &apA[i * super_NB, j *
                                                                                                                 super_NB],
                                                                                              aLdA);
                        } else {
                            // off diagonal super block


                            zlaset_full_kernel<<< grid, threads, 0, aContext.GetStream()>>>
                                    (mm, nn, aOffdiag, aOffdiag, &apA[i * super_NB, j * super_NB], aLdA);
                        }
                    }
                }
            } else if (aMatrixType == common::MatrixType::Upper) {
                for (size_t i = 0; i < super_grid.x; ++i) {
                    mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                    grid.x = ceil((mm + THREADS - 1) / THREADS);
                    for (size_t j = i; j < super_grid.y; ++j) {  // from diagonal to right
                        nn = (j == super_grid.y - 1 ? aN % super_NB : super_NB);
                        grid.y = ceil((nn + THREADS - 1) / THREADS);
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
                for (size_t i = 0; i < super_grid.x; ++i) {
                    mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                    grid.x = ceil((mm + THREADS - 1) / THREADS);
                    for (size_t j = 0; j < super_grid.y; ++j) {  // full row
                        nn = (j == super_grid.y - 1 ? aN % super_NB : super_NB);
                        grid.y = ceil((nn + THREADS - 1) / THREADS);
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
        HCoreCudaKernels<T>::LaCpy(common::MatrixType aType, size_t aM, size_t aN, T *apA, size_t aLdA, T *apB,
                                   size_t aLdB, const kernels::RunContext &aContext) {
#define dA(i_, j_) (dA + (i_) + (j_)*ldda)
#define dB(i_, j_) (dB + (i_) + (j_)*lddb)

            size_t info = 0;
            if (aType != common::MatrixType::Lower && aType != common::MatrixType::Upper &&
                aType != common::MatrixType::General)
                info = -1;
            else if (aM < 0)
                info = -2;
            else if (aN < 0)
                info = -3;
            else if (aLdA < std::max((size_t) 1, aM))
                info = -5;
            else if (aLdB < std::max((size_t) 1, aM))
                info = -7;

            if (info != 0) {
                return;
            }

            if (aM == 0 || aN == 0) {
                return;
            }

            const size_t super_NB = max_blocks * THREADS;
            double divider = super_NB;
            dim3 super_grid(ceil(aM / divider), ceil(aN / divider));
            dim3 threads(THREADS, THREADS);
            dim3 grid;

            size_t mm, nn;
            if (aType == common::MatrixType::Lower) {
                for (size_t i = 0; i < super_grid.x; ++i) {
                    mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                    grid.x = ceil((mm + THREADS - 1) / THREADS);
                    for (size_t j = 0; j < super_grid.y && j <= i; ++j) {  // from left to diagonal
                        nn = (j == super_grid.y - 1 ? aN % super_NB : super_NB);
                        grid.y = ceil((nn + THREADS - 1) / THREADS);
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
                for (size_t i = 0; i < super_grid.x; ++i) {
                    mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                    grid.x = (mm + THREADS - 1) / THREADS;
                    for (size_t j = i; j < super_grid.y; ++j) {  // from diagonal to right
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
                    for (size_t i = 0; i < super_grid.x; ++i) {
                        mm = (i == super_grid.x - 1 ? aM % super_NB : super_NB);
                        grid.x = (mm + THREADS - 1) / THREADS;
                        for (size_t j = 0; j < super_grid.y; ++j) {  // full row
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
        HCoreCudaKernels<T>::ungqr(size_t aM, size_t aN, size_t aK, T *apA, size_t aLdA, T *apTau,
                                   T *aWorkspace, size_t aWorkspace_size, const kernels::RunContext &aContext) {
            ///https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/orgqr/cusolver_orgqr_example.cu
            size_t lwork_orgqr = 0;
            size_t lwork = aWorkspace_size;

            T *d_work = aWorkspace;
            if (aWorkspace == nullptr || aWorkspace_size == 0) {
                lwork = 0;
                /* step 3: query working space of geqrf and orgqr */
                if constexpr (is_complex<T>()) {
                    if constexpr (is_complex_float<T>()) {
                        cusolverDnCorgqr_bufferSize(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau,
                                                    (size_t *) (&lwork_orgqr));
                    } else {
                        cusolverDnZorgqr_bufferSize(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau,
                                                    (size_t *) (&lwork_orgqr));
                    }
                } else {
                    if constexpr (is_double<T>()) {
                        cusolverDnDorgqr_bufferSize(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau,
                                                    (int *) (&lwork_orgqr));
                    } else {
                        cusolverDnSorgqr_bufferSize(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau,
                                                    (int *) (&lwork_orgqr));
                    }
                }


                lwork = lwork_orgqr;

                d_work = (T *) aContext.RequestWorkBuffer(sizeof(T) * lwork);
            }
            int *d_info = aContext.GetInfoPointer();
            /* step 5: compute Q */
            if constexpr (is_complex<T>()) {
                if constexpr (is_complex_float<T>()) {
                    cusolverDnCorgqr(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau, d_work, lwork,
                                     d_info);
                } else {
                    cusolverDnZorgqr(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau, d_work, lwork,
                                     d_info);
                }
            } else {
                if constexpr (is_double<T>()) {
                    cusolverDnDorgqr(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau, d_work, lwork,
                                     d_info);
                } else {
                    cusolverDnSorgqr(aContext.GetCusolverDnHandle(), aM, aN, aK, apA, aLdA, apTau, d_work, lwork,
                                     d_info);
                }
            }
        }

        template<typename T>
        void
        HCoreCudaKernels<T>::potrf(blas::Uplo aUplo, T *aWorkspace, size_t aWorkspaceSize, size_t aHostSize,
                                   size_t aMatrixOrder, T *apMatrix, size_t aLeadingDim,
                                   const kernels::RunContext &aContext) {

            size_t d_lwork = aWorkspaceSize;     /* size of workspace */
            size_t h_lwork = aHostSize;     /* size of workspace */
            void *h_work = nullptr; /* host workspace */
            T *d_work = aWorkspace;

            cublasFillMode_t upper_lower;
            if (aUplo == blas::Uplo::Upper) {
                upper_lower = cublasFillMode_t::CUBLAS_FILL_MODE_UPPER;
            } else {
                upper_lower = cublasFillMode_t::CUBLAS_FILL_MODE_LOWER;
            }

            if (aWorkspace == nullptr || aWorkspaceSize == 0) {
                cusolverDnXpotrf_bufferSize(aContext.GetCusolverDnHandle(), NULL, upper_lower, aMatrixOrder,
                                            traits<T>::cuda_data_type, (const void *) apMatrix, aLeadingDim,
                                            traits<T>::cuda_data_type, &d_lwork, &h_lwork);

                d_work = (T *) aContext.RequestWorkBuffer(sizeof(T) * d_lwork);
            }

            cusolverDnXpotrf(aContext.GetCusolverDnHandle(), NULL, upper_lower, aMatrixOrder, traits<T>::cuda_data_type,
                             apMatrix, aLeadingDim, traits<T>::cuda_data_type, d_work, d_lwork, h_work, h_lwork,
                             aContext.GetInfoPointer());

        }

        template<typename T>
        void HCoreCudaKernels<T>::FillMatrixTriangle(blas::Uplo aUplo, size_t aRows, T *apMatrix,
                                                     blas::Layout aLayout, size_t aValue,
                                                     const kernels::RunContext &aContext) {
            dim3 dimBlock(THREADS, THREADS);
            dim3 dimGrid((aRows + dimBlock.x - 1) / dimBlock.x, (aRows + dimBlock.y - 1) / dimBlock.y);

            fill_matrix_triangle_kernel<<< dimGrid, dimBlock, 0, aContext.GetStream()>>>(aRows, apMatrix,
                                                                                         aLayout, aUplo, aValue);
        }

        template<typename T>
        void HCoreCudaKernels<T>::SymmetrizeMatrix(blas::Uplo aUplo, size_t aRows, T *apMatrix, blas::Layout aLayout,
                                                   const kernels::RunContext &aContext) {

            dim3 dimBlock(THREADS, THREADS);
            dim3 dimGrid((aRows + dimBlock.x - 1) / dimBlock.x, (aRows + dimBlock.y - 1) / dimBlock.y);


            symmetrize_matrix_kernel<<< dimGrid, dimBlock, 0, aContext.GetStream()>>>(aRows, apMatrix,
                                                                                      aUplo, aLayout);

        }

        template<typename T>
        void HCoreCudaKernels<T>::TransposeMatrix(size_t aOuterLoopRange, size_t aInnerLoopRange, const T *aA,
                                                  size_t aLeadingDimA, T *aOut, size_t aLeadingDimOut,
                                                  const kernels::RunContext &aContext) {

            dim3 dimBlock(THREADS, THREADS);

            dim3 dimGrid((aOuterLoopRange + dimBlock.x - 1) / dimBlock.x,
                         (aInnerLoopRange + dimBlock.y - 1) / dimBlock.y);

            transpose_matrix_kernel<<<dimGrid, dimBlock, 0, aContext.GetStream()>>>(aOuterLoopRange, aInnerLoopRange,
                                                                                    aA, aLeadingDimA, aOut,
                                                                                    aLeadingDimOut);

        }


        HCOREPP_INSTANTIATE_CLASS(HCoreCudaKernels)

    }
}