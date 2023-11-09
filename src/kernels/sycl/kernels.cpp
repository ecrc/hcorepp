/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <iostream>

#include <CL/sycl.hpp>

#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/lapack.hpp>
#include <mkl.h>
#include <hcorepp/kernels/kernels.hpp>
#include <hcorepp/kernels/memory.hpp>

using namespace std;
using namespace sycl;
using namespace oneapi::mkl::blas;
using namespace oneapi::mkl::lapack;

#define JOBSVD_TO_MKLJOBSVD(in_var, out_var)        if (in_var == common::Job::NoVec) {              \
                                                        out_var = oneapi::mkl::jobsvd::novec;         \
                                                    } else if (in_var == common::Job::SomeVec) {     \
                                                        out_var = oneapi::mkl::jobsvd::somevec;       \
                                                    } else if (in_var == common::Job::AllVec) {      \
                                                        out_var = oneapi::mkl::jobsvd::vectors;       \
                                                    }

namespace hcorepp {
    namespace kernels {
        template<typename T>
        void HCoreKernels<T>::Gemm(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, size_t aM, size_t aN,
                                   size_t aK,
                                   T &aAlpha, T const *apA, size_t aLdA, T const *apB, size_t aLdB, T &aBeta, T *apC,
                                   size_t aLdC, const RunContext &aContext) {

            auto &q = aContext.GetQueue();

            oneapi::mkl::transpose trans_a;
            oneapi::mkl::transpose trans_b;
            if (aTransA == blas::Op::NoTrans) {
                trans_a = oneapi::mkl::transpose::nontrans;
            } else {
                trans_a = oneapi::mkl::transpose::trans;
            }
            if (aTransB == blas::Op::NoTrans) {
                trans_b = oneapi::mkl::transpose::nontrans;
            } else {
                trans_b = oneapi::mkl::transpose::trans;
            }
            std::vector<DepPair> actions = {
                    {(void *) apA, VariableDependency::READ},
                    {(void *) apB, VariableDependency::READ},
                    {(void *) apC, VariableDependency::READWRITE}
            };
            auto dep = aContext.GetVariableEvents(actions);
            sycl::event ev;
            if (aLayout == blas::Layout::ColMajor) {
                ev = oneapi::mkl::blas::column_major::gemm(q, trans_a, trans_b, aM, aN, aK, aAlpha, apA, aLdA,
                                                           apB, aLdB,
                                                           aBeta,
                                                           apC, aLdC, dep);
            } else {
                ev = oneapi::mkl::blas::row_major::gemm(q, trans_a, trans_b, aM, aN, aK, aAlpha, apA, aLdA,
                                                        apB, aLdB,
                                                        aBeta,
                                                        apC, aLdC, dep);
            }
            aContext.SetVariableEvent(actions, ev);
        }

        template<typename T>
        void
        HCoreKernels<T>::MultiplyByAlpha(T *apArray, size_t aRows, size_t aCols, size_t aM, size_t aRank,
                                         T &aAlpha, const RunContext &aContext) {
            queue q = aContext.GetQueue();

            size_t size_a = aRows * aCols;
            auto deps = aContext.GetVariableEvents({{(void *) apArray, VariableDependency::READWRITE}});
            auto ev = q.submit([&](handler &h) {
                h.depends_on(deps);
                h.parallel_for(size_a, [=](id<1> idx) {
                    apArray[aM * aRank + idx] *= aAlpha;
                });
            });
            aContext.SetVariableEvent({{(void *) apArray, VariableDependency::READWRITE}}, ev);
        }

        template<typename T>
        void
        HCoreKernels<T>::ProcessVpointer(size_t aN, size_t aCRank, bool aGetUngqr, size_t Vm, T &aBeta, T *apCV,
                                         size_t aLdcV, T *V, size_t aArank, const T *apBdata,
                                         const RunContext &aContext, bool aCholesky) {
            auto &q = aContext.GetQueue();

            size_t size_v = Vm * (aArank + aCRank);
            size_t size_cv = aN * aLdcV;
            size_t size_b = aN * aArank;

            std::vector<DepPair> actions = {
                    {V,    VariableDependency::WRITE},
                    {apCV, VariableDependency::READ}
            };
            auto deps = aContext.GetVariableEvents(actions);

            auto ev = q.submit([&](handler &h) {
                h.depends_on(deps);
                h.parallel_for(range < 2 > {(size_t) aN, (size_t) aCRank}, [=](id<2> idx) {
                    size_t j = idx[0];
                    size_t i = idx[1];
                    size_t v_index = j + i * Vm;
                    if (aCholesky) {
                        v_index = i + j * aLdcV;
                    }

                    if (aGetUngqr) {
                        V[v_index] = blas::conj(aBeta * apCV[i + j * aLdcV]);
                    } else {
                        V[v_index] = aBeta * apCV[i + j * aLdcV];
                    }
                });
            });
            aContext.SetVariableEvent(actions, ev);
            std::vector<DepPair> actions_2 = {
                    {(void *) V,       VariableDependency::WRITE},
                    {(void *) apBdata, VariableDependency::READ}
            };
            deps = aContext.GetVariableEvents(actions_2);
            ev = q.submit([&](handler &h) {
                h.depends_on(deps);
                h.parallel_for(range < 2 > {(size_t) aN, (size_t) aArank}, [=](id<2> idx) {
                    size_t j = idx[0];
                    size_t i = idx[1];
                    T *Vptr = &V[aN * aCRank];
                    size_t vptr_index = j + i * Vm;
                    if (aCholesky) {
                        vptr_index = i + j * aArank;
                    }

                    if (aGetUngqr) {
                        Vptr[vptr_index] = blas::conj(apBdata[i + j * aArank]);
                    } else {
                        Vptr[vptr_index] = apBdata[i + j * aArank];
                    }
                });
            });
            aContext.SetVariableEvent(actions_2, ev);
        }

        template<typename T>
        void HCoreKernels<T>::CalculateNewRank(size_t &aNewRank, bool aTruncatedSvd, blas::real_type<T> *apSigma,
                                               size_t sizeS, blas::real_type<T> accuracy,
                                               const RunContext &aContext) {
            auto host_sigma = new blas::real_type<T>[sizeS];
            hcorepp::memory::Memcpy<blas::real_type<T>>(host_sigma, apSigma, sizeS, aContext,
                                                        memory::MemoryTransfer::DEVICE_TO_HOST,
                                                        true);
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
        void HCoreKernels<T>::CalculateUVptr(size_t aRank, size_t aVm, T *UVptr, const T *Vnew,
                                             const RunContext &aContext) {
///TODO: in the blas implementation if it is complex an assert warning shows up, unlike in the cuda kernels, ask why.
///this checking is repeated across the kernels check for them as well.
            queue q = aContext.GetQueue();
            std::vector<DepPair> actions = {
                    {(void *) Vnew,  VariableDependency::READ},
                    {(void *) UVptr, VariableDependency::WRITE}
            };
            auto deps = aContext.GetVariableEvents(actions);
            auto ev = q.submit([&](handler &h) {
                h.depends_on(deps);
                h.parallel_for(range < 2 > {(size_t) aRank, (size_t) aVm}, [=](id<2> idx) {
                    size_t j = idx[0];
                    size_t i = idx[1];
                    UVptr[j + i * aRank] = blas::conj(Vnew[i + j * aVm]);
                });
            });
            aContext.SetVariableEvent(actions, ev);
        }

        template<typename T>
        void
        HCoreKernels<T>::CalculateVTnew(size_t aRkNew, bool aUngqr, size_t aMinVmVn, blas::real_type<T> *apSigma,
                                        T *apVTnew, size_t aSizeS, size_t aVm, const RunContext &aContext) {
            auto &q = aContext.GetQueue();
            std::vector<DepPair> actions = {
                    {apSigma, VariableDependency::READ},
                    {apVTnew, VariableDependency::READWRITE}
            };
            auto deps = aContext.GetVariableEvents(actions);
            sycl::event ev;
            if (aUngqr) {
                ev = q.submit([&](handler &h) {
                    h.depends_on(deps);
                    h.parallel_for(range < 2 > {(size_t) aRkNew, (size_t) aMinVmVn}, [=](id<2> idx) {
                        auto x = idx[0];
                        auto y = idx[1];
                        if (x >= aRkNew || y >= aMinVmVn) {
                            return;
                        }

                        size_t index = y * aSizeS;

                        T alpha = apSigma[x];
                        T *vt = &apVTnew[x];
                        vt[index] *= alpha;
                    });
                });
            } else {
                ev = q.submit([&](handler &h) {
                    h.depends_on(deps);
                    h.parallel_for(range < 2 > {(size_t) aRkNew, (size_t) aVm}, [=](id<2> idx) {
                        auto x = idx[0];
                        auto y = idx[1];
                        if (x >= aRkNew || y >= aVm) {
                            return;
                        }

                        size_t index = y * aSizeS;
                        T alpha = apSigma[x];
                        T *vt = &apVTnew[x];
                        vt[index] *= alpha;
                        apVTnew[index] = apVTnew[index];
                    });
                });
            }
            aContext.SetVariableEvent(actions, ev);
        }


        template<typename T>
        void
        HCoreKernels<T>::CalculateUVptrConj(size_t aRank, size_t aVm, T *UVptr, const RunContext &aContext) {
            queue q = aContext.GetQueue();

            auto ev = q.submit([&](handler &h) {
                h.depends_on(aContext.GetVariableEvents({{UVptr, VariableDependency::READWRITE}}));
                h.parallel_for(range < 2 > {(size_t) aRank, (size_t) aVm}, [=](id<2> idx) {
                    size_t i = idx[0];
                    size_t j = idx[1];
                    if (std::is_same<T, std::complex<double>>::value or
                        std::is_same<T, std::complex<float>>::value) {

                    } else {
                        UVptr[i + aRank * j] = UVptr[i + aRank * j];
                    }
                });
            });
            aContext.SetVariableEvent({{UVptr, VariableDependency::READWRITE}}, ev);
        }


        template<typename T>
        void
        HCoreKernels<T>::FillIdentityMatrix(size_t aNumOfElements, T *apMatrix, const RunContext &aContext) {
            auto &q = aContext.GetQueue();

            auto ev = q.submit([&](handler &h) {
                h.depends_on(aContext.GetVariableEvents({{apMatrix, VariableDependency::WRITE}}));
                h.parallel_for(aNumOfElements, [=](id<1> idx) {
                    apMatrix[idx * aNumOfElements + idx] = 1;
                });
            });
            aContext.SetVariableEvent({{apMatrix, VariableDependency::WRITE}}, ev);
        }


        template<typename T>
        void
        HCoreKernels<T>::LaCpy(common::MatrixType aType, size_t aM, size_t aN, T *apA, size_t aLdA, T *apB,
                               size_t aLdB, const RunContext &aContext) {
            ///no mkl implementation for lacpy.
            auto &q = aContext.GetQueue();
#define apA(i_, j_) (apA + (i_) + (j_)*aLdA)
#define apB(i_, j_) (apB + (i_) + (j_)*aLdB)

            int info = 0;
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

            const size_t threads_num = 32;
            const size_t super_NB = 65535 * threads_num;
            double divider = super_NB;
            size_t super_x = ceil(aM / divider);
            size_t super_y = ceil(aN / divider);

            size_t mm, nn;
            bool memcpy_dep = false;
            std::vector<DepPair> actions = {
                    {apA, VariableDependency::READ},
                    {apB, VariableDependency::WRITE}
            };
            auto deps = aContext.GetVariableEvents(actions);
            if (aType == common::MatrixType::Lower) {
                for (size_t i = 0; i < super_x; ++i) {
                    mm = (i == super_x - 1 ? aM % super_NB : super_NB);
                    size_t grid_x = ceil((mm + threads_num - 1) / threads_num);
                    for (size_t j = 0; j < super_y && j <= i; ++j) {  // from left to diagonal
                        nn = (j == super_y - 1 ? aN % super_NB : super_NB);
                        size_t grid_y = ceil((nn + threads_num - 1) / threads_num);
                        if (i == j) {  // diagonal super block
                            auto a = apA(i * super_NB, j * super_NB);
                            auto b = apB(i * super_NB, j * super_NB);
                            auto ev = q.submit([&](handler &h) {
                                h.depends_on(deps);
                                h.parallel_for(range<2>({grid_x * threads_num,
                                                         grid_y * threads_num}), [=](id<2> idx) {
                                    //TODO lower kernel zlacpy
                                    auto dA = a;
                                    auto dB = b;
                                    size_t ind = idx[0];
                                    size_t iby = idx[1];
                                    /* check if full block-column && (below diag) */
                                    bool full = (iby + 1 <= nn && (ind >= iby + 1));
                                    /* do only rows inside matrix, and blocks not above diag */
                                    if (ind < mm && ind + threads_num > iby) {
                                        dA += ind + iby * aLdA;
                                        dB += ind + iby * aLdB;
                                        if (full) {
                                            // full block-column, off-diagonal block
#pragma unroll
                                            for (size_t j = 0; j < 1; ++j) {
                                                dB[j * aLdB] = dA[j * aLdA];
                                            }
                                        } else {
                                            // either partial block-column or diagonal block
                                            for (size_t j = 0; j < 1 && iby + j < nn && ind >= iby + j; ++j) {
                                                dB[j * aLdB] = dA[j * aLdA];
                                            }
                                        }
                                    }
                                });
                            });
                            deps.push_back(ev);
                        } else {           // off diagonal super block
                            auto a = apA(i * super_NB, j * super_NB);
                            auto b = apB(i * super_NB, j * super_NB);
                            auto ev = q.submit([&](handler &h) {
                                h.depends_on(deps);
                                h.parallel_for(range<2>({grid_x * threads_num,
                                                         grid_y * threads_num}), [=](id<2> idx) {
                                    size_t ind = idx[0];
                                    size_t iby = idx[1];
                                    /* do only rows inside matrix */
                                    if (ind < mm) {
                                        if (iby < nn) {
                                            b[ind + iby * aLdB] = a[ind + iby * aLdA];
                                        }
                                    }
                                });
                            });
                            deps.push_back(ev);
                        }
                    }
                }
            } else if (aType == common::MatrixType::Upper) {
                for (size_t i = 0; i < super_x; ++i) {
                    mm = (i == super_x - 1 ? aM % super_NB : super_NB);
                    size_t grid_x = (mm + threads_num - 1) / threads_num;
                    for (size_t j = i; j < super_y; ++j) {  // from diagonal to right
                        nn = (j == super_y - 1 ? aN % super_NB : super_NB);
                        size_t grid_y = (nn + threads_num - 1) / threads_num;
                        if (i == j) {  // diagonal super block
                            auto a = apA(i * super_NB, j * super_NB);
                            auto b = apB(i * super_NB, j * super_NB);
                            auto ev = q.submit([&](handler &h) {
                                h.depends_on(deps);
                                h.parallel_for(range<2>({grid_x * threads_num,
                                                         grid_y * threads_num}), [=](id<2> idx) {
                                    size_t ind = idx[0];
                                    size_t iby = idx[1];
                                    /* check if full block-column && (above diag) */
                                    /* do only rows inside matrix, and blocks not below diag */
                                    if (ind < mm && ind <= iby && iby < nn) {
                                        b[ind + iby * aLdB] = a[ind + iby * aLdA];
                                    }
                                });
                            });
                            deps.push_back(ev);
                        } else {           // off diagonal super block
                            auto a = apA(i * super_NB, j * super_NB);
                            auto b = apB(i * super_NB, j * super_NB);
                            auto ev = q.submit([&](handler &h) {
                                h.depends_on(deps);
                                h.parallel_for(range<2>({grid_x * threads_num,
                                                         grid_y * threads_num}), [=](id<2> idx) {
                                    size_t ind = idx[0];
                                    size_t iby = idx[1];
                                    /* do only rows inside matrix */
                                    if (ind < mm) {
                                        if (iby < nn) {
                                            b[ind + iby * aLdB] = a[ind + iby * aLdA];
                                        }
                                    }
                                });
                            });
                            deps.push_back(ev);
                        }
                    }
                }
            } else {
                if (aLdA == aLdB) {
                    memory::Memcpy<T>(apB, apA, aM * aN, aContext);
                    memcpy_dep = true;
                } else {
                    for (size_t i = 0; i < super_x; ++i) {
                        mm = (i == super_x - 1 ? aM % super_NB : super_NB);
                        size_t grid_x = (mm + threads_num - 1) / threads_num;
                        for (size_t j = 0; j < super_y; ++j) {  // full row
                            nn = (j == super_y - 1 ? aN % super_NB : super_NB);
                            size_t grid_y = (nn + threads_num - 1) / threads_num;
                            auto a = apA(i * super_NB, j * super_NB);
                            auto b = apB(i * super_NB, j * super_NB);
                            auto ev = q.submit([&](handler &h) {
                                h.depends_on(deps);
                                h.parallel_for(range<2>({grid_x * threads_num,
                                                         grid_y * threads_num}), [=](id<2> idx) {
                                    size_t ind = idx[0];
                                    size_t iby = idx[1];
                                    /* do only rows inside matrix */
                                    if (ind < mm) {
                                        if (iby < nn) {
                                            b[ind + iby * aLdB] = a[ind + iby * aLdA];
                                        }
                                    }
                                });
                            });
                            deps.push_back(ev);
                        }
                    }
                }
            }
            if (!deps.empty() && !memcpy_dep) {
                aContext.SetVariableEvent(actions, deps[deps.size() - 1]);
            }
        }

        template<typename T>
        void HCoreKernels<T>::Geqrf(size_t aM, size_t aN, T *apA, size_t aLdA, T *apTau,
                                    T *aWorkspace, size_t aWorkspace_size, size_t aHostSize,
                                    const RunContext &aContext) {
            queue q = aContext.GetQueue();

            size_t size_a = aM * aN;

            size_t size_tau = std::min(aM, aN);

            size_t scratchpad_size = aWorkspace_size;
            auto *p_scratchpad = aWorkspace;


            if (aWorkspace == nullptr || aWorkspace_size == 0) {
                scratchpad_size = oneapi::mkl::lapack::geqrf_scratchpad_size<T>(q, aM, aN, aLdA);
                p_scratchpad = malloc_device<T>(scratchpad_size * sizeof(T), q);
            }


            std::vector<DepPair> actions = {
                    {apA,   VariableDependency::READWRITE},
                    {apTau, VariableDependency::WRITE}
            };
            if (aWorkspace && aWorkspace_size > 0) actions.push_back({aWorkspace, VariableDependency::WRITE});
            auto deps = aContext.GetVariableEvents(actions);
            auto ev = oneapi::mkl::lapack::geqrf(q, aM, aN, apA, aLdA, apTau, p_scratchpad, scratchpad_size, deps);
            aContext.SetVariableEvent(actions, ev);
            if (aWorkspace == nullptr || aWorkspace_size == 0) {
                ev.wait();
                free(p_scratchpad, q);
            }

        }

        template<typename T>
        void HCoreKernels<T>::Laset(common::MatrixType aMatrixType, size_t aM, size_t aN, T aOffdiag, T aDiag,
                                    T *apA, size_t aLdA, const RunContext &aContext) {
            ///no mkl implementation for laset.
            auto &q = aContext.GetQueue();

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

            const size_t threads_num = 32;
            const size_t super_NB = 65535 * threads_num;
            double divider = super_NB;
            size_t super_x = ceil(aM / divider);
            size_t super_y = ceil(aN / divider);

            size_t mm, nn;
            std::vector<DepPair> actions = {{apA, VariableDependency::WRITE}};
            auto deps = aContext.GetVariableEvents(actions);
            if (aMatrixType == common::MatrixType::Lower) {
                for (size_t i = 0; i < super_x; ++i) {
                    mm = (i == super_x - 1 ? aM % super_NB : super_NB);
                    size_t grid_x = ceil((mm + threads_num - 1) / threads_num);
                    for (size_t j = 0; j < super_y && j <= i; ++j) {  // from left to diagonal
                        nn = (j == super_y - 1 ? aN % super_NB : super_NB);
                        size_t grid_y = ceil((nn + threads_num - 1) / threads_num);
                        if (i == j) {  // diagonal super block
                            auto a = apA(i * super_NB, j * super_NB);
                            auto ev = q.submit([&](handler &h) {
                                h.depends_on(deps);
                                h.parallel_for(range<2>({grid_x * threads_num,
                                                         grid_y}), [=](id<2> idx) {
                                    size_t ind = idx[0];
                                    size_t iby = idx[1] * threads_num;
                                    /* check if full block-column && (below diag) */
                                    bool full = (iby + threads_num <= nn && (ind >= iby + threads_num));
                                    /* do only rows inside matrix, and blocks not above diag */
                                    if (ind < mm && ind + threads_num > iby) {
                                        auto A = a;
                                        A += ind + iby * aLdA;
                                        if (full) {
                                            // full block-column, off-diagonal block
#pragma unroll
                                            for (size_t j = 0; j < threads_num; ++j) {
                                                A[j * aLdA] = aOffdiag;
                                            }
                                        } else {
                                            // either partial block-column or diagonal block
                                            for (size_t j = 0; j < threads_num && iby + j < nn; ++j) {
                                                if (iby + j == ind)
                                                    A[j * aLdA] = aDiag;
                                                else if (ind > iby + j)
                                                    A[j * aLdA] = aOffdiag;
                                            }
                                        }
                                    }
                                });
                            });
                            deps.push_back(ev);
                        } else {           // off diagonal super block
                            auto a = apA(i * super_NB, j * super_NB);
                            auto ev = q.submit([&](handler &h) {
                                h.depends_on(deps);
                                h.parallel_for(range<2>({grid_x * threads_num,
                                                         grid_y}), [=](id<2> idx) {
                                    size_t ind = idx[0];
                                    size_t iby = idx[1] * threads_num;
                                    /* check if full block-column && (below diag || above diag || offdiag == diag) */
                                    bool full = (iby + threads_num <= nn &&
                                                 (ind >= iby + threads_num || ind + threads_num <= iby ||
                                                  (aOffdiag == aOffdiag)));
                                    //                         MAGMA_Z_EQUAL(offdiag, diag)));
                                    /* do only rows inside matrix */
                                    if (ind < mm) {
                                        auto A = a;
                                        A += ind + iby * aLdA;
                                        if (full) {
                                            // full block-column, off-diagonal block or offdiag == diag
#pragma unroll
                                            for (size_t j = 0; j < threads_num; ++j) {
                                                A[j * aLdA] = aOffdiag;
                                            }
                                        } else {
                                            // either partial block-column or diagonal block
                                            for (size_t j = 0; j < threads_num && iby + j < nn; ++j) {
                                                if (iby + j == ind)
                                                    A[j * aLdA] = aDiag;
                                                else
                                                    A[j * aLdA] = aOffdiag;
                                            }
                                        }
                                    }
                                });
                            });
                            deps.push_back(ev);
                        }
                    }
                }
            } else if (aMatrixType == common::MatrixType::Upper) {
                for (size_t i = 0; i < super_x; ++i) {
                    mm = (i == super_x - 1 ? aM % super_NB : super_NB);
                    size_t grid_x = ceil((mm + threads_num - 1) / threads_num);
                    for (size_t j = i; j < super_y; ++j) {  // from diagonal to right
                        nn = (j == super_y - 1 ? aN % super_NB : super_NB);
                        size_t grid_y = ceil((nn + threads_num - 1) / threads_num);
                        if (i == j) {  // diagonal super block
                            auto a = apA(i * super_NB, j * super_NB);
                            auto ev = q.submit([&](handler &h) {
                                h.depends_on(deps);
                                h.parallel_for(range<2>({grid_x * threads_num,
                                                         grid_y}), [=](id<2> idx) {
                                    size_t ind = idx[0];
                                    size_t iby = idx[1] * threads_num;
                                    /* check if full block-column && (above diag) */
                                    bool full = (iby + threads_num <= nn && (ind + threads_num <= iby));
                                    /* do only rows inside matrix, and blocks not below diag */
                                    if (ind < mm && ind < iby + threads_num) {
                                        auto A = a;
                                        A += ind + iby * aLdA;
                                        if (full) {
                                            // full block-column, off-diagonal block
#pragma unroll
                                            for (size_t j = 0; j < threads_num; ++j) {
                                                A[j * aLdA] = aOffdiag;
                                            }
                                        } else {
                                            // either partial block-column or diagonal block
                                            for (size_t j = 0; j < threads_num && iby + j < nn; ++j) {
                                                if (iby + j == ind)
                                                    A[j * aLdA] = aDiag;
                                                else if (ind < iby + j)
                                                    A[j * aLdA] = aOffdiag;
                                            }
                                        }
                                    }
                                });
                            });
                            deps.push_back(ev);
                        } else {           // off diagonal super block
                            auto a = apA(i * super_NB, j * super_NB);
                            auto ev = q.submit([&](handler &h) {
                                h.depends_on(deps);
                                h.parallel_for(range<2>({grid_x * threads_num,
                                                         grid_y * threads_num}), [=](id<2> idx) {
                                    size_t ind = idx[0];
                                    size_t iby = idx[1] * threads_num;
                                    /* check if full block-column && (below diag || above diag || offdiag == diag) */
                                    bool full = (iby + threads_num <= nn &&
                                                 (ind >= iby + threads_num || ind + threads_num <= iby ||
                                                  (aOffdiag == aOffdiag)));
                                    //                         MAGMA_Z_EQUAL(offdiag, diag)));
                                    /* do only rows inside matrix */
                                    if (ind < mm) {
                                        auto A = a;
                                        A += ind + iby * aLdA;
                                        if (full) {
                                            // full block-column, off-diagonal block or offdiag == diag
#pragma unroll
                                            for (size_t j = 0; j < threads_num; ++j) {
                                                A[j * aLdA] = aOffdiag;
                                            }
                                        } else {
                                            // either partial block-column or diagonal block
                                            for (size_t j = 0; j < threads_num && iby + j < nn; ++j) {
                                                if (iby + j == ind)
                                                    A[j * aLdA] = aDiag;
                                                else
                                                    A[j * aLdA] = aOffdiag;
                                            }
                                        }
                                    }
                                });
                            });
                            deps.push_back(ev);
                        }
                    }
                }
            } else {
                for (size_t i = 0; i < super_x; ++i) {
                    mm = (i == super_x - 1 ? aM % super_NB : super_NB);
                    size_t grid_x = ceil((mm + threads_num - 1) / threads_num);
                    for (size_t j = 0; j < super_y; ++j) {  // full row
                        nn = (j == super_y - 1 ? aN % super_NB : super_NB);
                        size_t grid_y = ceil((nn + threads_num - 1) / threads_num);
                        if (i == j) {  // diagonal super block
                            auto a = apA(i * super_NB, j * super_NB);
                            auto ev = q.submit([&](handler &h) {
                                h.depends_on(deps);
                                h.parallel_for(range<2>({grid_x * threads_num,
                                                         grid_y * threads_num}), [=](id<2> idx) {
                                    size_t ind = idx[0];
                                    size_t iby = idx[1];
                                    /* check if full block-column && (below diag || above diag || offdiag == diag) */
                                    bool full = (iby + threads_num <= nn &&
                                                 (ind >= iby + threads_num || ind + threads_num <= iby ||
                                                  (aOffdiag == aDiag)));
                                    //                         MAGMA_Z_EQUAL(offdiag, diag)));
                                    /* do only rows inside matrix */
                                    if (ind < mm) {
                                        auto A = a;
                                        A += ind + iby * aLdA;
                                        if (full) {
                                            // full block-column, off-diagonal block or offdiag == diag
#pragma unroll
                                            for (size_t j = 0; j < threads_num; ++j) {
                                                A[j * aLdA] = aOffdiag;
                                            }
                                        } else {
                                            // either partial block-column or diagonal block
                                            for (size_t j = 0; j < threads_num && iby + j < nn; ++j) {
                                                if (iby + j == ind)
                                                    A[j * aLdA] = aDiag;
                                                else
                                                    A[j * aLdA] = aOffdiag;
                                            }
                                        }
                                    }
                                });
                            });
                            deps.push_back(ev);
                        } else {           // off diagonal super block
                            auto a = apA(i * super_NB, j * super_NB);
                            auto ev = q.submit([&](handler &h) {
                                h.depends_on(deps);
                                h.parallel_for(range<2>({grid_x * threads_num,
                                                         grid_y}), [=](id<2> idx) {
                                    size_t ind = idx[0];
                                    size_t iby = idx[1] * threads_num;
                                    /* check if full block-column && (below diag || above diag || offdiag == diag) */
                                    bool full = (iby + threads_num <= nn &&
                                                 (ind >= iby + threads_num || ind + threads_num <= iby ||
                                                  (aOffdiag == aOffdiag)));
                                    //                         MAGMA_Z_EQUAL(offdiag, diag)));
                                    /* do only rows inside matrix */
                                    if (ind < mm) {
                                        auto A = a;
                                        A += ind + iby * aLdA;
                                        if (full) {
                                            // full block-column, off-diagonal block or offdiag == diag
#pragma unroll
                                            for (size_t j = 0; j < threads_num; ++j) {
                                                A[j * aLdA] = aOffdiag;
                                            }
                                        } else {
                                            // either partial block-column or diagonal block
                                            for (size_t j = 0; j < threads_num && iby + j < nn; ++j) {
                                                if (iby + j == ind)
                                                    A[j * aLdA] = aDiag;
                                                else
                                                    A[j * aLdA] = aOffdiag;
                                            }
                                        }
                                    }
                                });
                            });
                            deps.push_back(ev);
                        }
                    }
                }
            }
            if (deps.size() > 0) {
                aContext.SetVariableEvent(actions, deps[deps.size() - 1]);
            }
        }

        template<typename T>
        void HCoreKernels<T>::Trmm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans,
                                   blas::Diag aDiag, size_t aM, size_t aN, T aAlpha, T const *apA, size_t aLdA,
                                   T *apB, size_t aLdB, const RunContext &aContext) {
            queue q = aContext.GetQueue();

            oneapi::mkl::uplo upper_lower;
            if (aUplo == blas::Uplo::Upper) {
                upper_lower = oneapi::mkl::uplo::upper;
            } else {
                upper_lower = oneapi::mkl::uplo::lower;
            }

            oneapi::mkl::transpose trans;
            if (aTrans == blas::Op::NoTrans) {
                trans = oneapi::mkl::transpose::nontrans;
            } else {
                trans = oneapi::mkl::transpose::trans;
            }

            oneapi::mkl::diag unit_diag;
            if (aDiag == blas::Diag::NonUnit) {
                unit_diag = oneapi::mkl::diag::nonunit;
            } else {
                unit_diag = oneapi::mkl::diag::unit;
            }

            size_t size_a;
            oneapi::mkl::side left_right;
            if (aSide == blas::Side::Right) {
                left_right = oneapi::mkl::side::right;
                size_a = aLdA * aM;
            } else {
                left_right = oneapi::mkl::side::left;
                size_a = aLdA * aN;
            }

            std::vector<DepPair> actions = {
                    {(void *) apA, VariableDependency::READ},
                    {(void *) apB, VariableDependency::WRITE}
            };
            auto deps = aContext.GetVariableEvents(actions);
#if __SYCL_COMPILER_VERSION >= 20221201
            auto ev = oneapi::mkl::blas::trmm(q, left_right, upper_lower, trans, unit_diag,
                                              aM, aN, aAlpha, apA,
                                              aLdA, apB, aLdB, compute_mode::unset, deps);
#else
            auto ev = oneapi::mkl::blas::trmm(q, left_right, upper_lower, trans, unit_diag,
                                              aM, aN, aAlpha, apA,
                                              aLdA, apB, aLdB, deps);
#endif
            aContext.SetVariableEvent(actions, ev);
        }

        template<typename T>
        void
        HCoreKernels<T>::SVD(common::Job aJobu, common::Job aJobvt, size_t aM, size_t aN, T *apA, size_t aLdA,
                             T *apS, T *apU, size_t aLdU, T *apVT, size_t aLdVt, common::CompressionType aSVDType,
                             T *aWorkspace, size_t aWorkspace_size, size_t aHostSize, const RunContext &aContext) {
            auto &q = aContext.GetQueue();
            oneapi::mkl::jobsvd job_u;
            oneapi::mkl::jobsvd job_v;
            JOBSVD_TO_MKLJOBSVD(aJobu, job_u);
            JOBSVD_TO_MKLJOBSVD(aJobvt, job_v);
            size_t scratchpad_size = aWorkspace_size;
            auto *scratchpad = aWorkspace;
            if (aWorkspace == nullptr || aWorkspace_size == 0) {
                scratchpad_size = oneapi::mkl::lapack::gesvd_scratchpad_size<T>(q,
                                                                                job_u,
                                                                                job_v,
                                                                                aM, aN, aLdA, aLdU,
                                                                                aLdVt);
                scratchpad = malloc_device<T>(scratchpad_size * sizeof(T), q);
            }
            std::vector<DepPair> actions = {
                    {apA,  VariableDependency::READ},
                    {apS,  VariableDependency::WRITE},
                    {apU,  VariableDependency::WRITE},
                    {apVT, VariableDependency::WRITE}
            };

            if (aWorkspace && aWorkspace_size > 0) actions.push_back({aWorkspace, VariableDependency::WRITE});
            auto deps = aContext.GetVariableEvents(actions);
            auto ev = oneapi::mkl::lapack::gesvd(q, job_u, job_v, aM, aN,
                                                 apA,
                                                 aLdA,
                                                 apS, apU, aLdU, apVT,
                                                 aLdVt, scratchpad, scratchpad_size, deps);
            aContext.SetVariableEvent(actions, ev);
            if (aWorkspace == nullptr || aWorkspace_size == 0) {
                ev.wait();
                free(scratchpad, q);
            }
        }

        template<typename T>
        void
        HCoreKernels<T>::Unmqr(common::SideMode aSide, common::BlasOperation aTrans, size_t aM, size_t aN,
                               size_t aK, T const *apA, size_t aLdA, T const *apTau, T *apC, size_t aLdC,
                               T *aWorkspace, size_t aWorkspace_size, const RunContext &aContext) {
            queue q = aContext.GetQueue();

            oneapi::mkl::transpose trans;
            oneapi::mkl::side side;

            if (aTrans == common::BlasOperation::OP_NoTRANS) {
                trans = oneapi::mkl::transpose::nontrans;
            } else {
                trans = oneapi::mkl::transpose::trans;
            }
            if (aSide == common::SideMode::SIDE_LEFT) {
                side = oneapi::mkl::side::left;
            } else {
                side = oneapi::mkl::side::right;
            }

            std::vector<DepPair> actions = {
                    {(void *) apA,   VariableDependency::READ},
                    {(void *) apTau, VariableDependency::READ},
                    {(void *) apC,   VariableDependency::WRITE}
            };
            sycl::event ev;
            if (aWorkspace && aWorkspace_size > 0) actions.push_back({aWorkspace, VariableDependency::WRITE});
            auto deps = aContext.GetVariableEvents(actions);
            
            size_t scratchpad_size = aWorkspace_size;
            auto *scratchpad = aWorkspace;
            if constexpr (blas::is_complex<T>()) {
                if (aWorkspace == nullptr || aWorkspace_size == 0) {
                    scratchpad_size = oneapi::mkl::lapack::unmqr_scratchpad_size<T>(q, side, trans, aM,
                                                                                    aN,
                                                                                    aK, aLdA, aLdC);
                    scratchpad = malloc_device<T>(scratchpad_size * sizeof(T), q);
                }
                ev = oneapi::mkl::lapack::unmqr(q, side, trans, aM, aN, aK, apA, aLdA, apTau, apC, aLdC,
                                                scratchpad, scratchpad_size, deps);
            } else {
                if (aWorkspace == nullptr || aWorkspace_size == 0) {
                    scratchpad_size = oneapi::mkl::lapack::ormqr_scratchpad_size<T>(q, side, trans, aM,
                                                                                    aN,
                                                                                    aK, aLdA, aLdC);
                    scratchpad = malloc_device<T>(scratchpad_size * sizeof(T), q);
                }
                ev = oneapi::mkl::lapack::ormqr(q, side, trans, aM, aN, aK, (T *) apA, aLdA, (T *) apTau, apC, aLdC,
                                                scratchpad, scratchpad_size, deps);
            }
            aContext.SetVariableEvent(actions, ev);
            if (aWorkspace == nullptr || aWorkspace_size == 0) {
                ev.wait();
                free(scratchpad, q);
            }
        }

        template<typename T>
        void
        HCoreKernels<T>::ungqr(size_t aM, size_t aN, size_t aK, T *apA, size_t aLdA, T *apTau, T *aWorkspace,
                               size_t aWorkspace_size, const RunContext &aContext) {
            queue q = aContext.GetQueue();
            T *p_scratchpad = aWorkspace;
            size_t scratchpad_size = aWorkspace_size;
            std::vector<DepPair> actions = {
                    {(void *) apA,   VariableDependency::WRITE},
                    {(void *) apTau, VariableDependency::READ}
            };
            if (aWorkspace && aWorkspace_size > 0) actions.push_back({aWorkspace, VariableDependency::WRITE});
            auto deps = aContext.GetVariableEvents(actions);
            sycl::event ev;
            if constexpr (blas::is_complex<T>()) {
                if (aWorkspace == nullptr || aWorkspace_size == 0) {
                    scratchpad_size = oneapi::mkl::lapack::ungqr_scratchpad_size<T>(q, aM, aN, aK, aLdA);
                    p_scratchpad = malloc_device<T>(scratchpad_size * sizeof(T), q);
                }
                ev = oneapi::mkl::lapack::ungqr(q, aM, aN, aK, apA, aLdA, apTau, p_scratchpad, scratchpad_size, deps);
            } else {
                if (aWorkspace == nullptr || aWorkspace_size == 0) {
                    scratchpad_size = oneapi::mkl::lapack::orgqr_scratchpad_size<T>(q, aM, aN, aK, aLdA);
                    p_scratchpad = malloc_device<T>(scratchpad_size * sizeof(T), q);
                }
                ev = oneapi::mkl::lapack::orgqr(q, aM, aN, aK, apA, aLdA, apTau, p_scratchpad, scratchpad_size, deps);
            }
            aContext.SetVariableEvent(actions, ev);
            if (aWorkspace == nullptr || aWorkspace_size == 0) {
                ev.wait();
                free(p_scratchpad, q);
            }
        }

        template<typename T>
        size_t
        HCoreKernels<T>::CalculateGemmWorkspaceSize(size_t aUm, size_t aUn, size_t aVm, size_t aVn, size_t aSizeS,
                                                    const operators::CompressionParameters &aHelpers,
                                                    size_t &aHostSize, const RunContext &aContext) {
            queue q = aContext.GetQueue();
            std::vector<size_t> scratchpad_sizes;
            oneapi::mkl::jobsvd job_u;
            oneapi::mkl::jobsvd job_v;
            JOBSVD_TO_MKLJOBSVD(common::Job::SomeVec, job_u);
            JOBSVD_TO_MKLJOBSVD(common::Job::SomeVec, job_v);
            size_t min_Um_Un = std::min(aUm, aUn);
            size_t min_Vm_Vn = std::min(aVm, aVn);

            scratchpad_sizes.push_back(oneapi::mkl::lapack::geqrf_scratchpad_size<T>(q, aUm, aUn, aUm));
            scratchpad_sizes.push_back(oneapi::mkl::lapack::geqrf_scratchpad_size<T>(q, aVm, aVn, aVm));

            if (aHelpers.GetTrmm()) {
                if (aHelpers.GetUngqr()) {
                    scratchpad_sizes.push_back(oneapi::mkl::lapack::gesvd_scratchpad_size<T>(q,
                                                                                             job_u,
                                                                                             job_v,
                                                                                             min_Um_Un, aUn, min_Um_Un,
                                                                                             min_Um_Un,
                                                                                             aSizeS));
                } else {
                    scratchpad_sizes.push_back(oneapi::mkl::lapack::gesvd_scratchpad_size<T>(q,
                                                                                             job_u,
                                                                                             job_v,
                                                                                             min_Um_Un, aUn, min_Um_Un,
                                                                                             aUm,
                                                                                             aSizeS));
                }
            } else {
                if (aHelpers.GetUngqr()) {
                    scratchpad_sizes.push_back(oneapi::mkl::lapack::gesvd_scratchpad_size<T>(q,
                                                                                             job_u,
                                                                                             job_v,
                                                                                             min_Um_Un, min_Vm_Vn,
                                                                                             min_Um_Un, min_Um_Un,
                                                                                             aSizeS));
                } else {
                    scratchpad_sizes.push_back(oneapi::mkl::lapack::gesvd_scratchpad_size<T>(q,
                                                                                             job_u,
                                                                                             job_v,
                                                                                             min_Um_Un, min_Vm_Vn,
                                                                                             min_Um_Un, aUm,
                                                                                             aSizeS));
                }
            }

            if (aHelpers.GetUngqr()) {
                if constexpr (blas::is_complex<T>()) {
                    scratchpad_sizes.push_back(
                            oneapi::mkl::lapack::ungqr_scratchpad_size<T>(q, aUm, min_Um_Un, min_Um_Un, aUm));
                    scratchpad_sizes.push_back(
                            oneapi::mkl::lapack::ungqr_scratchpad_size<T>(q, aVm, min_Vm_Vn, min_Vm_Vn, aVm));
                } else {
                    scratchpad_sizes.push_back(
                            oneapi::mkl::lapack::orgqr_scratchpad_size<T>(q, aUm, min_Um_Un, min_Um_Un, aUm));
                    scratchpad_sizes.push_back(
                            oneapi::mkl::lapack::orgqr_scratchpad_size<T>(q, aVm, min_Vm_Vn, min_Vm_Vn, aVm));
                }
            } else {
                if constexpr (blas::is_complex<T>()) {
                    scratchpad_sizes.push_back(oneapi::mkl::lapack::unmqr_scratchpad_size<T>(q, oneapi::mkl::side::left,
                                                                                             oneapi::mkl::transpose::nontrans,
                                                                                             aUm,
                                                                                             aSizeS, min_Um_Un, aUm,
                                                                                             aUm));
                    scratchpad_sizes.push_back(
                            oneapi::mkl::lapack::unmqr_scratchpad_size<T>(q, oneapi::mkl::side::right,
                                                                          oneapi::mkl::transpose::trans, aSizeS,
                                                                          aVm, min_Vm_Vn, aVm, aSizeS));
                } else {
                    scratchpad_sizes.push_back(oneapi::mkl::lapack::ormqr_scratchpad_size<T>(q, oneapi::mkl::side::left,
                                                                                             oneapi::mkl::transpose::nontrans,
                                                                                             aUm,
                                                                                             aSizeS, min_Um_Un, aUm,
                                                                                             aUm));
                    scratchpad_sizes.push_back(
                            oneapi::mkl::lapack::ormqr_scratchpad_size<T>(q, oneapi::mkl::side::right,
                                                                          oneapi::mkl::transpose::trans, aSizeS,
                                                                          aVm, min_Vm_Vn, aVm, aSizeS));
                }
            }

            aHostSize = 0;
            return *max_element(scratchpad_sizes.begin(), scratchpad_sizes.end());
        }

        template<typename T>
        int
        HCoreKernels<T>::potrf(blas::Uplo aUplo, T *aWorkspace, size_t aWorkspaceSize, size_t aHostSize,
                               size_t aMatrixOrder, T *apMatrix, size_t aLeadingDim, blas::Layout aLayout,
                               const kernels::RunContext &aContext) {
            auto &q = aContext.GetQueue();

            oneapi::mkl::uplo upper_lower;
            if (aUplo == blas::Uplo::Upper) {
                upper_lower = oneapi::mkl::uplo::upper;
            } else {
                upper_lower = oneapi::mkl::uplo::lower;
            }

            std::vector<DepPair> actions = {
                    {(void *) apMatrix, VariableDependency::WRITE}
            };

            if (aWorkspace && aWorkspaceSize > 0) actions.push_back({aWorkspace, VariableDependency::WRITE});
            auto deps = aContext.GetVariableEvents(actions);

            size_t scratchpad_size = aWorkspaceSize;
            auto *p_scratchpad = aWorkspace;

            if (aWorkspace == nullptr || aWorkspaceSize == 0) {
                scratchpad_size = oneapi::mkl::lapack::potrf_scratchpad_size<T>(q, upper_lower, aMatrixOrder,
                                                                                aLeadingDim);
                p_scratchpad = malloc_device<T>(scratchpad_size * sizeof(T), q);
            }

            sycl::event ev = oneapi::mkl::lapack::potrf(q, upper_lower, aMatrixOrder, apMatrix, aLeadingDim,
                                                        p_scratchpad, scratchpad_size, deps);

            aContext.SetVariableEvent(actions, ev);

            if (aWorkspace == nullptr || aWorkspaceSize == 0) {
                ev.wait();
                free(p_scratchpad, q);
            }

            return 0;
        }

        template<typename T>
        void HCoreKernels<T>::FillMatrixTriangle(blas::Uplo aUplo, size_t aRows, size_t aCols, T *apMatrix,
                                                 blas::Layout aLayout, size_t aValue,
                                                 const kernels::RunContext &aContext) {
            if (aRows != aCols) {
                return;
            }
            oneapi::mkl::uplo upper_lower;
            if (aUplo == blas::Uplo::Upper) {
                upper_lower = oneapi::mkl::uplo::upper;
            } else {
                upper_lower = oneapi::mkl::uplo::lower;
            }

            queue q = aContext.GetQueue();

            auto ev = q.submit([&](handler &h) {
                h.depends_on(aContext.GetVariableEvents({{apMatrix, VariableDependency::WRITE}}));
                h.parallel_for(range < 2 > {(size_t) aRows, (size_t) aRows}, [=](id<2> idx) {
                    size_t i = idx[0];
                    size_t j = idx[1];
                    if (j > i) {
                        auto index = 0;
                        if (upper_lower == oneapi::mkl::uplo::upper) {
                            index = (aLayout == blas::Layout::RowMajor) ? i * aRows + j : j * aRows + i;
                        } else if (upper_lower == oneapi::mkl::uplo::lower) {
                            index = (aLayout == blas::Layout::RowMajor) ? j * aRows + i : i * aRows + j;
                        }
                        apMatrix[index] = aValue;
                    }
                });
            });

            aContext.SetVariableEvent({{apMatrix, VariableDependency::WRITE}}, ev);
        }

        template<typename T>
        void HCoreKernels<T>::trsm(blas::Layout aLayout, blas::Side aSide, blas::Uplo aUplo, blas::Op aTrans,
                                   blas::Diag aDiag, size_t aRows, size_t aCols, T aAlpha, const T *apMatrixA,
                                   size_t aLeadingDimA, T *apMatrixB, size_t aLeadingDimB,
                                   const kernels::RunContext &aContext) {

            auto &q = aContext.GetQueue();

            oneapi::mkl::uplo upper_lower;
            if (aUplo == blas::Uplo::Upper) {
                upper_lower = oneapi::mkl::uplo::upper;
            } else {
                upper_lower = oneapi::mkl::uplo::lower;
            }

            oneapi::mkl::diag diagonal;
            if (aDiag == blas::Diag::Unit) {
                diagonal = oneapi::mkl::diag::unit;
            } else {
                diagonal = oneapi::mkl::diag::nonunit;
            }

            oneapi::mkl::side left_right;
            if (aSide == blas::Side::Left) {
                left_right = oneapi::mkl::side::left;
            } else {
                left_right = oneapi::mkl::side::right;
            }

            oneapi::mkl::transpose transpose;
            if (aTrans == blas::Op::Trans) {
                transpose = oneapi::mkl::transpose::trans;
            } else if (aTrans == blas::Op::ConjTrans) {
                transpose = oneapi::mkl::transpose::conjtrans;
            } else {
                transpose = oneapi::mkl::transpose::nontrans;
            }

            std::vector<DepPair> actions = {
                    {(void *) apMatrixA, VariableDependency::READ},
                    {(void *) apMatrixB, VariableDependency::READWRITE}
            };

            auto deps = aContext.GetVariableEvents(actions);

            sycl::event ev;
            if (aLayout == blas::Layout::ColMajor) {
                ev = oneapi::mkl::blas::column_major::trsm(q, left_right, upper_lower, transpose, diagonal, aRows,
                                                           aCols, aAlpha, apMatrixA, aLeadingDimA, apMatrixB,
                                                           aLeadingDimB, deps);
            } else if (aLayout == blas::Layout::RowMajor) {
                ev = oneapi::mkl::blas::row_major::trsm(q, left_right, upper_lower, transpose, diagonal, aRows, aCols,
                                                        aAlpha, apMatrixA, aLeadingDimB, apMatrixB, aLeadingDimB, deps);
            }

            aContext.SetVariableEvent(actions, ev);
        }

        template<typename T>
        void
        HCoreKernels<T>::syrk(blas::Layout aLayout, blas::Uplo aUplo, blas::Op aTrans, size_t aRows, size_t aCols,
                              T aAlpha, const T *apMatrixA, size_t aLeadingDimA, T aBeta, T *apMatrixB,
                              size_t aLeadingDimB, const RunContext &aContext) {

            auto &q = aContext.GetQueue();

            oneapi::mkl::uplo upper_lower;
            if (aUplo == blas::Uplo::Upper) {
                upper_lower = oneapi::mkl::uplo::upper;
            } else {
                upper_lower = oneapi::mkl::uplo::lower;
            }

            oneapi::mkl::transpose transpose;
            if (aTrans == blas::Op::Trans) {
                transpose = oneapi::mkl::transpose::trans;
            } else if (aTrans == blas::Op::ConjTrans) {
                transpose = oneapi::mkl::transpose::conjtrans;
            } else {
                transpose = oneapi::mkl::transpose::nontrans;
            }

            std::vector<DepPair> actions = {
                    {(void *) apMatrixA, VariableDependency::READ},
                    {(void *) apMatrixB, VariableDependency::READWRITE}
            };

            auto deps = aContext.GetVariableEvents(actions);

            sycl::event ev;
            if (aLayout == blas::Layout::ColMajor) {
                ev = oneapi::mkl::blas::column_major::syrk(q, upper_lower, transpose, aRows, aCols, aAlpha, apMatrixA,
                                                           aLeadingDimA, aBeta,
                                                           apMatrixB, aLeadingDimB, deps);
            } else if (aLayout == blas::Layout::RowMajor) {
                ev = oneapi::mkl::blas::row_major::syrk(q, upper_lower, transpose, aRows, aCols, aAlpha, apMatrixA,
                                                        aLeadingDimA, aBeta,
                                                        apMatrixB, aLeadingDimB, deps);
            }

            aContext.SetVariableEvent(actions, ev);
        }

        template<typename T>
        void HCoreKernels<T>::Symmetrize(blas::Layout aLayout, T *apMatrixA, size_t aRows, size_t aCols,
                                         blas::Uplo aUplo, const RunContext &aContext) {

            if (aRows != aCols) {
                return;
            }

            queue q = aContext.GetQueue();

            auto ev = q.submit([&](handler &h) {
                h.depends_on(aContext.GetVariableEvents({{apMatrixA, VariableDependency::WRITE}}));

                h.parallel_for(range < 2 > {(size_t) aRows, (size_t) aRows}, [=](id<2> idx) {
                    size_t i = idx[0];
                    size_t j = idx[1];
                    if (j > i) {
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
                });
            });

            aContext.SetVariableEvent({{apMatrixA, VariableDependency::WRITE}}, ev);
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

            queue q = aContext.GetQueue();
            std::vector<DepPair> actions = {
                    {(void *) aA,   VariableDependency::READ},
                    {(void *) aOut, VariableDependency::WRITE}
            };
            auto deps = aContext.GetVariableEvents(actions);
            size_t outer_loop_range = std::min(y, aLeadingDimA);
            size_t inner_loop_range = std::min(x, aLeadingDimOut);
            auto ev = q.submit([&](handler &h) {
                h.depends_on(deps);
                h.parallel_for(range < 2 > {(size_t) outer_loop_range, (size_t) inner_loop_range}, [=](id<2> idx) {
                    size_t i = idx[0];
                    size_t j = idx[1];
                    aOut[(size_t) i * aLeadingDimOut + j] = aA[(size_t) j * aLeadingDimA + i];
                });
            });
            aContext.SetVariableEvent(actions, ev);

        }

        template<typename T>
        size_t HCoreKernels<T>::CalculatePotrfWorkspaceSize(T *apMatrix, blas::Uplo aUplo, size_t aMatrixOrder,
                                                            size_t aLeadingDim, size_t &aHostSize,
                                                            const RunContext &aContext) {
            auto &q = aContext.GetQueue();

            oneapi::mkl::uplo upper_lower;
            if (aUplo == blas::Uplo::Upper) {
                upper_lower = oneapi::mkl::uplo::upper;
            } else {
                upper_lower = oneapi::mkl::uplo::lower;
            }

            aHostSize = 0;
            return oneapi::mkl::lapack::potrf_scratchpad_size<T>(q, upper_lower, aMatrixOrder,
                                                                 aLeadingDim);
        }

        HCOREPP_INSTANTIATE_CLASS(HCoreKernels)

    }
}
