/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <hcorepp/operators/concrete/Compressed.hpp>
#include <hcorepp/operators/helpers/CompressionParameters.hpp>
#include <hcorepp/kernels/kernels.hpp>
#include <hcorepp/kernels/memory.hpp>
#include "hcorepp/data-units/memory-handlers/MemoryHandler.hpp"
#include <hcorepp/helpers/DebuggingTimer.hpp>

using namespace hcorepp::dataunits;
using namespace hcorepp::common;
using namespace hcorepp::kernels;
using namespace hcorepp::helpers;

namespace hcorepp::operators {
    template<typename T>
    CompressedTile<T>::CompressedTile(size_t aNumOfRows, size_t aNumOfCols, T *apDataU, T *apDataV,
                                      size_t aLeadingDim, size_t aRank, blas::Layout aLayout,
                                      const kernels::RunContext &aContext) {
        this->mLayout = aLayout;
        this->mRank = aRank;
        this->mLeadingDim = aLeadingDim;
        this->mNumOfRows = aNumOfRows;
        this->mNumOfCols = aNumOfCols;
        this->mMaxRank = aRank;
        if (this->mLayout == blas::Layout::ColMajor) {
            mULeadingDim = this->mNumOfRows;
            mVLeadingDim = this->mRank;
        } else {
            mULeadingDim = mVLeadingDim = this->mLeadingDim;
        }
        auto numElements = (this->mNumOfRows * this->mMaxRank) + (this->mMaxRank * this->mNumOfCols);

        auto *UV = hcorepp::memory::AllocateArray<T>(numElements, aContext);
        hcorepp::memory::Memset(UV, 0, numElements, aContext);

        hcorepp::memory::Memcpy(UV, apDataU, this->mNumOfRows * this->mRank, aContext);
        hcorepp::memory::Memcpy(&UV[this->mNumOfRows * this->mMaxRank], apDataV, this->mRank * this->mNumOfCols,
                                aContext);

        this->mpDataArray = new DataHolder<T>(numElements, 1, numElements, UV, aContext);

        hcorepp::memory::DestroyArray(UV, aContext);
    }

    template<typename T>
    CompressedTile<T>::CompressedTile(size_t aNumOfRows, size_t aNumOfCols, T *apData, size_t aLeadingDim,
                                      size_t aRank, blas::Layout aLayout, const RunContext &aContext) {
        this->mLayout = aLayout;
        this->mRank = aRank;
        this->mLeadingDim = aLeadingDim;
        this->mNumOfRows = aNumOfRows;
        this->mNumOfCols = aNumOfCols;
        this->mMaxRank = aRank;
        auto numElements = (this->mNumOfRows * this->mMaxRank) + (this->mMaxRank * this->mNumOfCols);
        if (this->mLayout == blas::Layout::ColMajor) {
            mULeadingDim = this->mNumOfRows;
            mVLeadingDim = this->mRank;
        } else {
            mULeadingDim = mVLeadingDim = this->mLeadingDim;
        }

        this->mpDataArray = new DataHolder<T>(numElements, 1, numElements, nullptr, aContext);
        if (apData) {
            hcorepp::memory::Memcpy(this->GetUMatrix(), apData, this->mNumOfRows * this->mRank, aContext);
            hcorepp::memory::Memcpy(this->GetVMatrix(), &apData[this->mNumOfRows * this->mRank],
                                    this->mRank * this->mNumOfCols, aContext);
        }

    }

    template<typename T>
    CompressedTile<T>::CompressedTile(size_t aNumOfRows, size_t aNumOfCols, T *apData, size_t aLeadingDim,
                                      const CompressionParameters &aParameters, blas::Layout aLayout,
                                      const RunContext &aContext) {
        this->mLayout = aLayout;
        this->mLeadingDim = aLeadingDim;
        this->mNumOfRows = aNumOfRows;
        this->mNumOfCols = aNumOfCols;
        this->mMaxRank = std::max(std::min(aNumOfRows, aNumOfCols) / MAX_RANK_RATIO, 1UL);
        auto &context = aContext;
        auto numElements = (this->mNumOfRows * this->mMaxRank) + (this->mMaxRank * this->mNumOfCols);
        this->mRank = this->mMaxRank;
        this->mpDataArray = new DataHolder<T>(numElements, 1, numElements, nullptr, aContext);
        if (apData) {
            size_t rk;
            size_t min_m_n = std::min(aNumOfRows, aNumOfCols);
            auto sigma = hcorepp::memory::AllocateArray<blas::real_type<T>>(min_m_n, context);
            DataHolder<T> u_dataholder(aNumOfRows, min_m_n, aNumOfRows, nullptr, context);
            auto u = u_dataholder.GetData();
            DataHolder<T> vt_dataholder(min_m_n, aNumOfCols, min_m_n, nullptr, context);
            auto vt = vt_dataholder.GetData();

            DataHolder<T> a_temp_dataholder(aNumOfRows, aNumOfCols, aNumOfRows, apData, context);
            auto a_temp = a_temp_dataholder.GetData();
            hcorepp::kernels::HCoreKernels<T>::SVD(common::Job::SomeVec, common::Job::SomeVec,
                                                   aNumOfRows, aNumOfCols, a_temp, aNumOfRows, sigma, u, aNumOfRows,
                                                   vt,
                                                   min_m_n, aParameters.GetOperationType(), nullptr, 0, 0, context);
            rk = 0;
            if (aParameters.GetFixedRank()) {
                /// truncate according to fixed_rk
                rk = aParameters.GetFixedRank();
                if (aParameters.GetFixedRank() > min_m_n) {
                    rk = min_m_n;
                }
            } else { // truncate according to accuracy
                hcorepp::kernels::HCoreKernels<T>::CalculateNewRank(rk, aParameters.GetTruncatedSvd(), sigma, min_m_n,
                                                                    aParameters.GetAccuracy(), context);
            }
            // Ensure at least rank is 1.
            rk = std::max(rk, 1UL);

            if (rk > this->mMaxRank) {
                rk = this->mMaxRank;
            }
            // VT eats Sigma.
            hcorepp::kernels::HCoreKernels<T>::CalculateVTnew(rk, aParameters.GetUngqr(),
                                                              aNumOfCols, sigma, vt, min_m_n,
                                                              vt_dataholder.GetNumOfRows(),
                                                              context);
            // Prepare UV array.
            auto auv = hcorepp::memory::AllocateArray<T>((aNumOfRows + aNumOfCols) * rk, context);
            hcorepp::memory::Memcpy<T>(auv, u, (aNumOfRows * rk), context,
                                       memory::MemoryTransfer::DEVICE_TO_DEVICE);
            hcorepp::kernels::HCoreKernels<T>::LaCpy(common::MatrixType::General, rk, aNumOfCols, vt, min_m_n,
                                                     &auv[aNumOfRows * rk], rk, context);
            hcorepp::memory::DestroyArray(sigma, context);
            this->mRank = rk;
            hcorepp::memory::Memcpy(this->GetUMatrix(), auv, this->mNumOfRows * this->mRank, aContext);
            hcorepp::memory::Memcpy(this->GetVMatrix(), &auv[this->mNumOfRows * this->mRank],
                                    this->mRank * this->mNumOfCols, aContext);
            memory::DestroyArray(auv, context);
        }

        if (this->mLayout == blas::Layout::ColMajor) {
            mULeadingDim = this->mNumOfRows;
            mVLeadingDim = this->mRank;
        } else {
            mULeadingDim = mVLeadingDim = this->mLeadingDim;
        }

    }


    template<typename T>
    CompressedTile<T>::~CompressedTile<T>() {
        delete this->mpDataArray;
    }

    template<typename T>
    void CompressedTile<T>::DeleteData() {
        delete this->mpDataArray;
    }

    template<typename T>
    void CompressedTile<T>::ReadjustTileRank(size_t aRank, const RunContext &aContext) {

        if (aRank == -1) {
            this->mRank = std::min(this->mNumOfRows, this->mNumOfCols);
        } else {
            this->mRank = aRank;
        }
        if (this->mLayout == blas::Layout::ColMajor) {
            mULeadingDim = this->mNumOfRows;
            mVLeadingDim = this->mRank;
        } else {
            mULeadingDim = mVLeadingDim = this->mLeadingDim;
        }
    }

    template<typename T>
    T *CompressedTile<T>::GetUMatrix() const {
        return this->mpDataArray->GetData();
    }

    template<typename T>
    T *CompressedTile<T>::GetVMatrix() const {
        if (this->mpDataArray->GetData())
            return &(this->mpDataArray->GetData()[this->mNumOfRows * this->mMaxRank]);
        return nullptr;
    }

    template<typename T>
    T *CompressedTile<T>::GetTileSubMatrix(size_t aIndex) const {
        if (aIndex > 1) {
            throw std::invalid_argument(
                    "CompressedTile::GetTileSubMatrix:: Index out of range, should be 0 or 1 in case of compressed tile.\n");
        }
        if (aIndex == 0) return GetUMatrix();
        else return GetVMatrix();
    }

    template<typename T>
    [[nodiscard]] size_t CompressedTile<T>::GetTileStride(size_t aIndex) const {
        if (aIndex > 1) {
            throw std::invalid_argument(
                    "CompressedTile::GetTileStride::Index out of range, should be 0 or 1 in case of compressed tile.\n");
        }

        if (aIndex == 0) return mULeadingDim;
        else return mVLeadingDim;
    }

    template<typename T>
    void
    CompressedTile<T>::Gemm(T &aAlpha, DataHolder <T> const &aTileA, blas::Op aTileAOp,
                            DataHolder <T> const &aTileB,
                            blas::Op aTileBOp, T &aBeta, size_t aLdAu, size_t aARank,
                            const CompressionParameters &aHelpers, const RunContext &aContext, size_t &aFlops,
                            MemoryUnit <T> &aMemoryUnit, bool aCholesky) {
        size_t flops = 0;
        using blas::conj;

        T zero = 0.0;
        T one = 1.0;

        size_t m = this->mNumOfRows;
        size_t n = this->mNumOfCols;
        T *CU = this->GetUMatrix();
        size_t CU_leading_dim = mULeadingDim;
        bool memoryHandlerfree = false;

        T *CV = this->GetVMatrix();
        size_t CV_leading_dim = mVLeadingDim;

        size_t Crk = this->GetTileRank();

        blas::real_type<T> accuracy = aHelpers.GetAccuracy();

        size_t Um = m;
        size_t Un = aARank + Crk;
        size_t min_Um_Un = std::min(Um, Un);

        size_t Vm = n;
        size_t Vn = aARank + Crk;

        size_t min_Vm_Vn = std::min(Vm, Vn);

        size_t max_rows;
        size_t max_cols;
        if (aHelpers.GetUngqr()) {
            max_rows = min_Um_Un;
            max_cols = min_Vm_Vn;
        } else {
            max_rows = Um;
            max_cols = Vm;
        }

        size_t sizeS;
        if (aHelpers.GetTrmm()) {
            sizeS = min_Um_Un;
        } else {
            sizeS = std::min(m, n);
            sizeS = std::min(sizeS, (aARank + Crk));
        }

        size_t ldcu = CU_leading_dim;

        DebuggingTimer *timer = DebuggingTimer::GetDebuggingTimer();
        size_t host_size = 0;

        size_t U_size = Um * Un;
        size_t Utau_size = min_Um_Un;
        size_t RU_size = min_Um_Un * Un;
        size_t V_size = Vm * Vn;
        size_t Vtau_size = min_Vm_Vn;
        size_t Unew_size = max_rows * sizeS;
        size_t VTnew_size = sizeS * max_cols;
        if (aCholesky) {
            Unew_size = U_size;
            VTnew_size = std::max(this->mNumOfCols, this->mNumOfRows) * this->mNumOfRows;
        }
        size_t UV_size = (ldcu + n) * sizeS;
        size_t workspace_size = hcorepp::kernels::HCoreKernels<T>::CalculateGemmWorkspaceSize(Um, Un, Vm, Vn,
                                                                                              sizeS,
                                                                                              aHelpers, host_size,
                                                                                              aContext);
        size_t RV_size = 0;
        size_t RURV_size = 0;
        size_t Vnew_size = 0;

        size_t total_size = U_size + Utau_size + RU_size + V_size + Vtau_size + Unew_size + VTnew_size + UV_size +
                            workspace_size;

        if (!aHelpers.GetTrmm()) {
            RV_size = min_Vm_Vn * Vn;
            RURV_size = min_Um_Un * min_Vm_Vn;
            total_size += RV_size + RURV_size;
        }

        if (aHelpers.GetUngqr()) {
            Vnew_size = Vm * sizeS;
            total_size += Vnew_size;
        }

        if (!aMemoryUnit.IsInitialized()) {
            memoryHandlerfree = true;
            timer->StartSnapshot("CompressedTile::Gemm::AllocatingTempBuffers");
            aMemoryUnit.Initialize(total_size);
            timer->Snapshot("CompressedTile::Gemm::AllocatingTempBuffers");
        }

        T *U = aMemoryUnit.RequestAllocation(U_size);
        T *Utau = aMemoryUnit.RequestAllocation(Utau_size);
        T *RU = aMemoryUnit.RequestAllocation(RU_size);
        T *V = aMemoryUnit.RequestAllocation(V_size);
        T *Vtau = aMemoryUnit.RequestAllocation(Vtau_size);
        T *Unew = aMemoryUnit.RequestAllocation(Unew_size);
        T *VTnew = aMemoryUnit.RequestAllocation(VTnew_size);
        T *UV = aMemoryUnit.RequestAllocation(UV_size);
        T *workspace = nullptr;
        if (workspace_size > 0)
            workspace = aMemoryUnit.RequestAllocation(workspace_size);
        T *RV = nullptr;
        T *RURV = nullptr;
        T *Vnew = nullptr;

        if (!aHelpers.GetTrmm()) {
            RV = aMemoryUnit.RequestAllocation(RV_size);
            RURV = aMemoryUnit.RequestAllocation(RURV_size);
            if (aHelpers.GetUngqr()) {
                Vnew = aMemoryUnit.RequestAllocation(Vnew_size);
            }
        } else if (aHelpers.GetUngqr()) {
            Vnew = aMemoryUnit.RequestAllocation(Vnew_size);
        }
        timer->StartSnapshot("CompressedTile::Gemm::LaCpy#1");
        hcorepp::kernels::HCoreKernels<T>::LaCpy(MatrixType::General, m, Crk, CU, ldcu, U, Um, aContext);
        timer->Snapshot("CompressedTile::Gemm::LaCpy#1");

        flops += (m * Crk);

        timer->StartSnapshot("CompressedTile::Gemm::LaCpy#2");
        hcorepp::kernels::HCoreKernels<T>::LaCpy(MatrixType::General, m, aARank, (T *) aTileA.GetData(), aLdAu,
                                                 &U[m * Crk], Um, aContext);
        timer->Snapshot("CompressedTile::Gemm::LaCpy#2");

        flops += (m * aARank);

        timer->StartSnapshot("CompressedTile::Gemm::MultiplyByAlpha");
        hcorepp::kernels::HCoreKernels<T>::MultiplyByAlpha(U, aTileA.GetNumOfRows(), aTileA.GetNumOfCols(), m, Crk,
                                                           aAlpha, aContext);
        timer->Snapshot("CompressedTile::Gemm::MultiplyByAlpha");

        flops += (aTileA.GetNumOfRows() * aTileA.GetNumOfCols());

        timer->StartSnapshot("CompressedTile::Gemm::Geqrf#1");

        hcorepp::kernels::HCoreKernels<T>::Geqrf(Um, Un, U, Um, Utau, workspace, workspace_size, host_size,
                                                 aContext);

        timer->Snapshot("CompressedTile::Gemm::Geqrf#1");

        flops += (2 * Um * Un * Un - (size_t) (2.0 * (float) Un * (float) Un * (float) Un / 3.0f) +
                  2 * Um * Un +
                  (size_t) (17 * (float) Un / 3.0f));

        timer->StartSnapshot("CompressedTile::Gemm::Laset#1");
        hcorepp::kernels::HCoreKernels<T>::Laset(MatrixType::Lower, min_Um_Un, Un, zero, zero, RU, min_Um_Un,
                                                 aContext);
        timer->Snapshot("CompressedTile::Gemm::Laset#1");

        flops += (min_Um_Un * ((Un + 1) / 2));

        timer->StartSnapshot("CompressedTile::Gemm::LaCpy#3");
        hcorepp::kernels::HCoreKernels<T>::LaCpy(MatrixType::Upper, min_Um_Un, Un, U, Um, RU, min_Um_Un, aContext);
        timer->Snapshot("CompressedTile::Gemm::LaCpy#3");

        flops += (min_Um_Un * ((Un + 1) / 2));

        size_t ldcv = CV_leading_dim;
        timer->StartSnapshot("CompressedTile::Gemm::ProcessVpointer");

        hcorepp::kernels::HCoreKernels<T>::ProcessVpointer(n, Crk, aHelpers.GetUngqr(), Vm, aBeta, CV, ldcv, V,
                                                           aARank, aTileB.GetData(), aContext, aCholesky);
        timer->Snapshot("CompressedTile::Gemm::ProcessVpointer");

        flops += (n * Crk + n * aARank);
        timer->StartSnapshot("CompressedTile::Gemm::Geqrf#2");
        hcorepp::kernels::HCoreKernels<T>::Geqrf(Vm, Vn, V, Vm, Vtau, workspace, workspace_size, host_size,
                                                 aContext);
        timer->Snapshot("CompressedTile::Gemm::Geqrf#2");

        flops += (2 * Vm * Vn * Vn - (size_t) (2 * (float) Vn * (float) Vn * (float) Vn / 3.0f) +
                  2 * Vm * Vn +
                  (size_t) (17 * (float) Vn / 3.0f));

        blas::real_type<T> *Sigma;

        timer->StartSnapshot("CompressedTile::Gemm::AllocatingTempBuffers");
        Sigma = hcorepp::memory::AllocateArray<blas::real_type<T>>(sizeS, aContext);
        timer->Snapshot("CompressedTile::Gemm::AllocatingTempBuffers");

        if (aHelpers.GetTrmm()) {

            if (aHelpers.GetUngqr()) {

                timer->StartSnapshot("CompressedTile::Gemm::Trmm#1");
                hcorepp::kernels::HCoreKernels<T>::Trmm(blas::Layout::ColMajor, blas::Side::Right,
                                                        blas::Uplo::Upper, blas::Op::ConjTrans,
                                                        blas::Diag::NonUnit, min_Um_Un, Un,
                                                        one, V, Vm, RU,
                                                        min_Um_Un, aContext);
                timer->Snapshot("CompressedTile::Gemm::Trmm#1");

                flops += (min_Um_Un * Un * Un);

                timer->StartSnapshot("CompressedTile::Gemm::SVD#1");
                hcorepp::kernels::HCoreKernels<T>::SVD(Job::SomeVec, Job::SomeVec,
                                                       min_Um_Un, Un, RU, min_Um_Un, Sigma, Unew, min_Um_Un,
                                                       VTnew, sizeS, aHelpers.GetOperationType(), workspace,
                                                       workspace_size, host_size, aContext);
                timer->Snapshot("CompressedTile::Gemm::SVD#1");

                flops += (22 * min_Um_Un * min_Um_Un * min_Um_Un);
            } else {
                timer->StartSnapshot("CompressedTile::Gemm::Trmm#2");
                hcorepp::kernels::HCoreKernels<T>::Trmm(blas::Layout::ColMajor, blas::Side::Right,
                                                        blas::Uplo::Upper,
                                                        blas::Op::Trans, blas::Diag::NonUnit,
                                                        min_Um_Un, Un, one, V, Vm, RU, min_Um_Un, aContext);
                timer->Snapshot("CompressedTile::Gemm::Trmm#2");

                flops += (min_Um_Un * Un * Un);

                timer->StartSnapshot("CompressedTile::Gemm::SVD#2");
                hcorepp::kernels::HCoreKernels<T>::SVD(Job::SomeVec, Job::SomeVec,
                                                       min_Um_Un, Un, RU, min_Um_Un,
                                                       Sigma, Unew, Um, VTnew, sizeS, aHelpers.GetOperationType(),
                                                       workspace, workspace_size, host_size, aContext);
                timer->Snapshot("CompressedTile::Gemm::SVD#2");

                flops += (22 * min_Um_Un * min_Um_Un * min_Um_Un);
            }
        } else {

            timer->StartSnapshot("CompressedTile::Gemm::Laset#2");
            hcorepp::kernels::HCoreKernels<T>::Laset(MatrixType::Lower, min_Vm_Vn, Vn, zero, zero,
                                                     RV, min_Vm_Vn, aContext);
            timer->Snapshot("CompressedTile::Gemm::Laset#2");

            flops += (min_Vm_Vn * ((Vn + 1) / 2));
            timer->StartSnapshot("CompressedTile::Gemm::LaCpy#4");
            hcorepp::kernels::HCoreKernels<T>::LaCpy(MatrixType::Upper, min_Vm_Vn, Vn, V, Vm, RV, min_Vm_Vn,
                                                     aContext);
            timer->Snapshot("CompressedTile::Gemm::LaCpy#4");

            flops += (min_Vm_Vn * ((Vn + 1) / 2));

            if (aHelpers.GetUngqr()) {
                timer->StartSnapshot("CompressedTile::Gemm::Gemm#1");
                hcorepp::kernels::HCoreKernels<T>::Gemm(blas::Layout::ColMajor,
                                                        blas::Op::NoTrans,
                                                        (aCholesky) ? blas::Op::Trans : blas::Op::ConjTrans,
                                                        min_Um_Un, min_Vm_Vn, (aARank + Crk),
                                                        one, RU, min_Um_Un, RV, min_Vm_Vn,
                                                        zero, RURV, min_Um_Un, aContext);
                timer->Snapshot("CompressedTile::Gemm::Gemm#1");

                flops += (min_Um_Un * min_Vm_Vn * (aARank + Crk));

                timer->StartSnapshot("CompressedTile::Gemm::SVD#3");

                if (aCholesky) {
                    size_t ld_Unew = std::max(std::max(this->mNumOfCols, this->mNumOfRows), m);
                    size_t ld_VTnew = std::max(this->mNumOfCols, this->mNumOfRows);

                    hcorepp::kernels::HCoreKernels<T>::SVD(Job::AllVec, Job::AllVec, min_Um_Un,
                                                           min_Vm_Vn, RURV, min_Um_Un, Sigma, Unew, ld_Unew,
                                                           VTnew, ld_VTnew, aHelpers.GetOperationType(), workspace,
                                                           workspace_size, host_size, aContext);
                } else {
                    hcorepp::kernels::HCoreKernels<T>::SVD(Job::SomeVec, Job::SomeVec, min_Um_Un,
                                                           min_Vm_Vn, RURV, min_Um_Un, Sigma, Unew, min_Um_Un,
                                                           VTnew, sizeS, aHelpers.GetOperationType(), workspace,
                                                           workspace_size, host_size, aContext);
                }
                timer->Snapshot("CompressedTile::Gemm::SVD#3");

                flops += (22 * min_Vm_Vn * min_Vm_Vn * min_Vm_Vn);

            } else {
                timer->StartSnapshot("CompressedTile::Gemm::Gemm#2");
                hcorepp::kernels::HCoreKernels<T>::Gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                                                        blas::Op::Trans, min_Um_Un, min_Vm_Vn,
                                                        (aARank + Crk), one, RU, min_Um_Un, RV,
                                                        min_Vm_Vn, zero, RURV, min_Um_Un, aContext);
                timer->Snapshot("CompressedTile::Gemm::Gemm#2");

                flops += (min_Um_Un * min_Vm_Vn * (aARank + Crk));

                timer->StartSnapshot("CompressedTile::Gemm::SVD#4");
                hcorepp::kernels::HCoreKernels<T>::SVD(Job::SomeVec, Job::SomeVec,
                                                       min_Um_Un, min_Vm_Vn, RURV,
                                                       min_Um_Un, Sigma, Unew, Um, VTnew, sizeS,
                                                       aHelpers.GetOperationType(), workspace, workspace_size,
                                                       host_size, aContext);
                timer->Snapshot("CompressedTile::Gemm::SVD#4");

                flops += (22 * min_Um_Un * min_Um_Un * min_Um_Un);

            }
        }

        size_t rk_new;
        if (aHelpers.GetFixedRank()) {
            /// truncate according to fixed_rk
            rk_new = aHelpers.GetFixedRank();
            if (aHelpers.GetFixedRank() > (aARank + Crk)) {
                rk_new = (aARank + Crk);
            }
        } else { // truncate according to accuracy

            timer->StartSnapshot("CompressedTile::Gemm::CalculateNewRank");
            hcorepp::kernels::HCoreKernels<T>::CalculateNewRank(rk_new, aHelpers.GetTruncatedSvd(), Sigma,
                                                                sizeS, accuracy, aContext);
            timer->Snapshot("CompressedTile::Gemm::CalculateNewRank");
            flops += sizeS;
        }


        rk_new = std::max(rk_new, 1UL);
        rk_new = std::min(rk_new, this->mMaxRank);

        if (aCholesky) {
            hcorepp::kernels::HCoreKernels<T>::CalculateVTnew(rk_new, true, sizeS, Sigma, VTnew,
                                                              Vm, Vm, aContext);
        }

        if (aCholesky) {
            size_t ld_Unew = std::max(this->mMaxRank, m);
            size_t nrows = this->mNumOfRows - Un;
            hcorepp::kernels::HCoreKernels<T>::Laset(MatrixType::General, nrows, rk_new, zero, zero,
                                                     &Unew[Un], ld_Unew,
                                                     aContext);

            hcorepp::kernels::HCoreKernels<T>::Unmqr(SideMode::SIDE_LEFT, BlasOperation::OP_NoTRANS,
                                                     Um, rk_new, min_Um_Un, U, Um, Utau, Unew, ld_Unew, workspace,
                                                     workspace_size, aContext);

            hcorepp::kernels::HCoreKernels<T>::LaCpy(MatrixType::General, Um, rk_new, Unew, ld_Unew, UV, ldcu,
                                                     aContext);

        } else {
            if (aHelpers.GetUngqr()) {
                timer->StartSnapshot("CompressedTile::Gemm::ungqr#1");
                hcorepp::kernels::HCoreKernels<T>::ungqr(Um, min_Um_Un, min_Um_Un, U, Um, Utau, workspace,
                                                         workspace_size, aContext);
                timer->Snapshot("CompressedTile::Gemm::ungqr#1");

                flops += ((8 / 3) * (size_t) pow((double) min_Um_Un, 2) * (3 * Um - min_Um_Un));

                timer->StartSnapshot("CompressedTile::Gemm::Gemm#3");
                hcorepp::kernels::HCoreKernels<T>::Gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                                                        blas::Op::NoTrans, Um, rk_new, min_Um_Un, one, U,
                                                        Um, Unew, min_Um_Un, zero, UV, ldcu, aContext);
                timer->Snapshot("CompressedTile::Gemm::Gemm#3");

                flops += (Um * rk_new * min_Um_Un);

            } else {
                timer->StartSnapshot("CompressedTile::Gemm::unmqr#1");
                hcorepp::kernels::HCoreKernels<T>::Unmqr(SideMode::SIDE_LEFT, BlasOperation::OP_NoTRANS,
                                                         Um, rk_new, min_Um_Un, U, Um, Utau, Unew, Um, workspace,
                                                         workspace_size, aContext);
                timer->Snapshot("CompressedTile::Gemm::unmqr#1");

                flops += (4 * rk_new * Um * min_Um_Un - 2 * rk_new * min_Um_Un * min_Um_Un + 3 * rk_new * min_Um_Un);

                timer->StartSnapshot("CompressedTile::Gemm::LaCpy#5");
                hcorepp::kernels::HCoreKernels<T>::LaCpy(MatrixType::General, Um, rk_new, Unew, Um, UV, ldcu, aContext);
                timer->Snapshot("CompressedTile::Gemm::LaCpy#5");

                flops += (Um * rk_new);

            }
        }

        if (aCholesky) {
            auto n_cols = this->mNumOfRows - Vn;
            size_t offset = Vn * Vm;

            hcorepp::kernels::HCoreKernels<T>::Laset(MatrixType::General, rk_new, n_cols, zero, zero,
                                                     &VTnew[offset], Vm, aContext);

            hcorepp::kernels::HCoreKernels<T>::Unmqr(SideMode::SIDE_RIGHT, BlasOperation::OP_TRANS,
                                                     rk_new, Vm, min_Vm_Vn, V, Vm, Vtau, VTnew, Vm, workspace,
                                                     workspace_size, aContext);

            hcorepp::kernels::HCoreKernels<T>::transpose(blas::Layout::ColMajor, rk_new, Vm, VTnew, Vm, V, Vm,
                                                         aContext);
        } else {
            timer->StartSnapshot("CompressedTile::Gemm::CalculateVTnew");
            hcorepp::kernels::HCoreKernels<T>::CalculateVTnew(rk_new, aHelpers.GetUngqr(), min_Vm_Vn, Sigma, VTnew,
                                                              sizeS, Vm, aContext);
            timer->Snapshot("CompressedTile::Gemm::CalculateVTnew");
            if (aHelpers.GetUngqr()) {
                flops += rk_new;
            } else {
                flops += (rk_new * Vm);
            }

            T *UVptr = UV + ldcu * rk_new;

            if (aHelpers.GetUngqr()) {
                timer->StartSnapshot("CompressedTile::Gemm::ungqr#2");
                hcorepp::kernels::HCoreKernels<T>::ungqr(Vm, min_Vm_Vn, min_Vm_Vn, V, Vm, Vtau, workspace,
                                                         workspace_size, aContext);
                timer->Snapshot("CompressedTile::Gemm::ungqr#2");


                flops += ((8 / 3) * (size_t) pow((double) min_Vm_Vn, 2) * (3 * Vm - min_Vm_Vn));

                timer->StartSnapshot("CompressedTile::Gemm::Gemm#4");
                hcorepp::kernels::HCoreKernels<T>::Gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
                                                        blas::Op::ConjTrans, Vm, rk_new,
                                                        min_Vm_Vn,
                                                        one, V, Vm, VTnew, sizeS, zero, Vnew, Vm, aContext);
                timer->Snapshot("CompressedTile::Gemm::Gemm#4");

                flops += (Vm * rk_new * min_Vm_Vn);

                timer->StartSnapshot("CompressedTile::Gemm::CalculateUVptr");
                hcorepp::kernels::HCoreKernels<T>::CalculateUVptr(rk_new, Vm, UVptr, Vnew, aContext);
                timer->Snapshot("CompressedTile::Gemm::CalculateUVptr");
            } else {
                timer->StartSnapshot("CompressedTile::Gemm::unmqr#2");
                hcorepp::kernels::HCoreKernels<T>::Unmqr(SideMode::SIDE_RIGHT, BlasOperation::OP_CONJG,
                                                         rk_new, Vm, min_Vm_Vn, V, Vm, Vtau, VTnew, sizeS, workspace,
                                                         workspace_size, aContext);
                timer->Snapshot("CompressedTile::Gemm::unmqr#2");

                flops += (4 * Vm * rk_new * min_Vm_Vn - 2 * rk_new * min_Vm_Vn * min_Vm_Vn + 2 * rk_new * min_Vm_Vn +
                          Vm * min_Vm_Vn - min_Vm_Vn * min_Vm_Vn / 2 + min_Vm_Vn / 2);

                timer->StartSnapshot("CompressedTile::Gemm::LaCpy#6");
                hcorepp::kernels::HCoreKernels<T>::LaCpy(MatrixType::General, rk_new, Vm, VTnew, sizeS, UVptr,
                                                         rk_new, aContext);
                timer->Snapshot("CompressedTile::Gemm::LaCpy#6");

                flops += (rk_new * Vm);

                timer->StartSnapshot("CompressedTile::Gemm::CalculateUVptrConj");
                hcorepp::kernels::HCoreKernels<T>::CalculateUVptrConj(rk_new, Vm, UVptr, aContext);
                timer->Snapshot("CompressedTile::Gemm::CalculateUVptrConj");

                flops += (rk_new * Vm);
            }
        }
        hcorepp::memory::DestroyArray(Sigma, aContext);

        this->mRank = rk_new;
        if (this->mLayout == blas::Layout::ColMajor) {
            mULeadingDim = this->mNumOfRows;
            mVLeadingDim = this->mRank;
        } else {
            mULeadingDim = mVLeadingDim = this->mLeadingDim;
        }
        size_t u_size =
                this->mNumOfRows * this->mRank;
        size_t v_size =
                this->mRank * this->mNumOfCols;


        if (aCholesky) {
            hcorepp::memory::Memcpy<T>(this->GetUMatrix(), Unew,
                                       u_size, aContext, memory::MemoryTransfer::DEVICE_TO_DEVICE);
            hcorepp::memory::Memcpy<T>(this->GetVMatrix(), V,
                                       v_size, aContext, memory::MemoryTransfer::DEVICE_TO_DEVICE);
        } else {
            timer->StartSnapshot("CompressedTile::Gemm::Memcpy#1");
            hcorepp::memory::Memcpy<T>(this->GetUMatrix(), UV, u_size,
                                       aContext, memory::MemoryTransfer::DEVICE_TO_DEVICE);
            timer->Snapshot("CompressedTile::Gemm::Memcpy#1");
            timer->StartSnapshot("CompressedTile::Gemm::Memcpy#2");

            hcorepp::memory::Memcpy<T>(this->GetVMatrix(), &UV[u_size],
                                       v_size, aContext, memory::MemoryTransfer::DEVICE_TO_DEVICE);
            timer->Snapshot("CompressedTile::Gemm::Memcpy#2");
        }
        aFlops += flops;

        if (memoryHandlerfree) {
            aMemoryUnit.FreeAllocations();

        } else {
            aMemoryUnit.Reset();
        }

    }

    template<typename T>
    void
    CompressedTile<T>::ReadjustTile(size_t aNumOfRows, size_t aNumOfCols, T *aPdata, size_t aLeadingDim,
                                    size_t aRank, const RunContext &aContext) {
        if (aRank == -1) {
            this->mRank = std::min(aNumOfRows, aNumOfCols);
        } else {
            this->mRank = aRank;
        }

        this->mLeadingDim = aLeadingDim;
        this->mNumOfRows = aNumOfRows;
        this->mNumOfCols = aNumOfCols;
        this->mMaxRank = this->mRank;
        auto numElements = (this->mNumOfRows * this->mMaxRank) + (this->mMaxRank * this->mNumOfCols);
        if (this->mLayout == blas::Layout::ColMajor) {
            mULeadingDim = this->mNumOfRows;
            mVLeadingDim = this->mRank;
        } else {
            mULeadingDim = mVLeadingDim = this->mLeadingDim;
        }

        this->GetDataHolder().get().Resize(numElements, 1, 1);

        size_t u_size = this->mNumOfRows * this->mRank;
        size_t v_size = this->mRank * this->mNumOfCols;
        hcorepp::memory::Memcpy<T>(this->GetUMatrix(), aPdata, u_size, aContext,
                                   memory::MemoryTransfer::DEVICE_TO_DEVICE);

        if (aRank == -1) {
            T *identity_matrix = this->GetVMatrix();

            hcorepp::kernels::HCoreKernels<T>::FillIdentityMatrix(this->mNumOfCols, identity_matrix, aContext);

        } else {
            hcorepp::memory::Memcpy<T>(this->GetVMatrix(),
                                       &aPdata[u_size], v_size, aContext,
                                       memory::MemoryTransfer::DEVICE_TO_DEVICE);
        }

//            if (aRank == -1)  {
//                this->mRank = std::min(aNumOfRows, aNumOfCols);
//            } else {
//                this->mRank = aRank;
//            }
//            this->mLeadingDim = aLeadingDim;
//            this->mNumOfRows = aNumOfRows;
//            this->mNumOfCols = aNumOfCols;
//            if(this->mLayout == blas::Layout::ColMajor) {
//                mULeadingDim = this->mNumOfRows;
//                mVLeadingDim = this->mRank;
//            }
//            else {
//                mULeadingDim = mVLeadingDim = this->mLeadingDim;
//            }
//
//            size_t u_size = this->mNumOfRows * this->mRank;
//            size_t v_size = this->mRank * this->mNumOfCols;
//            hcorepp::memory::Memcpy<T>(this->GetUMatrix(), aPdata, u_size,
//                                       aContext, memory::MemoryTransfer::DEVICE_TO_DEVICE);
//
//            if (aRank == -1) {
//                T *identity_matrix = this->GetVMatrix();
//
//                hcorepp::kernels::HCoreKernels<T>::FillIdentityMatrix(this->mNumOfCols, identity_matrix, aContext);
//
//            } else {
//                hcorepp::memory::Memcpy<T>(this->GetVMatrix(),
//                                           &aPdata[this->mNumOfRows * this->mMaxRank], v_size, aContext,
//                                           memory::MemoryTransfer::DEVICE_TO_DEVICE);
//            }
    }

    template<typename T>
    std::pair<TileMetadata *, T *> CompressedTile<T>::UnPackTile(const RunContext &aContext) {

        std::vector<T *> data_vector;

        auto *metadata = new TileMetadata(this->mNumOfRows, this->mNumOfCols, this->mRank, this->mMaxRank,
                                          this->mLeadingDim,
                                          this->mLayout, COMPRESSED);

        return {metadata, this->mpDataArray->GetData()};
    }

    template<typename T>
    void
    CompressedTile<T>::PackTile(TileMetadata aMetadata, T *aDataArray, const RunContext &aContext) {

        this->UpdateMetadata(aMetadata);
        auto numElements = (this->mNumOfRows * this->mMaxRank) + (this->mMaxRank * this->mNumOfCols);

        this->mpDataArray = new DataHolder<T>(numElements, 1, numElements, aDataArray, aContext, false);
    }

    template<typename T>
    void CompressedTile<T>::UpdateMetadata(TileMetadata aMetadata) {
        this->mLayout = aMetadata.mLayout;
        this->mRank = aMetadata.mMatrixRank;
        this->mLeadingDim = aMetadata.mLeadingDimension;
        this->mNumOfRows = aMetadata.mNumOfRows;
        this->mNumOfCols = aMetadata.mNumOfCols;
        this->mMaxRank = aMetadata.mMaxRank;
        if (this->mLayout == blas::Layout::ColMajor) {
            mULeadingDim = this->mNumOfRows;
            mVLeadingDim = this->mRank;
        } else {
            mULeadingDim = mVLeadingDim = this->mLeadingDim;
        }
    }

    HCOREPP_INSTANTIATE_CLASS(CompressedTile)
}
