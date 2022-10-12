#include <functional>
#include <iostream>
//#include "lapack/wrappers.hh"
#include <hcorepp/operators/helpers/SvdHelpers.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>
#include <hcorepp/kernels/kernels.hpp>

using namespace hcorepp::dataunits;
using namespace hcorepp::helpers;

namespace hcorepp {
    namespace operators {

        template<typename T>
        CompressedTile<T>::CompressedTile() {

        }

        template<typename T>
        CompressedTile<T>::CompressedTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                                          int64_t aRank, blas::real_type<T> aAccuracy, blas::Layout aLayout,
                                          blas::Op aOperation, blas::Uplo aUplo) {
            if (aRank < 0 || aAccuracy < 0) {
            }
            this->mOperation = aOperation;
            this->mLayout = aLayout;
            this->mUpLo = aUplo;
            this->mAccuracy = aAccuracy;
            this->mMatrixRank = aRank;
            this->mLeadingDim = aLeadingDim;
            this->mNumOfRows = aNumOfRows;
            this->mNumOfCols = aNumOfCols;

            ///     The m-by-n matrix compressed tile (A=UV): A is m-by-n, U is m-by-rk, and V is rk-by-n.
            if (this->mLayout == blas::Layout::ColMajor) {
                /**     If layout = blas::Layout::ColMajor,
                 * the data array of A is stored in an ld-by-n array buffer; the
                 * data array of U is stored in an ld-by-rk array buffer; and
                 * data array of V is stored in an rk-by-n array buffer.
                 */
                this->mDataArrays.push_back(
                        new DataHolder<T>(this->mNumOfRows, this->mMatrixRank, this->mNumOfRows, aPdata));
                this->mDataArrays.push_back(new DataHolder<T>(this->mMatrixRank, this->mNumOfCols, this->mMatrixRank,
                                                              aPdata + aNumOfRows * this->mMatrixRank));

            } else {
                /** layout = blas::Layout::RowMajor: the data array of A is stored in an
                 * m-by-ld array buffer, the data array of U is stored in an
                 * m-by-rk array buffer, and data array of V is stored in an
                 * rk-by-ld array buffer.
                 */
                this->mDataArrays.push_back(new DataHolder<T>(aNumOfRows, this->mMatrixRank, aLeadingDim, aPdata));
                this->mDataArrays.push_back(new DataHolder<T>(this->mMatrixRank, this->mLeadingDim, aLeadingDim,
                                                              aPdata + aNumOfRows * this->mMatrixRank));
            }
        }

        template<typename T>
        CompressedTile<T>::~CompressedTile<T>() {
            for (auto data_holder: this->mDataArrays) {
                delete data_holder;
            }
        }

        template<typename T>
        std::reference_wrapper<dataunits::DataHolder<T>> CompressedTile<T>::GetTileSubMatrix(size_t aIndex) {
            if (aIndex > 1 || aIndex < 0) {
                throw std::invalid_argument(
                        "CompressedTile::GetTileSubMatrix:: Index out of range, should be 0 or 1 in case of compressed tile.\n");
            }
            return *mDataArrays[aIndex];
        }

        template<typename T>
        const std::reference_wrapper<dataunits::DataHolder<T>>
        CompressedTile<T>::GetTileSubMatrix(size_t aIndex) const {
            if (aIndex > 1 || aIndex < 0) {
                throw std::invalid_argument(
                        "CompressedTile::GetTileSubMatrix:: Index out of range, should be 0 or 1 in case of compressed tile.\n");
            }
            return *mDataArrays[aIndex];
        }

        template<typename T>
        int64_t CompressedTile<T>::GetTileStride(size_t aIndex) const {
            if (aIndex > 1 || aIndex < 0) {
                throw std::invalid_argument(
                        "CompressedTile::GetTileStride::Index out of range, should be 0 or 1 in case of compressed tile.\n");
            }

            if (aIndex == 0) {
                if (this->mLayout == blas::Layout::ColMajor) {
                    return this->mLeadingDim;
                } else {
                    return this->mMatrixRank;
                }
            } else if (aIndex == 1) {
                if (this->mLayout == blas::Layout::ColMajor) {
                    return this->mMatrixRank;
                } else {
                    return this->mLeadingDim;
                }
            }
            return 0;
        }

        template<typename T>
        void
        CompressedTile<T>::Gemm(T &aAlpha, DataHolder<T> const &aTileA, blas::Op aTileAOp, DataHolder<T> const &aTileB,
                                blas::Op aTileBOp, T &aBeta, int64_t aLdAu, int64_t aARank,
                                const SvdHelpers &aHelpers) {
            using blas::conj;

            T zero = 0.0;
            T one = 1.0;

            int64_t m = this->GetNumOfRows();
            int64_t n = this->GetNumOfCols();

            T *CU = this->GetTileSubMatrix(0).get().GetData();
            size_t CU_leading_dim = this->GetTileSubMatrix(0).get().GetLeadingDim();

            T *CV = this->GetTileSubMatrix(1).get().GetData();
            size_t CV_leading_dim = this->GetTileSubMatrix(1).get().GetLeadingDim();

            int64_t Crk = this->GetTileRank();

            blas::real_type<T> accuracy = this->GetAccuracy();

            int64_t Um = m;
            int64_t Un = aARank + Crk;

            int64_t ldcu = CU_leading_dim;

            auto U_dataholder = new DataHolder<T>(Um, Un, Um);

            T *U = U_dataholder->GetData();

            hcorepp::kernels::LaCpy(lapack::MatrixType::General, m, Crk, CU, ldcu, U, Um);

            hcorepp::kernels::LaCpy(lapack::MatrixType::General, m, aARank, (T *) aTileA.GetData(), aLdAu, &U[m * Crk],
                                    Um);

            hcorepp::kernels::MultiplyByAlpha(U, aTileA.GetNumOfRows(), aTileA.GetNumOfCols(), m, Crk, aAlpha);

            int64_t min_Um_Un = std::min(Um, Un);

            auto Utau_dataholder = new DataHolder<T>(min_Um_Un, 1, min_Um_Un);
            T *Utau = Utau_dataholder->GetData();

            hcorepp::kernels::Geqrf(Um, Un, U, Um, Utau);

            auto ru_dataholder = new DataHolder<T>(min_Um_Un, Un, min_Um_Un);
            auto RU = ru_dataholder->GetData();

            hcorepp::kernels::Laset(lapack::MatrixType::Lower, min_Um_Un, Un, zero, zero, RU, min_Um_Un);
            hcorepp::kernels::LaCpy(lapack::MatrixType::Upper,
                                    min_Um_Un, Un, U, Um, RU, min_Um_Un);

            int64_t Vm = n;
            int64_t Vn = aARank + Crk;

            int64_t ldcv = CV_leading_dim;

            auto V_dataholder = new DataHolder<T>(Vm, Vn, Vm);

            T *V = V_dataholder->GetData();

            hcorepp::kernels::ProcessVpointer<T>(n, Crk, aHelpers.GetUngqr(), Vm, aBeta, CV, ldcv, V,
                                                 aARank, aTileB.GetData());

            int64_t min_Vm_Vn = std::min(Vm, Vn);

            auto Vtau_dataholder = new DataHolder<T>(min_Vm_Vn, 1, min_Vm_Vn);
            T *Vtau = Vtau_dataholder->GetData();

            hcorepp::kernels::Geqrf(Vm, Vn, V, Vm, Vtau);

            int64_t sizeS;
            if (aHelpers.GetTrmm()) {
                sizeS = min_Um_Un;
            } else {
                sizeS = std::min({m, n, (aARank + Crk)});
            }

            size_t max_rows;
            size_t max_cols;
            if (aHelpers.GetUngqr()) {
                max_rows = min_Um_Un;
                max_cols = min_Vm_Vn;
            } else {
                max_rows = Um;
                max_cols = Vm;
            }

            /// allocate max rows (m) because we truncate columns not rows, after unmqr
            auto Unew_dataHolder = new DataHolder<T>(max_rows, sizeS, max_rows);
            /// allocate max colums (n) because we truncate rows not columns, after unmqr
            auto VTnew_dataHolder = new DataHolder<T>(sizeS, max_cols, sizeS);

            T *Unew = Unew_dataHolder->GetData();
            T *VTnew = VTnew_dataHolder->GetData();

            blas::real_type<T> *Sigma;

            Sigma = hcorepp::kernels::AllocateSigma<T>(sizeS);

            if (aHelpers.GetTrmm()) {
                if (aHelpers.GetUngqr()) {
                    hcorepp::kernels::Trmm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper,
                                           blas::Op::ConjTrans, blas::Diag::NonUnit, min_Um_Un, Un, one, V, Vm, RU,
                                           min_Um_Un);

                    hcorepp::kernels::Gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec, min_Um_Un, Un, RU, min_Um_Un,
                                            Sigma, Unew, min_Um_Un, VTnew, sizeS);
                } else {
                    hcorepp::kernels::Trmm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper,
                                           blas::Op::Trans, blas::Diag::NonUnit, min_Um_Un, Un, one, V, Vm, RU,
                                           min_Um_Un);

                    hcorepp::kernels::Gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec, min_Um_Un, Un, RU, min_Um_Un,
                                            Sigma, Unew, Um, VTnew, sizeS);
                }
            } else {

                auto rv_dataHolder = new DataHolder<T>(min_Vm_Vn, Vn, min_Vm_Vn);
                auto rurv_dataHolder = new DataHolder<T>(min_Um_Un, min_Vm_Vn, min_Um_Un);

                T *RV = rv_dataHolder->GetData();
                T *RURV = rurv_dataHolder->GetData();

                hcorepp::kernels::Laset(lapack::MatrixType::Lower, min_Vm_Vn, Vn, zero, zero, RV, min_Vm_Vn);
                hcorepp::kernels::LaCpy(lapack::MatrixType::Upper, min_Vm_Vn, Vn, V, Vm, RV, min_Vm_Vn);

                if (aHelpers.GetUngqr()) {
                    hcorepp::kernels::Gemm<T>(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans,
                                              min_Um_Un, min_Vm_Vn, (aARank + Crk), one, RU, min_Um_Un, RV, min_Vm_Vn,
                                              zero, RURV, min_Um_Un);

                    hcorepp::kernels::Gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec, min_Um_Un, min_Vm_Vn, RURV,
                                            min_Um_Un, Sigma, Unew, min_Um_Un, VTnew, sizeS);
                } else {
                    hcorepp::kernels::Gemm<T>(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans, min_Um_Un,
                                              min_Vm_Vn, (aARank + Crk), one, RU, min_Um_Un, RV, min_Vm_Vn, zero, RURV,
                                              min_Um_Un);

                    hcorepp::kernels::Gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec, min_Um_Un, min_Vm_Vn, RURV,
                                            min_Um_Un, Sigma, Unew, Um, VTnew, sizeS);
                }
                delete rv_dataHolder;
                delete rurv_dataHolder;
            }
            delete ru_dataholder;

            int64_t rk_new;
            if (aHelpers.GetFixedRank()) {
                /// truncate according to fixed_rk
                rk_new = aHelpers.GetFixedRank();
                if (aHelpers.GetFixedRank() > (aARank + Crk)) {
                    rk_new = (aARank + Crk);
                }
            } else { // truncate according to accuracy
                hcorepp::kernels::CalculateNewRank<T>(rk_new, aHelpers.GetTruncatedSvd(), Sigma, sizeS, accuracy);
            }

            auto uv_dataHolder = new DataHolder<T>((ldcu + n), rk_new, (ldcu + n));
            T *UV = uv_dataHolder->GetData();

            if (aHelpers.GetUngqr()) {
                lapack::ungqr(Um, min_Um_Un, min_Um_Un, U, Um, Utau);

                hcorepp::kernels::Gemm<T>(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, Um, rk_new,
                                          min_Um_Un, one, U, Um, Unew, min_Um_Un, zero, UV, ldcu);

            } else {
                hcorepp::kernels::Unmqr(blas::Side::Left, blas::Op::NoTrans, Um, rk_new, min_Um_Un, U, Um, Utau, Unew,
                                        Um);

                hcorepp::kernels::LaCpy(lapack::MatrixType::General, Um, rk_new, Unew, Um, UV, ldcu);
            }

            hcorepp::kernels::CalculateVTnew(rk_new, aHelpers.GetUngqr(), min_Vm_Vn, Sigma, VTnew, sizeS, Vm);

            hcorepp::kernels::DestroySigma<T>(Sigma);

            T *UVptr = UV + ldcu * rk_new;

            if (aHelpers.GetUngqr()) {
                lapack::ungqr(Vm, min_Vm_Vn, min_Vm_Vn, V, Vm, Vtau);

                auto vnew_dataHolder = new DataHolder<T>(Vm, rk_new, Vm);

                T *Vnew = vnew_dataHolder->GetData();

                hcorepp::kernels::Gemm<T>(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans, Vm, rk_new,
                                          min_Vm_Vn,
                                          one, V, Vm, VTnew, sizeS, zero, Vnew, Vm);

                hcorepp::kernels::CalculateUVptr<T>(rk_new, Vm, UVptr, Vnew);

                delete vnew_dataHolder;
            } else {
                hcorepp::kernels::Unmqr(blas::Side::Right, blas::Op::ConjTrans, rk_new, Vm, min_Vm_Vn, V, Vm, Vtau,
                                        VTnew, sizeS);

                hcorepp::kernels::LaCpy(lapack::MatrixType::General, rk_new, Vm, VTnew, sizeS, UVptr, rk_new);
                hcorepp::kernels::CalculateUVptrConj(rk_new, Vm, UVptr);
            }

            this->SetTileRank(rk_new);
            this->GetTileSubMatrix(0).get().Resize(this->GetTileSubMatrix(0).get().GetNumOfRows(), rk_new,
                                                   this->GetTileSubMatrix(0).get().GetNumOfRows());
            this->GetTileSubMatrix(1).get().Resize(rk_new, this->GetTileSubMatrix(1).get().GetNumOfCols(), rk_new);

            size_t u_size =
                    this->GetTileSubMatrix(0).get().GetNumOfRows() * this->GetTileSubMatrix(0).get().GetNumOfCols();
            size_t v_size =
                    this->GetTileSubMatrix(1).get().GetNumOfRows() * this->GetTileSubMatrix(1).get().GetNumOfCols();

            this->GetTileSubMatrix(0).get().CopyDataArray(0, UV, u_size);
            this->GetTileSubMatrix(1).get().CopyDataArray(0, &UV[u_size], v_size);

            delete U_dataholder;
            delete uv_dataHolder;
            delete VTnew_dataHolder;
            delete Unew_dataHolder;
            delete Vtau_dataholder;
            delete V_dataholder;
            delete Utau_dataholder;
        }

        template<typename T>
        size_t CompressedTile<T>::GetNumberOfMatrices() const {
            return mDataArrays.size();
        }

        template<typename T>
        void CompressedTile<T>::SetTileRank(int64_t &aMatrixRank) {
            if (aMatrixRank == std::min(this->mNumOfRows, this->mNumOfCols)) {
                this->mMatrixRank = FULL_RANK_;
            } else {
                this->mMatrixRank = aMatrixRank;
            }
        }

        template<typename T>
        int64_t CompressedTile<T>::GetTileRank() const {
            if (this->mMatrixRank == FULL_RANK_) {
                return std::min(this->mNumOfRows, this->mNumOfCols);
            }
            return this->mMatrixRank;
        }

        template<typename T>
        size_t CompressedTile<T>::GetNumOfRows() const {
            if (this->mOperation == blas::Op::NoTrans) {
                return mNumOfRows;
            }
            return mNumOfCols;
        }

        template<typename T>
        size_t CompressedTile<T>::GetNumOfCols() const {
            if (this->mOperation == blas::Op::NoTrans) {
                return mNumOfCols;
            }
            return mNumOfRows;
        }


        template<typename T>
        blas::real_type<T> CompressedTile<T>::GetAccuracy() {
            return this->mAccuracy;
        }

        template<typename T>
        void
        CompressedTile<T>::ReadjustTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                                        int64_t aRank) {

            if (aRank == -1) {
                this->mMatrixRank = std::min(aNumOfRows, aNumOfCols);
            } else {
                this->mMatrixRank = aRank;
            }

            this->mLeadingDim = aLeadingDim;
            this->mNumOfRows = aNumOfRows;
            this->mNumOfCols = aNumOfCols;

            this->GetTileSubMatrix(0).get().Resize(this->mNumOfRows, this->mMatrixRank, this->mNumOfRows);
            this->GetTileSubMatrix(1).get().Resize(this->mMatrixRank, this->mNumOfCols, this->mMatrixRank);

            size_t u_size = this->mNumOfRows * this->mMatrixRank;
            size_t v_size = this->mMatrixRank * this->mNumOfCols;

            this->GetTileSubMatrix(0).get().CopyDataArray(0, aPdata, u_size);

            if (aRank == -1) {
                T *identity_matrix = this->GetTileSubMatrix(1).get().GetData();

                hcorepp::kernels::FillIdentityMatrix(this->mNumOfCols, identity_matrix);

            } else {
                this->GetTileSubMatrix(1).get().CopyDataArray(0, &aPdata[u_size], v_size);
            }
        }

    }

}