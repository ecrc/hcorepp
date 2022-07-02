
#include <hcorePP/operators/concrete/Compressed.hpp>

namespace hcorepp {
    namespace operators {

        template<typename T>
        CompressedTile<T>::CompressedTile() {

        }

        template<typename T>
        CompressedTile<T>::CompressedTile(int64_t aNumOfRows, int64_t aNumOfCols, T *aPdata, int64_t aLeadingDim,
                                          int64_t aRank, real_t aAccuracy, blas::Layout aLayout, blas::Op aOperation,
                                          blas::Uplo aUplo) {
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

            ///     The m-by-n matrix compressed tile (A=UV): A is m-by-n, U is
            ///     m-by-rk, and V is rk-by-n.
            if (this->mLayout == blas::Layout::ColMajor) {
                ///     If layout = blas::Layout::ColMajor,
                ///     the data array of A is stored in an ld-by-n array buffer; the
                ///     data array of U is stored in an ld-by-rk array buffer; and
                ///     data array of V is stored in an rk-by-n array buffer.

                this->mDataArrays.emplace_back(
                        DataHolder<T>(this->mLeadingDim, this->mMatrixRank, aLeadingDim, aPdata));
                this->mDataArrays.emplace_back(
                        DataHolder<T>(this->mMatrixRank, aNumOfCols, aLeadingDim, aPdata + this->mLeadingDim *
                                                                                           this->mMatrixRank));
            } else {
                ///     layout = blas::Layout::RowMajor: the data array of A is stored in an
                ///     m-by-ld array buffer, the data array of U is stored in an
                ///     m-by-rk array buffer, and data array of V is stored in an
                ///     rk-by-ld array buffer.

                this->mDataArrays.emplace_back(DataHolder<T>(aNumOfRows, this->mMatrixRank, aLeadingDim, aPdata));
                this->mDataArrays.emplace_back(DataHolder<T>(this->mMatrixRank, this->mLeadingDim, aLeadingDim,
                                                             aPdata + aNumOfRows * this->mMatrixRank));

            }
        }

        template<typename T>
        CompressedTile<T>::~CompressedTile<T>() {

        }

        template<typename T>
        DataHolder<T> &CompressedTile<T>::GetTileSubMatrix(size_t aIndex) {
            return mDataArrays[aIndex];
        }

        template<typename T>
        DataHolder<T> const *CompressedTile<T>::GetTileSubMatrixConst(size_t aIndex) const {
            return &mDataArrays[aIndex];
        }

        template<typename T>
        int64_t CompressedTile<T>::GetTileStride(size_t aIndex) const {
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
        }

        template<typename T>
        void
        CompressedTile<T>::Gemm(T &aAlpha, DataHolder<T> const &aTileA, blas::Op aTileAOp, DataHolder<T> const &aTileB,
                                blas::Op aTileBOp, T &aBeta) {
//            void reduced_svd(
//                    T beta, T const* AU, T const* AV, int64_t ldau, int64_t Ark,
//                    CompressedTile<T>& C, bool use_trmm=false, bool use_ungqr=true,
//                    bool truncated_svd=false, int64_t fixed_rk=0)

//            internal::reduced_svd(
//                    beta, &W[0], B.Vdata(), C.m(), B.rk(), C,
//                    use_trmm, use_ungqr, truncated_svd, fixed_rk);

//            /** declaring some initial values */
//            bool use_trmm = false;
//            bool use_ungqr = true;
//            bool truncated_svd = false;
//            int64_t fixed_rk = 0;
//
//            using blas::conj;
//
//            T zero = 0.0;
//            T one = 1.0;
//
//            int64_t m = this->mNumOfRows;
//            int64_t n = this->mNumOfCols;
//
//            T *CU = this->GetTileSubMatrix(0);
//            T *CV = this->GetTileSubMatrix(1);
//
//            int64_t Crk = this->GetTileRank();
//
//            blas::real_type <T> accuracy = this->mAccuracy;
//
//            int64_t Um = m;
//            int64_t Un = Ark + Crk;
//
//            int64_t ldcu = this->GetTileSubMatrix(0).GetLeadingDim();
//
//            // U = [CU AU]
//            std::vector<T> U(Um * Un);
//            lapack::lacpy(lapack::MatrixType::General, m, Crk, &CU[0], ldcu, &U[0], Um);
//            lapack::lacpy(lapack::MatrixType::General, m, Ark, &AU[0], ldau, &U[m * Crk], Um);
//
//            int64_t min_Um_Un = std::min(Um, Un);
//
//            // [QU, RU] = qr(U, 0)
//            std::vector<T> Utau(min_Um_Un);
//            lapack::geqrf(Um, Un, &U[0], Um, &Utau[0]);
//
//            // RU: uppertriangular part of QR(U)
//            std::vector<T> RU(min_Um_Un * Un);
//            lapack::laset(lapack::MatrixType::Lower,
//                          min_Um_Un, Un, zero, zero, &RU[0], min_Um_Un);
//            lapack::lacpy(lapack::MatrixType::Upper,
//                          min_Um_Un, Un, &U[0], Um, &RU[0], min_Um_Un);
//
//            int64_t Vm = n;
//            int64_t Vn = Ark + Crk;
//
//            int64_t ldcv = this->GetTileSubMatrix(1).GetLeadingDim();
//
//            // V = [beta * CV.' ((alpha * AV * BU) * BV).']
//            std::vector<T> V(Vm * Vn);
//            for (int64_t j = 0; j < n; ++j) {
//                for (int64_t i = 0; i < Crk; ++i) {
//                    if (use_ungqr)
//                        V[j + i * Vm] = conj(aBeta * CV[i + j * ldcv]);
//                    else
//                        V[j + i * Vm] = aBeta * CV[i + j * ldcv];
//                }
//            }
//            T *AV = aTileB.GetData();
//            for (int64_t j = 0; j < n; ++j) {
//                T *Vptr = &V[n * Crk];
//                for (int64_t i = 0; i < Ark; ++i) {
//                    if (use_ungqr)
//                        Vptr[j + i * Vm] = conj(AV[i + j * Ark]);
//                    else
//                        Vptr[j + i * Vm] = AV[i + j * Ark];
//                }
//            }
//
//            int64_t min_Vm_Vn = std::min(Vm, Vn);
//
//            // [QV, RV] = qr(V, 0)
//            std::vector<T> Vtau(min_Vm_Vn);
//            lapack::geqrf(Vm, Vn, &V[0], Vm, &Vtau[0]);
//
//            int64_t sizeS = (use_trmm ? min_Um_Un : std::min({m, n, (Ark + Crk)}));
//            std::vector<blas::real_type < T>> Sigma(sizeS);
//
//            // allocate max rows (m) because we truncate columns not rows, after unmqr
//            std::vector<T> Unew((use_ungqr ? min_Um_Un : Um) * sizeS);
//            // allocate max colums (n) because we truncate rows not columns, after unmqr
//            std::vector<T> VTnew(sizeS * (use_ungqr ? min_Vm_Vn : Vm));
//
//            if (use_trmm) {
//                blas::trmm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper,
//                           (use_ungqr ? blas::Op::ConjTrans : blas::Op::Trans),
//                           blas::Diag::NonUnit, min_Um_Un, Un, one, &V[0], Vm, &RU[0], min_Um_Un);
//
//                // orthogonal QU and QV
//                // [Unew, Sigma, VTnew] = svd(RU * RV.');
//                lapack::gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec, min_Um_Un, Un, &RU[0], min_Um_Un, &Sigma[0],
//                              &Unew[0], (use_ungqr ? min_Um_Un : Um), &VTnew[0], sizeS);
//            } else {
//                // RV: uppertriangular part of QR(V)
//                std::vector<T> RV(min_Vm_Vn * Vn);
//                lapack::laset(lapack::MatrixType::Lower,
//                              min_Vm_Vn, Vn, zero, zero, &RV[0], min_Vm_Vn);
//                lapack::lacpy(lapack::MatrixType::Upper,
//                              min_Vm_Vn, Vn, &V[0], Vm, &RV[0], min_Vm_Vn);
//
//                // RU * RV.'
//                std::vector<T> RURV(min_Um_Un * min_Vm_Vn);
//                blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans,
//                           (use_ungqr ? blas::Op::ConjTrans : blas::Op::Trans), min_Um_Un, min_Vm_Vn, (Ark + Crk),
//                           one, &RU[0], min_Um_Un, &RV[0], min_Vm_Vn, zero, &RURV[0], min_Um_Un);
//
//                // orthogonal QU and QV
//                // [Unew, Sigma, VTnew] = svd(RU * RV.');
//                lapack::gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec, min_Um_Un, min_Vm_Vn, &RURV[0], min_Um_Un,
//                              &Sigma[0], &Unew[0], (use_ungqr ? min_Um_Un : Um), &VTnew[0], sizeS);
//            }
//
//            int64_t rk_new;
//            if (fixed_rk) { // truncate according to fixed_rk
//                rk_new = fixed_rk;
//                if (fixed_rk > (Ark + Crk)) {
//                    rk_new = (Ark + Crk);
//                }
//            } else { // truncate according to accuracy
//                rk_new = sizeS;
//                if (truncated_svd) {
//                    blas::real_type <T> Sigma_0 = Sigma[0];
//                    for (int64_t i = 1; i < sizeS; i++) {
//                        if (Sigma[i] < accuracy * Sigma_0) {
//                            Sigma_0 = Sigma[i];
//                            rk_new = i;
//                            break;
//                        }
//                    }
//                } else {
//                    for (int64_t i = 1; i < sizeS; i++) {
//                        if (Sigma[i] < accuracy) {
//                            rk_new = i;
//                            break;
//                        }
//                    }
//                }
//            }
//
//            hcore_error_if_msg(
//                    rk_new > std::min(m, n),
//                    "Rank (%lld) after truncation (%lld) is greater than max rank (%lld)",
//                    (long long) Crk, (long long) rk_new, (long long) std::min(m, n));
//
//            T *UV = new T[(ldcu + n) * rk_new];
//
//            if (use_ungqr) {
//                lapack::ungqr(Um, min_Um_Un, min_Um_Un, &U[0], Um, &Utau[0]);
//                blas::gemm(
//                        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
//                        Um, rk_new, min_Um_Un,
//                        one, &U[0], Um,
//                        &Unew[0], min_Um_Un,
//                        zero, &UV[0], ldcu);
//            } else {
//                lapack::unmqr(
//                        blas::Side::Left, blas::Op::NoTrans,
//                        Um, rk_new, min_Um_Un,
//                        &U[0], Um, &Utau[0],
//                        &Unew[0], Um);
//                lapack::lacpy(lapack::MatrixType::General,
//                              Um, rk_new, &Unew[0], Um, UV, ldcu);
//            }
//
//            // VTnew eats Sigma.
//            // todo: we may need to have uplo parameter:
//            //       scale VT, if Lower, or scale U otherwise.
//            for (int64_t i = 0; i < rk_new; ++i) {
//                blas::scal((use_ungqr ? min_Vm_Vn : Vm), Sigma[i], &VTnew[i], sizeS);
//
//                if (!use_ungqr) {
//                    for (int64_t j = 0; j < Vm; ++j) {
//                        VTnew[i + j * sizeS] = conj(VTnew[i + j * sizeS]);
//                    }
//                }
//            }
//
//            T *UVptr = UV + ldcu * rk_new;
//
//            if (use_ungqr) {
//                lapack::ungqr(Vm, min_Vm_Vn, min_Vm_Vn, &V[0], Vm, &Vtau[0]);
//                std::vector<T> Vnew(Vm * rk_new);
//                blas::gemm(
//                        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans,
//                        Vm, rk_new, min_Vm_Vn,
//                        one, &V[0], Vm,
//                        &VTnew[0], sizeS,
//                        zero, &Vnew[0], Vm);
//                for (int64_t j = 0; j < rk_new; ++j) {
//                    for (int64_t i = 0; i < Vm; ++i) {
//                        UVptr[j + i * rk_new] = conj(Vnew[i + j * Vm]);
//                    }
//                }
//            } else {
//                lapack::unmqr(
//                        blas::Side::Right, blas::Op::ConjTrans,
//                        rk_new, Vm, min_Vm_Vn,
//                        &V[0], Vm, &Vtau[0],
//                        &VTnew[0], sizeS);
//                lapack::lacpy(lapack::MatrixType::General,
//                              rk_new, Vm, &VTnew[0], sizeS, &UVptr[0], rk_new);
//                for (int64_t i = 0; i < rk_new; ++i) {
//                    for (int64_t j = 0; j < Vm; ++j) {
//                        UVptr[i + j * rk_new] = conj(UVptr[i + j * rk_new]);
//                    }
//                }
//            }
//
//            //@todo Update the data holders and the new rank
//
////            C.UVdata(UV);
////            C.rk(rk_new);
        }

        template<typename T>
        size_t CompressedTile<T>::GetNumberOfMatrices() {
            return mDataArrays.size();
        }


        template<typename T>
        void CompressedTile<T>::SetTileRank(int64_t &aMatrixRank) const {
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
            } else {
                this->mMatrixRank;
            }
        }

/*
        template<typename T>
        real_t CompressedTile<T>::GetAccuracy() {
            return this->mAccuracy;
        }

        template<typename T>
        void CompressedTile<T>::SetAccuracy(real_t aAccuracy) {
            this->mAccuracy = aAccuracy;
        }

        template<typename T>
        bool CompressedTile<T>::IsFullRank() const {
            if (this->mMatrixRank == FULL_RANK_) {
                return true;
            }
            return false;
        }

*/
    }

}