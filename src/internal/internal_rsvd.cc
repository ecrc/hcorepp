// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include <initializer_list>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <complex>
#include <vector>
#include <new>

#include "blas.hh"
#include "lapack.hh"

#include "hcore/compressed_tile.hh"
#include "internal.hh"

namespace hcore {
namespace internal {

template <typename T>
void rsvd(blas::Op transAU, blas::Op transAV,
          T beta, T const* AU, int64_t ldau,
                  T const* AV, int64_t ldav, int64_t Ark,
                  CompressedTile<T>& C,
          bool use_gemm, bool use_Segma0_as_tol, int64_t fixed_rk) {
    using blas::conj;

    T zero = 0.0;
    T one  = 1.0;

    int64_t m = C.layout() == blas::Layout::ColMajor ? C.mb() : C.nb();
    int64_t n = C.layout() == blas::Layout::ColMajor ? C.nb() : C.mb();

    T* CU = C.layout() == blas::Layout::ColMajor ? C.Udata() : C.Vdata();
    T* CV = C.layout() == blas::Layout::ColMajor ? C.Vdata() : C.Udata();

    int64_t ldcu = C.layout() == blas::Layout::ColMajor ? C.Ustride()
                                                        : C.Vstride();
    int64_t ldcv = C.layout() == blas::Layout::ColMajor ? C.Vstride()
                                                        : C.Ustride();
    int64_t Crk = C.rk();

    blas::real_type<T> tol = C.tol();

    int64_t Um = m;
    int64_t Un = Ark + Crk;

    // U = [CU AU]
    std::vector<T> U(Um*Un);
    lapack::lacpy(lapack::MatrixType::General, m, Crk, &CU[0], ldcu, &U[0], Um);
    // lapack::lacpy(lapack::MatrixType::General, m, Ark, &AU[0], ldau,
    //               &U[m*Crk], Um);
    {
        #define AU(i_, j_) (transAU == blas::Op::NoTrans   \
                            ? AU[(i_) + (j_)*size_t(ldau)] \
                            : AU[(j_) + (i_)*size_t(ldau)])
        T* U_ = &U[m*Crk];
        for (int64_t j = 0; j < Ark; ++j) {
            for (int64_t i = 0; i < m; ++i) {
                U_[i + j*Um] = transAU == blas::Op::ConjTrans ? conj(AU(i, j))
                                                              :      AU(i, j);
            }
        }
        #undef AU
    }

    int64_t min_Um_Un = std::min(Um, Un);

    // [QU, RU] = qr(U, 0)
    std::vector<T> Utau(min_Um_Un);
    lapack::geqrf(Um, Un, &U[0], Um, &Utau[0]);

    // RU: uppertriangular part of QR(U)
    std::vector<T> RU(min_Um_Un * Un);
    lapack::laset(lapack::MatrixType::Lower, min_Um_Un, Un, zero, zero,
                  &RU[0], min_Um_Un);
    lapack::lacpy(lapack::MatrixType::Upper, min_Um_Un, Un, &U[0], Um,
                  &RU[0], min_Um_Un);

    int64_t Vm = n;
    int64_t Vn = Ark + Crk;

    // V = [beta * CV.' ((alpha * AV * BU) * BV).']
    std::vector<T> V(Vm*Vn);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < Crk; ++i)
            V[j + i*Vm] = conj(beta*CV[i + j*ldcv]);

    {
        #define AV(i_, j_) (transAV == blas::Op::NoTrans   \
                            ? AV[(i_) + (j_)*size_t(ldav)] \
                            : AV[(j_) + (i_)*size_t(ldav)])
        T* V_ = &V[n*Crk];
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < Ark; ++i) {
                V_[j + i*Vm] = transAV == blas::Op::ConjTrans ?      AV(i, j)
                                                              : conj(AV(i, j));
            }
        }
        #undef AV
    }

    int64_t min_Vm_Vn = std::min(Vm, Vn);

    // [QV, RV] = qr(V, 0)
    std::vector<T> Vtau(min_Vm_Vn);
    lapack::geqrf(Vm, Vn, &V[0], Vm, &Vtau[0]);

    // int64_t sizeS = (use_trmm ? min_Um_Un : std::min({m, n, (Ark + Crk)}));
    int64_t sizeS = std::min({m, n, (Ark + Crk)});
    std::vector<blas::real_type<T>> Sigma(sizeS);

    // allocate max rows (m) because we truncate columns not rows, after unmqr
    std::vector<T> Unew((use_gemm ? min_Um_Un : Um)*sizeS);
    // allocate max colums (n) because we truncate rows not columns, after unmqr
    std::vector<T> VTnew(sizeS*(use_gemm ? min_Vm_Vn : Vm));

    if (!use_gemm) {
        blas::trmm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper,
                   blas::Op::ConjTrans, blas::Diag::NonUnit,
                   min_Um_Un, Un,
                   one, &V[0],  Vm,
                        &RU[0], min_Um_Un);
        // orthogonal QU and QV
        // [Unew, Sigma, VTnew] = svd(RU * RV.');
        lapack::gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec,
                      min_Um_Un, min_Vm_Vn,
                      &RU[0],    min_Um_Un, &Sigma[0],
                      &Unew[0],  Um,
                      &VTnew[0], sizeS);

        // blas::trmm(
        //     blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper,
        //     (use_ungqr ? blas::Op::ConjTrans : blas::Op::Trans),
        //     blas::Diag::NonUnit,
        //     min_Um_Un, Un,
        //     one, &V[0],  Vm,
        //          &RU[0], min_Um_Un);
        // lapack::gesvd(
        //     lapack::Job::SomeVec, lapack::Job::SomeVec,
        //     min_Um_Un, Un,
        //     &RU[0], min_Um_Un, &Sigma[0],
        //     &Unew[0], (use_ungqr ? min_Um_Un : Um),
        //     &VTnew[0], sizeS);
    }
    else {
        // RV: uppertriangular part of QR(V)
        std::vector<T> RV(min_Vm_Vn*Vn);
        lapack::laset(lapack::MatrixType::Lower, min_Vm_Vn, Vn, zero, zero,
                      &RV[0], min_Vm_Vn);
        lapack::lacpy(lapack::MatrixType::Upper, min_Vm_Vn, Vn, &V[0], Vm,
                      &RV[0], min_Vm_Vn);

        // RU * RV.'
        std::vector<T> RURV(min_Um_Un*min_Vm_Vn);
        blas::gemm(blas::Layout::ColMajor,
                   blas::Op::NoTrans, blas::Op::ConjTrans,
                   min_Um_Un, min_Vm_Vn, (Ark + Crk),
                   one,  &RU[0],   min_Um_Un,
                         &RV[0],   min_Vm_Vn,
                   zero, &RURV[0], min_Um_Un);
        // orthogonal QU and QV
        // [Unew, Sigma, VTnew] = svd(RU * RV.');
        lapack::gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec,
                      min_Um_Un, min_Vm_Vn,
                      &RURV[0],  min_Um_Un, &Sigma[0],
                      &Unew[0],  min_Um_Un,
                      &VTnew[0], sizeS);
    }

    int64_t rk_new;
    if (fixed_rk) { // truncate according to fixed rank
        rk_new = fixed_rk;
        if (fixed_rk > (Ark + Crk))
            rk_new = (Ark + Crk);
    }
    else { // truncate according to tolerance
        rk_new = sizeS;
        blas::real_type<T> Sigma_0 = use_Segma0_as_tol ? Sigma[0] : 1.0;
        for (int64_t i = 1; i < sizeS; i++) {
            if (Sigma[i] < tol*Sigma_0) {
                // Sigma_0 = Sigma[i];
                rk_new = i;
                break;
            }
        }

        // if (use_Segma0_as_tol) {
        // }
        // else {
        //     for (int64_t i = 1; i < sizeS; i++) {
        //         if (Sigma[i] < tol) {
        //             rk_new = i;
        //             break;
        //         }
        //     }
        // }
    }

    if (rk_new > std::min(m, n)) {
        throw Error("Rank (" + std::to_string(Crk) + ") after truncation ("
                    + std::to_string(rk_new)
                    + ") is greater than maximum possible rank ("
                    + std::to_string(std::min(m, n)) + ").");
    }

    // VTnew eats (swallow) Sigma.
    // todo: we may need to have uplo parameter:
    //       scale VT, if Lower, or scale U otherwise.
    for(int64_t i = 0; i < rk_new; ++i)// {
        blas::scal(use_gemm ? min_Vm_Vn : Vm, Sigma[i], &VTnew[i], sizeS);

        //if (!use_gemm) {
        //    for (int64_t j = 0; j < Vm; ++j)
        //        VTnew[i + j*sizeS] = conj(VTnew[i + j*sizeS]);
        //}
    // }

    int64_t ldu = ldcu;
    int64_t ldv = rk_new;

    T* UV = new T[ldu*rk_new + ldv*n];
    T* UTilda = UV;
    T* VTilda = UV + ldu*rk_new;

    // T* UV = new T[(ldcu + n) * rk_new];

    if (use_gemm) {
        lapack::ungqr(Um, min_Um_Un, min_Um_Un, &U[0], Um, &Utau[0]);
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   Um, rk_new, min_Um_Un,
                   one,  &U[0],    Um,
                         &Unew[0], min_Um_Un,
                   zero, UTilda,   ldu);
    }
    else {
        lapack::unmqr(blas::Side::Left, blas::Op::NoTrans,
                      Um, rk_new, min_Um_Un,
                      &U[0],    Um, &Utau[0],
                      &Unew[0], Um);
        lapack::lacpy(lapack::MatrixType::General, Um, rk_new, &Unew[0], Um,
                      UTilda, ldu);
    }

    // T* UVptr = UV + ldcu * rk_new;

    if (use_gemm) {
        lapack::ungqr(Vm, min_Vm_Vn, min_Vm_Vn, &V[0], Vm, &Vtau[0]);
        std::vector<T> Vnew(Vm*rk_new);
        blas::gemm(blas::Layout::ColMajor,
                   blas::Op::NoTrans, blas::Op::ConjTrans,
                   Vm, rk_new, min_Vm_Vn,
                   one,  &V[0],     Vm,
                         &VTnew[0], sizeS,
                   zero, &Vnew[0],  Vm);

        for (int64_t j = 0; j < rk_new; ++j)
            for (int64_t i = 0; i < Vm; ++i)
                VTilda[j + i*ldv] = conj(Vnew[i + j*Vm]);
    }
    else {
        lapack::unmqr(blas::Side::Right, blas::Op::ConjTrans,
                      rk_new, Vm, min_Vm_Vn,
                      &V[0],     Vm, &Vtau[0],
                      &VTnew[0], sizeS);
        lapack::lacpy(lapack::MatrixType::General, rk_new, Vm, &VTnew[0], sizeS,
                      VTilda, ldv);

        //for (int64_t j = 0; j < Vm; ++j)
        //    for (int64_t i = 0; i < rk_new; ++i)
        //        VTilda[i + j*ldv] = conj(VTilda[i + j*ldv]);
    }

    C.resize(UV, ldu, ldv, rk_new);
}

template
void rsvd(blas::Op transAU, blas::Op transAV,
          float beta, float const* AU, int64_t ldau,
                      float const* AV, int64_t ldav, int64_t Ark,
                      CompressedTile<float>& C,
          bool use_gemm, bool use_Segma0_as_tol, int64_t fixed_rk);
template
void rsvd(blas::Op transAU, blas::Op transAV,
          double beta, double const* AU, int64_t ldau,
                       double const* AV, int64_t ldav, int64_t Ark,
                       CompressedTile<double>& C,
          bool use_gemm, bool use_Segma0_as_tol, int64_t fixed_rk);
template
void rsvd(blas::Op transAU, blas::Op transAV,
          std::complex<float> beta, std::complex<float> const* AU, int64_t ldau,
                                    std::complex<float> const* AV, int64_t ldav,
                                    int64_t Ark,
                                    CompressedTile<std::complex<float>>& C,
          bool use_gemm, bool use_Segma0_as_tol, int64_t fixed_rk);
template
void rsvd(blas::Op transAU, blas::Op transAV,
          std::complex<double> beta, std::complex<double> const* AU,
                                     int64_t ldau,
                                     std::complex<double> const* AV,
                                     int64_t ldav, int64_t Ark,
                                     CompressedTile<std::complex<double>>& C,
          bool use_gemm, bool use_Segma0_as_tol, int64_t fixed_rk);

} // namespace internal
} // namespace hcore
