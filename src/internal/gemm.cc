// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "internal/internal.hh"
#include "hcore/tile/compressed.hh"

#include "blas.hh"
#include "lapack.hh"

#include <complex>
#include <cassert>
#include <algorithm>
#include <initializer_list>

namespace hcore {
namespace internal {

template <typename T>
void gemm(
    T beta, T const* AU, T const* AV, int64_t ldau, int64_t Ark,
    CompressedTile<T>& C, bool use_trmm, bool use_ungqr,
    bool truncation_with_tol, int64_t rk)
{
    int64_t m = C.m();
    int64_t n = C.n();

    T* CU = C.Udata();
    T* CV = C.Vdata();

    int64_t Crk = C.rk();

    blas::real_type<T> accuracy = C.accuracy();

    using blas::conj;

    int64_t Um = m;
    int64_t Un = Ark + Crk;

    int64_t ldcu = C.ldu();

    // U = [CU AU]
    std::vector<T> U(Um * Un);
    lapack::lacpy(
        lapack::MatrixType::General, m, Crk, &CU[0], ldcu, &U[0], Um);
    lapack::lacpy(
        lapack::MatrixType::General, m, Ark, &AU[0], ldau, &U[m * Crk], Um);

    int64_t min_Um_Un = std::min(Um, Un);

    // [QU, RU] = qr(U, 0)
    std::vector<T> Utau(min_Um_Un);
    lapack::geqrf(Um, Un, &U[0], Um, &Utau[0]);

    // RU: uppertriangular part of QR(U)
    std::vector<T> RU(min_Um_Un * Un);
    lapack::laset(lapack::MatrixType::Lower,
        min_Um_Un, Un, 0.0, 0.0, &RU[0], min_Um_Un);
    lapack::lacpy(lapack::MatrixType::Upper,
        min_Um_Un, Un, &U[0], Um, &RU[0], min_Um_Un);

    int64_t Vm = n;
    int64_t Vn = Ark + Crk;

    int64_t ldcv = C.ldv();

    // V = [beta * CV.' ((alpha * AV * BU) * BV).']
    std::vector<T> V(Vm * Vn);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < Crk; ++i) {
            if (use_ungqr)
                V[j + i * Vm] = conj(beta * CV[i + j * ldcv]);
            else
                V[j + i * Vm] = beta * CV[i + j * ldcv];
        }
    }
    for (int64_t j = 0; j < n; ++j) {
        T* Vptr = &V[n * Crk];
        for (int64_t i = 0; i < Ark; ++i) {
            if (use_ungqr)
                Vptr[j + i * Vm] = conj(AV[i + j * Ark]);
            else
                Vptr[j + i * Vm] = AV[i + j * Ark];
        }
    }

    int64_t min_Vm_Vn = std::min(Vm, Vn);

    // [QV, RV] = qr(V, 0)
    std::vector<T> Vtau(min_Vm_Vn);
    lapack::geqrf(Vm, Vn, &V[0], Vm, &Vtau[0]);

    int64_t sizeS = (use_trmm ? min_Um_Un : std::min({m, n, (Ark + Crk)}));
    std::vector<blas::real_type<T>> Sigma(sizeS);

    // allocate max rows (m) because we truncate columns not rows, after unmqr
    std::vector<T> Unew((use_ungqr ? min_Um_Un : Um) * sizeS);
    // allocate max colums (n) because we truncate rows not columns, after unmqr
    std::vector<T> VTnew(sizeS * (use_ungqr ? min_Vm_Vn : Vm));

    if (use_trmm) {
        blas::trmm(
            blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Upper,
            (use_ungqr ? blas::Op::ConjTrans : blas::Op::Trans),
            blas::Diag::NonUnit,
            min_Um_Un, Un,
            1.0, &V[0],  Vm,
                 &RU[0], min_Um_Un);

        // orthogonal QU and QV
        // [Unew, Sigma, VTnew] = svd(RU * RV.');
        lapack::gesvd(
            lapack::Job::SomeVec, lapack::Job::SomeVec,
            min_Um_Un, Un,
            &RU[0], min_Um_Un, &Sigma[0],
            &Unew[0], (use_ungqr ? min_Um_Un : Um),
            &VTnew[0], sizeS);
    }
    else {
        // RV: uppertriangular part of QR(V)
        std::vector<T> RV(min_Vm_Vn * Vn);
        lapack::laset(lapack::MatrixType::Lower,
            min_Vm_Vn, Vn, 0.0, 0.0, &RV[0], min_Vm_Vn);
        lapack::lacpy(lapack::MatrixType::Upper,
            min_Vm_Vn, Vn, &V[0], Vm, &RV[0], min_Vm_Vn);

        // RU * RV.'
        std::vector<T> RURV(min_Um_Un * min_Vm_Vn);
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans,
            (use_ungqr ? blas::Op::ConjTrans : blas::Op::Trans),
            min_Um_Un, min_Vm_Vn, (Ark + Crk),
            1.0, &RU[0],   min_Um_Un,
                 &RV[0],   min_Vm_Vn,
            0.0, &RURV[0], min_Um_Un);

        // orthogonal QU and QV
        // [Unew, Sigma, VTnew] = svd(RU * RV.');
        lapack::gesvd(
            lapack::Job::SomeVec, lapack::Job::SomeVec,
            min_Um_Un, min_Vm_Vn,
            &RURV[0], min_Um_Un, &Sigma[0],
            &Unew[0], (use_ungqr ? min_Um_Un : Um),
            &VTnew[0], sizeS);
    }

    int64_t rk_new;
    if (rk) { // truncate according to rk
        rk_new = rk;
        if (rk > (Ark + Crk))
            rk_new = (Ark + Crk);
    }
    else { // truncate according to accuracy
        rk_new = sizeS;
        if (truncation_with_tol) {
            blas::real_type<T> Sigma_0 = Sigma[0];
            for (int64_t i = 1; i < sizeS; i++) {
                if (Sigma[i] < accuracy * Sigma_0) {
                    Sigma_0 = Sigma[i];
                    rk_new = i;
                    break;
                }
            }
        }
        else {
            for (int64_t i = 1; i < sizeS; i++) {
                if (Sigma[i] < accuracy) {
                    rk_new = i;
                    break;
                }
            }
        }
    }

    // todo
    // if (rk_new > max_rank) {
    //  "Rank after truncation is too big! rk_new:%d max_rank:%d";
    // }

    T* UV = new T[(m + n) * rk_new];

    if (use_ungqr) {
        lapack::ungqr(Um, min_Um_Un, min_Um_Un, &U[0], Um, &Utau[0]);
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            Um, rk_new, min_Um_Un,
            1.0, &U[0],    Um,
                 &Unew[0], min_Um_Un,
            0.0, &UV[0],   ldcu);
    }
    else {
        lapack::unmqr(
            blas::Side::Left, blas::Op::NoTrans,
            Um, rk_new, min_Um_Un,
            &U[0],    Um, &Utau[0],
            &Unew[0], Um);
        lapack::lacpy(lapack::MatrixType::General,
            Um, rk_new, &Unew[0], Um, UV, ldcu);
    }

    // VTnew eats Sigma.
    // todo: we may need to have uplo parameter:
    //       scale VT, if Lower, or scale U otherwise.
    for(int64_t i = 0; i < rk_new; ++i) {
        blas::scal((use_ungqr ? min_Vm_Vn : Vm), Sigma[i], &VTnew[i], sizeS);

        if (! use_ungqr) {
            for (int64_t j = 0; j < Vm; ++j) {
                VTnew[i + j * sizeS] = conj(VTnew[i + j * sizeS]);
            }
        }
    }

    T* UVptr = UV + m * rk_new;

    if (use_ungqr) {
        lapack::ungqr(Vm, min_Vm_Vn, min_Vm_Vn, &V[0], Vm, &Vtau[0]);
        std::vector<T> Vnew(Vm * rk_new);
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans,
            Vm, rk_new, min_Vm_Vn,
            1.0, &V[0],     Vm,
                 &VTnew[0], sizeS,
            0.0, &Vnew[0],  Vm);
        for (int64_t j = 0; j < rk_new; ++j) {
            for (int64_t i = 0; i < Vm; ++i) {
                UVptr[j + i * rk_new] = conj(Vnew[i + j * Vm]);
            }
        }
    }
    else {
        lapack::unmqr(
            blas::Side::Right, blas::Op::ConjTrans,
            rk_new, Vm, min_Vm_Vn,
            &V[0],     Vm, &Vtau[0],
            &VTnew[0], sizeS);
        lapack::lacpy(lapack::MatrixType::General,
            rk_new, Vm, &VTnew[0], sizeS, &UVptr[0], rk_new);
        for (int64_t i = 0; i < rk_new; ++i) {
            for (int64_t j = 0; j < Vm; ++j) {
                UVptr[i + j * rk_new] = conj(UVptr[i + j * rk_new]);
            }
        }
    }

    C.UVdata(UV);
    C.rk(rk_new);
}

// explicit instantaiton
template
void gemm(
    float beta,
    float const* AU, float const* AV,
    int64_t ldau, int64_t Ark,
    CompressedTile<float>& C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    double beta,
    double const* AU, double const* AV,
    int64_t ldau, int64_t Ark,
    CompressedTile<double>& C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    std::complex<float> beta,
    std::complex<float> const* AU, std::complex<float> const* AV,
    int64_t ldau, int64_t Ark,
    CompressedTile<std::complex<float>>& C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    std::complex<double> beta,
    std::complex<double> const* AU, std::complex<double> const* AV,
    int64_t ldau, int64_t Ark,
    CompressedTile<std::complex<double>>& C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);

} // namespace internal
} // namespace hcore
