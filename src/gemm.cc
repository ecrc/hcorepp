// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "hcore.hh"
#include "hcore/tile.hh"
#include "hcore/check.hh"
#include "hcore/exception.hh"
#include "hcore/base_tile.hh"
#include "hcore/compressed_tile.hh"

#include "blas.hh"
#include "lapack.hh"

#include <new>
#include <vector>
#include <cstdint>
#include <complex>
#include <cassert>
#include <algorithm>
#include <initializer_list>

namespace hcore {
namespace internal {

template <typename T>
void reduced_svd(
    T beta, T const* AU, T const* AV, int64_t ldau, int64_t Ark,
    CompressedTile<T>& C, bool use_trmm=false, bool use_ungqr=true,
    bool truncated_svd=false, int64_t fixed_rk=0)
{
    using blas::conj;

    T zero = 0.0;
    T one  = 1.0;

    int64_t m = C.m();
    int64_t n = C.n();

    T* CU = C.Udata();
    T* CV = C.Vdata();

    int64_t Crk = C.rk();

    blas::real_type<T> accuracy = C.accuracy();

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
        min_Um_Un, Un, zero, zero, &RU[0], min_Um_Un);
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
            one, &V[0],  Vm,
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
            min_Vm_Vn, Vn, zero, zero, &RV[0], min_Vm_Vn);
        lapack::lacpy(lapack::MatrixType::Upper,
            min_Vm_Vn, Vn, &V[0], Vm, &RV[0], min_Vm_Vn);

        // RU * RV.'
        std::vector<T> RURV(min_Um_Un * min_Vm_Vn);
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans,
            (use_ungqr ? blas::Op::ConjTrans : blas::Op::Trans),
            min_Um_Un, min_Vm_Vn, (Ark + Crk),
            one,  &RU[0],   min_Um_Un,
                  &RV[0],   min_Vm_Vn,
            zero, &RURV[0], min_Um_Un);

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
    if (fixed_rk) { // truncate according to fixed_rk
        rk_new = fixed_rk;
        if (fixed_rk > (Ark + Crk))
            rk_new = (Ark + Crk);
    }
    else { // truncate according to accuracy
        rk_new = sizeS;
        if (truncated_svd) {
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

    hcore_error_if_msg(
        rk_new > std::min(m, n),
        "Rank (%lld) after truncation (%lld) is greater than max rank (%lld)",
        (long long)Crk, (long long)rk_new, (long long)std::min(m, n));

    T* UV = new T[(ldcu + n) * rk_new];

    if (use_ungqr) {
        lapack::ungqr(Um, min_Um_Un, min_Um_Un, &U[0], Um, &Utau[0]);
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            Um, rk_new, min_Um_Un,
            one,  &U[0],    Um,
                  &Unew[0], min_Um_Un,
            zero, &UV[0],   ldcu);
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

    T* UVptr = UV + ldcu * rk_new;

    if (use_ungqr) {
        lapack::ungqr(Vm, min_Vm_Vn, min_Vm_Vn, &V[0], Vm, &Vtau[0]);
        std::vector<T> Vnew(Vm * rk_new);
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::ConjTrans,
            Vm, rk_new, min_Vm_Vn,
            one,  &V[0],     Vm,
                  &VTnew[0], sizeS,
            zero, &Vnew[0],  Vm);
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

} // namespace internal

/// General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k matrix.
/// @param[in] B
///     The k-by-n matrix.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n matrix.
///     On exit, overwritten by the result: alpha * op(A) * op(B) + beta C.
template <typename T>
void gemm(
    T alpha, Tile<T> const& A,
             Tile<T> const& B,
    T beta,  Tile<T>      & C) {
    internal::check::gemm(A, B, C);

    if (C.op() == blas::Op::NoTrans) {
        blas::gemm(C.layout(), A.op(), B.op(),
                   C.m(), C.n(), A.n(),
                   alpha, A.data(), A.ld(),
                          B.data(), B.ld(),
                   beta,  C.data(), C.ld());
    }
    else {
        hcore_error_if(C.is_complex &&
                       A.op() != blas::Op::NoTrans && A.op() != C.op());

        blas::Op opA;
        if (A.op() == blas::Op::NoTrans) {
            opA = C.op();
        }
        else if (A.op() == C.op() || !C.is_complex) {
            opA = blas::Op::NoTrans;
        }
        else {
            throw hcore::Error();
        }

        hcore_error_if(C.is_complex &&
                       B.op() != blas::Op::NoTrans && B.op() != C.op());

        blas::Op opB;
        if (B.op() == blas::Op::NoTrans) {
            opB = C.op();
        }
        else if (B.op() == C.op() || !C.is_complex) {
            opB = blas::Op::NoTrans;
        }
        else {
            throw hcore::Error();
        }

        using blas::conj;

        if (C.op() == blas::Op::ConjTrans) {
            alpha = conj(alpha);
            beta  = conj(beta);
        }

        blas::gemm(C.layout(), opB, opA,
                   C.n(), C.m(), A.n(),
                   alpha, B.data(), B.ld(),
                          A.data(), A.ld(),
                   beta,  C.data(), C.ld());
    }
}

template
void gemm(
    float alpha, Tile<float> const& A,
                 Tile<float> const& B,
    float beta,  Tile<float>      & C);
template
void gemm(
    double alpha, Tile<double> const& A,
                  Tile<double> const& B,
    double beta,  Tile<double>      & C);
template
void gemm(
    std::complex<float> alpha, Tile<std::complex<float>> const& A,
                               Tile<std::complex<float>> const& B,
    std::complex<float> beta,  Tile<std::complex<float>>      & C);
template
void gemm(
    std::complex<double> alpha, Tile<std::complex<double>> const& A,
                                Tile<std::complex<double>> const& B,
    std::complex<double> beta,  Tile<std::complex<double>>      & C);

/// General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k matrix.
/// @param[in] B
///     The k-by-n matrix.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n compressed matrix (A=UV), U: m-by-Crk; V: Crk-by-n.
///     On exit, overwritten by the result: alpha * op(A) * op(B) + beta C.
template <typename T>
void gemm(
    T alpha,      Tile<T> const& A,
                  Tile<T> const& B,
    T beta,  CompressedTile<T>      & C)
{
    assert(C.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check::gemm(A, B, C);

    T zero = 0.0;
    T one  = 1.0;

    T* W = new T[C.ldu() * C.n()];

    // W = alpha * A * B
    blas::gemm(C.layout(), A.op(), B.op(),
               A.m(), B.n(), A.n(),
               alpha, A.data(), A.ld(),
                      B.data(), B.ld(),
               zero,  &W[0],    C.ldu());

    // W += beta * CU * CV
    blas::gemm(C.layout(), blas::Op::NoTrans, blas::Op::NoTrans,
               C.m(), C.n(), C.rk(),
               beta, C.Udata(), C.ldu(),
                     C.Vdata(), C.ldv(),
               one,  &W[0],     C.ldu());


    C.UVdata(W);
    C.to_full_rk();
}

// explicit instantaiton
template
void gemm(
    float alpha,      Tile<float> const& A,
                      Tile<float> const& B,
    float beta,  CompressedTile<float>      & C);
template
void gemm(
    double alpha,      Tile<double> const& A,
                       Tile<double> const& B,
    double beta,  CompressedTile<double>      & C);
template
void gemm(
    std::complex<float> alpha,      Tile<std::complex<float>> const& A,
                                    Tile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>      & C);
template
void gemm(
    std::complex<double> alpha,      Tile<std::complex<double>> const& A,
                                     Tile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>      & C);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k dense tile.
/// @param[in] B
///     The k-by-n compressed tile (U: k-by-Brk; V: Brk-by-n).
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n dense tile.
///     On exit, overwritten by the result: alpha * op(A) * op(B) + beta C.
template <typename T>
void gemm(
    T alpha,      Tile<T> const& A,
             CompressedTile<T> const& B,
    T beta,       Tile<T>      & C)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check::gemm(A, B, C);

    T zero = 0.0;
    T one  = 1.0;

    std::vector<T> W(C.m() * B.rk());

    // W = alpha * A * BU
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        C.m(), B.rk(), A.n(),
        alpha, A.data(),  A.ld(),
               B.Udata(), B.ldu(),
        zero,  &W[0],     C.m());

    // C = W * BV + beta * C
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        C.m(), C.n(), B.rk(),
        one,  &W[0],     C.m(),
              B.Vdata(), B.ldv(),
        beta, C.data(),  C.ld());
}

// explicit instantaiton
template
void gemm(
    float alpha,      Tile<float> const& A,
                 CompressedTile<float> const& B,
    float beta,      Tile<float>      & C);
template
void gemm(
    double alpha,      Tile<double> const& A,
                  CompressedTile<double> const& B,
    double beta,       Tile<double>      & C);
template
void gemm(
    std::complex<float> alpha,      Tile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,       Tile<std::complex<float>>      & C);
template
void gemm(
    std::complex<double> alpha,      Tile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,       Tile<std::complex<double>>      & C);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k dense tile.
/// @param[in] B
///     The k-by-n compressed tile (U: k-by-Brk; V: Brk-by-n).
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n compressed tile (U: m-by-Crk; V: Crk-by-n).
///     On exit, overwritten by the result: alpha * op(A) * op(B) + beta C.
/// @param[in] use_trmm
///     Use trmm. Default false.
/// @param[in] use_ungqr
///     Use ungqr with gemm. Default true.
/// @param[in] truncated_svd
///     Truncation to fixed accuracy * tolerance. Default false.
/// @param[in] fixed_rk
///     Truncation to fixed rank. fixed_rk >= 0. Default 0 (false).
template <typename T>
void gemm(
    T alpha,      Tile<T> const& A,
             CompressedTile<T> const& B,
    T beta,  CompressedTile<T>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check::gemm(A, B, C);

    T zero = 0.0;

    std::vector<T> W(C.m() * B.rk());

    // W = alpha * A * BU
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        C.m(), B.rk(), A.n(),
        alpha, A.data(),  A.ld(),
               B.Udata(), B.ldu(),
        zero,  &W[0],     C.m());

    internal::reduced_svd(
        beta, &W[0], B.Vdata(), C.m(), B.rk(), C,
        use_trmm, use_ungqr, truncated_svd, fixed_rk);
}

// explicit instantaiton
template
void gemm(
    float alpha,      Tile<float> const& A,
                 CompressedTile<float> const& B,
    float beta,  CompressedTile<float>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);
template
void gemm(
    double alpha,      Tile<double> const& A,
                  CompressedTile<double> const& B,
    double beta,  CompressedTile<double>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);
template
void gemm(
    std::complex<float> alpha,      Tile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);
template
void gemm(
    std::complex<double> alpha,      Tile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k compressed tile (U: m-by-Ark; V: Ark-by-k).
/// @param[in] B
///     The k-by-n dense tile.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n dense tile.
///     On exit, overwritten by the result: alpha * op(A) * op(B) + beta C.
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
                  Tile<T> const& B,
    T beta,       Tile<T>      & C)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check::gemm(A, B, C);

    T zero = 0.0;
    T one  = 1.0;

    std::vector<T> W(A.rk() * C.n());

    // W = AV * B
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        A.rk(), C.n(), A.n(),
        one,  A.Vdata(), A.ldv(),
              B.data(),  B.ld(),
        zero, &W[0],     A.rk());

    // C = alpha * AU * W + beta * C
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        C.m(), C.n(), A.rk(),
        alpha, A.Udata(), A.ldu(),
               &W[0],     A.rk(),
        beta,  C.data(),  C.ld());
}

// explicit instantaiton
template
void gemm(
    float alpha, CompressedTile<float> const& A,
                      Tile<float> const& B,
    float beta,       Tile<float>      & C);
template
void gemm(
    double alpha, CompressedTile<double> const& A,
                       Tile<double> const& B,
    double beta,       Tile<double>      & C);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                                    Tile<std::complex<float>> const& B,
    std::complex<float> beta,       Tile<std::complex<float>>      & C);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                     Tile<std::complex<double>> const& B,
    std::complex<double> beta,       Tile<std::complex<double>>      & C);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k compressed tile (U: m-by-Ark; V: Ark-by-k).
/// @param[in] B
///     The k-by-n dense tile.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n compressed tile (U: m-by-Crk; V: Crk-by-n).
///     On exit, overwritten by the result: alpha * op(A) * op(B) + beta C.
/// @param[in] use_trmm
///     Use trmm. Default false.
/// @param[in] use_ungqr
///     Use ungqr with gemm. Default true.
/// @param[in] truncated_svd
///     Truncation to fixed accuracy * tolerance. Default false.
/// @param[in] fixed_rk
///     Truncation to fixed rank. fixed_rk >= 0. Default 0 (false).
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
                  Tile<T> const& B,
    T beta,  CompressedTile<T>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check::gemm(A, B, C);

    T zero = 0.0;

    std::vector<T> W(A.rk() * C.n());

    // W = alpha * AV * B
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        A.rk(), C.n(), A.n(),
        alpha, A.Vdata(), A.ldv(),
               B.data(),  B.ld(),
        zero,  &W[0],     A.rk());

    internal::reduced_svd(
        beta, A.Udata(), &W[0], A.ldu(), A.rk(), C,
        use_trmm, use_ungqr, truncated_svd, fixed_rk);
}

// explicit instantaiton
template
void gemm(
    float alpha, CompressedTile<float> const& A,
                      Tile<float> const& B,
    float beta,  CompressedTile<float>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);
template
void gemm(
    double alpha, CompressedTile<double> const& A,
                       Tile<double> const& B,
    double beta,  CompressedTile<double>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                                    Tile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                     Tile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k compressed tile (U: m-by-Ark; V: Ark-by-k).
/// @param[in] B
///     The k-by-n compressed tile (U: k-by-Brk; V: Brk-by-n).
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n dense tile.
///     On exit, overwritten by the result: alpha * op(A) * op(B) + beta C.
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
             CompressedTile<T> const& B,
    T beta,       Tile<T>      & C)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check::gemm(A, B, C);

    T zero = 0.0;
    T one  = 1.0;

    std::vector<T> W0(A.rk() * B.rk());

    // W0 = alpha * AV * BU
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        A.rk(), B.rk(), A.n(),
        alpha, A.Vdata(), A.ldv(),
               B.Udata(), B.ldu(),
        zero,  &W0[0],    A.rk());

    if (A.rk() <= B.rk()) {
        std::vector<T> W1(A.rk() * C.n());

        // W1 = W0 * BV
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            A.rk(), C.n(), B.rk(),
            one,  &W0[0],    A.rk(),
                  B.Vdata(), B.ldv(),
            zero, &W1[0],    A.rk());

        // C = AU * W1 + beta * C
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            C.m(), C.n(), A.rk(),
            one,  A.Udata(), A.ldu(),
                  &W1[0],    A.rk(),
            beta, C.data(),  C.ld());
    }
    else {
        std::vector<T> W1(C.m() * B.rk());

        // W1 = AU * W0
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            C.m(), B.rk(), A.rk(),
            one,  A.Udata(), A.ldu(),
                  &W0[0],    A.rk(),
            zero, &W1[0],    C.m());

        // C = W1 * BV + beta * C
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            C.m(), C.n(), B.rk(),
            one,  &W1[0],    C.m(),
                  B.Vdata(), B.ldv(),
            beta, C.data(),  C.ld());
    }
}

// explicit instantaiton
template
void gemm(
    float alpha, CompressedTile<float> const& A,
                 CompressedTile<float> const& B,
    float beta,       Tile<float>      & C);
template
void gemm(
    double alpha, CompressedTile<double> const& A,
                  CompressedTile<double> const& B,
    double beta,       Tile<double>      & C);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,       Tile<std::complex<float>>      & C);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,       Tile<std::complex<double>>      & C);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha * op(A) * op(B) + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k compressed tile (U: m-by-Ark; V: Ark-by-k).
/// @param[in] B
///     The k-by-n compressed tile (U: k-by-Brk; V: Brk-by-n).
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n compressed tile (U: m-by-Crk; V: Crk-by-n).
///     On exit, overwritten by the result: alpha * op(A) * op(B) + beta C.
/// @param[in] use_trmm
///     Use trmm. Default false.
/// @param[in] use_ungqr
///     Use ungqr with gemm. Default true.
/// @param[in] truncated_svd
///     Truncation to fixed accuracy * tolerance. Default false.
/// @param[in] fixed_rk
///     Truncation to fixed rank. fixed_rk >= 0. Default 0 (false).
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
             CompressedTile<T> const& B,
    T beta,  CompressedTile<T>& C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    T zero = 0.0;
    T one  = 1.0;

    internal::check::gemm(A, B, C);

    // W0 = alpha * AV * BU
    std::vector<T> W0(A.rk() * B.rk());
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        A.rk(), B.rk(), A.n(),
        alpha, A.Vdata(), A.ldv(),
               B.Udata(), B.ldu(),
        zero,  &W0[0],    A.rk());

    if (A.rk() <= B.rk()) {
        std::vector<T> W1(A.rk() * C.n());

        // W1 = W0 * BV
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            A.rk(), C.n(), B.rk(),
            one,  &W0[0],    A.rk(),
                  B.Vdata(), B.ldv(),
            zero, &W1[0],    A.rk());

        internal::reduced_svd(
            beta, A.Udata(), &W1[0], A.ldu(), A.rk(), C,
            use_trmm, use_ungqr, truncated_svd, fixed_rk);
    }
    else {
        std::vector<T> W1(C.m() * B.rk());

        // W1 = AU * W0
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            C.m(), B.rk(), A.rk(),
            one,  A.Udata(), A.ldu(),
                  &W0[0],    A.rk(),
            zero, &W1[0],    C.m());

        internal::reduced_svd(
            beta, &W1[0], B.Vdata(), C.m(), B.rk(), C,
            use_trmm, use_ungqr, truncated_svd, fixed_rk);
    }
}

// explicit instantaiton
template
void gemm(
    float alpha, CompressedTile<float> const& A,
                 CompressedTile<float> const& B,
    float beta,  CompressedTile<float>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);
template
void gemm(
    double alpha, CompressedTile<double> const& A,
                  CompressedTile<double> const& B,
    double beta,  CompressedTile<double>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>      & C,
    bool use_trmm, bool use_ungqr, bool truncated_svd, int64_t fixed_rk);

} // namespace hcore
