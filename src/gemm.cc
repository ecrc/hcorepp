// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "hcore.hh"
#include "internal/check.hh"
#include "hcore/tile/dense.hh"
#include "internal/internal.hh"
#include "hcore/tile/compressed.hh"

#include "blas.hh"

#include <vector>
#include <complex>
#include <cassert>
#include <string>
#include <stdexcept>

namespace hcore {

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha A * B + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k dense tile.
/// @param[in] B
///     The k-by-n dense tile.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n dense tile.
///     On exit, overwritten by the result: alpha * A * B + beta C.
template <typename T>
void gemm(
    T alpha, DenseTile<T> const& A,
             DenseTile<T> const& B,
    T beta,  DenseTile<T>      & C)
{
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check_gemm(A, B, C);

    if (C.op() == blas::Op::NoTrans) {
        blas::gemm(
            blas::Layout::ColMajor, A.op(), B.op(),
            C.m(), C.n(), A.n(),
            alpha, A.data(), A.ld(),
                   B.data(), B.ld(),
            beta,  C.data(), C.ld());
    }
    else {
        blas::Op opA;
        if (A.op() == blas::Op::NoTrans) {
            opA = C.op();
        }
        else if ((A.op() == C.op()) || (!blas::is_complex<T>::value)) {
            opA = blas::Op::NoTrans;
        }
        else {
            const std::string& what_arg =
            "C is complex, C != blas::Op::NoTrans, and A != blas::Op::NoTrans.";
            throw std::invalid_argument(what_arg);
        }

        blas::Op opB;
        if (B.op() == blas::Op::NoTrans) {
            opB = C.op();
        }
        else if ((B.op() == C.op()) || (!blas::is_complex<T>::value)) {
            opB = blas::Op::NoTrans;
        }
        else {
            const std::string& what_arg =
            "C is complex, C != blas::Op::NoTrans, and B != blas::Op::NoTrans.";
            throw std::invalid_argument(what_arg);
        }

        using blas::conj;

        if (C.op() == blas::Op::ConjTrans) {
            alpha = conj(alpha);
            beta  = conj(beta);
        }

        blas::gemm(
            blas::Layout::ColMajor, opB, opA,
            C.n(), C.m(), A.n(),
            alpha, B.data(), B.ld(),
                   A.data(), A.ld(),
            beta,  C.data(), C.ld());
    }
}

// explicit instantaiton
template
void gemm(
    float alpha, DenseTile<float> const& A,
                 DenseTile<float> const& B,
    float beta,  DenseTile<float>      & C);
template
void gemm(
    double alpha, DenseTile<double> const& A,
                  DenseTile<double> const& B,
    double beta,  DenseTile<double>      & C);
template
void gemm(
    std::complex<float> alpha, DenseTile<std::complex<float>> const& A,
                               DenseTile<std::complex<float>> const& B,
    std::complex<float> beta,  DenseTile<std::complex<float>>      & C);
template
void gemm(
    std::complex<double> alpha, DenseTile<std::complex<double>> const& A,
                                DenseTile<std::complex<double>> const& B,
    std::complex<double> beta,  DenseTile<std::complex<double>>      & C);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha A * B + beta * C.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k dense tile.
/// @param[in] B
///     The k-by-n dense tile.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n compressed tile (U: m-by-Crk; V: Crk-by-n).
///     On exit, overwritten by the result: alpha * A * B + beta C.
template <typename T>
void gemm(
    T alpha,      DenseTile<T> const& A,
                  DenseTile<T> const& B,
    T beta,  CompressedTile<T>      & C)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check_gemm(A, B, C);

    T* W = new T[C.ldu() * C.n()];

    // W = alpha * A * B
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        C.m(), C.n(), A.n(),
        alpha, A.data(), A.ld(),
               B.data(), B.ld(),
        0.0,   &W[0],    C.ldu());

    // W += beta * CU * VC
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        C.m(), C.n(), C.rk(),
        beta, C.Udata(), C.ldu(),
              C.Vdata(), C.ldv(),
        1.0,  &W[0],     C.ldu());

    C.UVdata(W);
    C.rk(std::max(C.m(), C.n()));
}

// explicit instantaiton
template
void gemm(
    float alpha,      DenseTile<float> const& A,
                      DenseTile<float> const& B,
    float beta,  CompressedTile<float>      & C);
template
void gemm(
    double alpha,      DenseTile<double> const& A,
                       DenseTile<double> const& B,
    double beta,  CompressedTile<double>      & C);
template
void gemm(
    std::complex<float> alpha,      DenseTile<std::complex<float>> const& A,
                                    DenseTile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>      & C);
template
void gemm(
    std::complex<double> alpha,      DenseTile<std::complex<double>> const& A,
                                     DenseTile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>      & C);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha A * B + beta * C.
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
///     On exit, overwritten by the result: alpha * A * B + beta C.
template <typename T>
void gemm(
    T alpha,      DenseTile<T> const& A,
             CompressedTile<T> const& B,
    T beta,       DenseTile<T>      & C)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check_gemm(A, B, C);

    std::vector<T> W(C.m() * B.rk());

    // W = alpha * A * BU
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        C.m(), B.rk(), A.n(),
        alpha, A.data(),  A.ld(),
               B.Udata(), B.ldu(),
        0.0,   &W[0],     C.m());

    // C = W * BV + beta * C
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        C.m(), C.n(), B.rk(),
        1.0,  &W[0],     C.m(),
              B.Vdata(), B.ldv(),
        beta, C.data(),  C.ld());
}

// explicit instantaiton
template
void gemm(
    float alpha,      DenseTile<float> const& A,
                 CompressedTile<float> const& B,
    float beta,      DenseTile<float>      & C);
template
void gemm(
    double alpha,      DenseTile<double> const& A,
                  CompressedTile<double> const& B,
    double beta,       DenseTile<double>      & C);
template
void gemm(
    std::complex<float> alpha,      DenseTile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,       DenseTile<std::complex<float>>      & C);
template
void gemm(
    std::complex<double> alpha,      DenseTile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,       DenseTile<std::complex<double>>      & C);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha A * B + beta * C.
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
///     On exit, overwritten by the result: alpha * A * B + beta C.
/// @param[in] use_trmm
///     Use trmm. Default false.
/// @param[in] use_ungqr
///     Use ungqr with gemm. Default true.
/// @param[in] truncation_with_tol
///     Truncation to fixed accuracy * tolerance. Default false.
/// @param[in] rk
///     Truncation to fixed rank. rk >= 0. Default 0 (false).
template <typename T>
void gemm(
    T alpha,      DenseTile<T> const& A,
             CompressedTile<T> const& B,
    T beta,  CompressedTile<T>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check_gemm(A, B, C);

    std::vector<T> W(C.m() * B.rk());

    // W = alpha * A * BU
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        C.m(), B.rk(), A.n(),
        alpha, A.data(),  A.ld(),
               B.Udata(), B.ldu(),
        0.0,   &W[0],     C.m());

    internal::gemm(
        beta, &W[0], B.Vdata(), C.m(), B.rk(), C,
        use_trmm, use_ungqr, truncation_with_tol, rk);
}

// explicit instantaiton
template
void gemm(
    float alpha,      DenseTile<float> const& A,
                 CompressedTile<float> const& B,
    float beta,  CompressedTile<float>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    double alpha,      DenseTile<double> const& A,
                  CompressedTile<double> const& B,
    double beta,  CompressedTile<double>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    std::complex<float> alpha,      DenseTile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    std::complex<double> alpha,      DenseTile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha A * B + beta * C.
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
///     On exit, overwritten by the result: alpha * A * B + beta C.
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
                  DenseTile<T> const& B,
    T beta,       DenseTile<T>      & C)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check_gemm(A, B, C);

    std::vector<T> W(A.rk() * C.n());

    // W = AV * B
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        A.rk(), C.n(), A.n(),
        1.0, A.Vdata(), A.ldv(),
             B.data(),  B.ld(),
        0.0, &W[0],     A.rk());

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
                      DenseTile<float> const& B,
    float beta,       DenseTile<float>      & C);
template
void gemm(
    double alpha, CompressedTile<double> const& A,
                       DenseTile<double> const& B,
    double beta,       DenseTile<double>      & C);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                                    DenseTile<std::complex<float>> const& B,
    std::complex<float> beta,       DenseTile<std::complex<float>>      & C);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                     DenseTile<std::complex<double>> const& B,
    std::complex<double> beta,       DenseTile<std::complex<double>>      & C);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha A * B + beta * C.
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
///     On exit, overwritten by the result: alpha * A * B + beta C.
/// @param[in] use_trmm
///     Use trmm. Default false.
/// @param[in] use_ungqr
///     Use ungqr with gemm. Default true.
/// @param[in] truncation_with_tol
///     Truncation to fixed accuracy * tolerance. Default false.
/// @param[in] rk
///     Truncation to fixed rank. rk >= 0. Default 0 (false).
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
                  DenseTile<T> const& B,
    T beta,  CompressedTile<T>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check_gemm(A, B, C);

    std::vector<T> W(A.rk() * C.n());

    // W = alpha * AV * B
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        A.rk(), C.n(), A.n(),
        alpha, A.Vdata(), A.ldv(),
               B.data(),  B.ld(),
        0.0,   &W[0],     A.rk());

    internal::gemm(
        beta, A.Udata(), &W[0], A.ldu(), A.rk(), C,
        use_trmm, use_ungqr, truncation_with_tol, rk);
}

// explicit instantaiton
template
void gemm(
    float alpha, CompressedTile<float> const& A,
                      DenseTile<float> const& B,
    float beta,  CompressedTile<float>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    double alpha, CompressedTile<double> const& A,
                       DenseTile<double> const& B,
    double beta,  CompressedTile<double>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                                    DenseTile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                     DenseTile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha A * B + beta * C.
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
///     On exit, overwritten by the result: alpha * A * B + beta C.
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
             CompressedTile<T> const& B,
    T beta,       DenseTile<T>      & C)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check_gemm(A, B, C);

    std::vector<T> W0(A.rk() * B.rk());

    // W0 = alpha * AV * BU
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        A.rk(), B.rk(), A.n(),
        alpha, A.Vdata(), A.ldv(),
               B.Udata(), B.ldu(),
        0.0,   &W0[0],    A.rk());

    if (A.rk() <= B.rk()) {
        std::vector<T> W1(A.rk() * C.n());

        // W1 = W0 * BV
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            A.rk(), C.n(), B.rk(),
            1.0, &W0[0],    A.rk(),
                 B.Vdata(), B.ldv(),
            0.0, &W1[0],    A.rk());

        // C = AU * W1 + beta * C
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            C.m(), C.n(), A.rk(),
            1.0,  A.Udata(), A.ldu(),
                  &W1[0],    A.rk(),
            beta, C.data(),  C.ld());
    }
    else {
        std::vector<T> W1(C.m() * B.rk());

        // W1 = AU * W0
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            C.m(), B.rk(), A.rk(),
            1.0,  A.Udata(), A.ldu(),
                  &W0[0],    A.rk(),
            0.0,  &W1[0],    C.m());

        // C = W1 * BV + beta * C
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            C.m(), C.n(), B.rk(),
            1.0,  &W1[0],    C.m(),
                  B.Vdata(), B.ldv(),
            beta, C.data(),  C.ld());
    }
}

// explicit instantaiton
template
void gemm(
    float alpha, CompressedTile<float> const& A,
                 CompressedTile<float> const& B,
    float beta,       DenseTile<float>      & C);
template
void gemm(
    double alpha, CompressedTile<double> const& A,
                  CompressedTile<double> const& B,
    double beta,       DenseTile<double>      & C);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,       DenseTile<std::complex<float>>      & C);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,       DenseTile<std::complex<double>>      & C);

// =============================================================================
//
/// General matrix-matrix multiplication, C = alpha A * B + beta * C.
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
///     On exit, overwritten by the result: alpha * A * B + beta C.
/// @param[in] use_trmm
///     Use trmm. Default false.
/// @param[in] use_ungqr
///     Use ungqr with gemm. Default true.
/// @param[in] truncation_with_tol
///     Truncation to fixed accuracy * tolerance. Default false.
/// @param[in] rk
///     Truncation to fixed rank. rk >= 0. Default 0 (false).
template <typename T>
void gemm(
    T alpha, CompressedTile<T> const& A,
             CompressedTile<T> const& B,
    T beta,  CompressedTile<T>& C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk)
{
    assert(A.op() == blas::Op::NoTrans); // todo
    assert(B.op() == blas::Op::NoTrans); // todo
    assert(C.layout() == blas::Layout::ColMajor); // todo

    internal::check_gemm(A, B, C);

    // W0 = alpha * AV * BU
    std::vector<T> W0(A.rk() * B.rk());
    blas::gemm(
        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        A.rk(), B.rk(), A.n(),
        alpha, A.Vdata(), A.ldv(),
               B.Udata(), B.ldu(),
        0.0,   &W0[0],    A.rk());

    if (A.rk() <= B.rk()) {
        std::vector<T> W1(A.rk() * C.n());

        // W1 = W0 * BV
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            A.rk(), C.n(), B.rk(),
            1.0, &W0[0],    A.rk(),
                 B.Vdata(), B.ldv(),
            0.0, &W1[0],    A.rk());

        internal::gemm(
            beta, A.Udata(), &W1[0], A.ldu(), A.rk(), C,
            use_trmm, use_ungqr, truncation_with_tol, rk);
    }
    else {
        std::vector<T> W1(C.m() * B.rk());

        // W1 = AU * W0
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            C.m(), B.rk(), A.rk(),
            1.0,  A.Udata(), A.ldu(),
                  &W0[0],    A.rk(),
            0.0,  &W1[0],    C.m());

        internal::gemm(
            beta, &W1[0], B.Vdata(), C.m(), B.rk(), C,
            use_trmm, use_ungqr, truncation_with_tol, rk);
    }
}

// explicit instantaiton
template
void gemm(
    float alpha, CompressedTile<float> const& A,
                 CompressedTile<float> const& B,
    float beta,  CompressedTile<float>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    double alpha, CompressedTile<double> const& A,
                  CompressedTile<double> const& B,
    double beta,  CompressedTile<double>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>      & C,
    bool use_trmm, bool use_ungqr, bool truncation_with_tol, int64_t rk);

} // namespace hcore
