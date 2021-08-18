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
#include <internal/internal.hh>

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

/// General matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C.
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
void gemm(T alpha, Tile<T> const& A,
                   Tile<T> const& B,
          T beta,  Tile<T>&       C) {
    internal::check::gemm(A, B, C);

    if (C.op() == blas::Op::NoTrans) {
        blas::gemm(C.layout(), A.op(), B.op(), C.m(), C.n(), A.n(),
                   alpha, A.data(), A.ld(),
                          B.data(), B.ld(),
                   beta,  C.data(), C.ld());
    }
    else {
        blas::Op opA;
        if (A.op() == blas::Op::NoTrans)
            opA = C.op();
        else if (A.op() == C.op() || !C.is_complex)
            opA = blas::Op::NoTrans;
        else
            throw Error();

        blas::Op opB;
        if (B.op() == blas::Op::NoTrans)
            opB = C.op();
        else if (B.op() == C.op() || !C.is_complex)
            opB = blas::Op::NoTrans;
        else
            throw Error();

        using blas::conj;

        if (C.op() == blas::Op::ConjTrans) {
            alpha = conj(alpha);
            beta  = conj(beta);
        }

        blas::gemm(C.layout(), opB, opA, C.n(), C.m(), A.n(),
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
void gemm(T alpha,           Tile<T> const& A,
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

    internal::rsvd(
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

    internal::rsvd(
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

        internal::rsvd(
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

        internal::rsvd(
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
