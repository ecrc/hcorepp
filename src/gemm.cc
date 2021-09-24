// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include <algorithm>
#include <cstdint>
#include <complex>
#include <cassert>
#include <vector>
#include <new>

#include "blas.hh"

#include "hcore/compressed_tile.hh"
#include "hcore/internal/check.hh"
#include "internal/internal.hh"
#include "hcore/exception.hh"
#include "hcore/options.hh"
#include "hcore/hcore.hh"
#include "hcore/tile.hh"

namespace hcore {

//------------------------------------------------------------------------------
/// General matrix-matrix multiplication. Performs the matrix-matrix operation:
/// $C = \alpha A B + \beta C$, where alpha and beta are scalars, and $A$, $B$,
/// and $C$ are matrices, with $A$ an m-by-k matrix, $B$ a k-by-n matrix, and
/// $C$ an m-by-n matrix. The matrices can be transposed or conjugate-transposed
/// beforehand. For example:
///     auto AT = hcore::transpose(A);
///     auto BT = hcore::conjugate_transpose(B);
///     hcore::gemm(alpha, AT, BT, beta, C);
///
/// @tparam T
///     One of float, double, std::complex<float>, std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k matrix $A$.
/// @param[in] B
///     The k-by-n matrix $B$.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n matrix $C$.
///     On exit, overwritten by the result $\alpha A B + \beta C$.
///
/// @ingroup gemm
template <typename T>
void gemm(T alpha, Tile<T> const& A,
                   Tile<T> const& B,
          T beta,  Tile<T>&       C)
{
    internal::check_gemm(A, B, C);

    if (C.op() == blas::Op::NoTrans) {
        blas::gemm(C.layout(), A.op(), B.op(),
                   C.mb(), C.nb(), A.nb(),
                   alpha, A.data(), A.stride(),
                          B.data(), B.stride(),
                   beta,  C.data(), C.stride());
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

        blas::gemm(C.layout(), opB, opA,
                   C.nb(), C.mb(), A.nb(),
                   alpha, B.data(), B.stride(),
                          A.data(), A.stride(),
                   beta,  C.data(), C.stride());
    }
}

template
void gemm(float alpha, Tile<float> const& A,
                       Tile<float> const& B,
          float beta,  Tile<float>&       C);
template
void gemm(double alpha, Tile<double> const& A,
                        Tile<double> const& B,
          double beta,  Tile<double>&       C);
template
void gemm(std::complex<float> alpha, Tile<std::complex<float>> const& A,
                                     Tile<std::complex<float>> const& B,
          std::complex<float> beta,  Tile<std::complex<float>>&       C);
template
void gemm(std::complex<double> alpha, Tile<std::complex<double>> const& A,
                                      Tile<std::complex<double>> const& B,
          std::complex<double> beta,  Tile<std::complex<double>>&       C);

//------------------------------------------------------------------------------
/// General matrix-matrix multiplication. Performs the matrix-matrix operation:
/// $C = \alpha A B + \beta C$, where alpha and beta are scalars, and $A$, $B$,
/// and $C$ are matrices, with $A$ an m-by-k compressed matrix ($A = U V$,
/// where $U$ an m-by-Ark and $V$ a Ark-by-k) $B$ a k-by-n matrix, and $C$ an
/// m-by-n matrix. The matrices can be transposed or conjugate-transposed
/// beforehand. For example:
///     auto AT = hcore::transpose(A);
///     auto BT = hcore::conjugate_transpose(B);
///     hcore::gemm(alpha, AT, BT, beta, C);
///
/// @tparam T
///     One of float, double, std::complex<float>, std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k compressed matrix $A = U V$, where $U$ the m-by-Ark matrix
///     and $V$ the Ark-by-k matrix.
/// @param[in] B
///     The k-by-n matrix $B$.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n matrix $C$.
///     On exit, overwritten by the result $\alpha A B + \beta C$.
///
/// @ingroup gemm
template <typename T>
void gemm(T alpha, CompressedTile<T> const& A,
                             Tile<T> const& B,
          T beta,            Tile<T>&       C)
{
    assert(C.op() == blas::Op::NoTrans); // todo

    internal::check_gemm(A, B, C);

    std::vector<T> W(A.rk()*C.nb());
    int64_t ldw = C.layout() == blas::Layout::ColMajor ? A.rk() : C.nb();

    T zero = 0.0;
    T one  = 1.0;

    // W = AV * B
    blas::gemm(C.layout(), A.op(), B.op(),
               A.rk(), C.nb(), A.nb(),
               one,  A.Vdata(), A.Vstride(),
                     B. data(), B. stride(),
               zero, &W[0],     ldw);

    // C = alpha * AU * W + beta * C
    blas::gemm(C.layout(), A.op(), blas::Op::NoTrans,
               C.mb(), C.nb(), A.rk(),
               alpha, A.Udata(), A.Ustride(),
                      &W[0],     ldw,
               beta,  C. data(), C. stride());
}

template
void gemm(float alpha, CompressedTile<float> const& A,
                                 Tile<float> const& B,
          float beta,            Tile<float>&       C);
template
void gemm(double alpha, CompressedTile<double> const& A,
                                  Tile<double> const& B,
          double beta,            Tile<double>&       C);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                                         Tile<std::complex<float>> const& B,
    std::complex<float> beta,            Tile<std::complex<float>>&       C);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                          Tile<std::complex<double>> const& B,
    std::complex<double> beta,            Tile<std::complex<double>>&       C);

//------------------------------------------------------------------------------
/// General matrix-matrix multiplication. Performs the matrix-matrix operation:
/// $C = \alpha A B + \beta C$, where alpha and beta are scalars, and $A$, $B$,
/// and $C$ are matrices, with $A$ an m-by-k matrix, $B$ a k-by-n compressed
/// matrix ($B = U V$, where $U$ an k-by-Brk and $V$ a Brk-by-n), and $C$ an
/// m-by-n matrix. The matrices can be transposed or conjugate-transposed
/// beforehand. For example:
///     auto AT = hcore::transpose(A);
///     auto BT = hcore::conjugate_transpose(B);
///     hcore::gemm(alpha, AT, BT, beta, C);
///
/// @tparam T
///     One of float, double, std::complex<float>, std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k matrix $A$.
/// @param[in] B
///     The k-by-n compressed matrix $B = U V$, where $U$ the k-by-Brk matrix
///     and $V$ the Brk-by-n matrix.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n matrix $C$.
///     On exit, overwritten by the result $\alpha A B + \beta C$.
///
/// @ingroup gemm
template <typename T>
void gemm(T alpha,           Tile<T> const& A,
                   CompressedTile<T> const& B,
          T beta,            Tile<T>&       C)
{
    assert(C.op() == blas::Op::NoTrans); // todo

    internal::check_gemm(A, B, C);

    std::vector<T> W(C.mb()*B.rk());
    int64_t ldw = C.layout() == blas::Layout::ColMajor ? C.mb() : B.rk();

    T zero = 0.0;

    // W = alpha * A * BU
    blas::gemm(C.layout(), A.op(), B.op(),
               C.mb(), B.rk(), A.nb(),
               alpha, A. data(), A. stride(),
                      B.Udata(), B.Ustride(),
               zero,  &W[0],     ldw);

    T one  = 1.0;

    // C = W * BV + beta * C
    blas::gemm(C.layout(), blas::Op::NoTrans, B.op(),
               C.mb(), C.nb(), B.rk(),
               one,  &W[0],     ldw,
                     B.Vdata(), B.Vstride(),
               beta, C. data(), C. stride());
}

template
void gemm(float alpha,           Tile<float> const& A,
                       CompressedTile<float> const& B,
          float beta,            Tile<float>&       C);
template
void gemm(double alpha,           Tile<double> const& A,
                        CompressedTile<double> const& B,
          double beta,            Tile<double>&       C);
template
void gemm(
    std::complex<float> alpha,           Tile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,            Tile<std::complex<float>>&       C);
template
void gemm(
    std::complex<double> alpha,           Tile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,            Tile<std::complex<double>>&       C);

//------------------------------------------------------------------------------
/// General matrix-matrix multiplication. Performs the matrix-matrix operation:
/// $C = \alpha A B + \beta C$, where alpha and beta are scalars, and $A$, $B$,
/// and $C$ are matrices, with $A$ an m-by-k compressed matrix ($A = U V$,
/// where $U$ an m-by-Ark and $V$ a Ark-by-k), $B$ a k-by-n compressed matrix
/// ($B = U V$, where $U$ an k-by-Brk and $V$ a Brk-by-n), and $C$ an
/// m-by-n matrix. The matrices can be transposed or conjugate-transposed
/// beforehand. For example:
///     auto AT = hcore::transpose(A);
///     auto BT = hcore::conjugate_transpose(B);
///     hcore::gemm(alpha, AT, BT, beta, C);
///
/// @tparam T
///     One of float, double, std::complex<float>, std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k compressed matrix $A = U V$, where $U$ the m-by-Ark matrix
///     and $V$ the Ark-by-k matrix.
/// @param[in] B
///     The k-by-n compressed matrix $B = U V$, where $U$ the k-by-Brk matrix
///     and $V$ the Brk-by-n matrix.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n matrix $C$.
///     On exit, overwritten by the result $\alpha A B + \beta C$.
///
/// @ingroup gemm
template <typename T>
void gemm(T alpha, CompressedTile<T> const& A,
                   CompressedTile<T> const& B,
          T beta,            Tile<T>&       C)
{
    assert(C.op() == blas::Op::NoTrans); // todo

    internal::check_gemm(A, B, C);

    T zero = 0.0;
    T one  = 1.0;

    std::vector<T> W0(A.rk() * B.rk());
    int64_t ldw0 = C.layout() == blas::Layout::ColMajor ? A.rk() : B.rk();

    // W0 = alpha * AV * BU
    blas::gemm(C.layout(), A.op(), B.op(),
               A.rk(), B.rk(), A.nb(),
               alpha, A.Vdata(), A.Vstride(),
                      B.Udata(), B.Ustride(),
               zero,  &W0[0],    ldw0);

    if (A.rk() <= B.rk()) {
        std::vector<T> W1(A.rk() * C.nb());
        int64_t ldw1 = C.layout() == blas::Layout::ColMajor ? A.rk() : C.nb();

        // W1 = W0 * BV
        blas::gemm(C.layout(), blas::Op::NoTrans, B.op(),
                   A.rk(), C.nb(), B.rk(),
                   one,  &W0[0],    ldw0,
                         B.Vdata(), B.Vstride(),
                   zero, &W1[0],    ldw1);

        // C = AU * W1 + beta * C
        blas::gemm(C.layout(), A.op(), blas::Op::NoTrans,
                   C.mb(), C.nb(), A.rk(),
                   one,  A.Udata(), A.Ustride(),
                         &W1[0],    ldw1,
                   beta, C.data(),  C.stride());
    }
    else {
        std::vector<T> W1(C.mb() * B.rk());
        int64_t ldw1 = C.layout() == blas::Layout::ColMajor ? C.mb() : B.rk();

        // W1 = AU * W0
        blas::gemm(C.layout(), A.op(), blas::Op::NoTrans,
                   C.mb(), B.rk(), A.rk(),
                   one,  A.Udata(), A.Ustride(),
                         &W0[0],    ldw0,
                   zero, &W1[0],    ldw1);

        // C = W1 * BV + beta * C
        blas::gemm(C.layout(), blas::Op::NoTrans, B.op(),
                   C.mb(), C.nb(), B.rk(),
                   one,  &W1[0],    ldw1,
                         B.Vdata(), B.Vstride(),
                   beta, C. data(), C. stride());
    }
}

template
void gemm(float alpha, CompressedTile<float> const& A,
                       CompressedTile<float> const& B,
          float beta,            Tile<float>&       C);
template
void gemm(double alpha, CompressedTile<double> const& A,
                        CompressedTile<double> const& B,
          double beta,            Tile<double>&       C);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,            Tile<std::complex<float>>&       C);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,            Tile<std::complex<double>>&       C);

//------------------------------------------------------------------------------
/// General matrix-matrix multiplication. Performs the matrix-matrix operation:
/// $C = \alpha A B + \beta C$, where alpha and beta are scalars, and $A$, $B$,
/// and $C$ are matrices, with $A$ an m-by-k matrix, $B$ a k-by-n matrix, and
/// $C$ an m-by-n compressed matrix ($C = U V$, where $U$ an m-by-Crk and $V$ a
/// Crk-by-n). The matrices can be transposed or conjugate-transposed
/// beforehand. For example:
///     auto AT = hcore::transpose(A);
///     auto BT = hcore::conjugate_transpose(B);
///     hcore::gemm(alpha, AT, BT, beta, C);
///
/// @tparam T
///     One of float, double, std::complex<float>, std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k matrix $A$.
/// @param[in] B
///     The k-by-n matrix $B$.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n compressed matrix $C = U V$, where $U$ the m-by-Crk
///               matrix and $V$ the Crk-by-n matrix.
///     On exit, overwritten by the result $\alpha A B + \beta C$.
///
/// @ingroup gemm
template <typename T>
void gemm(T alpha,           Tile<T> const& A,
                             Tile<T> const& B,
          T beta,  CompressedTile<T>&       C)
{
    assert(C.op() == blas::Op::NoTrans); // todo

    internal::check_gemm(A, B, C);

    int64_t ldw = C.layout() == blas::Layout::ColMajor ? C.Ustride()
                                                       : C.Vstride();
    T* W = new T[ldw*(C.layout() == blas::Layout::ColMajor ? C.nb() : C.mb())];

    T zero = 0.0;

    // W = alpha * A * B
    blas::gemm(C.layout(), A.op(), B.op(),
               A.mb(), B.nb(), A.nb(),
               alpha, A.data(), A.stride(),
                      B.data(), B.stride(),
               zero,  &W[0],    ldw);

    T one  = 1.0;

    // W += beta * CU * CV
    blas::gemm(C.layout(), blas::Op::NoTrans, blas::Op::NoTrans,
               C.mb(), C.nb(), C.rk(),
               beta, C.Udata(), C.Ustride(),
                     C.Vdata(), C.Vstride(),
               one,  &W[0],     ldw);

    int64_t min_mb_nb = std::min(C.mb(), C.nb());
    C.resize(W, ldw, min_mb_nb, min_mb_nb);
}

template
void gemm(float alpha,           Tile<float> const& A,
                                 Tile<float> const& B,
          float beta,  CompressedTile<float>&       C);
template
void gemm(double alpha,           Tile<double> const& A,
                                  Tile<double> const& B,
          double beta,  CompressedTile<double>&       C);
template
void gemm(
    std::complex<float> alpha,           Tile<std::complex<float>> const& A,
                                         Tile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>&       C);
template
void gemm(
    std::complex<double> alpha,           Tile<std::complex<double>> const& A,
                                          Tile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>&       C);

//------------------------------------------------------------------------------
/// General matrix-matrix multiplication. Performs the matrix-matrix operation:
/// $C = \alpha A B + \beta C$, where alpha and beta are scalars, and $A$, $B$,
/// and $C$ are matrices, with $A$ an m-by-k matrix, $B$ a k-by-n compressed
/// matrix ($B = U V$, where $U$ an k-by-Brk and $V$ a Brk-by-n), and $C$ an
/// m-by-n compressed matrix ($C = U V$, where $U$ an m-by-Crk and $V$ a
/// Crk-by-n). The matrices can be transposed or conjugate-transposed
/// beforehand. For example:
///     auto AT = hcore::transpose(A);
///     auto BT = hcore::conjugate_transpose(B);
///     hcore::gemm(alpha, AT, BT, beta, C);
///
/// @tparam T
///     One of float, double, std::complex<float>, std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k matrix $A$.
/// @param[in] B
///     The k-by-n compressed matrix $B = U V$, where $U$ the k-by-Brk matrix
///     and $V$ the Brk-by-n matrix.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n compressed matrix $C = U V$, where $U$ the m-by-Crk
///               matrix and $V$ the Crk-by-n matrix.
///     On exit, overwritten by the result $\alpha A B + \beta C$.
///
/// @ingroup gemm
template <typename T>
void gemm(T alpha,           Tile<T> const& A,
                   CompressedTile<T> const& B,
          T beta,  CompressedTile<T>&       C,
          Options const& opts)
{
    assert(C.op() == blas::Op::NoTrans); // todo

    internal::check_gemm(A, B, C);

    bool use_gemm = get_option(opts, Option::UseGEMM, true);
    int64_t fixed_rk = get_option(opts, Option::FixedRank, 0);
    bool truncate_with_tol = get_option(opts, Option::TruncateWithTol, false);

    T zero = 0.0;

    std::vector<T> W(C.mb()*B.rk());
    int64_t ldw = C.layout() == blas::Layout::ColMajor ? C.mb() : B.rk();

    // W = alpha * A * BU
    blas::gemm(C.layout(), A.op(), B.op(),
               C.mb(), B.rk(), A.nb(),
               alpha, A. data(), A. stride(),
                      B.Udata(), B.Ustride(),
               zero,  &W[0],     ldw);

    // C = W * BV + beta * C
    if (C.layout() == blas::Layout::RowMajor) {
        internal::rsvd(B.op(), blas::Op::NoTrans,
                       beta, B.Vdata(), B.Vstride(),
                             &W[0],     ldw, B.rk(),
                             C,
                       use_gemm, fixed_rk, truncate_with_tol);
    }
    else {
        internal::rsvd(blas::Op::NoTrans, B.op(),
                       beta, &W[0],     ldw,
                             B.Vdata(), B.Vstride(), B.rk(),
                             C,
                       use_gemm, fixed_rk, truncate_with_tol);
    }
}

template
void gemm(float alpha,           Tile<float> const& A,
                       CompressedTile<float> const& B,
          float beta,  CompressedTile<float>&       C,
          Options const& opts);
template
void gemm(double alpha,           Tile<double> const& A,
                        CompressedTile<double> const& B,
          double beta,  CompressedTile<double>&       C,
          Options const& opts);
template
void gemm(
    std::complex<float> alpha,           Tile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>&       C,
    Options const& opts);
template
void gemm(
    std::complex<double> alpha,           Tile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>&       C,
    Options const& opts);

//------------------------------------------------------------------------------
/// General matrix-matrix multiplication. Performs the matrix-matrix operation:
/// $C = \alpha A B + \beta C$, where alpha and beta are scalars, and $A$, $B$,
/// and $C$ are matrices, with $A$ an m-by-k compressed matrix ($A = U V$, where
/// $U$ an m-by-Ark and $V$ a Ark-by-n), $B$ a k-by-n matrix, and $C$ an
/// m-by-n compressed matrix ($C = U V$, where $U$ an m-by-Crk and $V$ a
/// Crk-by-n). The matrices can be transposed or conjugate-transposed
/// beforehand. For example:
///     auto AT = hcore::transpose(A);
///     auto BT = hcore::conjugate_transpose(B);
///     hcore::gemm(alpha, AT, BT, beta, C);
///
/// @tparam T
///     One of float, double, std::complex<float>, std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k compressed matrix $A = U V$, where $U$ the m-by-Ark matrix
///     and $V$ the Ark-by-k matrix.
/// @param[in] B
///     The k-by-n matrix $B$.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n compressed matrix $C = U V$, where $U$ the m-by-Crk
///               matrix and $V$ the Crk-by-n matrix.
///     On exit, overwritten by the result $\alpha A B + \beta C$.
///
/// @ingroup gemm
template <typename T>
void gemm(T alpha, CompressedTile<T> const& A,
                             Tile<T> const& B,
          T beta,  CompressedTile<T>&       C,
          Options const& opts)
{
    assert(C.op() == blas::Op::NoTrans); // todo

    internal::check_gemm(A, B, C);

    bool use_gemm = get_option(opts, Option::UseGEMM, true);
    int64_t fixed_rk = get_option(opts, Option::FixedRank, 0);
    bool truncate_with_tol = get_option(opts, Option::TruncateWithTol, false);

    std::vector<T> W(A.rk()*C.nb());
    int64_t ldw = C.layout() == blas::Layout::ColMajor ? A.rk() : C.nb();

    T zero = 0.0;

    // W = alpha * AV * B
    blas::gemm(C.layout(), A.op(), B.op(),
               A.rk(), C.nb(), A.nb(),
               alpha, A.Vdata(), A.Vstride(),
                      B. data(), B. stride(),
               zero,  &W[0],     ldw);

    // C = alpha * AU * W + beta * C
    if (C.layout() == blas::Layout::RowMajor) {
        internal::rsvd(blas::Op::NoTrans, A.op(),
                       beta, &W[0],     ldw,
                             A.Udata(), A.Ustride(), A.rk(),
                             C,
                       use_gemm, truncate_with_tol, fixed_rk);
    }
    else {
        internal::rsvd(A.op(), blas::Op::NoTrans,
                       beta, A.Udata(), A.Ustride(),
                             &W[0],     ldw, A.rk(),
                             C,
                       use_gemm, truncate_with_tol, fixed_rk);
    }
}

template
void gemm(float alpha, CompressedTile<float> const& A,
                                 Tile<float> const& B,
          float beta,  CompressedTile<float>&       C,
          Options const& opts);
template
void gemm(double alpha, CompressedTile<double> const& A,
                                  Tile<double> const& B,
          double beta,  CompressedTile<double>&       C,
          Options const& opts);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                                         Tile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>&       C,
    Options const& opts);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                          Tile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>&       C,
    Options const& opts);

//------------------------------------------------------------------------------
/// General matrix-matrix multiplication. Performs the matrix-matrix operation:
/// $C = \alpha A B + \beta C$, where alpha and beta are scalars, and $A$, $B$,
/// and $C$ are matrices, with $A$ an m-by-k compressed matrix ($A = U V$, where
/// $U$ an m-by-Ark and $V$ a Ark-by-n), $B$ a k-by-n compressed matrix
/// ($B = U V$, where $U$ an k-by-Brk and $V$ a Brk-by-n), and $C$ an m-by-n
/// compressed matrix ($C = U V$, where $U$ an m-by-Crk and $V$ a Crk-by-n). The
/// matrices can be transposed or conjugate-transposed beforehand. For example:
///     auto AT = hcore::transpose(A);
///     auto BT = hcore::conjugate_transpose(B);
///     hcore::gemm(alpha, AT, BT, beta, C);
///
/// @tparam T
///     One of float, double, std::complex<float>, std::complex<double>.
/// @param[in] alpha
///     The scalar alpha.
/// @param[in] A
///     The m-by-k compressed matrix $A = U V$, where $U$ the m-by-Ark matrix
///     and $V$ the Ark-by-k matrix.
/// @param[in] B
///     The k-by-n compressed matrix $B = U V$, where $U$ the k-by-Brk matrix
///     and $V$ the Brk-by-n matrix.
/// @param[in] beta
///     The scalar beta.
/// @param[in,out] C
///     On entry, the m-by-n compressed matrix $C = U V$, where $U$ the m-by-Crk
///               matrix and $V$ the Crk-by-n matrix.
///     On exit, overwritten by the result $\alpha A B + \beta C$.
///
/// @ingroup gemm
template <typename T>
void gemm(T alpha, CompressedTile<T> const& A,
                   CompressedTile<T> const& B,
          T beta,  CompressedTile<T>&       C,
          Options const& opts)
{
    assert(C.op() == blas::Op::NoTrans); // todo

    T zero = 0.0;
    T one  = 1.0;

    internal::check_gemm(A, B, C);

    bool use_gemm = get_option(opts, Option::UseGEMM, true);
    int64_t fixed_rk = get_option(opts, Option::FixedRank, 0);
    bool truncate_with_tol = get_option(opts, Option::TruncateWithTol, false);

    // W0 = alpha * AV * BU
    std::vector<T> W0(A.rk() * B.rk());
    int64_t ldw0 = C.layout() == blas::Layout::ColMajor ? A.rk() : B.rk();

    blas::gemm(C.layout(), A.op(), B.op(),
               A.rk(), B.rk(), A.nb(),
               alpha, A.Vdata(), A.Vstride(),
                      B.Udata(), B.Ustride(),
               zero,  &W0[0],    ldw0);

    if (A.rk() <= B.rk()) {
        std::vector<T> W1(A.rk() * C.nb());
        int64_t ldw1 = C.layout() == blas::Layout::ColMajor ? A.rk() : C.nb();

        // W1 = W0 * BV
        blas::gemm(C.layout(), blas::Op::NoTrans, B.op(),
                   A.rk(), C.nb(), B.rk(),
                   one,  &W0[0],    ldw0,
                         B.Vdata(), B.Vstride(),
                   zero, &W1[0],    ldw1);

        // C = AU * W1 + beta * C
        if (C.layout() == blas::Layout::RowMajor) {
            internal::rsvd(blas::Op::NoTrans, A.op(),
                           beta, &W1[0],    ldw1,
                                 A.Udata(), A.Ustride(), A.rk(),
                                 C,
                           use_gemm, truncate_with_tol, fixed_rk);
        }
        else {
            internal::rsvd(A.op(), blas::Op::NoTrans,
                           beta, A.Udata(), A.Ustride(),
                                 &W1[0],    ldw1, A.rk(),
                                 C,
                           use_gemm, truncate_with_tol, fixed_rk);
        }
    }
    else {
        std::vector<T> W1(C.mb() * B.rk());
        int64_t ldw1 = C.layout() == blas::Layout::ColMajor ? C.mb() : B.rk();

        // W1 = AU * W0
        blas::gemm(C.layout(), A.op(), blas::Op::NoTrans,
                   C.mb(), B.rk(), A.rk(),
                   one,  A.Udata(), A.Ustride(),
                         &W0[0],    ldw0,
                   zero, &W1[0],    ldw1);

        // C = W1 * BV + beta * C
        if (C.layout() == blas::Layout::RowMajor) {
            internal::rsvd(B.op(), blas::Op::NoTrans,
                           beta, B.Vdata(), B.Vstride(),
                                 &W1[0],    ldw1, B.rk(),
                                 C,
                           use_gemm, truncate_with_tol, fixed_rk);
        }
        else {
            internal::rsvd(blas::Op::NoTrans, B.op(),
                           beta, &W1[0],    ldw1,
                                 B.Vdata(), B.Vstride(), B.rk(),
                                 C,
                           use_gemm, truncate_with_tol, fixed_rk);
        }
    }
}

template
void gemm(float alpha, CompressedTile<float> const& A,
                       CompressedTile<float> const& B,
          float beta,  CompressedTile<float>&       C,
          Options const& opts);
template
void gemm(double alpha, CompressedTile<double> const& A,
                        CompressedTile<double> const& B,
           double beta, CompressedTile<double>&       C,
          Options const& opts);
template
void gemm(
    std::complex<float> alpha, CompressedTile<std::complex<float>> const& A,
                               CompressedTile<std::complex<float>> const& B,
    std::complex<float> beta,  CompressedTile<std::complex<float>>&       C,
    Options const& opts);
template
void gemm(
    std::complex<double> alpha, CompressedTile<std::complex<double>> const& A,
                                CompressedTile<std::complex<double>> const& B,
    std::complex<double> beta,  CompressedTile<std::complex<double>>&       C,
    Options const& opts);

} // namespace hcore
