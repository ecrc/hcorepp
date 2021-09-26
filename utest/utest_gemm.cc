// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include <algorithm>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <complex>
#include <vector>
#include <limits>

#include "blas.hh"
#include "lapack.hh"

#include "hcore/exception.hh"
#include "hcore/hcore.hh"
#include "hcore/tile.hh"
#include "utest.hh"

namespace hcore {
namespace utest {

blas::Op ops[] = {blas::Op::NoTrans, blas::Op::Trans, blas::Op::ConjTrans};
blas::Layout layouts[] = {blas::Layout::ColMajor, blas::Layout::RowMajor};

template <typename T>
void copy(Tile<T> const& A, T* B, int64_t ldb) {
    for (int64_t j = 0; j < A.nb(); ++j) {
        for (int64_t i = 0; i < A.mb(); ++i) {
            if (A.layout() == blas::Layout::ColMajor)
                B[i + j*ldb] = A(i, j);
            else
                B[j + i*ldb] = A(i, j);
        }
    }
}

template <typename T>
void err_check(Tile<T> const& A, T const* B, int ldb,
               blas::real_type<T> abs_tol, blas::real_type<T> rel_tol) {
    using blas::real;
    using blas::imag;

    for (int64_t j = 0; j < A.nb(); ++j) {
        for (int64_t i = 0; i < A.mb(); ++i) {
            if ((A.uplo() == blas::Uplo::General) ||
                (A.uplo() == blas::Uplo::Lower && i >= j) ||
                (A.uplo() == blas::Uplo::Upper && i <= j)) {
                T Bij = A.layout() == blas::Layout::ColMajor ? B[i + j*ldb]
                                                             : B[j + i*ldb];

                blas::real_type<T> abs_error = std::abs(A(i, j) - Bij);
                blas::real_type<T> rel_error = abs_error / std::abs(A(i, j));

                if (!(abs_error <= abs_tol || rel_error <= rel_tol)) {
                    printf("\n");
                    printf("A(%3d, %3d) %8.4f + %8.4fi\n"
                           "B           %8.4f + %8.4fi\n"
                           "abs_error %8.2e, rel_error %8.2e\n",
                            i, j, real(A(i, j)), imag(A(i, j)),
                                  real(Bij),     imag(Bij),
                            abs_error, rel_error);
                }

                hcore_assert(abs_error <= abs_tol || rel_error <= rel_tol);
            }
        }
    }
}

template <typename T>
void gemm() {
    blas::real_type<T> eps = std::numeric_limits<blas::real_type<T>>::epsilon();

    int64_t idist = 1;
    int64_t iseed[4] = {0, 1, 2, 3};

    T alpha;
    lapack::larnv(idist, iseed, 1, &alpha);

    T beta;
    lapack::larnv(idist, iseed, 1, &beta);

    int64_t m = 50, n = 40, k = 30;

    for (const auto& layout : layouts) {
        for (const auto& opc : ops) {
            for (const auto& opb : ops) {
                for (const auto& opa : ops) {
                    int64_t Am = opa == blas::Op::NoTrans ? m : k;
                    int64_t An = opa == blas::Op::NoTrans ? k : m;
                    int64_t Bm = opb == blas::Op::NoTrans ? k : n;
                    int64_t Bn = opb == blas::Op::NoTrans ? n : k;
                    int64_t Cm = opc == blas::Op::NoTrans ? m : n;
                    int64_t Cn = opc == blas::Op::NoTrans ? n : m;

                    if (layout == blas::Layout::RowMajor) {
                        std::swap(Am, An);
                        std::swap(Bm, Bn);
                        std::swap(Cm, Cn);
                    }

                    int64_t lda = Am + 1;
                    std::vector<T> Adata(lda*An);
                    lapack::larnv(idist, iseed, Adata.size(), Adata.data());

                    int64_t ldb = Bm + 1;
                    std::vector<T> Bdata(ldb*Bn);
                    lapack::larnv(idist, iseed, Bdata.size(), Bdata.data());

                    int64_t ldc = Cm + 1;
                    std::vector<T> Cdata(ldc*Cn);
                    lapack::larnv(idist, iseed, Cdata.size(), Cdata.data());

                    if (layout == blas::Layout::RowMajor) {
                        std::swap(Am, An);
                        std::swap(Bm, Bn);
                        std::swap(Cm, Cn);
                    }

                    Tile<T> A(Am, An, Adata.data(), lda, layout);
                    if (opa == blas::Op::Trans)
                        A = transpose(A);
                    else if (opa == blas::Op::ConjTrans)
                        A = conjugate_transpose(A);

                    assert(A.mb() == m);
                    assert(A.nb() == k);

                    Tile<T> B(Bm, Bn, Bdata.data(), ldb, layout);
                    if (opb == blas::Op::Trans)
                        B = transpose(B);
                    else if (opb == blas::Op::ConjTrans)
                        B = conjugate_transpose(B);

                    assert(B.mb() == k);
                    assert(B.nb() == n);

                    Tile<T> C(Cm, Cn, Cdata.data(), ldc, layout);
                    if (opc == blas::Op::Trans)
                        C = transpose(C);
                    else if (opc == blas::Op::ConjTrans)
                        C = conjugate_transpose(C);

                    assert(C.mb() == m);
                    assert(C.nb() == n);

                    int64_t opCrefm = m;
                    int64_t opCrefn = n;

                    if (layout == blas::Layout::RowMajor)
                        std::swap(opCrefm, opCrefn);

                    int64_t ldopc = opCrefm + 1;

                    std::vector<T> opCref(ldopc*opCrefn);
                    copy(C, opCref.data(), ldopc);

                    try {
                        gemm(alpha, A, B, beta, C);

                        hcore_assert(!(blas::is_complex<T>::value &&
                                     ((opc == blas::Op::Trans &&
                                     (opa == blas::Op::ConjTrans ||
                                     opb == blas::Op::ConjTrans)) ||
                                     (opc == blas::Op::ConjTrans &&
                                     (opa == blas::Op::Trans ||
                                     opb == blas::Op::Trans)))));
                    }
                    catch (const std::exception& e) {
                        hcore_assert(blas::is_complex<T>::value &&
                                     ((opc == blas::Op::Trans &&
                                     (opa == blas::Op::ConjTrans ||
                                     opb == blas::Op::ConjTrans)) ||
                                     (opc == blas::Op::ConjTrans &&
                                     (opa == blas::Op::Trans ||
                                     opb == blas::Op::Trans))));
                        continue;
                    }

                    blas::gemm(C.layout(), A.op(), B.op(), m, n, k,
                               alpha, Adata.data(),  lda,
                                      Bdata.data(),  ldb,
                               beta,  opCref.data(), ldopc);

                    err_check(C, opCref.data(), ldopc,
                              3*sqrt(k)*eps, 3*sqrt(k)*eps);
                }
            }
        }
    }
}

void gemm_dispatch() {
    gemm<float>();
    gemm<double>();
    gemm<std::complex<float>>();
    gemm<std::complex<double>>();
}

void launch() {
    run(gemm_dispatch, "gemm");
}

} // namespace utest
} // namespace hcore

int main(int argc, char* argv[]) { return hcore::utest::main(); }