// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "test.hh"
#include "hcore.hh"
#include "pretty_print.hh"
#include "matrix_utils.hh"

#include "blas.hh"
#include "blas/flops.hh"
#include "lapack.hh"
#include "testsweeper.hh"

#include <vector>
#include <cstdint>
#include <complex>
#include <cassert>
#include <stdexcept>

template <typename T>
void trsm_test_execute(Params& params, bool run)
{
    // todo
    // blas::Layout layout = params.layout();

    blas::Side side = params.side();
    blas::Uplo uplo = params.uplo();
    blas::Op trans = params.trans();
    blas::Diag diag = params.diag();
    T alpha = params.alpha();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t align = params.align();
    int verbose = params.verbose();
    double tol = params.tol();
    int mode = params.latms_mode();
    blas::real_type<T> cond = params.latms_cond();
    blas::real_type<T> accuracy =
        std::numeric_limits<blas::real_type<T>>::epsilon();

    if (!run) return;

    int64_t An = side == blas::Side::Left ? m : n;
    int64_t Bm = m;
    int64_t Bn = n;

    // todo
    // if (layout == Layout::RowMajor) {
    //     std::swap(Bm, Bn);
    // }

    int64_t lda = testsweeper::roundup(An, align);
    int64_t ldb = testsweeper::roundup(Bm, align);

    std::vector<T> Adata(lda * An);
    std::vector<T> Bdata(ldb * Bn);

    // int64_t idist = 1;
    int iseed[4] = {0, 0, 0, 1};

    // lapack::larnv(idist, iseed, lda * An, &Adata[0]);
    generate_dense_matrix(An, An, &Adata[0], lda, iseed, mode, cond);

    // lapack::larnv(idist, iseed, ldb * Bn, &Bdata[0]);
    generate_dense_matrix(Bm, Bn, &Bdata[0], ldb, iseed, mode, cond);

    blas::real_type<T> Anorm =
            lapack::lantr(lapack::Norm::Inf, uplo, diag, An, An, &Adata[0], lda);
    blas::real_type<T> Bnorm =
            lapack::lange(lapack::Norm::Inf, Bm, Bn, &Bdata[0], ldb);

    hcore::DenseTile<T> A(An, An, &Adata[0], lda);
    A.op(trans);
    A.uplo(uplo);
    hcore::DenseTile<T> B(Bm, Bn, &Bdata[0], ldb);
    // B.op(transB); // todo

    std::vector<T> Bref;
    if (params.check() == 'y') {
        Bref = Bdata;
    }

    T nan_ = nan("");
    if (uplo == blas::Uplo::Lower) {
        lapack::laset(lapack::MatrixType::Upper,
            An-1, An-1, nan_, nan_, &Adata[0 + 1 * lda], lda);
    }
    else {
        lapack::laset(lapack::MatrixType::Lower,
            An-1, An-1, nan_, nan_, &Adata[1 + 0 * lda], lda);
    }

    // brute force positive definiteness
    for (int j = 0; j < An; ++j)
        Adata[j + j * lda] += An;

    // factor to get well-conditioned triangle
    lapack::potrf(uplo, An, &Adata[0], lda);

    // todo
    // if row major, transpose A
    // if (layout == blas::Layout::RowMajor) {
    //     for (int64_t j = 0; j < Am; ++j) {
    //         for (int64_t i = 0; i < j; ++i) {
    //             std::swap(A[i + j * lda], A[j + i * lda]);
    //         }
    //     }
    // }

    // todo
    //assert(!(
    //    blas::is_complex<T>::value &&
    //    ((B.op() == blas::Op::Trans     && A.op() == blas::Op::ConjTrans) ||
    //     (B.op() == blas::Op::ConjTrans && A.op() == blas::Op::Trans    ))));

    if (verbose) {
        pretty_print(A, "A");
        pretty_print(B, "B");

        if (verbose > 1) {
            pretty_print(Bm, Bn, &Bref[0], ldb, "Bref");
        }
    }

    double time_start = testsweeper::get_wtime();
    {
        hcore::trsm<T>(side, diag, alpha, A, B);
    }
    double time_end = testsweeper::get_wtime();
    params.time() = time_end - time_start;
    params.gflops() = blas::Gflop<T>::trsm(side, m, n) / params.time();

    if (verbose) {
        pretty_print(B, "B");
    }

    if (params.check() == 'y') {
        double ref_time_start = testsweeper::get_wtime();
        {
            blas::trsm(
                blas::Layout::ColMajor, side, A.uplo_physical(), A.op(), diag,
                m, n,
                alpha, &Adata[0], lda,
                       &Bref[0],  ldb);
        }
        double ref_time_end = testsweeper::get_wtime();
        params.ref_time() = ref_time_end - ref_time_start;
        params.ref_gflops() =
            blas::Gflop<T>::trsm(side, m, n) / params.ref_time();

        if (verbose) {
            pretty_print(Bm, Bn, &Bref[0], ldb, "Bref");
        }

        // Compute the Residual ||Bref - B||_inf.

        for (int64_t j = 0; j < Bn; ++j) {
            for (int64_t i = 0; i < Bm; ++i) {
                Bref[i + j * ldb] -= Bdata[i + j * ldb];
            }
        }

        if (verbose) {
            pretty_print(Bm, Bn, &Bref[0], ldb, "Bref_diff_B");
        }

        params.error() =
                    lapack::lange(lapack::Norm::Inf, Bm, Bn, &Bref[0], ldb)
                    / (sqrt(blas::real_type<T>(An) + 2) * std::abs(alpha) *
                       Anorm * Bnorm + 2 * std::abs(0.0) * 0.0);

        // Complex number need extra factor.
        // See "Accuracy and Stability of Numerical Algorithms", by Nicholas J.
        // Higham, Section 3.6, 2002.
        if (blas::is_complex<T>::value) {
            params.error() /= 2 * sqrt(2);
        }
        params.okay() = (params.error() < tol * accuracy);
    }


}

void trsm_test_dispatch(Params& params, bool run)
{
    switch(params.datatype()) {
        case testsweeper::DataType::Single:
            trsm_test_execute<float>(params, run);
            break;
        case testsweeper::DataType::Double:
            trsm_test_execute<double>(params, run);
            break;
        case testsweeper::DataType::SingleComplex:
            trsm_test_execute<std::complex<float>>(params, run);
            break;
        case testsweeper::DataType::DoubleComplex:
            trsm_test_execute<std::complex<double>>(params, run);
            break;
        default:
            throw std::invalid_argument("Unsupported data type.");
            break;
    }
}
