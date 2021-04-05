// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "test.hh"
#include "hcore.hh"
#include "tile_utils.hh"
#include "matrix_utils.hh"

#include "blas.hh"
#include "blas/flops.hh"
#include "lapack.hh"
#include "testsweeper.hh"

#include <vector>
#include <cstdint>
#include <complex>
#include <stdexcept>

template <typename T>
void trsm_test_execute(Params& params, bool run)
{
    using real_t = blas::real_type<T>;

    // todo
    // blas::Layout layout = params.layout();

    blas::Side side = params.side();
    blas::Uplo uplo = params.uplo();
    blas::Diag diag = params.diag();
    blas::Op transA = params.transA();
    blas::Op transB = params.transB();

    T alpha = params.alpha();

    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t mode = params.latms_mode();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    real_t tol = params.tol();
    real_t cond = params.latms_cond();
    real_t accuracy = std::numeric_limits<real_t>::epsilon();

    if (!run) return;

    // quick returns
    if (blas::is_complex<T>::value) {
        if ((transB == blas::Op::Trans     && transA == blas::Op::ConjTrans) ||
            (transB == blas::Op::ConjTrans && transA == blas::Op::Trans)) {
            printf("skipping: wrong combinations of transA/transB.\n");
            return;
        }
    }

    int64_t An = side == blas::Side::Left ? m : n;
    int64_t Bm = transB == blas::Op::NoTrans ? m : n;
    int64_t Bn = transB == blas::Op::NoTrans ? n : m;

    // todo
    // if (layout == Layout::RowMajor) {
    //     std::swap(Bm, Bn);
    // }

    int64_t lda = testsweeper::roundup(An, align);
    int64_t ldb = testsweeper::roundup(Bm, align);

    std::vector<T> Adata(lda * An);
    std::vector<T> Bdata(ldb * Bn);

    // int64_t idist = 1;
    int64_t iseed[4] = {0, 0, 0, 1};

    // lapack::larnv(idist, iseed, lda * An, &Adata[0]);
    generate_dense_matrix(An, An, &Adata[0], lda, iseed, mode, cond);

    set_dense_uplo(uplo, An, An, &Adata[0], lda);

    dense_to_positive_definite(An, &Adata[0], lda);

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

    // lapack::larnv(idist, iseed, ldb * Bn, &Bdata[0]);
    generate_dense_matrix(Bm, Bn, &Bdata[0], ldb, iseed, mode, cond);

    lapack::Norm norm = lapack::Norm::Inf; // todo: variable norm type
    real_t Anorm = lapack::lantr(norm, uplo, diag, An, An, &Adata[0], lda);
    real_t Bnorm = lapack::lange(norm, Bm, Bn, &Bdata[0], ldb);

    hcore::DenseTile<T> A(An, An, &Adata[0], lda);
    A.op(transA);
    A.uplo(uplo);
    hcore::DenseTile<T> B(Bm, Bn, &Bdata[0], ldb);
    B.op(transB);

    int64_t ldbref = testsweeper::roundup(m, align);

    std::vector<T> Bref;
    if (params.check() == 'y') {
        Bref.resize(ldbref * n);
        copy(&Bref[0], ldbref, B);
    }

    if (verbose) {
        pretty_print(A, "A");
        pretty_print(B, "B");

        if (verbose > 1) {
            pretty_print(m, n, &Bref[0], ldbref, "Bref");
        }
    }

    double gflops = 0.0;
    double time_start = testsweeper::get_wtime();
    {
        hcore::trsm<T>(side, diag, alpha, A, B);
        gflops = blas::Gflop<T>::trsm(side, m, n);
    }
    double time_end = testsweeper::get_wtime();
    params.time() = time_end - time_start;
    params.gflops() = gflops / params.time();

    if (verbose) {
        pretty_print(B, "B");
    }

    if (params.check() == 'y') {
        double ref_time_start = testsweeper::get_wtime();
        {
            blas::trsm(
                blas::Layout::ColMajor, side, uplo, transA, diag,
                m, n,
                alpha, &Adata[0], lda,
                       &Bref[0],  ldbref);
        }
        double ref_time_end = testsweeper::get_wtime();
        params.ref_time() = ref_time_end - ref_time_start;
        params.ref_gflops() =
            blas::Gflop<T>::trsm(side, m, n) / params.ref_time();

        if (verbose) {
            pretty_print(m, n, &Bref[0], ldbref, "Bref");
        }

        // Compute the Residual ||Bref - B||_inf.

        diff(&Bref[0], ldbref, B);

        if (verbose) {
            pretty_print(m, n, &Bref[0], ldbref, "Bref_diff_B");
        }

        params.error() = lapack::lange(norm, m, n, &Bref[0], ldbref)
                        / (sqrt(real_t(An) + 2) * std::abs(alpha) *
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
