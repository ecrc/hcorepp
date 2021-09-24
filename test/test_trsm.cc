// Copyright (c) 2017-2021, King Abdullah University of Science and Technology
// (KAUST). All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include <vector>
#include <cstdint>
#include <complex>

#include "testsweeper.hh"
#include "blas/flops.hh"
#include "lapack.hh"
#include "blas.hh"

#include "matrix_utils.hh"
#include "hcore/hcore.hh"
#include "tile_utils.hh"
#include "test.hh"

namespace hcore {
namespace test {

template <typename T>
void trsm(Params& params, bool run) {
    using real_t = blas::real_type<T>;

    blas::Layout layout = params.layout();

    blas::Side side = params.side();
    blas::Uplo uplo = params.uplo();
    blas::Diag diag = params.diag();
    blas::Op transA = params.transA();
    blas::Op transB = params.transB();

    T alpha = params.alpha.get<T>();

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
    if (blas::is_complex<T>::value && transA != blas::Op::NoTrans && transA != transB) {
        printf("skipping: wrong combinations of transA/transB.\n");
        return;
    }

    int64_t An = side == blas::Side::Left ? m : n;
    int64_t Bm = transB == blas::Op::NoTrans ? m : n;
    int64_t Bn = transB == blas::Op::NoTrans ? n : m;

    if (layout == blas::Layout::RowMajor) {
        std::swap(Bm, Bn);
    }

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

    // lapack::larnv(idist, iseed, ldb * Bn, &Bdata[0]);
    generate_dense_matrix(Bm, Bn, &Bdata[0], ldb, iseed, mode, cond);

    lapack::Norm norm = lapack::Norm::Inf; // todo: variable norm type
    real_t Anorm = lapack::lantr(norm, uplo, diag, An, An, &Adata[0], lda);
    real_t Bnorm = lapack::lange(norm, Bm, Bn, &Bdata[0], ldb);

    // if row major, transpose A
    if (layout == blas::Layout::RowMajor) {
        for (int64_t j = 0; j < An; ++j)
            for (int64_t i = 0; i < j; ++i)
                std::swap(Adata[i + j*lda], Adata[j + i*lda]);
    }

    if (layout == blas::Layout::RowMajor) {
        std::swap(Bm, Bn);
    }

    hcore::Tile<T> A(An, An, &Adata[0], lda, layout);
    if (transA == blas::Op::Trans)
        A = transpose(A);
    else if (transA == blas::Op::ConjTrans)
        A = conjugate_transpose(A);

    A.uplo(uplo);

    hcore::Tile<T> B(Bm, Bn, &Bdata[0], ldb, layout);
    if (transB == blas::Op::Trans)
        B = transpose(B);
    else if (transB == blas::Op::ConjTrans)
        B = conjugate_transpose(B);

    int64_t Bref_m = m;
    int64_t Bref_n = n;

    if (layout == blas::Layout::RowMajor) {
        std::swap(Bref_m, Bref_n);
    }

    int64_t ldbref = testsweeper::roundup(Bref_m, align);

    std::vector<T> Bref;
    if (params.check() == 'y') {
        Bref.resize(ldbref * Bref_n);
        copy(&Bref[0], ldbref, B);
    }

    if (verbose) {
        pretty_print(A, "A");
        pretty_print(B, "B");

        if (verbose > 1) {
            pretty_print(Bref_m, Bref_n, &Bref[0], ldbref, "Bref");
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
                layout, side, uplo, transA, diag,
                m, n,
                alpha, &Adata[0], lda,
                       &Bref[0],  ldbref);
        }
        double ref_time_end = testsweeper::get_wtime();
        params.ref_time() = ref_time_end - ref_time_start;
        params.ref_gflops() =
            blas::Gflop<T>::trsm(side, m, n) / params.ref_time();

        if (verbose) {
            pretty_print(Bref_m, Bref_n, &Bref[0], ldbref, "Bref");
        }

        // Compute the Residual ||Bref - B||_inf.

        diff(&Bref[0], ldbref, B);

        if (verbose) {
            pretty_print(Bref_m, Bref_n, &Bref[0], ldbref, "Bref_diff_B");
        }

        params.error() = lapack::lange(norm, Bref_m, Bref_n, &Bref[0], ldbref)
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

void trsm_dispatch(Params& params, bool run) {
    switch(params.datatype()) {
        case testsweeper::DataType::Single:
            hcore::test::trsm<float>(params, run);
            break;
        case testsweeper::DataType::Double:
            hcore::test::trsm<double>(params, run);
            break;
        case testsweeper::DataType::SingleComplex:
            hcore::test::trsm<std::complex<float>>(params, run);
            break;
        case testsweeper::DataType::DoubleComplex:
            hcore::test::trsm<std::complex<double>>(params, run);
            break;
        default:
            throw hcore::Error("Unsupported data type.");
            break;
    }
}

} // namespace test
} // namespace hcore