// Copyright (c) 2017-2021, King Abdullah University of Science and Technology
// (KAUST). All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "test.hh"
#include "hcore.hh"
#include "tile_utils.hh"
#include "matrix_utils.hh"

#include "blas.hh"
#include "lapack.hh"
#include "lapack/flops.hh"
#include "testsweeper.hh"

#include <vector>
#include <cstdint>
#include <complex>
#include <cassert>

namespace hcore {
namespace test {

template <typename T>
void potrf(Params& params, bool run) {
    using real_t = blas::real_type<T>;

    blas::Layout layout = params.layout();

    blas::Uplo uplo = params.uplo();
    blas::Op trans = params.trans();

    int64_t n = params.dim.n();
    int64_t mode = params.latms_mode();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    real_t tol = params.tol();
    real_t cond = params.latms_cond();
    real_t accuracy = std::numeric_limits<real_t>::epsilon();

    if (!run) return;

    int64_t lda = testsweeper::roundup(n, align);

    std::vector<T> Adata(lda * n);

    // int64_t idist = 1;
    int64_t iseed[4] = {0, 0, 0, 1};

    // lapack::larnv(idist, iseed, lda * n, &Adata[0]);
    generate_dense_matrix(n, n, &Adata[0], lda, iseed, mode, cond);

    set_dense_uplo(uplo, n, n, &Adata[0], lda);

    dense_to_positive_definite(n, &Adata[0], lda);

    hcore::Tile<T> A(n, n, &Adata[0], lda, layout);
    if (trans == blas::Op::Trans)
        A = transpose(A);
    else if (trans == blas::Op::ConjTrans)
        A = conjugate_transpose(A);

    A.uplo(uplo);

    std::vector<T> Aref;
    if (params.check() == 'y') {
        Aref.resize(lda * n);
        copy(&Aref[0], lda, A);
    }

    if (verbose) {
        pretty_print(A, "A");

        if (verbose > 1) {
            pretty_print(n, n, &Aref[0], lda, "Aref");
        }
    }

    double gflops = 0.0;

    double time_start = testsweeper::get_wtime();
    int64_t info = hcore::potrf<T>(A);
    double time_end = testsweeper::get_wtime();
    if (info != 0) {
        throw hcore::Error("lapack::potrf returned error " + info);
    }

    gflops = lapack::Gflop<T>::potrf(n);
    params.time() = time_end - time_start;
    params.gflops() = gflops / params.time();

    if (verbose) {
        pretty_print(A, "A");
    }

    if (params.check() == 'y') {
        blas::Uplo uplo_ = uplo;
        if (trans != blas::Op::NoTrans) {
            uplo_ = (uplo_ == blas::Uplo::Lower ? blas::Uplo::Upper
                                                : blas::Uplo::Lower);
        }
        double ref_time_start = testsweeper::get_wtime();
        {
            int64_t info = lapack::potrf(uplo_, n, &Aref[0], lda);
            if (info != 0) {
                throw hcore::Error("lapack::potrf returned error " + info);
            }
        }
        double ref_time_end = testsweeper::get_wtime();
        params.ref_time() = ref_time_end - ref_time_start;
        params.ref_gflops() = lapack::Gflop<T>::potrf(n) / params.ref_time();        

        if (verbose) {
            pretty_print(n, n, &Aref[0], lda, "Aref");
        }

        // Compute the Residual ||Aref - A||_2.

        real_t temp = 0;
        real_t diff = 0;
        real_t norm = 0;
        for (int64_t j = 0; j < A.n(); ++j) {
            for (int64_t i = 0; i < A.m(); ++i) {
                if ((uplo_ == blas::Uplo::Lower && i >= j) ||
                    (uplo_ == blas::Uplo::Upper && i <= j)) {
                    T Aij = layout == blas::Layout::ColMajor ? A(i, j)
                                                             : A(j, i);

                    temp = std::abs(Aij - Aref[i + j * lda]);
                    diff += temp * temp;

                    temp = std::abs(Aref[i + j * lda]);
                    norm += temp * temp;
                }
            }
        }
        diff = sqrt(diff);
        norm = sqrt(norm);

        params.error() = diff / norm;

        // Complex number need extra factor.
        // See "Accuracy and Stability of Numerical Algorithms", by Nicholas J.
        // Higham, Section 3.6, 2002.
        if (blas::is_complex<T>::value) {
            params.error() /= 2 * sqrt(2);
        }
        params.okay() = (params.error() < tol * accuracy);
    }


}

void potrf_dispatch(Params& params, bool run) {
    switch(params.datatype()) {
        case testsweeper::DataType::Single:
            hcore::test::potrf<float>(params, run);
            break;
        case testsweeper::DataType::Double:
            hcore::test::potrf<double>(params, run);
            break;
        case testsweeper::DataType::SingleComplex:
            hcore::test::potrf<std::complex<float>>(params, run);
            break;
        case testsweeper::DataType::DoubleComplex:
            hcore::test::potrf<std::complex<double>>(params, run);
            break;
        default:
            throw hcore::Error("Unsupported data type.");
            break;
    }
}

} // namespace test
} // namespace hcore