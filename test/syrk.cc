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
void syrk_test_execute(Params& params, bool run)
{
    // todo
    // blas::Layout layout = params.layout();

    blas::Uplo uplo = params.uplo();
    blas::Op trans = params.trans();
    T alpha = params.alpha();
    T beta  = params.beta();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t align = params.align();
    int verbose = params.verbose();
    double tol = params.tol();
    int mode = params.latms_mode();
    blas::real_type<T> cond = params.latms_cond();
    blas::real_type<T> accuracy = params.routine == "syrk_dd" ?
        std::numeric_limits<blas::real_type<T>>::epsilon() : params.accuracy();

    if (params.routine == "syrk_dc" ||
        params.routine == "syrk_cc") {
        params.rk();
    }

    if (!run) return;

    // quick returns
    // todo: relax these assumptions
    if (params.routine != "syrk_dd") {
        if (trans != blas::Op::NoTrans) {
            printf("skipping: only trans=NoTrans is supported.\n");
            return;
        }
    }
    if (params.routine == "syrk_dc" || params.routine == "syrk_cc") {
        printf("skipping: syrk_dc and syrk_cc aren't yet supported.\n");
        return;
    }

    int64_t Am = trans == blas::Op::NoTrans ? n : k;
    int64_t An = trans == blas::Op::NoTrans ? k : n;
    int64_t Cm = n;
    int64_t Cn = n;

    // todo
    // if (layout == blas::Layout::RowMajor) {
    //     std::swap(Am, An);
    // }

    int64_t lda = testsweeper::roundup(Am, align);
    int64_t ldc = testsweeper::roundup(Cm, align);

    std::vector<T> Adata(lda * An);
    std::vector<T> Cdata(ldc * Cn);

    // int64_t idist = 1;
    int iseed[4] = {0, 0, 0, 1};

    // lapack::larnv(idist, iseed, lda * An, &Adata[0]);
    generate_dense_matrix(Am, An, &Adata[0], lda, iseed, mode, cond);

    // lapack::larnv(idist, iseed, ldc * n, &Cdata[0]);
    generate_dense_matrix(Cm, Cn, &Cdata[0], ldc, iseed, mode, cond);

    blas::real_type<T> Anorm =
                lapack::lange(lapack::Norm::Inf, Am, An, &Adata[0], lda);
    blas::real_type<T> Cnorm =
                lapack::lansy(lapack::Norm::Inf, uplo, Cn, &Cdata[0], ldc);

    hcore::DenseTile<T> A(Am, An, &Adata[0], lda);
    A.op(trans);
    hcore::DenseTile<T> C(Cm, Cn, &Cdata[0], ldc);
    C.op(trans);
    C.uplo(uplo);

    T nan_ = nan("");
    if (uplo == blas::Uplo::Lower) {
        lapack::laset(lapack::MatrixType::Upper,
            Cm-1, Cn-1, nan_, nan_, &Cdata[0 + 1 * ldc], ldc);
    }
    else {
        lapack::laset(lapack::MatrixType::Lower,
            Cm-1, Cn-1, nan_, nan_, &Cdata[1 + 0 * ldc], ldc);
    }

    std::vector<T> Cref;
    if (params.check() == 'y') {
        Cref = Cdata;
    }

    assert(!(
        blas::is_complex<T>::value &&
        (C.op() == blas::Op::ConjTrans || A.op() == blas::Op::ConjTrans)));

    if (verbose) {
        pretty_print(A, "A");
        pretty_print(C, "C");

        if (verbose > 1) {
            pretty_print(Cm, Cn, &Cref[0], ldc, "Cref");
        }
    }

    std::vector<T> AUVdata; //, CUVdata; // todo
    int64_t Ark; //, Crk; // todo
    hcore::CompressedTile<T> AUV; //, CUV; // todo

    if (params.routine == "syrk_cd" ||
        params.routine == "syrk_cc") {
        compress_dense_matrix(Am, An, Adata, lda, AUVdata, Ark, accuracy);

        AUV = hcore::CompressedTile<T>(Am, An, &AUVdata[0], lda, Ark, accuracy);
        AUV.op(trans);

        if (verbose) {
            pretty_print(AUV, "A");
        }
    }

    if (params.routine == "syrk_dc" ||
        params.routine == "syrk_cc") {
        assert(false);
        // todo
        // compress_dense_matrix(Cm, Cn, Cdata, ldc, CUVdata, Crk, accuracy);

        // CUV = hcore::CompressedTile<T>(Cm, Cn, &CUVdata[0], ldc, Crk, accuracy);
        // CUV.op(trans);
        // CUV.uplo(uplo);

        // if (verbose) {
        //     pretty_print(CUV, "C");
        // }
    }

    double gflops = 0.0;
    double time_start = testsweeper::get_wtime();

    if (params.routine == "syrk_dd") {
        hcore::syrk<T>(alpha, A, beta, C);
        gflops = blas::Gflop<T>::syrk(n, k);
    }
    else if (params.routine == "syrk_dc") {
        assert(false);
        // todo
        // hcore::syrk<T>(alpha, A, beta, CUV);
        // gflops = 
    }
    else if (params.routine == "syrk_cd") {
        hcore::syrk<T>(alpha, AUV, beta, C);
        gflops = blas::Gflop<T>::syrk(Ark, An) +
                 blas::Gflop<T>::gemm(Am, Ark, Ark) +
                 blas::Gflop<T>::gemm(Am, Am, Ark);
    }
    else if (params.routine == "syrk_cc") {
        assert(false);
        // todo
        // gflops =
    }

    double time_end = testsweeper::get_wtime();
    params.time() = time_end - time_start;
    params.gflops() = gflops / params.time();

    if (verbose) {
        if (params.routine == "syrk_dd" ||
            params.routine == "syrk_cd") {
            pretty_print(C, "C");
        }
        else if (params.routine == "syrk_dc" ||
                 params.routine == "syrk_cc") {
            assert(false);
            // todo
            // pretty_print(CUV, "C");
        }
    }

    if (params.routine == "syrk_dc" ||
        params.routine == "syrk_cc") {
        assert(false);
        // todo
        // params.rk() = std::to_string(Crk) + " -> " + std::to_string(CUV.rk());
    }

    if (params.check() == 'y') {
        double ref_time_start = testsweeper::get_wtime();
        {
            blas::syrk(
                blas::Layout::ColMajor, uplo, trans,
                n, k,
                alpha, &Adata[0], lda,
                beta,  &Cref[0],  ldc);
        }
        double ref_time_end = testsweeper::get_wtime();
        params.ref_time() = ref_time_end - ref_time_start;
        params.ref_gflops() = blas::Gflop<T>::syrk(n, k) / params.ref_time();

        if (verbose) {
            pretty_print(Cm, Cn, &Cref[0], ldc, "Cref");
        }

        T* C_ptr = &Cdata[0];

        // todo
        //if (params.routine == "syrk_cc") {
        //    // C = CU * CV.'
        //    blas::gemm(
        //        blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        //        Cm, Cn, CUV.rk(),
        //        1.0, CUV.Udata(), CUV.ldu(),
        //             CUV.Vdata(), CUV.ldv(),
        //        0.0, &C_ptr[0],   ldc);
        //}
        //else if (params.routine == "syrk_dc") {
        //    C_ptr = CUV.Udata();
        //}

        // Compute the Residual ||Cref - C||_inf.

        for (int64_t j = 0; j < Cn; ++j) {
            for (int64_t i = 0; i < Cm; ++i) {
                if ((uplo == blas::Uplo::Lower && i >= j) ||
                    (uplo == blas::Uplo::Upper && i <= j)) {
                    Cref[i + j * ldc] -= C_ptr[i + j * ldc];
                }
            }
        }

        if (verbose) {
            pretty_print(Cm, Cn, &Cref[0], ldc, "Cref_diff_C");
        }

        params.error() =
                    lapack::lansy(lapack::Norm::Inf, uplo, Cn, &Cref[0], ldc)
                    / (sqrt(blas::real_type<T>(k) + 2) * std::abs(alpha) *
                       Anorm * Anorm + 2 * std::abs(beta) * Cnorm);

        // Complex number need extra factor.
        // See "Accuracy and Stability of Numerical Algorithms", by Nicholas J.
        // Higham, Section 3.6, 2002.
        if (blas::is_complex<T>::value) {
            params.error() /= 2 * sqrt(2);
        }
        params.okay() = (params.error() < tol * accuracy);
    }

}

void syrk_test_dispatch(Params& params, bool run)
{
    switch(params.datatype()) {
        case testsweeper::DataType::Single:
            syrk_test_execute<float>(params, run);
            break;
        case testsweeper::DataType::Double:
            syrk_test_execute<double>(params, run);
            break;
        case testsweeper::DataType::SingleComplex:
            syrk_test_execute<std::complex<float>>(params, run);
            break;
        case testsweeper::DataType::DoubleComplex:
            syrk_test_execute<std::complex<double>>(params, run);
            break;
        default:
            throw std::invalid_argument("Unsupported data type.");
            break;
    }
}
