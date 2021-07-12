// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
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

namespace hcore {
namespace test {

template <typename T>
void syrk(Params& params, bool run)
{
    using real_t = blas::real_type<T>;

    blas::Layout layout = params.layout();

    blas::Uplo uplo = params.uplo();
    blas::Op transA = params.transA();
    blas::Op transC = params.transC();

    T alpha = params.alpha.get<T>();
    T beta  = params.beta.get<T>();

    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t mode = params.latms_mode();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    real_t tol = params.tol();
    real_t cond = params.latms_cond();
    real_t accuracy =
        params.routine == "syrk_dd" ? std::numeric_limits<real_t>::epsilon()
                                    : params.accuracy();

    // todo
    // if (params.routine == "syrk_dc" ||
    //     params.routine == "syrk_cc") {
    //     params.rk();
    // }

    if (!run) return;

    // quick returns
    if (params.routine == "syrk_dc" || params.routine == "syrk_cc") {
        printf("skipping: syrk_dc and syrk_cc aren't yet supported.\n");
        return;
    }

    if (blas::is_complex<T>::value) {
        if (transA == blas::Op::ConjTrans || transC == blas::Op::ConjTrans) {
            printf("skipping: wrong combinations of transA/transC.\n");
            return;
        }
    }

    // todo: relax these assumptions
    if (params.routine != "syrk_dd") {
        if (transA != blas::Op::NoTrans) {
            printf("skipping: only transA=NoTrans is supported.\n");
            return;
        }
        if (transC != blas::Op::NoTrans) {
            printf("skipping: only transC=NoTrans is supported.\n");
            return;
        }
        if (layout != blas::Layout::ColMajor) {
            printf("skipping: only layout=ColMajor is supported.\n");
            return;
        }
    }

    int64_t Am = transA == blas::Op::NoTrans ? n : k;
    int64_t An = transA == blas::Op::NoTrans ? k : n;

    if (layout == blas::Layout::RowMajor) {
        std::swap(Am, An);
    }

    int64_t lda = testsweeper::roundup(Am, align);
    int64_t ldc = testsweeper::roundup(n,  align);

    std::vector<T> Adata(lda * An);
    std::vector<T> Cdata(ldc *  n);

    // int64_t idist = 1;
    int64_t iseed[4] = {0, 0, 0, 1};

    // lapack::larnv(idist, iseed, lda * An, &Adata[0]);
    generate_dense_matrix(Am, An, &Adata[0], lda, iseed, mode, cond);

    // lapack::larnv(idist, iseed, ldc * n, &Cdata[0]);
    generate_dense_matrix(n, n, &Cdata[0], ldc, iseed, mode, cond);

    lapack::Norm norm = lapack::Norm::Inf; // todo: variable norm type
    real_t Anorm = lapack::lange(norm, Am, An, &Adata[0], lda);
    real_t Cnorm = lapack::lansy(norm, uplo, n, &Cdata[0], ldc);

    if (layout == blas::Layout::RowMajor) {
        std::swap(Am, An);
    }

    hcore::Tile<T> A(Am, An, &Adata[0], lda, layout);
    A.op(transA);
    hcore::Tile<T> C(n, n, &Cdata[0], ldc, layout);
    C.op(transC);
    C.uplo(uplo);

    set_dense_uplo(uplo, n, n, &Cdata[0], ldc);

    std::vector<T> Cref;
    if (params.check() == 'y') {
        Cref.resize(ldc * n);
        copy(&Cref[0], ldc, C);
    }

    if (verbose) {
        pretty_print(A, "A");
        pretty_print(C, "C");

        if (verbose > 1) {
            pretty_print(n, n, &Cref[0], ldc, "Cref");
        }
    }

    std::vector<T> AUVdata; //, CUVdata; // todo
    int64_t Ark; //, Crk; // todo
    hcore::CompressedTile<T> AUV; //, CUV; // todo

    if (params.routine == "syrk_cd" ||
        params.routine == "syrk_cc") {
        compress_dense_matrix(Am, An, Adata, lda, AUVdata, Ark, accuracy);

        AUV = hcore::CompressedTile<T>(Am, An, &AUVdata[0], lda, Ark, accuracy);
        AUV.op(transA);

        if (verbose) {
            pretty_print(AUV, "A");
        }
    }

//    if (params.routine == "syrk_dc" ||
//        params.routine == "syrk_cc") {
//        compress_dense_matrix(n, n, Cdata, ldc, CUVdata, Crk, accuracy);
//
//        CUV = hcore::CompressedTile<T>(n, n, &CUVdata[0], ldc, Crk, accuracy);
//        CUV.op(transC);
//        CUV.uplo(uplo);
//
//        if (verbose) {
//            pretty_print(CUV, "C");
//        }
//    }

    double gflops = 0.0;
    double time_start = testsweeper::get_wtime();

    if (params.routine == "syrk_dd") {
        hcore::syrk<T>(alpha, A, beta, C);
        gflops = blas::Gflop<T>::syrk(n, k);
    }
    // else if (params.routine == "syrk_dc") {
    //     hcore::syrk<T>(alpha, A, beta, CUV);
    //     gflops =
    // }
    else if (params.routine == "syrk_cd") {
        hcore::syrk<T>(alpha, AUV, beta, C);
        gflops = blas::Gflop<T>::syrk(Ark, An) +
                 blas::Gflop<T>::gemm(Am, Ark, Ark) +
                 blas::Gflop<T>::gemm(Am, Am, Ark);
    }
    // else if (params.routine == "syrk_cc") {
    //    hcore::syrk<T>(alpha, AUV, beta, CUV);
    //    gflops =
    // }

    double time_end = testsweeper::get_wtime();
    params.time() = time_end - time_start;
    params.gflops() = gflops / params.time();

    if (verbose) {
        if (params.routine == "syrk_dd" ||
            params.routine == "syrk_cd") {
            pretty_print(C, "C");
        }
        // else if (params.routine == "syrk_dc" ||
        //          params.routine == "syrk_cc") {
        //     pretty_print(CUV, "C");
        // }
    }

    // if (params.routine == "syrk_dc" ||
    //     params.routine == "syrk_cc") {
    //     params.rk() = std::to_string(Crk) + "->" + std::to_string(CUV.rk());
    // }

    if (params.check() == 'y') {
        blas::Uplo uplo_ = uplo;
        if (transC != blas::Op::NoTrans) {
            uplo_ = (uplo_ == blas::Uplo::Lower ? blas::Uplo::Upper
                                                : blas::Uplo::Lower);
        }
        double ref_time_start = testsweeper::get_wtime();
        {
            blas::syrk(
                layout, uplo_, transA,
                n, k,
                alpha, &Adata[0], lda,
                beta,  &Cref[0],  ldc);
        }
        double ref_time_end = testsweeper::get_wtime();
        params.ref_time() = ref_time_end - ref_time_start;
        params.ref_gflops() = blas::Gflop<T>::syrk(n, k) / params.ref_time();

        if (verbose) {
            pretty_print(n, n, &Cref[0], ldc, "Cref");
        }

        // todo
        // if (params.routine == "syrk_dc" ||
        //     params.routine == "syrk_cc") {
        //     // C = CU * CV.'
        //     blas::gemm(
        //         blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
        //         Cm, Cn, CUV.rk(),
        //         1.0, CUV.Udata(), CUV.ldu(),
        //              CUV.Vdata(), CUV.ldv(),
        //         0.0, &Cdata[0],   ldc);
        // }

        // Compute the Residual ||Cref - C||_inf.        

        // if (params.routine == "syrk_dc") {
        //     diff(&Cref[0], ldc, CUV);
        // }
        // else {
        diff(&Cref[0], ldc, C);
        // }

        if (verbose) {
            pretty_print(n, n, &Cref[0], ldc, "Cref_diff_C");
        }

        params.error() = lapack::lansy(norm, uplo_, n, &Cref[0], ldc)
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

void syrk_dispatch(Params& params, bool run) {
    switch(params.datatype()) {
        case testsweeper::DataType::Single:
            hcore::test::syrk<float>(params, run);
            break;
        case testsweeper::DataType::Double:
            hcore::test::syrk<double>(params, run);
            break;
        case testsweeper::DataType::SingleComplex:
            hcore::test::syrk<std::complex<float>>(params, run);
            break;
        case testsweeper::DataType::DoubleComplex:
            hcore::test::syrk<std::complex<double>>(params, run);
            break;
        default:
            throw hcore::Error("Unsupported data type.");
            break;
    }
}

} // namespace test
} // namespace hcore