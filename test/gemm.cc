// Copyright (c) 2017-2021, King Abdullah University of Science and Technology
// (KAUST). All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "test.hh"
#include "hcore.hh"
#include "tile_utils.hh"
#include "hcore/flops.hh"
#include "matrix_utils.hh"

#include "blas.hh"
#include "blas/flops.hh"
#include "lapack.hh"
#include "testsweeper.hh"

#include <vector>
#include <cstdint>
#include <complex>
#include <algorithm>
#include <initializer_list>

namespace hcore {
namespace test {

template <typename T>
void gemm(Params& params, bool run) {
    using real_t = blas::real_type<T>;

    blas::Layout layout = params.layout();

    blas::Op transA = params.transA();
    blas::Op transB = params.transB();
    blas::Op transC = params.transC();

    T alpha = params.alpha.get<T>();
    T beta  = params.beta.get<T>();

    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t mode = params.latms_mode();
    int64_t align = params.align();
    int64_t verbose = params.verbose();
    int64_t truncate_with_fixed_rk = params.truncate_with_fixed_rk();

    real_t tol = params.tol();
    real_t cond = params.latms_cond();
    real_t accuracy =
        params.routine == "gemm_ddd" ? std::numeric_limits<real_t>::epsilon()
                                     : params.accuracy();

    bool use_trmm = params.use_trmm() == 'y';
    bool use_ungqr = params.use_ungqr() == 'y';
    bool truncate_with_tol = params.truncate_with_tol() == 'y';

    if (params.routine == "gemm_ddc" || params.routine == "gemm_dcc" ||
        params.routine == "gemm_cdc" || params.routine == "gemm_ccc") {
        params.rk();
    }

    if (!run) return;

    // quick returns
    if (params.routine == "gemm_ddd") {
        if (blas::is_complex<T>::value) {
            if ((transC == blas::Op::Trans &&
                 (transA == blas::Op::ConjTrans || transB == blas::Op::ConjTrans)
                ) ||
                (transC == blas::Op::ConjTrans &&
                 (transA == blas::Op::Trans || transB == blas::Op::Trans)
                )) {
                printf("skipping: wrong combinations of transA/transB/transC.\n");
                return;
            }
        }
    }

    // todo: relax these assumptions
    if (params.routine != "gemm_ddd" && params.routine != "gemm_ddc") {
        if (transA != blas::Op::NoTrans) {
            printf("skipping: only transA=NoTrans is supported.\n");
            return;
        }
        if (transB != blas::Op::NoTrans) {
            printf("skipping: only transB=NoTrans is supported.\n");
            return;
        }
    }

    if (params.routine != "gemm_ddd") {
        if (transC != blas::Op::NoTrans) {
            printf("skipping: only transC=NoTrans is supported.\n");
            return;
        }
        if (layout != blas::Layout::ColMajor) {
            printf("skipping: only layout=ColMajor is supported.\n");
            return;
        }
    }

    int64_t Am = transA == blas::Op::NoTrans ? m : k;
    int64_t An = transA == blas::Op::NoTrans ? k : m;
    int64_t Bm = transB == blas::Op::NoTrans ? k : n;
    int64_t Bn = transB == blas::Op::NoTrans ? n : k;
    int64_t Cm = transC == blas::Op::NoTrans ? m : n;
    int64_t Cn = transC == blas::Op::NoTrans ? n : m;

    if (layout == blas::Layout::RowMajor) {
        std::swap(Am, An);
        std::swap(Bm, Bn);
        std::swap(Cm, Cn);
    }

    int64_t lda = testsweeper::roundup(Am, align);
    int64_t ldb = testsweeper::roundup(Bm, align);
    int64_t ldc = testsweeper::roundup(Cm, align);

    std::vector<T> Adata(lda * An);
    std::vector<T> Bdata(ldb * Bn);
    std::vector<T> Cdata(ldc * Cn);

    // int64_t idist = 1;
    int64_t iseed[4] = {0, 0, 0, 1};

    // lapack::larnv(idist, iseed, lda * An, &Adata[0]);
    generate_dense_matrix(Am, An, &Adata[0], lda, iseed, mode, cond);

    // lapack::larnv(idist, iseed, ldb * Bn, &Bdata[0]);
    generate_dense_matrix(Bm, Bn, &Bdata[0], ldb, iseed, mode, cond);

    // lapack::larnv(idist, iseed, ldc * Cn, &Cdata[0]);
    generate_dense_matrix(Cm, Cn, &Cdata[0], ldc, iseed, mode, cond);

    lapack::Norm norm = lapack::Norm::Inf; // todo: variable norm type
    real_t Anorm = lapack::lange(norm, Am, An, &Adata[0], lda);
    real_t Bnorm = lapack::lange(norm, Bm, Bn, &Bdata[0], ldb);
    real_t Cnorm = lapack::lange(norm, Cm, Cn, &Cdata[0], ldc);

    if (layout == blas::Layout::RowMajor) {
        std::swap(Am, An);
        std::swap(Bm, Bn);
        std::swap(Cm, Cn);
    }

    hcore::Tile<T> A(Am, An, &Adata[0], lda, layout);
    A.op(transA);
    hcore::Tile<T> B(Bm, Bn, &Bdata[0], ldb, layout);
    B.op(transB);
    hcore::Tile<T> C(Cm, Cn, &Cdata[0], ldc, layout);
    C.op(transC);

    int64_t Cref_m = m;
    int64_t Cref_n = n;

    if (layout == blas::Layout::RowMajor) {
        std::swap(Cref_m, Cref_n);
    }

    int64_t ldcref = testsweeper::roundup(Cref_m, align);

    std::vector<T> Cref;
    if (params.check() == 'y') {
        Cref.resize(ldcref * Cref_n);
        copy(&Cref[0], ldcref, C);
    }

    if (verbose) {
        pretty_print(A, "A");
        pretty_print(B, "B");
        pretty_print(C, "C");

        if (verbose > 1) {
            pretty_print(Cref_m, Cref_n, &Cref[0], ldcref, "Cref");
        }
    }

    std::vector<T> AUVdata, BUVdata, CUVdata;
    int64_t Ark, Brk, Crk;
    hcore::CompressedTile<T> AUV, BUV, CUV;

    if (params.routine == "gemm_cdd" ||
        params.routine == "gemm_cdc" ||
        params.routine == "gemm_ccd" ||
        params.routine == "gemm_ccc") {
        compress_dense_matrix(Am, An, Adata, lda, AUVdata, Ark, accuracy);

        AUV = hcore::CompressedTile<T>(Am, An, &AUVdata[0], lda, Ark, accuracy, layout);
        AUV.op(transA);

        if (verbose) {
            pretty_print(AUV, "A");
        }
    }
    if (params.routine == "gemm_dcd" ||
        params.routine == "gemm_dcc" ||
        params.routine == "gemm_ccd" ||
        params.routine == "gemm_ccc") {
        compress_dense_matrix(Bm, Bn, Bdata, ldb, BUVdata, Brk, accuracy);

        BUV = hcore::CompressedTile<T>(Bm, Bn, &BUVdata[0], ldb, Brk, accuracy, layout);
        BUV.op(transB);

        if (verbose) {
            pretty_print(BUV, "B");
        }
    }
    if (params.routine == "gemm_ddc" ||
        params.routine == "gemm_dcc" ||
        params.routine == "gemm_cdc" ||
        params.routine == "gemm_ccc") {
        compress_dense_matrix(Cm, Cn, Cdata, ldc, CUVdata, Crk, accuracy);

        CUV = hcore::CompressedTile<T>(Cm, Cn, &CUVdata[0], ldc, Crk, accuracy, layout);
        CUV.op(transC);

        if (verbose) {
            pretty_print(CUV, "C");
        }
    }

    double gflops = 0.0;
    double time_start = testsweeper::get_wtime();

    if (params.routine == "gemm_ddd") {
        hcore::gemm<T>(alpha, A, B, beta, C);
        gflops = hcore::Gflop<T>::gemm(A, B, C);
        // gflops = blas::Gflop<T>::gemm(m, n, k);
    }
    else if (params.routine == "gemm_ddc") {
        hcore::gemm<T>(alpha, A, B, beta, CUV);
        gflops = hcore::Gflop<T>::gemm(A, B, CUV, Crk);
        // gflops = blas::Gflop<T>::gemm(Cm, Cn, An) +
        //          blas::Gflop<T>::gemm(Cm, Cn, Crk);
    }
    else if (params.routine == "gemm_dcd") {
        hcore::gemm<T>(alpha, A, BUV, beta, C);
        gflops = hcore::Gflop<T>::gemm(A, BUV, C);
        // gflops = blas::Gflop<T>::gemm(Cm, Brk, An) +
        //          blas::Gflop<T>::gemm(Cm, Cn, Brk);
    }
    else if (params.routine == "gemm_dcc") {
        hcore::gemm<T>(alpha, A, BUV, beta, CUV, 
                use_trmm, use_ungqr, truncate_with_tol, truncate_with_fixed_rk);
        gflops = hcore::Gflop<T>::gemm(A, BUV, CUV, Crk);
        // gflops = blas::Gflop<T>::gemm(Cm, Brk, An) +
        //          hcore::internal::Gflop<T>::rsvd(Cm, Cn, Brk, Crk, CUV.rk());
    }
    else if (params.routine == "gemm_cdd") {
        hcore::gemm<T>(alpha, AUV, B, beta, C);
        gflops = hcore::Gflop<T>::gemm(AUV, B, C);
        // gflops = blas::Gflop<T>::gemm(Ark, Cn, An) +
        //          blas::Gflop<T>::gemm(Cm, Cn, Ark);
    }
    else if (params.routine == "gemm_cdc") {
        hcore::gemm<T>(alpha, AUV, B, beta, CUV, 
                use_trmm, use_ungqr, truncate_with_tol, truncate_with_fixed_rk);
        gflops = hcore::Gflop<T>::gemm(AUV, B, CUV, Crk);
        // gflops = blas::Gflop<T>::gemm(Ark, Cn, An) +
        //          hcore::internal::Gflop<T>::rsvd(Cm, Cn, Ark, Crk, CUV.rk());
    }
    else if (params.routine == "gemm_ccd") {
        hcore::gemm<T>(alpha, AUV, BUV, beta, C);
        gflops = hcore::Gflop<T>::gemm(AUV, BUV, C);
        // gflops = blas::Gflop<T>::gemm(Ark, Brk, An) +
        //          (Ark <= Brk ? blas::Gflop<T>::gemm(Ark, Cn, Brk) +
        //                        blas::Gflop<T>::gemm(Cm, Cn, Ark)
        //                      : blas::Gflop<T>::gemm(Cm, Brk, Ark) +
        //                        blas::Gflop<T>::gemm(Cm, Cn, Brk));
    }
    else if (params.routine == "gemm_ccc") {
        hcore::gemm<T>(alpha, AUV, BUV, beta, CUV,
                use_trmm, use_ungqr, truncate_with_tol, truncate_with_fixed_rk);
        gflops = hcore::Gflop<T>::gemm(AUV, BUV, CUV, Crk);
        // gflops = blas::Gflop<T>::gemm(Ark, Brk, An) +
        //          (Ark <= Brk ? blas::Gflop<T>::gemm(Ark, Cn, Brk) +
        //                        hcore::internal::Gflop<T>::rsvd(Cm, Cn, Ark, Crk, CUV.rk())
        //                      : blas::Gflop<T>::gemm(Cm, Brk, Ark) +
        //                        hcore::internal::Gflop<T>::rsvd(Cm, Cn, Brk, Crk, CUV.rk()));
        // todo: for now use PASC paper, which assumes square matrices
        // int64_t max_Ark_Brk_Crk = std::max({Ark, Brk, Crk});
        // int64_t max_m_n_k = std::max({m, n, k});
        // gflops = (1e-9 * ((blas::is_complex<T>::value ? 3 : 1)
        //              * 36 * max_m_n_k * (max_Ark_Brk_Crk
        //              * max_Ark_Brk_Crk) + 157 * (max_Ark_Brk_Crk
        //              * max_Ark_Brk_Crk * max_Ark_Brk_Crk)));
    }

    double time_end = testsweeper::get_wtime();
    params.time() = time_end - time_start;
    params.gflops() = gflops / params.time();

    if (params.routine == "gemm_ddc") {
        C = hcore::Tile<T>(CUV);
    }

    if (verbose) {
        if (params.routine == "gemm_ddd" ||
            params.routine == "gemm_dcd" ||
            params.routine == "gemm_cdd" ||
            params.routine == "gemm_ccd" ||
            params.routine == "gemm_ddc") {
            pretty_print(C, "C");
        }
        else if (params.routine == "gemm_dcc" ||
                 params.routine == "gemm_cdc" ||
                 params.routine == "gemm_ccc") {
            pretty_print(CUV, "C");
        }
    }

    if (params.routine == "gemm_ddc" ||
        params.routine == "gemm_dcc" ||
        params.routine == "gemm_cdc" ||
        params.routine == "gemm_ccc") {
        params.rk() = std::to_string(Crk) + "->" + std::to_string(CUV.rk());
    }

    if (params.check() == 'y') {
        double ref_time_start = testsweeper::get_wtime();
        {
            blas::gemm(
                layout, transA, transB,
                m, n, k,
                alpha, &Adata[0], lda,
                       &Bdata[0], ldb,
                beta,  &Cref[0],  ldcref);
        }
        double ref_time_end = testsweeper::get_wtime();
        params.ref_time() = ref_time_end - ref_time_start;
        params.ref_gflops() =
            blas::Gflop<T>::gemm(m, n, k) / params.ref_time();

        if (verbose) {
            pretty_print(Cref_m, Cref_n, &Cref[0], ldcref, "Cref");
        }

        if (params.routine == "gemm_dcc" ||
            params.routine == "gemm_cdc" ||
            params.routine == "gemm_ccc") {
            // C = CU * CV.'
            blas::gemm(
                blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                Cm, Cn, CUV.rk(),
                1.0, CUV.Udata(), CUV.ldu(),
                     CUV.Vdata(), CUV.ldv(),
                0.0, &Cdata[0],   ldc);
        }

        // Compute the Residual ||Cref - C||_inf.        
        diff(&Cref[0], ldcref, C);

        if (verbose) {
            pretty_print(Cref_m, Cref_n, &Cref[0], ldcref, "Cref_diff_C");
        }

        params.error() = lapack::lange(norm, Cref_m, Cref_n, &Cref[0], ldcref)
                        / (sqrt(real_t(k) + 2) * std::abs(alpha) *
                           Anorm * Bnorm + 2 * std::abs(beta) * Cnorm);

        // Complex number need extra factor.
        // See "Accuracy and Stability of Numerical Algorithms", by Nicholas J.
        // Higham, Section 3.6, 2002.
        if (blas::is_complex<T>::value) {
            params.error() /= 2 * sqrt(2);
        }
        params.okay() = (params.error() < tol * accuracy);
    }

    if (params.routine == "gemm_ddc" ||
        params.routine == "gemm_dcc" ||
        params.routine == "gemm_cdc" ||
        params.routine == "gemm_ccc") {
        delete [] CUV.Udata();
    }

}

void gemm_dispatch(Params& params, bool run) {
    switch(params.datatype()) {
        case testsweeper::DataType::Single:
            hcore::test::gemm<float>(params, run);
            break;
        case testsweeper::DataType::Double:
            hcore::test::gemm<double>(params, run);
            break;
        case testsweeper::DataType::SingleComplex:
            hcore::test::gemm<std::complex<float>>(params, run);
            break;
        case testsweeper::DataType::DoubleComplex:
            hcore::test::gemm<std::complex<double>>(params, run);
            break;
        default:
            throw hcore::Error("Unsupported data type.");
            break;
    }
}

} // namespace test
} // namespace hcore