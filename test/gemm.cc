// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "test.hh"
#include "hcore.hh"
#include "flops.hh"
#include "pretty_print.hh"
#include "lapacke_wrappers.hh"

#include "blas.hh"
#include "lapack.hh"
#include "blas/flops.hh"

#include <vector>
#include <complex>
#include <algorithm>

template <typename T>
void generate_dense_matrix(
    int64_t m, int64_t n, T* A, int64_t lda, blas::real_type<T>& norm,
    lapack::Norm norm_type, int* iseed, int mode, blas::real_type<T> cond)
{
    int16_t min_m_n = std::min(m, n);

    std::vector<blas::real_type<T>> D(min_m_n);

    for (int64_t i = 0; i < min_m_n; ++i)
        D[i] = std::pow(10, -1*i);

    lapacke_latms(
        m, n, 'U', iseed, 'N', &D[0], mode, cond, -1.0, m-1, n-1, 'N', A, lda);

    norm = lapack::lange(norm_type, m, n, A, lda);
}

template <typename T>
void compress_dense_matrix(
    int64_t m, int64_t n, std::vector<T> A, int64_t lda,
    std::vector<T>& UV, int64_t& rk, blas::real_type<T> accuracy)
{
    int16_t min_m_n = std::min(m, n);

    std::vector<blas::real_type<T>> Sigma(min_m_n);

    std::vector<T> U(lda * min_m_n);
    std::vector<T> VT(min_m_n * n);

    lapack::gesvd(
        lapack::Job::SomeVec, lapack::Job::SomeVec,
        m, n, &A[0], lda, &Sigma[0],
              &U[0], lda,
              &VT[0], min_m_n);

    rk = 0;
    while (Sigma[rk] >= accuracy && rk < min_m_n)
        rk++;

    // todo: more conservative max rank assumption, e.g., min_m_n / 3.
    int64_t max_rk = min_m_n / 2;
    if (rk > max_rk)
        rk = max_rk;

    // VT eats Sigma.
    // todo: we may need to have uplo parameter:
    //       scale VT, if Lower, or scale U otherwise.
    for (int64_t i = 0; i < rk; ++i)
        blas::scal(n, Sigma[i], &VT[i], min_m_n);

    UV.reserve((lda + n) * rk);

    // copy first rk columns of U; UV = U(:,1:rk)
    // todo: assume column-major, what about row-major?
    UV.insert(UV.end(), U.begin(), U.begin() + (lda * rk));

    // copy first rk rows of VT; UV = VT(1:rk,:)
    // todo: assume column-major, what about row-major?
    lapack::lacpy(
        lapack::MatrixType::General, rk, n, &VT[0], min_m_n, &UV[lda * rk], rk);
}

template <typename T>
void gemm_test_execute(Params& params, bool run)
{
    blas::Op transA = params.transA();
    blas::Op transB = params.transB();

    T alpha = params.alpha();
    T beta  = params.beta();

    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();

    int64_t align = params.align();
    int64_t verbose = params.verbose();

    bool ref = params.check() == 'y';

    double tol = params.tol();

    int mode = params.latms_mode();

    blas::real_type<T> cond = params.latms_cond();
    blas::real_type<T> accuracy = params.routine == "gemm_ddd" ?
        std::numeric_limits<blas::real_type<T>>::epsilon() : params.accuracy();

    bool use_trmm = params.use_trmm() == 'y';
    bool use_ungqr = params.use_ungqr() == 'y';
    bool truncate_with_tol = params.truncate_with_tol() == 'y';
    int64_t truncate_with_fixed_rk = params.truncate_with_fixed_rk();

    if (params.routine == "gemm_ddc" ||
        params.routine == "gemm_dcc" ||
        params.routine == "gemm_cdc" ||
        params.routine == "gemm_ccc") {
        params.rk();
    }

    params.time();
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // quick returns
    // todo: relax these assumptions
    if (params.routine != "gemm_ddd") {
        if (transA != blas::Op::NoTrans) {
            printf("skipping: only transA=NoTrans is supported.\n");
            return;
        }
        if (transB != blas::Op::NoTrans) {
            printf("skipping: only transB=NoTrans is supported.\n");
            return;
        }
    }

    int64_t Am = transA == blas::Op::NoTrans ? m : k;
    int64_t An = transA == blas::Op::NoTrans ? k : m;
    int64_t Bm = transB == blas::Op::NoTrans ? k : n;
    int64_t Bn = transB == blas::Op::NoTrans ? n : k;
    int64_t Cm = m;
    int64_t Cn = n;

    // todo: relax this assumption
    blas::Layout layout = blas::Layout::ColMajor;
    // if (layout == blas::Layout::RowMajor) {
    //     std::swap(Am, An);
    //     std::swap(Bm, Bn);
    //     std::swap(Cm, Cn);
    // }

    // todo: relax this assumption
    lapack::Norm norm_type = lapack::Norm::Inf;

    int64_t lda = testsweeper::roundup(Am, align);
    int64_t ldb = testsweeper::roundup(Bm, align);
    int64_t ldc = testsweeper::roundup(Cm, align);

    std::vector<T> Adata(lda * An);
    std::vector<T> Bdata(ldb * Bn);
    std::vector<T> Cdata(ldc * Cn);

    blas::real_type<T> Anorm = 0.0;
    blas::real_type<T> Bnorm = 0.0;
    blas::real_type<T> Cnorm = 0.0;

    // int64_t idist = 1;
    int iseed[4] = {0, 0, 0, 1};

    // lapack::larnv(idist, iseed, lda * An, &Adata[0]);
    generate_dense_matrix(
        Am, An, &Adata[0], lda, Anorm, norm_type, iseed, mode, cond);

    // lapack::larnv(idist, iseed, ldb * Bn, &Bdata[0]);
    generate_dense_matrix(
        Bm, Bn, &Bdata[0], ldb, Bnorm, norm_type, iseed, mode, cond);

    // lapack::larnv(idist, iseed, ldc * Cn, &Cdata[0]);
    generate_dense_matrix(
        Cm, Cn, &Cdata[0], ldc, Cnorm, norm_type, iseed, mode, cond);

    std::vector<T> Cref;
    if (ref) {
        Cref = Cdata;
    }

    hcore::DenseTile<T> A(Am, An, &Adata[0], lda);
    A.op(transA);
    hcore::DenseTile<T> B(Bm, Bn, &Bdata[0], ldb);
    B.op(transB);
    hcore::DenseTile<T> C(Cm, Cn, &Cdata[0], ldc);

    if (verbose) {
        pretty_print(A, "A");
        pretty_print(B, "B");
        pretty_print(C, "C");

        if (verbose > 1) {
            pretty_print(Cm, Cn, &Cref[0], ldc, "Cref");
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

        AUV = hcore::CompressedTile<T>(Am, An, &AUVdata[0], lda, Ark, accuracy);

        if (verbose) {
            pretty_print(AUV, "A");
        }
    }
    if (params.routine == "gemm_dcd" ||
        params.routine == "gemm_dcc" ||
        params.routine == "gemm_ccd" ||
        params.routine == "gemm_ccc") {
        compress_dense_matrix(Bm, Bn, Bdata, ldb, BUVdata, Brk, accuracy);

        BUV = hcore::CompressedTile<T>(Bm, Bn, &BUVdata[0], ldb, Brk, accuracy);

        if (verbose) {
            pretty_print(BUV, "B");
        }
    }
    if (params.routine == "gemm_ddc" ||
        params.routine == "gemm_dcc" ||
        params.routine == "gemm_cdc" ||
        params.routine == "gemm_ccc") {
        compress_dense_matrix(Cm, Cn, Cdata, ldc, CUVdata, Crk, accuracy);

        CUV = hcore::CompressedTile<T>(Cm, Cn, &CUVdata[0], ldc, Crk, accuracy);

        if (verbose) {
            pretty_print(CUV, "C");
        }
    }

    double time_start = testsweeper::get_wtime();
    double gflops = 0;

    if (params.routine == "gemm_ddd") {
        hcore::gemm<T>(alpha, A, B, beta, C);
        gflops = blas::Gflop<T>::gemm(m, n, k);
    }
    else if (params.routine == "gemm_ddc") {
        hcore::gemm<T>(alpha, A, B, beta, CUV);
        gflops = blas::Gflop<T>::gemm(Cm, Cn, An) +
                 blas::Gflop<T>::gemm(Cm, Cn, Crk);
    }
    else if (params.routine == "gemm_dcd") {
        hcore::gemm<T>(alpha, A, BUV, beta, C);
        gflops = blas::Gflop<T>::gemm(Cm, Brk, An) +
                 blas::Gflop<T>::gemm(Cm, Cn, Brk);
    }
    else if (params.routine == "gemm_dcc") {
        hcore::gemm<T>(alpha, A, BUV, beta, CUV, 
                use_trmm, use_ungqr, truncate_with_tol, truncate_with_fixed_rk);
        // todo
        // gflops = blas::Gflop<T>::gemm(Cm, Brk, An) +
        //          internal::gemm;
    }
    else if (params.routine == "gemm_cdd") {
        hcore::gemm<T>(alpha, AUV, B, beta, C);
        gflops = blas::Gflop<T>::gemm(Ark, Cn, An) +
                 blas::Gflop<T>::gemm(Cm, Cn, Ark);
    }
    else if (params.routine == "gemm_cdc") {
        hcore::gemm<T>(alpha, AUV, B, beta, CUV, 
                use_trmm, use_ungqr, truncate_with_tol, truncate_with_fixed_rk);
        // todo
        // gflops = blas::Gflop<T>::gemm(Ark, Cn, An) +
        //          internal::gemm;
    }
    else if (params.routine == "gemm_ccd") {
        hcore::gemm<T>(alpha, AUV, BUV, beta, C);
        gflops = blas::Gflop<T>::gemm(Ark, Brk, An) +
                 (Ark <= Brk ? blas::Gflop<T>::gemm(Ark, Cn, Brk) +
                               blas::Gflop<T>::gemm(Cm, Cn, Ark)
                             : blas::Gflop<T>::gemm(Cm, Brk, Ark) +
                               blas::Gflop<T>::gemm(Cm, Cn, Brk));
    }
    else if (params.routine == "gemm_ccc") {
        hcore::gemm<T>(alpha, AUV, BUV, beta, CUV, 
                use_trmm, use_ungqr, truncate_with_tol, truncate_with_fixed_rk);
        gflops = hcore::Gflop<T>::gemm(m, n, k, Ark, Brk, CUV.rk());
        // todo
        // gflops = blas::Gflop<T>::gemm(Ark, Brk, An) +
        //          (Ark <= Brk ? blas::Gflop<T>::gemm(Ark, Cn, Brk) +
        //                        internal::gemm
        //                      : blas::Gflop<T>::gemm(Cm, Brk, Ark) +
        //                        internal::gemm;        
    }

    params.time() = testsweeper::get_wtime() - time_start;
    params.gflops() = gflops / params.time();

    if (verbose) {
        if (params.routine == "gemm_ddd" ||
            params.routine == "gemm_dcd" ||
            params.routine == "gemm_cdd" ||
            params.routine == "gemm_ccd") {
            pretty_print(C, "C");
        }
        else if (params.routine == "gemm_ddc" ||
                 params.routine == "gemm_dcc" ||
                 params.routine == "gemm_cdc" ||
                 params.routine == "gemm_ccc") {
            pretty_print(CUV, "C");
        }
    }

    if (params.routine == "gemm_ddc" ||
        params.routine == "gemm_dcc" ||
        params.routine == "gemm_cdc" ||
        params.routine == "gemm_ccc") {
        params.rk() = std::to_string(Crk) + " -> " + std::to_string(CUV.rk());
    }

    if (ref) {
        double ref_time_start = testsweeper::get_wtime();
        {
            blas::gemm(
                layout, transA, transB,
                m, n, k,
                alpha, &Adata[0], lda,
                       &Bdata[0], ldb,
                beta,  &Cref[0],  ldc);
        }
        double ref_time_end = testsweeper::get_wtime();
        params.ref_time() = ref_time_end - ref_time_start;
        params.ref_gflops() = blas::Gflop<T>::gemm(m, n, k) / params.ref_time();

        if (verbose) {
            pretty_print(Cm, Cn, &Cref[0], ldc, "Cref");
        }

        T* C_ptr = &Cdata[0];

        if (params.routine == "gemm_dcc" ||
            params.routine == "gemm_cdc" ||
            params.routine == "gemm_ccc") {
            // C = CU * CV.'
            blas::gemm(
                layout, blas::Op::NoTrans, blas::Op::NoTrans,
                Cm, Cn, CUV.rk(),
                1.0, CUV.Udata(), CUV.ldu(),
                     CUV.Vdata(), CUV.ldv(),
                0.0, &C_ptr[0],   ldc);
        }
        else if (params.routine == "gemm_ddc") {
            C_ptr = CUV.Udata();
        }

        // Compute the Residual ||Cref - C||_inf.

        for (int64_t j = 0; j < Cn; ++j) {
            for (int64_t i = 0; i < Cm; ++i) {
                Cref[i + j * ldc] -= C_ptr[i + j * ldc];
            }
        }

        if (verbose) {
            pretty_print(Cm, Cn, &Cref[0], ldc, "Cref_diff_C");
        }

        params.error() = lapack::lange(norm_type, Cm, Cn, &Cref[0], ldc)
                       / (sqrt(blas::real_type<T>(k) + 2) * std::abs(alpha) *
                          Anorm * Bnorm + 2 * std::abs(beta) * Cnorm);

        // Complex number need extra factor.
        // See "Accuracy and Stability of Numerical Algorithms", by Nicholas J.
        // Higham, Section 3.6, 2002.
        if (blas::is_complex<T>::value) {
            params.error() /= 2 * sqrt(2);
        }
        params.okay() = (params.error() < tol * accuracy);
    }
}

void gemm_test_dispatch(Params& params, bool run)
{
    switch(params.datatype()) {
        case testsweeper::DataType::Single:
            gemm_test_execute<float>(params, run);
            break;
        case testsweeper::DataType::Double:
            gemm_test_execute<double>(params, run);
            break;
        case testsweeper::DataType::SingleComplex:
            gemm_test_execute<std::complex<float>>(params, run);
            break;
        case testsweeper::DataType::DoubleComplex:
            gemm_test_execute<std::complex<double>>(params, run);
            break;
        default:
            throw std::invalid_argument("Unsupported data type.");
            break;
    }
}
