// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include <cstdint>
#include <complex>
#include <vector>
#include <limits>

#include "testsweeper.hh"
#include "blas/flops.hh"
#include "lapack.hh"
#include "blas.hh"

#include "matrix_generator.hh"
#include "print_matrix.hh"
#include "hcore/hcore.hh"
#include "hcore/flops.hh"
#include "test.hh"

namespace hcore {
namespace test {

template <typename T>
void gemm(Params& params, bool run) {
    blas::Layout layout = params.layout();

    blas::Op transA = params.transA();
    blas::Op transB = params.transB();

    T alpha = params.alpha.get<T>();
    T beta  = params.beta.get<T>();

    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t align = params.align();
    int64_t verbose = params.verbose();

    lapack::Norm norm = params.norm();

    blas::real_type<T> tol = params.tol();

    int64_t mode = params.latms_mode();
    blas::real_type<T> cond = params.latms_cond();
    blas::real_type<T> dmax = params.latms_dmax();

    blas::real_type<T> accuracy = params.routine == "gemm_ddd"
                            ? std::numeric_limits<blas::real_type<T>>::epsilon()
                            : params.accuracy();

    bool use_gemm = params.use_gemm() == 'y';
    bool use_Segma0_as_tol = params.use_Segma0_as_tol() == 'y';
    int64_t truncate_with_fixed_rk = params.truncate_with_fixed_rk();

    if (params.routine == "gemm_ddc" || params.routine == "gemm_dcc" ||
        params.routine == "gemm_cdc" || params.routine == "gemm_ccc") {
        params.rk();
    }

    if (!run) return;

    int64_t Am = transA == blas::Op::NoTrans ? m : k;
    int64_t An = transA == blas::Op::NoTrans ? k : m;
    int64_t Bm = transB == blas::Op::NoTrans ? k : n;
    int64_t Bn = transB == blas::Op::NoTrans ? n : k;
    int64_t Cm = m;
    int64_t Cn = n;

    int64_t Am_orig = Am;
    int64_t An_orig = An;
    int64_t Bm_orig = Bm;
    int64_t Bn_orig = Bn;
    int64_t Cm_orig = Cm;
    int64_t Cn_orig = Cn;

    if (layout == blas::Layout::RowMajor) {
        std::swap(Am, An);
        std::swap(Bm, Bn);
        std::swap(Cm, Cn);
    }

    int64_t lda = testsweeper::roundup(Am, align);
    int64_t ldb = testsweeper::roundup(Bm, align);
    int64_t ldc = testsweeper::roundup(Cm, align);

    std::vector<T> Adata(lda*An);
    std::vector<T> Bdata(ldb*Bn);
    std::vector<T> Cdata(ldc*Cn);

    int64_t iseed[4] = {0, 0, 0, 1};

    generate_matrix(Am, An, Adata, lda, iseed, mode, cond, dmax);
    generate_matrix(Bm, Bn, Bdata, ldb, iseed, mode, cond, dmax);
    generate_matrix(Cm, Cn, Cdata, ldc, iseed, mode, cond, dmax);

    blas::real_type<T> Anorm = lapack::lange(norm, Am, An, &Adata[0], lda);
    blas::real_type<T> Bnorm = lapack::lange(norm, Bm, Bn, &Bdata[0], ldb);
    blas::real_type<T> Cnorm = lapack::lange(norm, Cm, Cn, &Cdata[0], ldc);

    if (verbose) {
        print_matrix(Am, An, &Adata[0], lda, "A");
        print_matrix(Bm, Bn, &Bdata[0], ldb, "B");
        print_matrix(Cm, Cn, &Cdata[0], ldc, "C");

        printf("||A||_%c = %.2e, ||B||_%c = %.2e, ||C||_%c = %.2e\n",
               char(norm), Anorm, char(norm), Bnorm, char(norm), Cnorm);
    }

    std::vector<T> Cref;
    if (params.check() == 'y') {
        Cref = Cdata;

        if (verbose > 1)
            print_matrix(Cm, Cn, &Cref[0], ldc, "Cref");
    }

    hcore::Tile<T> A, B, C;
    if (params.routine == "gemm_ddd" || params.routine == "gemm_ddc"
        || params.routine == "gemm_dcd" || params.routine == "gemm_dcc") {
        A = hcore::Tile<T>(Am_orig, An_orig, &Adata[0], lda, layout);
        if (transA == blas::Op::Trans)
            A = transpose(A);
        else if (transA == blas::Op::ConjTrans)
            A = conjugate_transpose(A);
    }
    if (params.routine == "gemm_ddd" || params.routine == "gemm_ddc"
        || params.routine == "gemm_cdd" || params.routine == "gemm_cdc") {
        B = hcore::Tile<T>(Bm_orig, Bn_orig, &Bdata[0], ldb, layout);
        if (transB == blas::Op::Trans)
        B = transpose(B);
    else if (transB == blas::Op::ConjTrans)
        B = conjugate_transpose(B);
    }
    if (params.routine == "gemm_ddd" || params.routine == "gemm_dcd"
        || params.routine == "gemm_cdd" || params.routine == "gemm_ccd") {
        C = hcore::Tile<T>(Cm_orig, Cn_orig, &Cdata[0], ldc, layout);
    }

    T *AUVdata, *BUVdata, *CUVdata;
    hcore::CompressedTile<T> AUV, BUV, CUV;
    if (params.routine == "gemm_cdd" || params.routine == "gemm_cdc"
        || params.routine == "gemm_ccd" || params.routine == "gemm_ccc") {
        int64_t ldu, ldv, rk;
        compress_matrix(Am, An, Adata, lda, &AUVdata, ldu, ldv, rk, accuracy,
                        align, (verbose > 3));

        AUV = hcore::CompressedTile<T>(Am_orig, An_orig, &AUVdata[0], ldu, ldv,
                                       rk, accuracy, layout);
        if (verbose)
            print_matrix(AUV, "A");

        if (transA == blas::Op::Trans)
            AUV = transpose(AUV);
        else if (transA == blas::Op::ConjTrans)
            AUV = conjugate_transpose(AUV);
    }
    if (params.routine == "gemm_dcd" || params.routine == "gemm_dcc"
        || params.routine == "gemm_ccd" || params.routine == "gemm_ccc") {
        int64_t ldu, ldv, rk;
        compress_matrix(Bm, Bn, Bdata, ldb, &BUVdata, ldu, ldv, rk, accuracy,
                        align, (verbose > 3));

        BUV = hcore::CompressedTile<T>(Bm_orig, Bn_orig, &BUVdata[0], ldu, ldv,
                                       rk, accuracy, layout);
        if (verbose)
            print_matrix(BUV, "B");

        if (transB == blas::Op::Trans)
            BUV = transpose(BUV);
        else if (transB == blas::Op::ConjTrans)
            BUV = conjugate_transpose(BUV);
    }
    if (params.routine == "gemm_ddc" || params.routine == "gemm_dcc"
        || params.routine == "gemm_cdc" || params.routine == "gemm_ccc") {
        int64_t ldu, ldv, rk;
        compress_matrix(Cm, Cn, Cdata, ldc, &CUVdata, ldu, ldv, rk, accuracy,
                        align, (verbose > 3));

        CUV = hcore::CompressedTile<T>(Cm_orig, Cn_orig, &CUVdata[0], ldu, ldv,
                                       rk, accuracy, layout);
        if (verbose)
            print_matrix(CUV, "C");

        params.rk() = std::to_string(rk);
    }

    hcore::Options const opts = {
        { hcore::Option::UseGEMM,        use_gemm               },
        { hcore::Option::FixedRank,      truncate_with_fixed_rk },
        { hcore::Option::UseSegma0AsTol, use_Segma0_as_tol      }
    };

    double gflops = 0.0;
    double time_start = testsweeper::get_wtime();

    if (params.routine == "gemm_ddd") {
        hcore::gemm<T>(alpha, A, B, beta, C);
        gflops = hcore::Gflop<T>::gemm(A, B, C);
    }
    else if (params.routine == "gemm_ddc") {
        hcore::gemm<T>(alpha, A, B, beta, CUV);
        gflops = hcore::Gflop<T>::gemm(A, B, CUV);
    }
    else if (params.routine == "gemm_dcd") {
        hcore::gemm<T>(alpha, A, BUV, beta, C);
        gflops = hcore::Gflop<T>::gemm(A, BUV, C);
    }
    else if (params.routine == "gemm_dcc") {
        hcore::gemm<T>(alpha, A, BUV, beta, CUV, opts);
        gflops = hcore::Gflop<T>::gemm(A, BUV, CUV);
    }
    else if (params.routine == "gemm_cdd") {
        hcore::gemm<T>(alpha, AUV, B, beta, C);
        gflops = hcore::Gflop<T>::gemm(AUV, B, C);
    }
    else if (params.routine == "gemm_cdc") {
        hcore::gemm<T>(alpha, AUV, B, beta, CUV, opts);
        gflops = hcore::Gflop<T>::gemm(AUV, B, CUV);
    }
    else if (params.routine == "gemm_ccd") {
        hcore::gemm<T>(alpha, AUV, BUV, beta, C);
        gflops = hcore::Gflop<T>::gemm(AUV, BUV, C);
    }
    else if (params.routine == "gemm_ccc") {
        hcore::gemm<T>(alpha, AUV, BUV, beta, CUV, opts);
        gflops = hcore::Gflop<T>::gemm(AUV, BUV, CUV);
    }

    double time_end = testsweeper::get_wtime();
    params.time() = time_end - time_start;
    params.gflops() = gflops / params.time();

    if (params.routine == "gemm_ddc")
        C = hcore::Tile<T>(CUV); // decompress to full spectrum

    if (verbose) {
        if (params.routine == "gemm_ddd" || params.routine == "gemm_dcd" ||
            params.routine == "gemm_cdd" || params.routine == "gemm_ccd" ||
            params.routine == "gemm_ddc") {
            print_matrix(C, "C");
        }
        else if (params.routine == "gemm_dcc" || params.routine == "gemm_cdc" ||
                 params.routine == "gemm_ccc") {
            print_matrix(CUV, "C");
        }
    }

    if (params.routine == "gemm_ddc" || params.routine == "gemm_dcc" ||
        params.routine == "gemm_cdc" || params.routine == "gemm_ccc") {
        params.rk() += "->" + std::to_string(CUV.rk());
    }

    if (params.check() == 'y') {
        double ref_time_start = testsweeper::get_wtime();
        blas::gemm(layout, transA, transB, m, n, k,
                   alpha, &Adata[0], lda,
                          &Bdata[0], ldb,
                   beta,  &Cref[0],  ldc);
        double ref_time_end = testsweeper::get_wtime();
        params.ref_time()   = ref_time_end - ref_time_start;
        params.ref_gflops() = blas::Gflop<T>::gemm(m, n, k) / params.ref_time();

        if (verbose)
            print_matrix(Cm, Cn, &Cref[0], ldc, "Cref");

        T* Cdata_ = &Cdata[0];

        if (params.routine == "gemm_ddc" || params.routine == "gemm_dcc"
            || params.routine == "gemm_cdc" || params.routine == "gemm_ccc") {
            if (params.routine == "gemm_ddc") {
                Cdata_ = C.data(); // get the new copy after being updated
            }
            else {
                T const* CU = layout == blas::Layout::ColMajor ? CUV.Udata()
                                                               : CUV.Vdata();
                T const* CV = layout == blas::Layout::ColMajor ? CUV.Vdata()
                                                               : CUV.Udata();

                int64_t ldcu = layout == blas::Layout::ColMajor ? CUV.Ustride()
                                                                : CUV.Vstride();
                int64_t ldcv = layout == blas::Layout::ColMajor ? CUV.Vstride()
                                                                : CUV.Ustride();
                // C = CU * CV.'
                blas::gemm(blas::Layout::ColMajor,
                           blas::Op::NoTrans, blas::Op::NoTrans,
                           Cm, Cn, CUV.rk(),
                           1.0, CU,        ldcu,
                                CV,        ldcv,
                           0.0, &Cdata[0], ldc);
            }
        }

        // compute the Residual ||Cref - C||_inf
        for (int64_t j = 0; j < Cn; ++j)
            for (int64_t i = 0; i < Cm; ++i)
                Cref[i + j*ldc] -= Cdata_[i + j*ldc];

        if (verbose)
            print_matrix(Cm, Cn, &Cref[0], ldc, "Cref_diff_C");

        params.error() = lapack::lange(norm, Cm, Cn, &Cref[0], ldc)
                       / (sqrt(blas::real_type<T>(k) + 2) * std::abs(alpha)
                       * Anorm * Bnorm + 2 * std::abs(beta) * Cnorm);

        // Complex number need extra factor.
        // See "Accuracy and Stability of Numerical Algorithms", by Nicholas J.
        // Higham, Section 3.6, 2002.
        if (blas::is_complex<T>::value)
            params.error() /= 2 * sqrt(2);

        params.okay() = (params.error() < tol * accuracy);
    }

    // cleanup
    if (params.routine == "gemm_cdd" || params.routine == "gemm_cdc"
        || params.routine == "gemm_ccd" || params.routine == "gemm_ccc") {
        delete [] AUVdata;
    }
    if (params.routine == "gemm_dcd" || params.routine == "gemm_dcc"
        || params.routine == "gemm_ccd" || params.routine == "gemm_ccc") {
        delete [] BUVdata;
    }
    if (params.routine == "gemm_ddc" || params.routine == "gemm_dcc"
        || params.routine == "gemm_cdc" || params.routine == "gemm_ccc") {
        delete [] CUVdata;

        // the data array has already been replaced with a new one inside the
        // gemm routine, so no need to keep the old local copy; however, the
        // data pointer ownership is still belong to the user. Thus the data
        // pointer must be delete by the user, since the concept of the Tile
        // class is to not own the data array pointer
        CUV.clear();
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
            throw Error("Unsupported data type.");
            break;
    }
}

} // namespace test
} // namespace hcore
