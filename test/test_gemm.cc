// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include <algorithm>
#include <cstdint>
#include <complex>
#include <vector>

#include "testsweeper.hh"
#include "blas/flops.hh"
#include "lapack.hh"
#include "blas.hh"

#include "lapack_wrappers.hh"
#include "hcore/hcore.hh"
#include "hcore/flops.hh"
#include "test.hh"

namespace hcore {
namespace test {

template <typename T>
void print_matrix(int64_t m, int64_t n, T const* A, int64_t lda,
                  char const* label, const char* format = "%9.4f")
{
    #define A(i_, j_) A[(i_) + size_t(lda)*(j_)]

    assert(m >= 0);
    assert(n >= 0);
    assert(lda >= m);

    char format2[32];
    if (blas::is_complex<T>::value) {
        snprintf(format2, sizeof(format2), " %s + %si", format, format);
    }
    else {
        snprintf(format2, sizeof(format2), " %s", format);
    }

    using blas::real;
    using blas::imag;

    printf("%s = [\n", label);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            if (blas::is_complex<T>::value) {
                printf(format2, real(A(i, j)), imag(A(i, j)));
            }
            else {
                printf(format2, A(i, j));
            }
        }
        printf("\n");
    }
    printf("];\n");

    #undef A
}

template <typename T>
void compress_matrix(int64_t m, int64_t n, std::vector<T> A, int64_t lda,
                     T** UV, int64_t& ldu, int64_t& ldv, int64_t& rk,
                     blas::real_type<T> tol, int64_t align, bool verbose=false)
{
    int16_t min_m_n = std::min(m, n);

    std::vector<blas::real_type<T>> Sigma(min_m_n);

    // A
    // m-by-n
    // m-by-min_m_n      (U)
    //      min_m_n-by-n (VT)
    //
    // ldu  = lda
    // ldvt = min_m_n
    //
    // std::swap(m, n)
    // op(A)
    // n-by-m
    // n-by-min_m_n      (op(U))
    //      min_m_n-by-m (op(VT))
    //
    // ldut  = lda
    // ldvtt = min_m_n

    ldu = lda;
    int64_t ldvt = min_m_n;

    std::vector<T> U(ldu * min_m_n);
    std::vector<T> VT(ldvt * n);

    lapack::gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec,
                  m, n, &A[0],  lda, &Sigma[0],
                        &U[0],  ldu,
                        &VT[0], ldvt);

    if (verbose) {
        print_matrix( m,       n,       &A[0],  lda,  "Asvd"  );
        print_matrix( m,       min_m_n, &U[0],  ldu,  "Usvd"  );
        print_matrix( min_m_n, n,       &VT[0], ldvt, "VTsvd" );
    }

    rk = 0;
    while (Sigma[rk] >= tol && rk < min_m_n)
        rk++;

    // todo: more conservative max rank assumption, e.g., min_m_n / 3.
    int64_t max_rk = min_m_n / 2;
    if (rk > max_rk)
        rk = max_rk;

    // A
    // mb-by-nb
    // mb-by-rk       (U)
    //       rk-by-nb (V)
    //
    // ldu = lda
    // ldv = rk
    //
    // U(:,1:rk) * V(1:rk,:)
    //
    // std::swap(mb, nb)
    // op(A)
    // nb-by-mb
    // nb-by-rk       (VT)
    //       rk-by-mb (U)
    //
    // ldv = lda
    // ldu = rk
    //
    // VT(:,1:rk) * U(1:rk,:)

    // VT eats Sigma.
    // todo: we may need to have uplo parameter:
    //       scale VT, if Lower, or scale U otherwise.
    for (int64_t i = 0; i < rk; ++i)
        blas::scal(n, Sigma[i], &VT[i], ldvt);

    ldv = testsweeper::roundup(rk, align);

    *UV = new T[ldu*rk + ldv*n];
    T* UTilda = *UV;
    T* VTilda = *UV + ldu*rk;

    // copy first rk cols of U; UV = U(:,1:rk)
    std::copy(U.begin(), U.begin() + ldu*rk, UTilda);

    // copy first rk rows of VT; UV = VT(1:rk,:)
    lapack::lacpy(lapack::MatrixType::General, rk, n, &VT[0], ldvt,
                                                      VTilda, ldv);
}

template <typename T>
void generate_matrix(int64_t m, int64_t n, std::vector<T>& A, int64_t lda,
                     int64_t* iseed, int64_t mode, blas::real_type<T> cond,
                     blas::real_type<T> dmax)
{
    int16_t min_m_n = std::min(m, n);

    std::vector<blas::real_type<T>> D(min_m_n);

    for (int64_t i = 0; i < min_m_n; ++i)
        D[i] = std::pow(10, -1*i);

    lapack::latms(m, n, lapack::Dist::Uniform, iseed, lapack::Sym::Nonsymmetric,
                  &D[0], mode, cond, dmax, m-1, n-1, lapack::Pack::NoPacking,
                  &A[0], lda);
}

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
    bool truncate_with_tol = params.truncate_with_tol() == 'y';
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

    int64_t Crk;
    T *AUVdata, *BUVdata, *CUVdata;
    hcore::CompressedTile<T> AUV, BUV, CUV;
    if (params.routine == "gemm_cdd" || params.routine == "gemm_cdc"
        || params.routine == "gemm_ccd" || params.routine == "gemm_ccc") {
        int64_t ldu, ldv, rk;
        compress_matrix(Am, An, Adata, lda, &AUVdata, ldu, ldv, rk, accuracy,
                        align, (verbose > 3));

        if (verbose) {
            print_matrix(Am, rk, &AUVdata[0],      ldu, "AU");
            print_matrix(rk, An, &AUVdata[ldu*rk], ldv, "AV");
        }

        AUV = hcore::CompressedTile<T>(Am_orig, An_orig, &AUVdata[0], ldu, ldv,
                                       rk, accuracy, layout);
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

        if (verbose) {
            print_matrix(Bm, rk, &BUVdata[0],      ldu, "BU");
            print_matrix(rk, Bn, &BUVdata[ldu*rk], ldv, "BV");
        }

        BUV = hcore::CompressedTile<T>(Bm_orig, Bn_orig, &BUVdata[0], ldu, ldv,
                                       rk, accuracy, layout);
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

        if (verbose) {
            print_matrix(Cm, rk, &CUVdata[0],      ldu, "CU");
            print_matrix(rk, Cn, &CUVdata[ldu*rk], ldv, "CV");
        }

        CUV = hcore::CompressedTile<T>(Cm_orig, Cn_orig, &CUVdata[0], ldu, ldv,
                                       rk, accuracy, layout);
        Crk = rk;
    }

    hcore::Options const opts = {
        { hcore::Option::UseGEMM,         use_gemm               },
        { hcore::Option::FixedRank,       truncate_with_fixed_rk },
        { hcore::Option::TruncateWithTol, truncate_with_tol      }
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
            print_matrix(Cm, Cn, C.data(), ldc, "C");
        }
        else if (params.routine == "gemm_dcc" || params.routine == "gemm_cdc" ||
                 params.routine == "gemm_ccc") {
            print_matrix(Cm, CUV.rk(), CUV.Udata(), CUV.Ustride(), "CU");
            print_matrix(CUV.rk(), Cn, CUV.Vdata(), CUV.Vstride(), "CV");
        }
    }

    if (params.routine == "gemm_ddc" || params.routine == "gemm_dcc" ||
        params.routine == "gemm_cdc" || params.routine == "gemm_ccc") {
        params.rk() = std::to_string(Crk) + "->" + std::to_string(CUV.rk());
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