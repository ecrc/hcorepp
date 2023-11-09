/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <catch2/catch_all.hpp>
#include <iostream>
#include <hcorepp/api/HCore.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <cstdlib>
#include <cstring>
#include "blas/flops.hh"
#include <hcorepp/helpers/MatrixHelpers.hpp>
#include <hcorepp/helpers/LapackWrappers.hpp>
#include <hcorepp/test-helpers/testHelpers.hpp>
#include <hcorepp/data-units/memory-handlers/MemoryHandler.hpp>

using namespace std::chrono;
using namespace hcorepp::operators;
using namespace hcorepp::helpers::matrixhelpers;
using namespace hcorepp::test_helpers;

template<typename T>
void TEST_GEMM_ADVANCED(TILE_COMBINATION Combination, int64_t n_elements) {
    hcorepp::kernels::RunContext& context = hcorepp::kernels::ContextManager::GetInstance().GetContext();
    using real_t = blas::real_type<T>;

    blas::Op transA = blas::Op::NoTrans;
    blas::Op transB = blas::Op::NoTrans;
    blas::Op transC = blas::Op::NoTrans;
    T alpha = 3.5;
    T beta = 2.5;

    int64_t m = n_elements;
    int64_t n = n_elements;
    int64_t k = n_elements;
    int64_t mode = 0;
    int64_t align = 1;

    real_t tol = 3;
    real_t cond = 1;
    real_t accuracy = 0.0001;

    int64_t Am = transA == blas::Op::NoTrans ? m : k;
    int64_t An = transA == blas::Op::NoTrans ? k : m;
    int64_t Bm = transB == blas::Op::NoTrans ? k : n;
    int64_t Bn = transB == blas::Op::NoTrans ? n : k;
    int64_t Cm = transC == blas::Op::NoTrans ? m : n;
    int64_t Cn = transC == blas::Op::NoTrans ? n : m;

    int64_t lda = ((Am + align - 1) / align) * align;
    int64_t ldb = ((Bm + align - 1) / align) * align;
    int64_t ldc = ((Cm + align - 1) / align) * align;

    T *Adata = new T[lda * An];
    T *Bdata = new T[ldb * Bn];
    T *Cdata = new T[ldc * Cn];
    int64_t iseed[4] = {0, 0, 0, 1};

    generate_dense_matrix(Am, An, Adata, lda, iseed, mode, cond);

    generate_dense_matrix(Bm, Bn, Bdata, ldb, iseed, mode, cond);

    generate_dense_matrix(Cm, Cn, Cdata, ldc, iseed, mode, cond);

    hcorepp::common::Norm norm = hcorepp::common::Norm::INF; // todo: variable norm type
    real_t Anorm = lapack_lange(norm, Am, An, Adata, lda);
    real_t Bnorm = lapack_lange(norm, Bm, Bn, Bdata, ldb);
    real_t Cnorm = lapack_lange(norm, Cm, Cn, Cdata, ldc);

    DenseTile<T> A(Am, An, Adata, lda, blas::Layout::ColMajor, context);
    DenseTile<T> B(Bm, Bn, Bdata, ldb, blas::Layout::ColMajor, context);
    DenseTile<T> C(Cm, Cn, Cdata, ldc, blas::Layout::ColMajor, context);
    context.Sync();

    int64_t ldcref = ((m + align - 1) / align) * align;

    T *Cref = nullptr;
    Cref = copy_output(C.GetDataHolder().get(), context);

    int64_t Ark, Brk, Crk;
    Ark = 0;
    Brk = 0;
    Crk = 0;
    T *AUVdata;
    T *BUVdata;
    T *CUVdata;

    CompressedTile<T> *AUV;
    CompressedTile<T> *BUV;
    CompressedTile<T> *CUV;
    if (Combination == CDD || Combination == CDC || Combination == CCD || Combination == CCC) {
        compress_dense_matrix(Am, An, Adata, lda, &AUVdata, Ark, accuracy);
        AUV = new CompressedTile<T>(Am, An, AUVdata, lda, Ark, blas::Layout::ColMajor, context);
        context.Sync();
        free(AUVdata);
    }
    if (Combination == DCD || Combination == DCC || Combination == CCD || Combination == CCC) {
        compress_dense_matrix(Bm, Bn, Bdata, ldb, &BUVdata, Brk, accuracy);

        BUV = new CompressedTile<T>(Bm, Bn, BUVdata, ldb, Brk, blas::Layout::ColMajor, context);
        context.Sync();
        free(BUVdata);
    }
    if (Combination == DDC || Combination == DCC || Combination == CDC || Combination == CCC) {
        compress_dense_matrix(Cm, Cn, Cdata, ldc, &CUVdata, Crk, accuracy);

        CUV = new CompressedTile<T>(Cm, Cn, CUVdata, ldc, Crk, blas::Layout::ColMajor, context);
        context.Sync();
        free(CUVdata);
    }

    double gflops = 0.0;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();
    size_t flops_ = 0;
    hcorepp::operators::CompressionParameters helpers(accuracy);
    hcorepp::dataunits::MemoryHandler<T>& memoryHandler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();

    switch (Combination) {
        case DDD:
            hcorepp::api::HCore<T>::Gemm(alpha, A, transA, B, transB, beta, C, context, flops_, memoryHandler.GetMemoryUnit(), helpers);
            gflops = blas::Gflop<T>::gemm(m, n, k);
            break;
        case DDC:
            hcorepp::api::HCore<T>::Gemm(alpha, A, transA, B, transB, beta, *CUV, context, flops_, memoryHandler.GetMemoryUnit(), helpers);
            gflops = blas::Gflop<T>::gemm(Cm, Cn, An) + blas::Gflop<T>::gemm(Cm, Cn, Crk);
            break;
        case DCD:
            hcorepp::api::HCore<T>::Gemm(alpha, A, transA, *BUV, transB, beta, C, context, flops_, memoryHandler.GetMemoryUnit(), helpers);
            gflops = blas::Gflop<T>::gemm(Cm, Brk, An) + blas::Gflop<T>::gemm(Cm, Cn, Brk);
            break;
        case DCC:
            hcorepp::api::HCore<T>::Gemm(alpha, A, transA, *BUV, transB, beta, *CUV, context, flops_, memoryHandler.GetMemoryUnit(), helpers);
            // todo
            // gflops = blas::Gflop<T>::gemm(Cm, Brk, An) +
            //          internal::gemm;
            break;
        case CDD:
            hcorepp::api::HCore<T>::Gemm(alpha, *AUV, transA, B, transB, beta, C, context, flops_, memoryHandler.GetMemoryUnit(), helpers);
            gflops = blas::Gflop<T>::gemm(Ark, Cn, An) + blas::Gflop<T>::gemm(Cm, Cn, Ark);
            break;
        case CDC:
            hcorepp::api::HCore<T>::Gemm(alpha, *AUV, transA, B, transB, beta, *CUV, context, flops_, memoryHandler.GetMemoryUnit(), helpers);
            // todo
            // gflops = blas::Gflop<T>::gemm(Ark, Cn, An) +
            //          internal::gemm;
            break;
        case CCD:
            hcorepp::api::HCore<T>::Gemm(alpha, *AUV, transA, *BUV, transB, beta, C, context, flops_, memoryHandler.GetMemoryUnit(), helpers);
            gflops = blas::Gflop<T>::gemm(Ark, Brk, An) +
                     (Ark <= Brk ? blas::Gflop<T>::gemm(Ark, Cn, Brk) +
                                   blas::Gflop<T>::gemm(Cm, Cn, Ark)
                                 : blas::Gflop<T>::gemm(Cm, Brk, Ark) +
                                   blas::Gflop<T>::gemm(Cm, Cn, Brk));
            break;
        case CCC:
            hcorepp::api::HCore<T>::Gemm(alpha, *AUV, transA, *BUV, transB, beta, *CUV, context, flops_, memoryHandler.GetMemoryUnit(), helpers);
            // todo: for now use PASC paper, which assumes square matrices
            int64_t max_Ark_Brk_Crk = std::max({Ark, Brk, Crk});
            int64_t max_m_n_k = std::max({m, n, k});
            gflops = (1e-9 * ((blas::is_complex<T>::value ? 3 : 1)
                              * 36 * max_m_n_k * (max_Ark_Brk_Crk
                                                  * max_Ark_Brk_Crk) + 157 * (max_Ark_Brk_Crk
                                                                              * max_Ark_Brk_Crk * max_Ark_Brk_Crk)));
            // todo
            // gflops = blas::Gflop<T>::gemm(Ark, Brk, An) +
            //          (Ark <= Brk ? blas::Gflop<T>::gemm(Ark, Cn, Brk) +
            //                        internal::gemm
            //                      : blas::Gflop<T>::gemm(Cm, Brk, Ark) +
            //                        internal::gemm;
            break;
    }
    context.Sync();
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    auto elapsed_time = elapsed_seconds.count();
    auto flops = gflops / elapsed_time;

    double ref_elapsed_time;
    double ref_flops;
    double error = 0;
    bool pass = false;
    T *C_output = nullptr;
    {
        std::chrono::time_point<std::chrono::system_clock> ref_time_start = std::chrono::system_clock::now();
        {

            blas::gemm(blas::Layout::ColMajor, transA, transB, m, n, k, alpha, Adata, lda, Bdata, ldb, beta, Cref,
                       ldcref);

        }
        std::chrono::time_point<std::chrono::system_clock> ref_time_end = std::chrono::system_clock::now();

        std::chrono::duration<double> ref_elapsed_seconds = ref_time_end - ref_time_start;
        ref_elapsed_time = ref_elapsed_seconds.count();
        double ref_gflops = blas::Gflop<T>::gemm(m, n, k);

        ref_flops = ref_gflops / ref_elapsed_time;

        if (Combination == DCC || Combination == CDC || Combination == CCC) {
            auto cu_m_new = CUV->GetNumOfRows();
            auto cu_n_new = CUV->GetTileRank();
            auto cv_m_new = CUV->GetTileRank();
            auto cv_n_new = CUV->GetNumOfCols();

            auto cu = copy_output(CUV->GetUMatrix(), cu_m_new * cu_n_new, context);
            auto cv = copy_output(CUV->GetVMatrix(), cv_m_new * cv_n_new, context);
            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                       Cm, Cn, CUV->GetTileRank(), 1.0, cu,
                       CUV->GetULeadingDim(), cv,
                       CUV->GetVLeadingDim(), 0.0, Cdata, ldc);
            delete[] cu;
            delete[] cv;
            C_output = new T[Cm * Cn];

            memcpy((void *) C_output, (void *) Cdata, Cm * Cn * sizeof(T));
            diff(Cref, ldcref, C_output, Cm, Cn, ldc);

        } else if (Combination == DDC) {
            auto cu_m_new = CUV->GetNumOfRows();
            auto cu_n_new = CUV->GetTileRank();
            auto cv_m_new = CUV->GetTileRank();
            auto cv_n_new = CUV->GetNumOfCols();

            C_output = new T[cu_m_new * cu_n_new + cv_m_new * cv_n_new];
            auto cu_raw = copy_output(CUV->GetUMatrix(), cu_m_new * cu_n_new, context);
            memcpy((void *) C_output, (void *) cu_raw,
                   cu_m_new * cu_n_new * sizeof(T));
            delete[] cu_raw;
            auto cv_raw = copy_output(CUV->GetVMatrix(), cv_m_new * cv_n_new, context);
            memcpy((void *) &C_output[cu_m_new * cu_n_new], (void *) cv_raw,
                   cv_m_new * cv_n_new * sizeof(T));
            delete[] cv_raw;
            diff(Cref, ldcref, C_output, Cm, Cn, ldc);

        } else {
            C_output = copy_output(C.GetDataHolder().get(), context);
            diff(Cref, ldcref, C_output, Cm, Cn, ldc);

        }

        auto temp = (sqrt(real_t(k) + 2) * std::abs(alpha) *
                     Anorm * Bnorm + 2 * std::abs(beta) * Cnorm);

        auto lange = lapack_lange(norm, m, n, Cref, ldcref);

        error = lange / temp;

        if (blas::is_complex<T>::value) {
            error /= 2 * sqrt(2);
        }
        pass = (error < tol * accuracy);
        REQUIRE(pass);
    }

    delete[] Cref;
    delete[] Adata;
    delete[] Bdata;
    delete[] Cdata;
    delete[] C_output;
    if (Combination == CDD || Combination == CDC || Combination == CCD || Combination == CCC) {
        delete AUV;
    }
    if (Combination == DCD || Combination == DCC || Combination == CCD || Combination == CCC) {
        delete BUV;
    }
    if (Combination == DDC || Combination == DCC || Combination == CDC || Combination == CCC) {
        delete CUV;
    }
    printf("|%-5s|%-10s|%-10s|%-10s|%-10s|%-5ld|%-5ld|%-5ld|%-8.3f|%-8.3f|%-15f|%-15f|%-15f|%-15f|%-15e|%-5ld|%-5ld|%-5ld|%-10s|\n",
           tile_combination_strings[Combination], typeid(T).name(),
           op2str(transA), op2str(transB), op2str(transC), m, n, k, alpha,
           beta, elapsed_time, gflops, ref_elapsed_time, ref_flops, error, Ark, Brk, Crk, ((pass) ? "Pass" : "Fail"));
    memoryHandler.FreeAllocations();


}

TEMPLATE_TEST_CASE("AdvancedGemmTest", "[ADVANCEDGEMMTESTING]", float, double) {
    std::vector<blas::Op> blas_ops = {blas::Op::NoTrans};
    printf("%s\n", std::string(196, '=').c_str());
    printf("|%-5s|%-10s|%-10s|%-10s|%-10s|%-5s|%-5s|%-5s|%-8s|%-8s|%-15s|%-15s|%-15s|%-15s|%-15s|%-5s|%-5s|%-5s|%-10s|\n",
           "Gemm", "Datatype", "opA", "opB", "opC", "m", "n", "k", "alpha",
           "beta", "time(s)", "gflops", "ref_time(s)", "ref_gflops", "error", "Ark", "Brk", "Crk", "status");
    printf("%s\n", std::string(196, '=').c_str());
    std::vector<TILE_COMBINATION> combinations = {DDD, DDC, DCD, DCC, CDD, CDC, CCD, CCC};

    std::vector<int64_t> n_elements = {100, 200, 300, 400, 500};
    for (auto C: combinations) {
        for (auto N: n_elements) {
            TEST_GEMM_ADVANCED<TestType>(C, N);
        }
    }
    printf("%s\n", std::string(196, '=').c_str());
    hcorepp::kernels::ContextManager::DestroyInstance();
    hcorepp::dataunits::MemoryHandler<TestType>::DestroyInstance();

}
