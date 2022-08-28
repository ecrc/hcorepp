#include <libraries/catch/catch.hpp>
#include <iostream>
#include <hcorepp/test-helpers/lapack_wrappers.hpp>
#include "lapack.hh"
#include <hcorepp/api/hcorepp.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "blas/flops.hh"

enum TILE_COMBINATION {
    DDD,
    DDC,
    DCD,
    DCC,
    CDD,
    CDC,
    CCD,
    CCC
};
static const char *tile_combination_strings[] =
        {"DenseDenseDense", "DenseDenseCompressed", "DenseCompressedDense", "DenseCompressedCompressed",
         "CompressedDenseDense", "CompressedDenseCompressed", "CompressedCompressedDense",
         "CompressedCompressedCompressed"};

using namespace std::chrono;
using namespace hcorepp::operators;

template<typename T>
void generate_dense_matrix(int64_t m, int64_t n, T *A, int64_t lda, int64_t *iseed, int64_t mode = 0,
                           blas::real_type<T> cond = 1) {
    int16_t min_m_n = std::min(m, n);

//    std::vector<blas::real_type<T>> D(min_m_n);
    blas::real_type<T> *D = (blas::real_type<T> *) malloc(min_m_n * sizeof(blas::real_type<T>));

    for (int64_t i = 0; i < min_m_n; ++i)
        D[i] = std::pow(10, -1 * i);

    lapack_latms(
            m, n, 'U', iseed, 'N', D, mode, cond, -1.0, m - 1, n - 1, 'N', A, lda);

    free(D);
}

template<typename T>
void compress_dense_matrix(int64_t m, int64_t n, T *A, int64_t lda, T **UV, int64_t &rk,
                           blas::real_type<T> accuracy) {
    int16_t min_m_n = std::min(m, n);

    blas::real_type<T> *Sigma = (blas::real_type<T> *) malloc(min_m_n * sizeof(blas::real_type<T>));
    T *U = (T *) malloc(lda * min_m_n * sizeof(T));
    T *VT = (T *) malloc(min_m_n * n * sizeof(T));

    lapack::gesvd(lapack::Job::SomeVec, lapack::Job::SomeVec, m, n, A, lda, Sigma, U, lda, VT, min_m_n);

    rk = 0;
    while (Sigma[rk] >= accuracy && rk < min_m_n) {
        rk++;
    }

    // todo: more conservative max rank assumption, e.g., min_m_n / 3.
    int64_t max_rk = min_m_n / 2;
    if (rk > max_rk) {
        rk = max_rk;
    }

    // VT eats Sigma.
    // todo: we may need to have uplo parameter:
    //       scale VT, if Lower, or scale U otherwise.
    for (int64_t i = 0; i < rk; ++i) {
        blas::scal(n, Sigma[i], &VT[i], min_m_n);
    }

    *UV = (T *) malloc((lda + n) * rk * sizeof(T));

    memcpy((void *) (*UV), (void *) U, (lda + rk) * sizeof(T));

    // copy first rk rows of VT; UV = VT(1:rk,:)
    // todo: assume column-major, what about row-major?
    lapack::lacpy(
            lapack::MatrixType::General, rk, n, VT, min_m_n, &(*UV)[lda * rk], rk);

    free(U);
    free(VT);
    free(Sigma);

}

template<typename T>
void diff(T *Aref, int64_t lda_ref, T const *A, int64_t m, int64_t n) {
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            Aref[i + j * lda_ref] -= A[i + j * lda_ref];
        }
    }
}

template<typename T>
void TEST_GEMM_ADVANCED(TILE_COMBINATION Combination, int64_t n_elements) {
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
    int64_t verbose = 0;
    int64_t truncate_with_fixed_rk = 0;

    real_t tol = 3;
    real_t cond = 1;
    real_t accuracy = 2.22045e-16;

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


    std::cout << " INITIAL A DATA \n";
    for (int i = 0; i < An; i++) {
        for (int j = 0; j < Am; j++) {
            int index = i * An + j;
            printf(" A[%d][%d] = %lf \n", i, j, Adata[index]);
        }
    }
    std::cout << " INITIAL B DATA \n";
    for (int i = 0; i < Bn; i++) {
        for (int j = 0; j < Bm; j++) {
            int index = i * Bn + j;
            printf(" B[%d][%d] = %lf \n", i, j, Bdata[index]);
        }
    }
    std::cout << " INITIAL C DATA \n";
    for (int i = 0; i < Cn; i++) {
        for (int j = 0; j < Cm; j++) {
            int index = i * Cn + j;
            printf(" C[%d][%d] = %lf \n", i, j, Cdata[index]);
        }
    }

    lapack::Norm norm = lapack::Norm::Inf; // todo: variable norm type
    real_t Anorm = lapack::lange(norm, Am, An, Adata, lda);
    real_t Bnorm = lapack::lange(norm, Bm, Bn, Bdata, ldb);
    real_t Cnorm = lapack::lange(norm, Cm, Cn, Cdata, ldc);

    DenseTile<T> A(Am, An, Adata, lda, blas::Layout::ColMajor, transA, blas::Uplo::General);
    DenseTile<T> B(Bm, Bn, Bdata, ldb, blas::Layout::ColMajor, transB, blas::Uplo::General);
    DenseTile<T> C(Cm, Cn, Cdata, ldc, blas::Layout::ColMajor, transC, blas::Uplo::General);

    int64_t ldcref = ((m + align - 1) / align) * align;

    T *Cref = new T[ldcref * n];

    memcpy((void *) Cref, (void *) C.GetTileSubMatrix(0).get().GetData(), Cm * Cn * sizeof(T));

    int64_t Ark, Brk, Crk;
    T *AUVdata;
    T *BUVdata;
    T *CUVdata;

    CompressedTile<T> *AUV;
    CompressedTile<T> *BUV;
    CompressedTile<T> *CUV;
    if (Combination == CDD || Combination == CDC || Combination == CCD || Combination == CCC) {
        compress_dense_matrix(Am, An, Adata, lda, &AUVdata, Ark, accuracy);
        AUV = new CompressedTile<T>(Am, An, AUVdata, lda, Ark, accuracy, blas::Layout::ColMajor, transA,
                                    blas::Uplo::General);
        free(AUVdata);
    }
    if (Combination == DCD || Combination == DCC || Combination == CCD || Combination == CCC) {
        compress_dense_matrix(Bm, Bn, Bdata, ldb, &BUVdata, Brk, accuracy);
        BUV = new CompressedTile<T>(Bm, Bn, BUVdata, ldb, Brk, accuracy, blas::Layout::ColMajor, transB,
                                    blas::Uplo::General);
        free(BUVdata);
    }
    if (Combination == DDC || Combination == DCC || Combination == CDC || Combination == CCC) {
        compress_dense_matrix(Cm, Cn, Cdata, ldc, &CUVdata, Crk, accuracy);
        CUV = new CompressedTile<T>(Cm, Cn, CUVdata, ldc, Crk, accuracy, blas::Layout::ColMajor, transC,
                                    blas::Uplo::General);
        free(CUVdata);
    }

    double gflops = 0.0;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();

    hcorepp::helpers::SvdHelpers helpers;
    switch (Combination) {
        case DDD:
            hcorepp::api::gemm(alpha, A, transA, B, transB, beta, C, transC, helpers);
            gflops = blas::Gflop<T>::gemm(m, n, k);
            break;
        case DDC:
            hcorepp::api::gemm(alpha, A, transA, B, transB, beta, *CUV, transC, helpers);
            gflops = blas::Gflop<T>::gemm(Cm, Cn, An) + blas::Gflop<T>::gemm(Cm, Cn, Crk);
            break;
        case DCD:
            hcorepp::api::gemm(alpha, A, transA, *BUV, transB, beta, C, transC, helpers);
            gflops = blas::Gflop<T>::gemm(Cm, Brk, An) + blas::Gflop<T>::gemm(Cm, Cn, Brk);
            break;
        case DCC:
            hcorepp::api::gemm(alpha, A, transA, *BUV, transB, beta, *CUV, transC, helpers);
            // todo
            // gflops = blas::Gflop<T>::gemm(Cm, Brk, An) +
            //          internal::gemm;
            break;
        case CDD:
            hcorepp::api::gemm(alpha, *AUV, transA, B, transB, beta, C, transC, helpers);
            gflops = blas::Gflop<T>::gemm(Ark, Cn, An) + blas::Gflop<T>::gemm(Cm, Cn, Ark);
            break;
        case CDC:
            hcorepp::api::gemm(alpha, *AUV, transA, B, transB, beta, *CUV, transC, helpers);
            // todo
            // gflops = blas::Gflop<T>::gemm(Ark, Cn, An) +
            //          internal::gemm;
            break;
        case CCD:
            hcorepp::api::gemm(alpha, *AUV, transA, *BUV, transB, beta, C, transC, helpers);
            gflops = blas::Gflop<T>::gemm(Ark, Brk, An) +
                     (Ark <= Brk ? blas::Gflop<T>::gemm(Ark, Cn, Brk) +
                                   blas::Gflop<T>::gemm(Cm, Cn, Ark)
                                 : blas::Gflop<T>::gemm(Cm, Brk, Ark) +
                                   blas::Gflop<T>::gemm(Cm, Cn, Brk));
            break;
        case CCC:
            hcorepp::api::gemm(alpha, *AUV, transA, *BUV, transB, beta, *CUV, transC, helpers);
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

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    auto elapsed_time = elapsed_seconds.count();
    auto flops = gflops / elapsed_time;

    if (Combination == DDC) {
//        C = hcore::DenseTile<T>(CUV);
    }
    if (Combination == DDC || Combination == DCC || Combination == CDC || Combination == CCC) {
//        params.rk() = std::to_string(Crk) + "->" + std::to_string(CUV.rk());
    }

    double ref_elapsed_time;
    double ref_flops;
    double error = 0;
    bool pass = false;
    T *C_output;

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
            // C = CU * CV.'
            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                       Cm, Cn, CUV->GetTileRank(), 1.0, CUV->GetTileSubMatrix(0).get().GetData(),
                       CUV->GetTileSubMatrix(0).get().GetLeadingDim(), CUV->GetTileSubMatrix(1).get().GetData(),
                       CUV->GetTileSubMatrix(1).get().GetLeadingDim(), 0.0, Cdata, ldc);
        }

        if (Combination == DDC || Combination == DCC || Combination == CDC || Combination == CCC) {
            C_output = (T *) malloc(
                    CUV->GetTileSubMatrix(0).get().GetNumOfRows() * CUV->GetTileSubMatrix(1).get().GetNumOfCols() *
                    sizeof(T));
            std::cout << "SIZEOF C_OUTPUT == "
                      << CUV->GetTileSubMatrix(0).get().GetNumOfRows() * CUV->GetTileSubMatrix(1).get().GetNumOfCols()
                      << "\n";

            std::cout << "SIZEOF CREF == " << ldcref * n << "\n";

            memcpy((void *) C_output, (void *) CUV->GetTileSubMatrix(0).get().GetData(),
                   CUV->GetTileSubMatrix(0).get().GetNumOfRows() * CUV->GetTileSubMatrix(0).get().GetNumOfCols() *
                   sizeof(T));
            memcpy((void *) &C_output[CUV->GetTileSubMatrix(0).get().GetNumOfRows() *
                                      CUV->GetTileSubMatrix(0).get().GetNumOfCols()],
                   (void *) CUV->GetTileSubMatrix(1).get().GetData(),
                   CUV->GetTileSubMatrix(1).get().GetNumOfRows() * CUV->GetTileSubMatrix(1).get().GetNumOfCols() *
                   sizeof(T));
            diff(Cref, ldcref, C_output, CUV->GetTileSubMatrix(0).get().GetNumOfRows(),
                 CUV->GetTileSubMatrix(1).get().GetNumOfCols());

        } else {
            C_output = (T *) malloc(
                    C.GetTileSubMatrix(0).get().GetNumOfRows() * C.GetTileSubMatrix(0).get().GetNumOfCols() *
                    sizeof(T));
            memcpy((void *) C_output, (void *) C.GetTileSubMatrix(0).get().GetData(),
                   C.GetTileSubMatrix(0).get().GetNumOfRows() * C.GetTileSubMatrix(0).get().GetNumOfCols() *
                   sizeof(T));
            diff(Cref, ldcref, C_output, C.GetTileSubMatrix(0).get().GetNumOfRows(),
                 C.GetTileSubMatrix(0).get().GetNumOfCols());

        }

        error = lapack::lange(norm, m, n, Cref, ldcref)
                / (sqrt(real_t(k) + 2) * std::abs(alpha) *
                   Anorm * Bnorm + 2 * std::abs(beta) * Cnorm);

        if (blas::is_complex<T>::value) {
            error /= 2 * sqrt(2);
        }
        pass = (error < tol * accuracy);
//        std::cout << " OKAYYYY ????? " << okay << "\n";

    }

    std::cout << " OUTPUT CDATA \n";
    for (int i = 0; i < Cn; i++) {
        for (int j = 0; j < Cm; j++) {
            int index = i * Cn + j;
            printf(" C[%d][%d] = %lf \n", i, j, C_output[index]);
        }
    }

    delete[]Cref;
    std::cout << "HELLO WORLD \n";
    delete[]Adata;
    delete[]Bdata;
    delete[]Cdata;
    if (Combination == CDD || Combination == CDC || Combination == CCD || Combination == CCC) {
        delete AUV;
    }
    if (Combination == DCD || Combination == DCC || Combination == CCD || Combination == CCC) {
        delete BUV;
    }
    if (Combination == DDC || Combination == DCC || Combination == CDC || Combination == CCC) {
        delete CUV;
    }
    free(C_output);


    std::cout
            << "GemmType \t DataType \t transA \t transB \t transC \t m \t n \t k \t alpha \t beta \t time(s) \t gflops \t ref_time(s) \t ref_gflops \t error \t status \n";

    std::cout << tile_combination_strings[Combination] << " \t " << typeid(T).name() << " \t " << op2str(transA)
              << " \t " << op2str(transB) << " \t " << op2str(transC) << "\t" << n_elements << " \t " << n_elements
              << " \t " << n_elements << " \t " << alpha << " \t " << beta << " \t " << elapsed_time << " \t " << gflops
              << " \t " << ref_elapsed_time << " \t " << ref_flops << " \t " << error << " \t "
              << ((pass) ? "Pass" : "Fail") << "\n";


}


TEMPLATE_TEST_CASE("AdvancedGemmTest", "[ADVANCEDGEMMTESTING]", double) {
    std::vector<TILE_COMBINATION> combinations = {CDC};//, DDC, DCD, DCC, CDD, CDC, CCD, CCC};

    //    std::vector<blas::Op> blas_ops = {blas::Op::NoTrans};

    std::vector<int64_t> n_elements = {3};//, 200, 300, 400, 500};

    for (auto C: combinations) {
        for (auto N: n_elements) {
            TEST_GEMM_ADVANCED<TestType>(C, N);
        }
    }
}
