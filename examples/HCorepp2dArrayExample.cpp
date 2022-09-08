#include <cstring>
#include <chrono>
#include "blas/flops.hh"
#include "lapack.hh"
#include <hcorepp/api/hcorepp.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/helpers/MatrixHelpers.hpp>
#include <iostream>

using namespace std::chrono;
using namespace hcorepp::operators;
using namespace hcorepp::helpers::matrixhelpers;

template<typename T>
void CalculateExact(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, int64_t aLdA, int64_t aLdB, int64_t aLdC,
                    T *aAData, T *aBData, T *aCData, int64_t aM, int64_t aN, int64_t aK, T aAlpha, T aBeta) {

    blas::gemm(aLayout, aTransA, aTransB, aM, aN, aK, aAlpha, aAData, aLdA, aBData, aLdB, aBeta, aCData, aLdC);
}

template<typename T>
void CalculateApprox(Tile<T> &A, blas::Op aTransA, Tile<T> &B, blas::Op aTransB, Tile<T> &C,
                     blas::Op aTransC, int64_t aLdC, T *aCData, T aAlpha, T aBeta, T **aCOutput,
                     hcorepp::helpers::SvdHelpers aHelpers, TILE_COMBINATION aCombination) {
    hcorepp::api::gemm(aAlpha, A, aTransA, B, aTransB, aBeta, C, aTransC, aHelpers);

    if (aCombination == DCC || aCombination == CDC || aCombination == CCC) {
        auto Cm = C.GetTileSubMatrix(0).get().GetNumOfRows();
        auto Cn = C.GetTileSubMatrix(1).get().GetNumOfCols();
        auto rank = C.GetTileSubMatrix(0).get().GetNumOfCols();

        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   Cm, Cn, rank, 1.0, C.GetTileSubMatrix(0).get().GetData(),
                   C.GetTileSubMatrix(0).get().GetLeadingDim(), C.GetTileSubMatrix(1).get().GetData(),
                   C.GetTileSubMatrix(1).get().GetLeadingDim(), 0.0, aCData, aLdC);
        *aCOutput = (T *) malloc(Cm * Cn * sizeof(T));
        memcpy((void *) *aCOutput, (void *) aCData, Cm * Cn * sizeof(T));


    } else if (aCombination == DDC) {
        auto cu_m_new = C.GetTileSubMatrix(0).get().GetNumOfRows();
        auto cu_n_new = C.GetTileSubMatrix(0).get().GetNumOfCols();
        auto cv_m_new = C.GetTileSubMatrix(1).get().GetNumOfRows();
        auto cv_n_new = C.GetTileSubMatrix(1).get().GetNumOfCols();

        *aCOutput = (T *) malloc((cu_m_new * cu_n_new + cv_m_new * cv_n_new) * sizeof(T));

        memcpy((void *) *aCOutput, (void *) C.GetTileSubMatrix(0).get().GetData(), cu_m_new * cu_n_new * sizeof(T));

        memcpy((void *) &(*aCOutput[cu_m_new * cu_n_new]), (void *) C.GetTileSubMatrix(1).get().GetData(),
               cv_m_new * cv_n_new * sizeof(T));

    } else {
        auto Cm = C.GetTileSubMatrix(0).get().GetNumOfRows();
        auto Cn = C.GetTileSubMatrix(0).get().GetNumOfCols();

        *aCOutput = (T *) malloc(Cm * Cn * sizeof(T));

        memcpy((void *) *aCOutput, (void *) C.GetTileSubMatrix(0).get().GetData(), Cm * Cn * sizeof(T));
    }
}

template<typename T>
void run(blas::Op aTransA, blas::Op aTransB, blas::Op aTransC, T aAlpha, T aBeta, int64_t aM, int64_t aN, int64_t aK,
         T aTol, T aAcc, int64_t aMode, int64_t aAlignment, TILE_COMBINATION aCombination) {
    blas::Op transA = aTransA;
    blas::Op transB = aTransB;
    blas::Op transC = aTransC;
    T alpha = aAlpha;
    T beta = aBeta;

    int64_t m = aM;
    int64_t n = aN;
    int64_t k = aK;

    blas::real_type<T> tol = 3;
    blas::real_type<T> cond = 1;
    blas::real_type<T> accuracy = 0.0001;
    int64_t mode = aMode;
    int64_t align = aAlignment;

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

    lapack::Norm norm = lapack::Norm::Inf; // todo: variable norm type
    blas::real_type<T> Anorm = lapack::lange(norm, Am, An, Adata, lda);
    blas::real_type<T> Bnorm = lapack::lange(norm, Bm, Bn, Bdata, ldb);
    blas::real_type<T> Cnorm = lapack::lange(norm, Cm, Cn, Cdata, ldc);

    DenseTile<T> A(Am, An, Adata, lda, blas::Layout::ColMajor, transA, blas::Uplo::General);
    DenseTile<T> B(Bm, Bn, Bdata, ldb, blas::Layout::ColMajor, transB, blas::Uplo::General);
    DenseTile<T> C(Cm, Cn, Cdata, ldc, blas::Layout::ColMajor, transC, blas::Uplo::General);

    int64_t ldcref = ((m + align - 1) / align) * align;

    T *Cref = new T[ldcref * n];

    memcpy((void *) Cref, (void *) C.GetTileSubMatrix(0).get().GetData(), Cm * Cn * sizeof(T));

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
    if (aCombination == CDD || aCombination == CDC || aCombination == CCD || aCombination == CCC) {
        compress_dense_matrix(Am, An, Adata, lda, &AUVdata, Ark, accuracy);
        AUV = new CompressedTile<T>(Am, An, AUVdata, lda, Ark, accuracy, blas::Layout::ColMajor, transA,
                                    blas::Uplo::General);
        free(AUVdata);
    }
    if (aCombination == DCD || aCombination == DCC || aCombination == CCD || aCombination == CCC) {
        compress_dense_matrix(Bm, Bn, Bdata, ldb, &BUVdata, Brk, accuracy);
        BUV = new CompressedTile<T>(Bm, Bn, BUVdata, ldb, Brk, accuracy, blas::Layout::ColMajor, transB,
                                    blas::Uplo::General);
        free(BUVdata);
    }
    if (aCombination == DDC || aCombination == DCC || aCombination == CDC || aCombination == CCC) {
        compress_dense_matrix(Cm, Cn, Cdata, ldc, &CUVdata, Crk, accuracy);
        CUV = new CompressedTile<T>(Cm, Cn, CUVdata, ldc, Crk, accuracy, blas::Layout::ColMajor, transC,
                                    blas::Uplo::General);
        free(CUVdata);
    }

    double gflops = 0.0;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();

    T *C_output;
    hcorepp::helpers::SvdHelpers helpers;
    switch (aCombination) {
        case DDD:
            CalculateApprox(A, transA, B, transB, C, transC, ldc, Cdata, alpha, beta,
                            &C_output, helpers, aCombination);
            gflops = blas::Gflop<T>::gemm(m, n, k);
            break;
        case DDC:
            CalculateApprox(A, transA, B, transB, *CUV, transC, ldc, Cdata, alpha, beta,
                            &C_output, helpers, aCombination);
            gflops = blas::Gflop<T>::gemm(Cm, Cn, An) + blas::Gflop<T>::gemm(Cm, Cn, Crk);
            break;
        case DCD:
            CalculateApprox(A, transA, *BUV, transB, C, transC, ldc, Cdata, alpha, beta,
                            &C_output, helpers, aCombination);
            gflops = blas::Gflop<T>::gemm(Cm, Brk, An) + blas::Gflop<T>::gemm(Cm, Cn, Brk);
            break;
        case DCC:
            CalculateApprox(A, transA, *BUV, transB, *CUV, transC, ldc, Cdata, alpha, beta,
                            &C_output, helpers, aCombination);
            break;
        case CDD:
            CalculateApprox(*AUV, transA, B, transB, C, transC, ldc, Cdata, alpha, beta,
                            &C_output, helpers, aCombination);
            gflops = blas::Gflop<T>::gemm(Ark, Cn, An) + blas::Gflop<T>::gemm(Cm, Cn, Ark);
            break;
        case CDC:
            CalculateApprox(*AUV, transA, B, transB, *CUV, transC, ldc, Cdata, alpha, beta,
                            &C_output, helpers, aCombination);
            break;
        case CCD:
            CalculateApprox(*AUV, transA, *BUV, transB, C, transC, ldc, Cdata, alpha, beta,
                            &C_output, helpers, aCombination);
            gflops = blas::Gflop<T>::gemm(Ark, Brk, An) +
                     (Ark <= Brk ? blas::Gflop<T>::gemm(Ark, Cn, Brk) +
                                   blas::Gflop<T>::gemm(Cm, Cn, Ark)
                                 : blas::Gflop<T>::gemm(Cm, Brk, Ark) +
                                   blas::Gflop<T>::gemm(Cm, Cn, Brk));
            break;
        case CCC:
            CalculateApprox(*AUV, transA, *BUV, transB, *CUV, transC, ldc, Cdata, alpha, beta,
                            &C_output, helpers, aCombination);
            int64_t max_Ark_Brk_Crk = std::max({Ark, Brk, Crk});
            int64_t max_m_n_k = std::max({m, n, k});
            gflops = (1e-9 * ((blas::is_complex<T>::value ? 3 : 1)
                              * 36 * max_m_n_k * (max_Ark_Brk_Crk
                                                  * max_Ark_Brk_Crk) + 157 * (max_Ark_Brk_Crk
                                                                              * max_Ark_Brk_Crk * max_Ark_Brk_Crk)));
            break;
    }

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    auto elapsed_time = elapsed_seconds.count();
    auto flops = gflops / elapsed_time;

    double ref_elapsed_time;
    double ref_flops;
    double error = 0;
    bool pass = false;

    {
        std::chrono::time_point<std::chrono::system_clock> ref_time_start = std::chrono::system_clock::now();
        {
            CalculateExact(blas::Layout::ColMajor, transA, transB, lda, ldb, ldcref, Adata, Bdata, Cref, m, n, k, alpha,
                           beta);
        }
        std::chrono::time_point<std::chrono::system_clock> ref_time_end = std::chrono::system_clock::now();

        std::chrono::duration<double> ref_elapsed_seconds = ref_time_end - ref_time_start;
        ref_elapsed_time = ref_elapsed_seconds.count();
        double ref_gflops = blas::Gflop<T>::gemm(m, n, k);

        ref_flops = ref_gflops / ref_elapsed_time;

        diff(Cref, ldcref, C_output, Cm, Cn, ldc);

        error = lapack::lange(norm, m, n, Cref, ldcref)
                / (sqrt(blas::real_type<T>(k) + 2) * std::abs(alpha) *
                   Anorm * Bnorm + 2 * std::abs(beta) * Cnorm);

        if (blas::is_complex<T>::value) {
            error /= 2 * sqrt(2);
        }
        pass = (error < tol * accuracy);

    }

    delete[]Cref;
    delete[]Adata;
    delete[]Bdata;
    delete[]Cdata;
    if (aCombination == CDD || aCombination == CDC || aCombination == CCD || aCombination == CCC) {
        delete AUV;
    }
    if (aCombination == DCD || aCombination == DCC || aCombination == CCD || aCombination == CCC) {
        delete BUV;
    }
    if (aCombination == DDC || aCombination == DCC || aCombination == CDC || aCombination == CCC) {
        delete CUV;
    }
    free(C_output);
    printf("%s\n",std::string(196,'=').c_str());
    printf("|%-5s|%-10s|%-10s|%-10s|%-10s|%-5s|%-5s|%-5s|%-8s|%-8s|%-15s|%-15s|%-15s|%-15s|%-15s|%-5s|%-5s|%-5s|%-10s|\n",
           "Gemm", "Datatype", "opA", "opB", "opC", "m", "n", "k", "alpha",
           "beta", "time(s)", "gflops", "ref_time(s)", "ref_gflops", "error", "Ark", "Brk", "Crk", "status");
    printf("%s\n",std::string(196,'=').c_str());
    printf("|%-5s|%-10s|%-10s|%-10s|%-10s|%-5ld|%-5ld|%-5ld|%-8.3f|%-8.3f|%-15f|%-15f|%-15f|%-15f|%-15e|%-5ld|%-5ld|%-5ld|%-10s|\n",
           tile_combination_strings[aCombination], typeid(T).name(),
           op2str(transA), op2str(transB), op2str(transC), m, n, k, alpha,
           beta, elapsed_time, gflops, ref_elapsed_time, ref_flops, error, Ark, Brk, Crk, ((pass) ? "Pass" : "Fail"));
    printf("%s\n",std::string(196,'=').c_str());
}

template<typename T>
void
run(blas::Op aTransA, blas::Op aTransB, blas::Op aTransC, T aAlpha, T aBeta, int64_t aM, int64_t aN, int64_t aK, T aTol,
    T aAcc, DenseTile<T> &dense_tileA, DenseTile<T> &dense_tileB, DenseTile<T> &dense_tileC,
    CompressedTile<T> &comp_tileA, CompressedTile<T> &comp_tileB, CompressedTile<T> &comp_tileC, T **Approx_C_output,
    T **Exact_C_Output) {

    int64_t lda = dense_tileA.GetTileSubMatrix(0).get().GetLeadingDim();
    int64_t ldb = dense_tileB.GetTileSubMatrix(0).get().GetLeadingDim();
    int64_t ldc = dense_tileC.GetTileSubMatrix(0).get().GetLeadingDim();

    T *Adata = dense_tileA.GetTileSubMatrix(0).get().GetData();
    T *Bdata = dense_tileB.GetTileSubMatrix(0).get().GetData();
    T *Cdata = dense_tileC.GetTileSubMatrix(0).get().GetData();

    int64_t m = aM;
    int64_t n = aN;
    int64_t k = aK;

    int64_t Am = dense_tileA.GetTileSubMatrix(0).get().GetNumOfRows();
    int64_t An = dense_tileA.GetTileSubMatrix(0).get().GetNumOfCols();
    int64_t Bm = dense_tileB.GetTileSubMatrix(0).get().GetNumOfRows();
    int64_t Bn = dense_tileB.GetTileSubMatrix(0).get().GetNumOfCols();
    int64_t Cm = dense_tileC.GetTileSubMatrix(0).get().GetNumOfRows();
    int64_t Cn = dense_tileC.GetTileSubMatrix(0).get().GetNumOfCols();

    lapack::Norm norm = lapack::Norm::Inf; // todo: variable norm type
    blas::real_type<T> Anorm = lapack::lange(norm, Am, An, Adata, lda);
    blas::real_type<T> Bnorm = lapack::lange(norm, Bm, Bn, Bdata, ldb);
    blas::real_type<T> Cnorm = lapack::lange(norm, Cm, Cn, Cdata, ldc);

    int64_t align = 1;

    int64_t ldcref = ((m + align - 1) / align) * align;

    *Exact_C_Output = (T *) malloc(ldcref * Cn * sizeof(T));
    memcpy((void *) *Exact_C_Output, Cdata, Cm * Cn * sizeof(T));

    CalculateExact(blas::Layout::ColMajor, aTransA, aTransB, lda, ldb, ldcref, Adata,
                   Bdata, *Exact_C_Output, m, n, k, aAlpha, aBeta);

    hcorepp::helpers::SvdHelpers helpers;
    CalculateApprox(comp_tileA, aTransA, comp_tileB, aTransB, comp_tileC, aTransC, ldc, Cdata, aAlpha, aBeta,
                    Approx_C_output, helpers, CCC);

    diff(*Exact_C_Output, ldcref, *Approx_C_output, Cm, Cn, ldc);


    double error = lapack::lange(norm, m, n, *Exact_C_Output, ldcref)
                   / (sqrt(blas::real_type<T>(k) + 2) * std::abs(aAlpha) *
                      Anorm * Bnorm + 2 * std::abs(aBeta) * Cnorm);

    if (blas::is_complex<T>::value) {
        error /= 2 * sqrt(2);
    }

    std::cout << " ERROR VALUE : " << error << "\n";
    bool pass = (error < aTol * aAcc);
    std::cout << " PASS : " << pass << "\n";

}

template<typename T>
void GenerateTiles(blas::Op aTransA, int64_t aM, int64_t aN, int64_t aK, int64_t aMode, T aAcc, int64_t aAlignment,
                   int64_t *iseed, DenseTile<T> **aDenseTile, CompressedTile<T> **aCompressedTile) {

    blas::real_type<T> cond = 1;
    int64_t align = aAlignment;

    int64_t Am = aTransA == blas::Op::NoTrans ? aM : aK;
    int64_t An = aTransA == blas::Op::NoTrans ? aK : aN;

    int64_t lda = ((Am + align - 1) / align) * align;

    T *Adata = new T[lda * An];

    generate_dense_matrix(Am, An, Adata, lda, iseed, aMode, cond);
    *aDenseTile = new DenseTile<T>(Am, An, Adata, lda, blas::Layout::ColMajor, aTransA, blas::Uplo::General);

    int64_t Ark;
    T *AUVdata;

    compress_dense_matrix(Am, An, Adata, lda, &AUVdata, Ark, aAcc);
    *aCompressedTile = new CompressedTile<T>(Am, An, AUVdata, lda, Ark, aAcc, blas::Layout::ColMajor, aTransA,
                                             blas::Uplo::General);
    free(AUVdata);
    delete[] Adata;
}

int main(int argc, char *argv[]) {
    int64_t m = 500;
    int64_t n = 500;
    int64_t k = 500;
    double alpha = 3.5;
    double beta = 2.5;
    blas::real_type<double> accuracy = 0.0001;
    blas::Op transA = blas::Op::NoTrans;
    blas::Op transB = blas::Op::NoTrans;
    blas::Op transC = blas::Op::NoTrans;
    int64_t mode = 0;
    int64_t align = 1;
    blas::real_type<double> cond = 1;
    blas::real_type<double> tol = 3;

    TILE_COMBINATION combination = CCC;

    int a_rows = 2;
    int a_cols = 2;
    int b_rows = 2;
    int b_cols = 2;
    DenseTile<double> ADense[a_rows][a_cols];
    DenseTile<double> BDense[b_rows][b_cols];
    DenseTile<double> CDense[a_rows][b_cols];
    CompressedTile<double> AComp[a_rows][a_cols];
    CompressedTile<double> BComp[b_rows][b_cols];
    CompressedTile<double> CComp[a_rows][b_cols];
    int64_t iseed[4] = {0, 0, 0, 1};

    double *Exact_c[a_rows][b_cols];
    double *Approx_c[a_rows][b_cols];

    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            Approx_c[i][j] = (double *) malloc(m * n * sizeof(double));
            auto accumulative_Approx_c_output = Approx_c[i][j];
            Exact_c[i][j] = (double *) malloc(m * n * sizeof(double));
            auto accumulative_Exact_c_output = Exact_c[i][j];

            memset(accumulative_Approx_c_output, 0, m * n * sizeof(double));
            memset(accumulative_Exact_c_output, 0, m * n * sizeof(double));
            for (int q = 0; q < a_cols; q++) {
                auto denseA = &ADense[i][q];
                auto denseB = &BDense[q][j];
                auto denseC = &CDense[i][j];
                auto compA = &AComp[i][q];
                auto compB = &BComp[q][j];
                auto compC = &CComp[i][j];

                double *Approx_c_output;
                double *Exact_c_output;
                GenerateTiles(transA, m, n, k, mode, accuracy, align, iseed,
                              &denseA, &compA);

                GenerateTiles(transB, m, n, k, mode, accuracy, align, iseed,
                              &denseB, &compB);

                GenerateTiles(transC, m, n, k, mode, accuracy, align, iseed,
                              &denseC, &compC);

                run(transA, transB, transC, alpha, beta, m, n, k, tol, accuracy,
                    *denseA, *denseB, *denseC, *compA, *compB, *compC, &Approx_c_output, &Exact_c_output);

                for (int h = 0; h < m * n; h++) {
                    accumulative_Exact_c_output[h] += Exact_c_output[h];
                    accumulative_Approx_c_output[h] += Approx_c_output[h];
                }

                free(Approx_c_output);
                free(Exact_c_output);
            }
        }
    }

    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < a_cols; j++) {
            free(Exact_c[i][j]);
            free(Approx_c[i][j]);
        }
    }

    run(transA, transB, transC, alpha, beta, m, n, k, tol,
        accuracy, mode, align, combination);
}