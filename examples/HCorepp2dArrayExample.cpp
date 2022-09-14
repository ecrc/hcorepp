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
                     blas::Op aTransC, int64_t aLdC, T aAlpha, T aBeta, T **aCOutput,
                     hcorepp::helpers::SvdHelpers aHelpers, TILE_COMBINATION aCombination) {
    hcorepp::api::gemm(aAlpha, A, aTransA, B, aTransB, aBeta, C, aTransC, aHelpers);

    if (aCombination == DCC || aCombination == CDC || aCombination == CCC) {
        auto Cm = C.GetTileSubMatrix(0).get().GetNumOfRows();
        auto Cn = C.GetTileSubMatrix(1).get().GetNumOfCols();
        auto rank = C.GetTileSubMatrix(0).get().GetNumOfCols();

        *aCOutput = (T *) malloc(Cm * Cn * sizeof(T));
        memset(*aCOutput, 0, Cm * Cn * sizeof(T));
        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   Cm, Cn, rank, 1.0, C.GetTileSubMatrix(0).get().GetData(),
                   C.GetTileSubMatrix(0).get().GetLeadingDim(), C.GetTileSubMatrix(1).get().GetData(),
                   C.GetTileSubMatrix(1).get().GetLeadingDim(), 0.0, *aCOutput, aLdC);
    } else if (aCombination == DDC) {
        auto cu_m_new = C.GetTileSubMatrix(0).get().GetNumOfRows();
        auto cu_n_new = C.GetTileSubMatrix(0).get().GetNumOfCols();

        *aCOutput = (T *) malloc((cu_m_new * cu_n_new) * sizeof(T));

        memcpy((void *) *aCOutput, (void *) C.GetTileSubMatrix(0).get().GetData(), cu_m_new * cu_n_new * sizeof(T));

    } else {
        auto Cm = C.GetTileSubMatrix(0).get().GetNumOfRows();
        auto Cn = C.GetTileSubMatrix(0).get().GetNumOfCols();

        *aCOutput = (T *) malloc(Cm * Cn * sizeof(T));

        memcpy((void *) *aCOutput, (void *) C.GetTileSubMatrix(0).get().GetData(), Cm * Cn * sizeof(T));
    }
}

template<typename T>
void
CalculateApproxAndExactTilesGemm(blas::Op aTransA, blas::Op aTransB, blas::Op aTransC, T aAlpha, T aBeta, int64_t aM,
                                 int64_t aN, int64_t aK, T aTol,
                                 T aAcc, DenseTile<T> &dense_tileA, DenseTile<T> &dense_tileB,
                                 DenseTile<T> &dense_tileC,
                                 CompressedTile<T> &comp_tileA, CompressedTile<T> &comp_tileB,
                                 CompressedTile<T> &comp_tileC, T **Approx_C_output,
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

    std::cout << " ldc :    " << ldc << "\n";
    std::cout << " ldcref :    " << ldcref << "\n";
    *Exact_C_Output = (T *) malloc(ldcref * Cn * sizeof(T));
    memcpy((void *) *Exact_C_Output, Cdata, Cm * Cn * sizeof(T));

    CalculateExact(blas::Layout::ColMajor, aTransA, aTransB, lda, ldb, ldcref, Adata,
                   Bdata, *Exact_C_Output, m, n, k, aAlpha, aBeta);
    memcpy(Cdata, (void *) *Exact_C_Output, Cm * Cn * sizeof(T));

    hcorepp::helpers::SvdHelpers helpers;
    CalculateApprox(comp_tileA, aTransA, comp_tileB, aTransB, comp_tileC, aTransC, ldc, aAlpha, aBeta,
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

    std::cout << "A NORM: " << Anorm << " B NORM : " << Bnorm << " CNORM : " << Cnorm << " ldc:  " << ldc
              << " ldcref:  " << ldcref << "  k: " << k << "\n";

}

template<typename T>
void
GenerateTiles(blas::Op aTransA, int64_t aM, int64_t aN, int64_t aK, int64_t aMode, blas::real_type<T> aCond, T aAcc,
              int64_t aAlignment, int64_t *iseed, DenseTile<T> **aDenseTile, CompressedTile<T> **aCompressedTile) {

//    blas::real_type<T> cond = 1;// to be passed with function parameters.cond = precision .
    //lamch >> Gets machine precision.
    int64_t align = aAlignment;

    int64_t Am = aTransA == blas::Op::NoTrans ? aM : aK;
    int64_t An = aTransA == blas::Op::NoTrans ? aK : aN;

    int64_t lda = ((Am + align - 1) / align) * align;

    T *Adata = new T[lda * An];

    generate_dense_matrix(Am, An, Adata, lda, iseed, aMode, aCond);
    *aDenseTile = new DenseTile<T>(Am, An, Adata, lda, blas::Layout::ColMajor, aTransA, blas::Uplo::General);

    int64_t Ark;
    T *AUVdata;

    compress_dense_matrix(Am, An, Adata, lda, &AUVdata, Ark, aAcc);
    *aCompressedTile = new CompressedTile<T>(Am, An, AUVdata, lda, Ark, aAcc, blas::Layout::ColMajor, aTransA,
                                             blas::Uplo::General);
    free(AUVdata);
    delete[] Adata;
}

template<typename T>
void
GenerateDenseAndCompressedTiles(int64_t aRows, int64_t aCols, blas::Op aTransA, int64_t aM, int64_t aN, int64_t aK,
                                int64_t aMode, blas::real_type<T> aCond, T aAcc, int64_t aAlignment, int64_t *iseed,
                                std::vector<std::vector<DenseTile<T> *>> &ADense,
                                std::vector<std::vector<CompressedTile<T> *>> &AComp) {
    ADense.resize(aRows);
    AComp.resize(aRows);
    for (int i = 0; i < aRows; i++) {
        ADense[i].resize(aCols);
        AComp[i].resize(aCols);
        for (int j = 0; j < aCols; j++) {
            GenerateTiles(aTransA, aM, aN, aK, aMode, aCond, aAcc, aAlignment, iseed,
                          &ADense[i][j], &AComp[i][j]);
        }
    }
}

template<typename T>
void MergeMatrixTilesIntoArray(int64_t aM, int64_t aN, int64_t aRows, int64_t aCols,
                               std::vector<std::vector<T *>> &apTiles, T *&apOutArray) {

    int64_t total_rows = aM * aRows;
    int64_t total_cols = aN * aCols;

    size_t full_array_index = 0;
    size_t tile_index_r = 0;
    size_t tile_index_c = 0;
    size_t index_in_tile = 0;

    for (int i = 0; i < total_cols; i++) {
        for (int j = 0; j < total_rows; j++) {
            full_array_index = i * total_rows + j;
            tile_index_r = j / aM;
            tile_index_c = i / aN;
            index_in_tile = (i % aN) * aM + (j % aM);
            apOutArray[full_array_index] = (apTiles[tile_index_r][tile_index_c])[index_in_tile];
        }
    }

}

template<typename T>
void DeleteDenseAndCompressedTiles(int64_t aRows, int64_t aCols, std::vector<std::vector<DenseTile<T> *>> &ADense,
                                   std::vector<std::vector<CompressedTile<T> *>> &AComp) {
    for (int i = 0; i < aRows; i++) {
        for (int j = 0; j < aCols; j++) {
            delete ADense[i][j];
            delete AComp[i][j];
        }
    }
}

int main() {
    int64_t m = 500;
    int64_t n = 500;
    int64_t k = 500;
    double alpha = 1;
    double beta = 1;
    blas::real_type<double> accuracy = 0.0001;
    blas::Op transA = blas::Op::NoTrans;
    blas::Op transB = blas::Op::NoTrans;
    blas::Op transC = blas::Op::NoTrans;
    int64_t mode = 0;
    int64_t align = 1;
    blas::real_type<double> cond = 1;
    blas::real_type<double> tol = 3;

    int a_rows = 2;
    int a_cols = 2;
    int b_rows = 2;
    int b_cols = 2;
    int c_rows = a_rows;
    int c_cols = b_cols;

    int64_t iseed[4] = {0, 0, 0, 1};

    ///Generate Matrix A Dense and compressed tiles.
    std::vector<std::vector<DenseTile<double> *>> ADense;
    std::vector<std::vector<CompressedTile<double> *>> AComp;
    GenerateDenseAndCompressedTiles(a_rows, a_cols, transA, m, n, k, mode, cond, accuracy, align, iseed,
                                    ADense, AComp);

    ///Generate Matrix B Dense and compressed tiles.
    std::vector<std::vector<DenseTile<double> *>> BDense;
    std::vector<std::vector<CompressedTile<double> *>> BComp;
    GenerateDenseAndCompressedTiles(b_rows, b_cols, transB, m, n, k, mode, cond, accuracy, align, iseed,
                                    BDense, BComp);

    ///Generate Matrix C Dense and compressed tiles.
    std::vector<std::vector<DenseTile<double> *>> CDense;
    std::vector<std::vector<CompressedTile<double> *>> CComp;
    GenerateDenseAndCompressedTiles(c_rows, c_cols, transC, m, n, k, mode, cond, accuracy, align, iseed,
                                    CDense, CComp);

    std::vector<std::vector<double *>> Exact_a;
    std::vector<std::vector<double *>> Exact_b;
    std::vector<std::vector<double *>> Exact_c;
    std::vector<std::vector<double *>> Approx_c;

    Exact_a.resize(a_rows);
    Exact_b.resize(b_rows);
    Exact_c.resize(c_rows);
    Approx_c.resize(c_rows);
    for (int i = 0; i < a_rows; i++) {
        Exact_a[i].resize(a_cols);
        Exact_b[i].resize(b_cols);
        Exact_c[i].resize(c_cols);
        Approx_c[i].resize(c_cols);
        for (int j = 0; j < b_cols; j++) {

            Approx_c[i][j] = (double *) malloc(m * n * sizeof(double));

            Exact_c[i][j] = (double *) malloc(m * n * sizeof(double));

            Exact_a[i][j] = ADense[i][j]->GetTileSubMatrix(0).get().GetData();
            Exact_b[i][j] = BDense[i][j]->GetTileSubMatrix(0).get().GetData();

            auto denseC = CDense[i][j];
            auto compC = CComp[i][j];

            for (int q = 0; q < a_cols; q++) {
                auto denseA = ADense[i][q];
                auto denseB = BDense[q][j];
                auto compA = AComp[i][q];
                auto compB = BComp[q][j];

                CalculateApproxAndExactTilesGemm(transA, transB, transC, alpha, beta, m, n, k, tol, accuracy,
                                                 *denseA, *denseB, *denseC, *compA, *compB, *compC, &Approx_c[i][j],
                                                 &Exact_c[i][j]);
            }
        }
    }

    auto *full_approx_c = (double *) malloc(a_rows * b_cols * (m * n * sizeof(double)));
    auto *full_exact_c = (double *) malloc(a_rows * b_cols * (m * n * sizeof(double)));
    auto *full_exact_a = (double *) malloc(a_rows * a_cols * (m * n * sizeof(double)));
    auto *full_exact_b = (double *) malloc(b_rows * b_cols * (m * n * sizeof(double)));

    int64_t new_m = m * a_rows;
    int64_t new_n = n * b_cols;

    MergeMatrixTilesIntoArray(m, n, c_rows, c_cols, Approx_c, full_approx_c);
    MergeMatrixTilesIntoArray(m, n, c_rows, c_cols, Exact_c, full_exact_c);
    MergeMatrixTilesIntoArray(m, n, a_rows, a_cols, Exact_a, full_exact_a);
    MergeMatrixTilesIntoArray(m, n, b_rows, b_cols, Exact_b, full_exact_b);

    int64_t lda = m * a_rows;
    int64_t ldb = m * b_rows;
    int64_t ldc = m * a_rows;

    int64_t ldcref = (((m * a_rows) + align - 1) / align) * align;

    k = n * a_cols;
    lapack::Norm norm = lapack::Norm::Inf; // todo: variable norm type
    blas::real_type<double> Anorm = lapack::lange(norm, new_m, new_n, full_exact_a, lda);
    blas::real_type<double> Bnorm = lapack::lange(norm, new_m, new_n, full_exact_b, ldb);
    blas::real_type<double> Cnorm = lapack::lange(norm, new_m, new_n, full_exact_c, ldc);

    diff(full_exact_c, ldcref, full_approx_c, new_m, new_n, ldc);

    double error = lapack::lange(norm, new_m, new_n, full_exact_c, ldcref)
                   / (sqrt(blas::real_type<double>(k) + 2) * std::abs(alpha) *
                      Anorm * Bnorm + 2 * std::abs(beta) * Cnorm);

    std::cout << "FINAL ERROR VALUE : " << error << " expected tolerance:  " << tol * accuracy << "\n";
    bool pass = (error < tol * accuracy);
    std::cout << " FINAL RESULT : " << pass << "\n";


    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            free(Exact_c[i][j]);
            free(Approx_c[i][j]);
        }
    }
    ///Delete Matrix A Dense and compressed tiles.
    DeleteDenseAndCompressedTiles(a_rows, a_cols, ADense, AComp);
    ///Delete Matrix B Dense and compressed tiles.
    DeleteDenseAndCompressedTiles(b_rows, b_cols, BDense, BComp);
    ///Delete Matrix C Dense and compressed tiles.
    DeleteDenseAndCompressedTiles(c_rows, c_cols, CDense, CComp);

    free(full_exact_c);
    free(full_approx_c);
    free(full_exact_a);
    free(full_exact_b);
}
