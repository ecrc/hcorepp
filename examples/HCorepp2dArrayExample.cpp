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
              int64_t aAlignment,
              int64_t *iseed, DenseTile<T> **aDenseTile, CompressedTile<T> **aCompressedTile) {

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

int main(int argc, char *argv[]) {
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

    TILE_COMBINATION combination = CCC;

    int a_rows = 2;
    int a_cols = 2;
    int b_rows = 2;
    int b_cols = 2;

    DenseTile<double> *ADense[a_rows][a_cols];
    DenseTile<double> *BDense[b_rows][b_cols];
    DenseTile<double> *CDense[a_rows][b_cols];
    CompressedTile<double> *AComp[a_rows][a_cols];
    CompressedTile<double> *BComp[b_rows][b_cols];
    CompressedTile<double> *CComp[a_rows][b_cols];

    int64_t iseed[4] = {0, 0, 0, 1};

    double *Exact_c[a_rows][b_cols];
    double *Approx_c[a_rows][b_cols];

    ///Generate Matrix A Dense and compressed tiles.
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < a_cols; j++) {
            GenerateTiles(transA, m, n, k, mode, cond, accuracy, align, iseed,
                          &ADense[i][j], &AComp[i][j]);
        }
    }

    ///Generate Matrix B Dense and compressed tiles.
    for (int i = 0; i < b_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            GenerateTiles(transB, m, n, k, mode, cond, accuracy, align, iseed,
                          &BDense[i][j], &BComp[i][j]);
        }
    }

    ///Generate Matrix C Dense and compressed tiles.
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            GenerateTiles(transC, m, n, k, mode, cond, accuracy, align, iseed,
                          &CDense[i][j], &CComp[i][j]);
        }
    }


    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {

            Approx_c[i][j] = (double *) malloc(m * n * sizeof(double));

            Exact_c[i][j] = (double *) malloc(m * n * sizeof(double));

            auto denseC = CDense[i][j];
            auto compC = CComp[i][j];

            for (int q = 0; q < a_cols; q++) {
                auto denseA = ADense[i][q];
                auto denseB = BDense[q][j];
                auto compA = AComp[i][q];
                auto compB = BComp[q][j];

                double *Approx_c_output;
                double *Exact_c_output;

                run(transA, transB, transC, alpha, beta, m, n, k, tol, accuracy,
                    *denseA, *denseB, *denseC, *compA, *compB, *compC, &Approx_c_output, &Exact_c_output);

                memcpy((void *) (Approx_c[i][j]), (void *) Approx_c_output, m * n * sizeof(double));
                memcpy((void *) (Exact_c[i][j]), (void *) Exact_c_output, m * n * sizeof(double));

                free(Approx_c_output);
                free(Exact_c_output);
            }
        }
    }


    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            free(Exact_c[i][j]);
            free(Approx_c[i][j]);
        }
    }

    ///Delete Matrix A Dense and compressed tiles.
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < a_cols; j++) {
            delete ADense[i][j];
            delete AComp[i][j];
        }
    }

    ///Delete Matrix B Dense and compressed tiles.
    for (int i = 0; i < b_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            delete BDense[i][j];
            delete BComp[i][j];
        }
    }

    ///Delete Matrix C Dense and compressed tiles.
    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < b_cols; j++) {
            delete CDense[i][j];
            delete CComp[i][j];
        }
    }


}