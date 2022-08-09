#include <libraries/catch/catch.hpp>
#include <iostream>

#include <hcorepp/api/hcorepp.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>
#include <hcorepp/data-units/DataHolder.hpp>
#include <hcorepp/test-helpers/testHelpers.hpp>
#include <cstring>

#include "blas.hh"

using namespace std;
using namespace hcorepp::dataunits;
using namespace hcorepp::operators;
using namespace hcorepp::api;
using namespace hcorepp::test_helpers;

void TEST_GEMM() {

    SECTION("Gemm test 1") {
        std::cout << "Test1: DDD \n =========================== \n";

        float matrix_A[3][3] = {{1, 2, 3},
                                {4, 5, 6},
                                {7, 8, 9}};
        float matrix_B[3][3] = {{2,  4,  6},
                                {8,  10, 12},
                                {14, 16, 18}};
        float output_matrix[3][3] = {{60,  72,  84},
                                     {132, 162, 192},
                                     {204, 252, 300}};
        float matrix_D[3][3] = {{0, 0, 0},
                                {0, 0, 0},
                                {0, 0, 0}};

        float alpha = 1;
        float beta = 1;

        int64_t m = 3;
        int64_t n = 3;
        int64_t k = 3;
        int64_t lda = 3;
        int64_t ldb = 3;
        int64_t ldc = 3;

        int size_a = m * n;
        int size_b = size_a;

        float *a_input = new float[size_a];
        float *b_input = new float[size_b];

        rowMajorToColumnMajor<float>((float *) matrix_A, n, m, a_input);
        rowMajorToColumnMajor<float>((float *) matrix_B, n, m, b_input);

        DenseTile<float> dense_tile_A(m, n, (float *) a_input, lda, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        DenseTile<float> dense_tile_B(m, n, (float *) b_input, ldb, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        DenseTile<float> dense_tile_C(m, n, (float *) nullptr, ldc, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        gemm<float>(alpha, dense_tile_A, dense_tile_B, beta, dense_tile_C);
        float *output = dense_tile_C.GetTileSubMatrix(0).get().GetData();

        columnMajorToRowMajor<float>(output, n, m, (float *) matrix_D);

        //        std::cout << "C Output \n";
        validateOutput((float *) matrix_D, m, n, (float *) output_matrix);
//        printMatrix(output, c_m, c_n);

        delete[] a_input;
        delete[] b_input;

    }

    SECTION("Gemm test 2") {
        std::cout << "Test2: DDD \n =========================== \n";

        float matrix_A[3][1] = {{5},
                                {10},
                                {15}};
        float matrix_B[1][3] = {{2, 4, 6}};

        float output_matrix[3][3] = {{10, 20, 30},
                                     {20, 40, 60},
                                     {30, 60, 90}};
        float matrix_D[3][3] = {{0, 0, 0},
                                {0, 0, 0},
                                {0, 0, 0}};

        float alpha = 1;
        float beta = 1;

        int64_t m = 3;
        int64_t n = 3;
        int64_t k = 1;
        int64_t lda = 3;
        int64_t ldb = 1;
        int64_t ldc = 3;

        int size_a = m * k;
        int size_b = k * n;

        float *a_input = new float[size_a];
        float *b_input = new float[size_b];

        rowMajorToColumnMajor<float>((float *) matrix_A, k, m, a_input);
        rowMajorToColumnMajor<float>((float *) matrix_B, n, k, b_input);

        DenseTile<float> dense_tile_A(m, k, (float *) matrix_A, lda, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        DenseTile<float> dense_tile_B(k, n, (float *) matrix_B, ldb, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        DenseTile<float> dense_tile_C(m, n, (float *) nullptr, ldc, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        gemm<float>(alpha, dense_tile_A, dense_tile_B, beta, dense_tile_C);
        float *output = dense_tile_C.GetTileSubMatrix(0).get().GetData();

        columnMajorToRowMajor<float>(output, n, m, (float *) matrix_D);

        //        std::cout << "C Output \n";
        validateOutput((float *) matrix_D, m, n, (float *) output_matrix);
//        printMatrix(output, c_m, c_n);

        delete[] a_input;
        delete[] b_input;

    }

    SECTION("Gemm Test 3") {
        std::cout << "Test3: DDD \n =========================== \n";

        float matrix_A[1][3] = {10, 11, 12};

        float matrix_B[3][2] = {{2,  4},
                                {8,  10},
                                {14, 16}};

        float output_matrix[1][2] = {{276, 342}};

        float matrix_D[1][2] = {{0, 0}};

        float alpha = 1;
        float beta = 1;

        int64_t a_m = 1;
        int64_t a_n = 3;
        int64_t b_m = 3;
        int64_t b_n = 2;

        int64_t m = 1;
        int64_t n = 2;
        int64_t k = 3;

        int64_t lda = a_m;
        int64_t ldb = b_m;
        int64_t ldc = m;

        float *a_input = new float[a_m * a_n];
        float *b_input = new float[b_m * b_n];

        rowMajorToColumnMajor<float>((float *) matrix_A, a_n, a_m, a_input);
        rowMajorToColumnMajor<float>((float *) matrix_B, b_n, b_m, b_input);

        DenseTile<float> dense_tile_A(a_m, a_n, a_input, lda, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        DenseTile<float> dense_tile_B(b_m, b_n, b_input, ldb, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        DenseTile<float> dense_tile_C(m, n, nullptr, ldc, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        gemm<float>(alpha, dense_tile_A, dense_tile_B, beta, dense_tile_C);

        float *output = dense_tile_C.GetTileSubMatrix(0).get().GetData();

        columnMajorToRowMajor<float>(output, n, m, (float *) matrix_D);

        //        std::cout << "C Output \n";
        validateOutput((float *) matrix_D, m, n, (float *) output_matrix);
//        printMatrix(output, c_m, c_n);

        delete[] a_input;
        delete[] b_input;
    }

    SECTION("Gemm Test 4") {
        std::cout << "Test4: CDD \n =========================== \n";

        float matrix_Au[3][1] = {{1},
                                 {4},
                                 {7}};

        float matrix_Av[1][3] = {{10, 11, 12}};

        float matrix_B[3][2] = {{2,  4},
                                {8,  10},
                                {14, 16}};


        float output_matrix[3][2] = {{276,  342},
                                     {1104, 1368},
                                     {1932, 2394}};
        float matrix_D[3][2] = {{0, 0},
                                {0, 0},
                                {0, 0}};

        float alpha = 1;
        float beta = 1;

        int64_t au_m = 3;
        int64_t au_n = 1;
        int64_t av_m = 1;
        int64_t av_n = 3;
        int64_t b_m = 3;
        int64_t b_n = 2;
        int64_t c_m = 3;
        int64_t c_n = 2;
        int64_t rank = 1;
        int64_t ldau = 3;
        int64_t ldav = 1;
        int64_t ldb = 3;
        int64_t ldc = 3;

        size_t au_size = au_m * au_n;
        size_t av_size = av_m * av_n;
        size_t b_size = b_m * b_n;

        float *au_input = new float[au_size];
        float *av_input = new float[av_size];
        float *b_input = new float[b_size];

        rowMajorToColumnMajor<float>((float *) matrix_Au, au_n, au_m, au_input);
        rowMajorToColumnMajor<float>((float *) matrix_Av, av_n, av_m, av_input);
        rowMajorToColumnMajor<float>((float *) matrix_B, b_n, b_m, b_input);

        float *compressed_tile_a_data = (float *) malloc((au_size + av_size) * sizeof(float));
        memcpy(compressed_tile_a_data, au_input, au_size * sizeof(float));
        memcpy(&compressed_tile_a_data[au_size], av_input, av_size * sizeof(float));

        CompressedTile<float> compressed_tile_A(au_m, av_n, compressed_tile_a_data, ldau, rank, 1,
                                                blas::Layout::ColMajor,
                                                blas::Op::NoTrans, blas::Uplo::General);

        DenseTile<float> dense_tile_B(b_m, b_n, (float *) b_input, ldb, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        DenseTile<float> dense_tile_C(c_m, c_n, nullptr, ldc, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        gemm<float>(alpha, compressed_tile_A, dense_tile_B, beta, dense_tile_C);
        float *output = dense_tile_C.GetTileSubMatrix(0).get().GetData();

        columnMajorToRowMajor<float>(output, c_n, c_m, (float *) matrix_D);

        //        std::cout << "C Output \n";
        validateOutput((float *) matrix_D, c_m, c_n, (float *) output_matrix);
//        printMatrix(output, c_m, c_n);

        delete[] au_input;
        delete[] av_input;
        delete[] b_input;

        free(compressed_tile_a_data);
    }

    SECTION("Gemm Test 5") {
        std::cout << "Test5: DCD \n =========================== \n";

        float matrix_A[3][3] = {{2,  1,  4},
                                {8,  5,  10},
                                {14, 10, 16}};
        float matrix_Bu[3][1] = {{1},
                                 {4},
                                 {7}};

        float matrix_Bv[1][3] = {{10, 11, 12}};

        float output_matrix[3][3] = {{340,  374,  408},
                                     {980,  1078, 1176},
                                     {1660, 1826, 1992}};
        float matrix_D[3][3] = {{0, 0, 0},
                                {0, 0, 0},
                                {0, 0, 0}};

        float alpha = 1;
        float beta = 1;

        int64_t a_m = 3;
        int64_t a_n = 3;
        int64_t bu_m = 3;
        int64_t bu_n = 1;
        int64_t bv_m = 1;
        int64_t bv_n = 3;

        int64_t c_m = 3;
        int64_t c_n = 3;

        int64_t rank = 1;

        int64_t lda = 3;
        int64_t ldbu = 3;
        int64_t ldbv = 1;
        int64_t ldc = 3;

        size_t a_size = a_m * a_n;
        size_t bu_size = bu_m * bu_n;
        size_t bv_size = bv_m * bv_n;

        float *a_input = new float[a_size];
        float *bu_input = new float[bu_size];
        float *bv_input = new float[bv_size];

        rowMajorToColumnMajor<float>((float *) matrix_A, a_n, a_m, a_input);
        rowMajorToColumnMajor<float>((float *) matrix_Bu, bu_n, bu_m, bu_input);
        rowMajorToColumnMajor<float>((float *) matrix_Bv, bv_n, bv_m, bv_input);

        float *compressed_tile_b_data = (float *) malloc((bu_size + bv_size) * sizeof(float));
        memcpy(compressed_tile_b_data, bu_input, bu_size * sizeof(float));
        memcpy(&compressed_tile_b_data[bu_size], bv_input, bv_size * sizeof(float));

        CompressedTile<float> compressed_tile_B(bu_m, bv_n, compressed_tile_b_data, ldbu, rank, 1);
        DenseTile<float> dense_tile_A(a_m, a_n, (float *) a_input, lda);
        DenseTile<float> dense_tile_C(c_m, c_n, nullptr, ldc);

        gemm<float>(alpha, dense_tile_A, compressed_tile_B, beta, dense_tile_C);
        float *output = dense_tile_C.GetTileSubMatrix(0).get().GetData();

        columnMajorToRowMajor<float>(output, c_n, c_m, (float *) matrix_D);

        //        std::cout << "C Output \n";
        validateOutput((float *) matrix_D, c_m, c_n, (float *) output_matrix);
//        printMatrix(output, c_m, c_n);

        delete[] a_input;
        delete[] bu_input;
        delete[] bv_input;
        free(compressed_tile_b_data);
    }

    SECTION("Gemm Test 6") {
        std::cout << "Test6: CCD \n =========================== \n";

        float matrix_Au[3][2] = {{2,  4},
                                 {8,  10},
                                 {14, 16}};
        float matrix_Av[2][4] = {{1, 5, 6, 7},
                                 {3, 4, 8, 9}};
        float matrix_Bu[4][3] = {{2, 10, 18},
                                 {4, 12, 20},
                                 {6, 14, 22},
                                 {8, 16, 24}};
        float matrix_Bv[3][2] = {{5,  25},
                                 {10, 30},
                                 {15, 35}};
        float output_matrix[3][2] = {{66760,  178840},
                                     {195400, 523480},
                                     {324040, 868120}};
        float matrix_D[3][2] = {{0, 0},
                                {0, 0},
                                {0, 0}};

        float alpha = 1;
        float beta = 1;

        int64_t au_m = 3;
        int64_t au_n = 2;
        int64_t av_m = 2;
        int64_t av_n = 4;
        int64_t bu_m = 4;
        int64_t bu_n = 3;
        int64_t bv_m = 3;
        int64_t bv_n = 2;

        int64_t c_m = 3;
        int64_t c_n = 2;

        int64_t a_rank = 2;
        int64_t b_rank = 3;

        int64_t ldau = 3;
        int64_t ldav = 2;
        int64_t ldbu = 4;
        int64_t ldbv = 3;
        int64_t ldc = 3;

        size_t au_size = au_m * au_n;
        size_t av_size = av_m * av_n;
        size_t bu_size = bu_m * bu_n;
        size_t bv_size = bv_m * bv_n;

        float *au_input = new float[au_size];
        float *av_input = new float[av_size];
        float *bu_input = new float[bu_size];
        float *bv_input = new float[bv_size];

        rowMajorToColumnMajor<float>((float *) matrix_Au, au_n, au_m, au_input);
        rowMajorToColumnMajor<float>((float *) matrix_Av, av_n, av_m, av_input);
        rowMajorToColumnMajor<float>((float *) matrix_Bu, bu_n, bu_m, bu_input);
        rowMajorToColumnMajor<float>((float *) matrix_Bv, bv_n, bv_m, bv_input);

        float *compressed_tile_a_data = (float *) malloc((au_size + av_size) * sizeof(float));
        memcpy(compressed_tile_a_data, au_input, au_size * sizeof(float));
        memcpy(&compressed_tile_a_data[au_size], av_input, av_size * sizeof(float));

        float *compressed_tile_b_data = (float *) malloc((bu_size + bv_size) * sizeof(float));
        memcpy(compressed_tile_b_data, bu_input, bu_size * sizeof(float));
        memcpy(&compressed_tile_b_data[bu_size], bv_input, bv_size * sizeof(float));

        CompressedTile<float> compressed_tile_A(au_m, av_n, compressed_tile_a_data, ldau, a_rank, 1);
        CompressedTile<float> compressed_tile_B(bu_m, bv_n, compressed_tile_b_data, ldbu, b_rank, 1);
        DenseTile<float> dense_tile_C(c_m, c_n, nullptr, ldc);

        gemm<float>(alpha, compressed_tile_A, compressed_tile_B, beta, dense_tile_C);
        float *output = dense_tile_C.GetTileSubMatrix(0).get().GetData();

        columnMajorToRowMajor<float>(output, c_n, c_m, (float *) matrix_D);


        //        std::cout << "C Output \n";
        validateOutput((float *) matrix_D, c_m, c_n, (float *) output_matrix);
//        printMatrix(output, c_m, c_n);


        delete[] au_input;
        delete[] av_input;
        delete[] bu_input;
        delete[] bv_input;
        free(compressed_tile_a_data);
        free(compressed_tile_b_data);
    }

    SECTION("Gemm Test 7") {
        std::cout << "Test6: CDC \n =========================== \n";

        float matrix_Au[3][2] = {{2, 8},
                                 {4, 10},
                                 {6, 12}};

        float matrix_Av[2][3] = {{5,  15, 25},
                                 {10, 20, 30}};

        float matrix_B[3][2] = {{1, 9},
                                {5, 10},
                                {7, 12}};

        float matrix_CU[3][2] = {{0.411242, -0.81499},
                                 {0.56376,  -0.124536},
                                 {0.716278, 0.565935}};

        float matrix_CV[2][2] = {{7488,    15040.6},
                                 {11.5109, -5.73071}};

        float output_matrix[3][2] = {{3070, 6190},
                                     {4220, 8480},
                                     {5370, 10770}};
        float matrix_D[3][2] = {{0, 0},
                                {0, 0},
                                {0, 0}};

        float alpha = 1;
        float beta = 1;

        int64_t au_m = 3;
        int64_t au_n = 2;
        int64_t av_m = 2;
        int64_t av_n = 3;
        int64_t b_m = 3;
        int64_t b_n = 2;
        int64_t cu_m = 3;
        int64_t cu_n = 1;
        int64_t cv_m = 1;
        int64_t cv_n = 2;
        int64_t c_m = 3;
        int64_t c_n = 2;
        int64_t a_rank = 2;
        int64_t c_rank = 1;
        int64_t ldau = 3;
        int64_t ldav = 2;
        int64_t ldb = 3;
        int64_t ldcu = 3;

        size_t au_size = au_m * au_n;
        size_t av_size = av_m * av_n;
        size_t b_size = b_m * b_n;
        size_t cu_size = cu_m * cu_n;
        size_t cv_size = cv_m * cv_n;

        float *au_input = new float[au_size];
        float *av_input = new float[av_size];
        float *b_input = new float[b_size];
        float *c_input = new float[cu_size + cv_size];

        rowMajorToColumnMajor<float>((float *) matrix_Au, au_n, au_m, au_input);
        rowMajorToColumnMajor<float>((float *) matrix_Av, av_n, av_m, av_input);
        rowMajorToColumnMajor<float>((float *) matrix_B, b_n, b_m, b_input);
        rowMajorToColumnMajor<float>((float *) matrix_D, c_n, c_m, c_input);

        float *compressed_tile_a_data = (float *) malloc((au_size + av_size) * sizeof(float));
        memcpy(compressed_tile_a_data, au_input, au_size * sizeof(float));
        memcpy(&compressed_tile_a_data[au_size], av_input, av_size * sizeof(float));

        CompressedTile<float> compressed_tile_A(au_m, av_n, compressed_tile_a_data, ldau, a_rank,
                                                std::numeric_limits<blas::real_type<float>>::epsilon());

        CompressedTile<float> compressed_tile_C(c_m, c_n, (float *) c_input, ldcu, c_rank,
                                                std::numeric_limits<blas::real_type<float>>::epsilon());

        DenseTile<float> dense_tile_B(b_m, b_n, (float *) b_input, ldb);

        gemm<float>(alpha, compressed_tile_A, dense_tile_B, beta, compressed_tile_C);

        float *cu_output = compressed_tile_C.GetTileSubMatrix(0).get().GetData();
        float *cv_output = compressed_tile_C.GetTileSubMatrix(1).get().GetData();

        cu_m = compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfRows();
        cu_n = compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfCols();
        cv_m = compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfRows();
        cv_n = compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfCols();

        float *cu_output_row = new float[cu_n * cu_m];
        float *cv_output_row = new float[cv_n * cv_m];

        columnMajorToRowMajor<float>(cu_output, cu_n, cu_m, (float *) cu_output_row);
        columnMajorToRowMajor<float>(cv_output, cv_n, cv_m, (float *) cv_output_row);

//        std::cout << "CU Output \n";
        validateOutput(cu_output_row, cu_m, cu_n, (float *) matrix_CU);
//        printMatrix(cu_output_row, cu_m, cu_n);

//        std::cout << "CV Output \n";
        validateOutput(cv_output_row, cv_m, cv_n, (float *) matrix_CV);
//        printMatrix(cv_output_row, cv_m, cv_n);

        float *cu_cv = new float[c_m * c_n];

        blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   c_m, c_n, cu_n, alpha, cu_output_row, cu_n, cv_output_row, cv_n,
                   0, cu_cv, c_n);

        //        std::cout << "C Output \n";
        validateOutput(cu_cv, c_m, c_n, (float *) output_matrix);
//        printMatrix(cu_cv, c_m, c_n);

        delete[] au_input;
        delete[] av_input;
        delete[] b_input;
        delete[] cv_output_row;
        delete[] cu_output_row;
        delete[] cu_cv;

        free(compressed_tile_a_data);
    }

    SECTION("Gemm Test 8") {
        std::cout << "Test8: DCC \n =========================== \n";

        float matrix_A[3][4] = {{2, 8,  14, 20},
                                {4, 10, 16, 22},
                                {6, 12, 18, 24}};

        float matrix_Bu[4][2] = {{1, 9},
                                 {3, 11},
                                 {5, 13},
                                 {7, 15}};

        float matrix_Bv[2][2] = {{5,  15},
                                 {10, 20}};

        float matrix_CU[3][2] = {{0.495891, -0.766442},
                                 {0.573832, -0.0636292},
                                 {0.651774, 0.639154}};

        float matrix_CV[2][2] = {{14256.4, 30844.6},
                                 {12.5638, -5.807}};

        float output_matrix[3][2] = {{7060, 15300},
                                     {8180, 17700},
                                     {9300, 20100}};

        float matrix_D[3][2] = {{0, 0},
                                {0, 0},
                                {0, 0}};

        float alpha = 1;
        float beta = 1;

        int64_t a_m = 3;
        int64_t a_n = 4;
        int64_t bu_m = 4;
        int64_t bu_n = 2;
        int64_t bv_m = 2;
        int64_t bv_n = 2;

        int64_t cu_m = 3;
        int64_t cu_n = 1;
        int64_t cv_m = 1;
        int64_t cv_n = 2;

        int64_t c_m = 3;
        int64_t c_n = 2;

        int64_t b_rank = 2;
        int64_t c_rank = 1;
        int64_t lda = 3;
        int64_t ldbu = 4;
        int64_t ldbv = 2;
        int64_t ldc = 3;

        size_t a_size = a_m * a_n;
        size_t bu_size = bu_m * bu_n;
        size_t bv_size = bv_m * bv_n;
        size_t cu_size = cu_m * cu_n;
        size_t cv_size = cv_m * cv_n;

        float *a_input = new float[a_size];
        float *bu_input = new float[bu_size];
        float *bv_input = new float[bv_size];
        float *c_input = new float[cu_size + cv_size];

        rowMajorToColumnMajor<float>((float *) matrix_A, a_n, a_m, a_input);
        rowMajorToColumnMajor<float>((float *) matrix_Bu, bu_n, bu_m, bu_input);
        rowMajorToColumnMajor<float>((float *) matrix_Bv, bv_n, bv_m, bv_input);

        rowMajorToColumnMajor<float>((float *) matrix_D, c_n, c_m, c_input);

        float *compressed_tile_b_data = (float *) malloc((bu_size + bv_size) * sizeof(float));
        memcpy(compressed_tile_b_data, bu_input, bu_size * sizeof(float));
        memcpy(&compressed_tile_b_data[bu_size], bv_input, bv_size * sizeof(float));

        CompressedTile<float> compressed_tile_B(bu_m, bv_n, compressed_tile_b_data, ldbu, b_rank,
                                                std::numeric_limits<blas::real_type<float>>::epsilon());

        CompressedTile<float> compressed_tile_C(c_m, c_n, (float *) c_input, ldc, c_rank,
                                                std::numeric_limits<blas::real_type<float>>::epsilon());

        DenseTile<float> dense_tile_A(a_m, a_n, (float *) a_input, lda);

        gemm<float>(alpha, dense_tile_A, compressed_tile_B, beta, compressed_tile_C);
        float *cu_output = compressed_tile_C.GetTileSubMatrix(0).get().GetData();
        float *cv_output = compressed_tile_C.GetTileSubMatrix(1).get().GetData();

        cu_m = compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfRows();
        cu_n = compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfCols();
        cv_m = compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfRows();
        cv_n = compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfCols();

        float *cu_output_row = new float[cu_n * cu_m];
        float *cv_output_row = new float[cv_n * cv_m];

        columnMajorToRowMajor<float>(cu_output, cu_n, cu_m, (float *) cu_output_row);
        columnMajorToRowMajor<float>(cv_output, cv_n, cv_m, (float *) cv_output_row);

//        std::cout << "CU Output \n";
        validateOutput(cu_output_row, cu_m, cu_n, (float *) matrix_CU);
//        printMatrix(cu_output_row, cu_m, cu_n);

//        std::cout << "CV Output \n";
        validateOutput(cv_output_row, cv_m, cv_n, (float *) matrix_CV);
//        printMatrix(cv_output_row, cv_m, cv_n);

        float *cu_cv = new float[c_m * c_n];

        blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   c_m, c_n, cu_n, alpha, cu_output_row, cu_n, cv_output_row, cv_n,
                   0, cu_cv, c_n);

        //        std::cout << "C Output \n";
        validateOutput(cu_cv, c_m, c_n, (float *) output_matrix);
//        printMatrix(cu_cv, c_m, c_n);

        delete[] a_input;
        delete[] bu_input;
        delete[] bv_input;
        delete[] cv_output_row;
        delete[] cu_output_row;
        delete[] cu_cv;

        free(compressed_tile_b_data);

    }

    SECTION("Gemm Test 9") {
        std::cout << "Test9: CCC \n =========================== \n";

        float matrix_Au[5][3] = {{1, 10, 20},
                                 {2, 11, 21},
                                 {3, 12, 22},
                                 {4, 13, 23},
                                 {5, 14, 24}};

        float matrix_Av[3][4] = {{2, 8,  14, 20},
                                 {4, 10, 16, 22},
                                 {6, 12, 18, 24}};

        float matrix_Bu[4][2] = {{1, 9},
                                 {3, 11},
                                 {5, 13},
                                 {7, 15}};

        float matrix_Bv[2][2] = {{5,  15},
                                 {10, 20}};

        float matrix_CU[5][2] = {{0.37726,  0.67687},
                                 {0.410963, 0.36179},
                                 {0.444666, 0.0474023},
                                 {0.47837,  -0.266773},
                                 {0.512073, -0.580978,}};

        float matrix_CV[2][2] = {{728497,  1.57534e+06},
                                 {40.5026, -18.7299}};

        float output_matrix[5][2] = {{274860, 594300},
                                     {299400, 647400},
                                     {323940, 700500},
                                     {348480, 753600},
                                     {373020, 806700}};

        float matrix_D[5][2] = {{0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0},
                                {0, 0}};

        float alpha = 1;
        float beta = 1;

        int64_t au_m = 5;
        int64_t au_n = 3;
        int64_t av_m = 3;
        int64_t av_n = 4;
        int64_t bu_m = 4;
        int64_t bu_n = 2;
        int64_t bv_m = 2;
        int64_t bv_n = 2;

        int64_t cu_m = 5;
        int64_t cu_n = 1;
        int64_t cv_m = 1;
        int64_t cv_n = 2;

        int64_t c_m = 5;
        int64_t c_n = 2;

        int64_t a_rank = 3;
        int64_t b_rank = 2;
        int64_t c_rank = 1;
        int64_t ldau = 5;
        int64_t ldav = 3;
        int64_t ldbu = 4;
        int64_t ldbv = 2;
        int64_t ldc = 5;

        size_t au_size = au_m * au_n;
        size_t av_size = av_m * av_n;
        size_t bu_size = bu_m * bu_n;
        size_t bv_size = bv_m * bv_n;
        size_t cu_size = cu_m * cu_n;
        size_t cv_size = cv_m * cv_n;

        float *au_input = new float[au_size];
        float *av_input = new float[av_size];
        float *bu_input = new float[bu_size];
        float *bv_input = new float[bv_size];
        float *c_input = new float[cu_size + cv_size];

        rowMajorToColumnMajor<float>((float *) matrix_Au, au_n, au_m, au_input);
        rowMajorToColumnMajor<float>((float *) matrix_Av, av_n, av_m, av_input);
        rowMajorToColumnMajor<float>((float *) matrix_Bu, bu_n, bu_m, bu_input);
        rowMajorToColumnMajor<float>((float *) matrix_Bv, bv_n, bv_m, bv_input);

        rowMajorToColumnMajor<float>((float *) matrix_D, c_n, c_m, c_input);

        float *compressed_tile_a_data = (float *) malloc((au_size + av_size) * sizeof(float));
        memcpy(compressed_tile_a_data, au_input, au_size * sizeof(float));
        memcpy(&compressed_tile_a_data[au_size], av_input, av_size * sizeof(float));

        float *compressed_tile_b_data = (float *) malloc((bu_size + bv_size) * sizeof(float));
        memcpy(compressed_tile_b_data, bu_input, bu_size * sizeof(float));
        memcpy(&compressed_tile_b_data[bu_size], bv_input, bv_size * sizeof(float));

        CompressedTile<float> compressed_tile_A(au_m, av_n, compressed_tile_a_data, ldau, a_rank,
                                                std::numeric_limits<blas::real_type<float>>::epsilon());

        CompressedTile<float> compressed_tile_B(bu_m, bv_n, compressed_tile_b_data, ldbu, b_rank,
                                                std::numeric_limits<blas::real_type<float>>::epsilon());

        CompressedTile<float> compressed_tile_C(c_m, c_n, (float *) c_input, ldc, c_rank,
                                                std::numeric_limits<blas::real_type<float>>::epsilon());

        gemm<float>(alpha, compressed_tile_A, compressed_tile_B, beta, compressed_tile_C);
        float *cu_output = compressed_tile_C.GetTileSubMatrix(0).get().GetData();
        float *cv_output = compressed_tile_C.GetTileSubMatrix(1).get().GetData();

        cu_m = compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfRows();
        cu_n = compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfCols();
        cv_m = compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfRows();
        cv_n = compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfCols();

        float *cu_output_row = new float[cu_n * cu_m];
        float *cv_output_row = new float[cv_n * cv_m];

        columnMajorToRowMajor<float>(cu_output, cu_n, cu_m, (float *) cu_output_row);
        columnMajorToRowMajor<float>(cv_output, cv_n, cv_m, (float *) cv_output_row);

//        std::cout << "CU Output \n";
        validateOutput(cu_output_row, cu_m, cu_n, (float *) matrix_CU);
//        printMatrix(cu_output_row, cu_m, cu_n);

//        std::cout << "CV Output \n";
        validateOutput(cv_output_row, cv_m, cv_n, (float *) matrix_CV);
//        printMatrix(cv_output_row, cv_m, cv_n);


        float *cu_cv = new float[c_m * c_n];

        blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   c_m, c_n, cu_n, alpha, cu_output_row, cu_n, cv_output_row, cv_n,
                   0, cu_cv, c_n);

//        std::cout << "C Output \n";
//        validateOutput(cu_cv, c_m, c_n, (float *) output_matrix);
//        printMatrix(cu_cv, c_m, c_n);

        delete[] au_input;
        delete[] av_input;
        delete[] bu_input;
        delete[] bv_input;
        delete[] cv_output_row;
        delete[] cu_output_row;
        delete[] cu_cv;

        free(compressed_tile_a_data);
        free(compressed_tile_b_data);

    }

    SECTION("Gemm Test 10") {
        std::cout << "Test10: DDC \n =========================== \n";

        float matrix_A[5][3] = {{1, 10, 20},
                                {2, 11, 21},
                                {3, 12, 22},
                                {4, 13, 23},
                                {5, 14, 24}};

        float matrix_B[3][4] = {{2, 8,  14, 20},
                                {4, 10, 16, 22},
                                {6, 12, 18, 24}};

        float matrix_CU[5][4] = {{162, 348, 534, 720},
                                 {174, 378, 582, 786},
                                 {186, 408, 630, 852},
                                 {198, 438, 678, 918},
                                 {210, 468, 726, 984}};

        float matrix_CV[4][4] = {{1, 0, 0, 0},
                                 {0, 1, 0, 0},
                                 {0, 0, 1, 0},
                                 {0, 0, 0, 1}};

        float output_matrix[5][4] = {{162, 348, 534, 720},
                                     {174, 378, 582, 786},
                                     {186, 408, 630, 852},
                                     {198, 438, 678, 918},
                                     {210, 468, 726, 984}};


        float alpha = 1;
        float beta = 1;

        int64_t a_m = 5;
        int64_t a_n = 3;
        int64_t b_m = 3;
        int64_t b_n = 4;

        int64_t cu_m = 5;
        int64_t cu_n = 1;
        int64_t cv_m = 1;
        int64_t cv_n = 4;

        int64_t c_m = 5;
        int64_t c_n = 4;

        int64_t c_rank = 1;
        int64_t lda = 5;
        int64_t ldb = 3;
        int64_t ldc = 5;

        size_t a_size = a_m * a_n;
        size_t b_size = b_m * b_n;
        size_t cu_size = cu_m * cu_n;
        size_t cv_size = cv_m * cv_n;

        float *a_input = new float[a_size];
        float *b_input = new float[b_size];
        float *c_input = new float[cu_size + cv_size];
        float *cu_input_R = (float *) malloc((cu_size) * sizeof(float));
        float *cv_input_R = (float *) malloc((cv_size) * sizeof(float));

        float *cu_input_C = (float *) malloc((cu_size) * sizeof(float));
        float *cv_input_C = (float *) malloc((cv_size) * sizeof(float));

        memset(cu_input_R, 0, (cu_size) * sizeof(float));
        memset(cv_input_R, 0, (cv_size) * sizeof(float));

        rowMajorToColumnMajor<float>((float *) matrix_A, a_n, a_m, a_input);
        rowMajorToColumnMajor<float>((float *) matrix_B, b_n, b_m, b_input);

        rowMajorToColumnMajor<float>((float *) cu_input_R, cu_n, cu_m, cu_input_C);
        rowMajorToColumnMajor<float>((float *) cv_input_R, cv_n, cv_m, cv_input_C);

        memcpy(c_input, cu_input_C, cu_size * sizeof(float));
        memcpy(&c_input[cu_size], cv_input_C, cv_size * sizeof(float));

        DenseTile<float> dense_tile_A(a_m, a_n, (float *) a_input, lda);
        DenseTile<float> dense_tile_B(b_m, b_n, (float *) b_input, ldb);

        CompressedTile<float> compressed_tile_C(c_m, c_n, (float *) c_input, ldc, c_rank,
                                                std::numeric_limits<blas::real_type<float>>::epsilon());


        gemm<float>(alpha, dense_tile_A, dense_tile_B, beta, compressed_tile_C);
        float *cu_output = compressed_tile_C.GetTileSubMatrix(0).get().GetData();
        float *cv_output = compressed_tile_C.GetTileSubMatrix(1).get().GetData();

        cu_m = compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfRows();
        cu_n = compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfCols();
        cv_m = compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfRows();
        cv_n = compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfCols();

        float *cu_output_row = new float[cu_n * cu_m];
        float *cv_output_row = new float[cv_n * cv_m];

        columnMajorToRowMajor<float>(cu_output, cu_n, cu_m, (float *) cu_output_row);
        columnMajorToRowMajor<float>(cv_output, cv_n, cv_m, (float *) cv_output_row);

//        std::cout << "CU Output \n";
        validateOutput(cu_output_row, cu_m, cu_n, (float *) matrix_CU);
//        printMatrix(cu_output_row, cu_m, cu_n);

//        std::cout << "CV Output \n";
        validateOutput(cv_output_row, cv_m, cv_n, (float *) matrix_CV);
//        printMatrix(cv_output_row, cv_m, cv_n);

        float *cu_cv = new float[c_m * c_n];

        blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   c_m, c_n, cu_n, alpha, cu_output_row, cu_n, cv_output_row, cv_n,
                   0, cu_cv, c_n);

//        std::cout << " C Output ==== \n";
        validateOutput(cu_cv, c_m, c_n, (float *) output_matrix);
//        printMatrix(cu_cv, c_m, c_n);

        delete[] a_input;
        delete[] b_input;
        delete[] cv_output_row;
        delete[] cu_output_row;
        delete[] cu_cv;
        free(cu_input_R);
        free(cu_input_C);
        free(cv_input_R);
        free(cv_input_C);

    }
}

TEST_CASE("GemmTest", "[GEMM]") {
    TEST_GEMM();
}
