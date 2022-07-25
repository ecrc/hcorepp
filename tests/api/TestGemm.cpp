#include <libraries/catch/catch.hpp>
#include <iostream>

#include <hcorepp/api/hcorepp.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/data-units/DataHolder.hpp>

#include "blas.hh"

using namespace std;
using namespace hcorepp::dataunits;
using namespace hcorepp::operators;
using namespace hcorepp::api;


void TEST_GEMM() {

    SECTION("Gemm test 1") {
        std::cout
                << "Test1: Square matrices multiplication using blas::gemm directly \n =========================== \n";
        float matrix_A[3][3] = {{1, 2, 3},
                                {4, 5, 6},
                                {7, 8, 9}};
        float matrix_B[3][3] = {{2,  4,  6},
                                {8,  10, 12},
                                {14, 16, 18}};
        float matrix_C[3][3] = {{60,  72,  84},
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

//        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, k,
//                   alpha, (const float *) matrix_A, lda, (const float *) matrix_B, ldb, beta, (float *) matrix_D, ldc);
//
//        for (int i = 0; i < m; i++) {
//            std::cout << "{ ";
//            for (int j = 0; j < n; j++) {
//                REQUIRE(matrix_D[i][j] == matrix_C[i][j]);
//                std::cout << matrix_D[i][j] << ", ";
//            }
//            std::cout << "} \n";
//        }

        DenseTile<float> dense_tile_A(m, n, (float *) matrix_A, lda, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        DenseTile<float> dense_tile_B(m, n, (float *) matrix_B, ldb, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        DenseTile<float> dense_tile_C(m, n, (float *) matrix_D, ldc, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        gemm<float>(alpha, dense_tile_A, dense_tile_B, beta, dense_tile_C);
        float *output = dense_tile_C.GetTileSubMatrix(0).get().GetData();

        for (int i = 0; i < m; i++) {
            std::cout << "{ ";
            for (int j = 0; j < n; j++) {
                int index = i * m + j;
                REQUIRE(output[index] == matrix_C[i][j]);
                std::cout << output[index] << ", ";
            }
            std::cout << "} \n";
        }

    }SECTION("Gemm test 2") {
        std::cout
                << " Test2: Non Square matrices multiplication using blas::gemm directly \n =========================== \n";

        float matrix_A[3][1] = {{5},
                                {10},
                                {15}};
        float matrix_B[1][3] = {{2, 4, 6}};

        float matrix_C[3][3] = {{10, 20, 30},
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

//        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, m, n, k,
//                   alpha, (const float *) matrix_A, lda, (const float *) matrix_B, ldb, beta, (float *) matrix_D, ldc);
//
//        for (int i = 0; i < m; i++) {
//            std::cout << "{ ";
//            for (int j = 0; j < n; j++) {
//                REQUIRE(matrix_D[i][j] == matrix_C[i][j]);
//                std::cout << matrix_D[i][j] << ", ";
//            }
//            std::cout << "} \n";
//
//        }

        DenseTile<float> dense_tile_A(m, k, (float *) matrix_A, lda, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        DenseTile<float> dense_tile_B(k, n, (float *) matrix_B, ldb, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        DenseTile<float> dense_tile_C(m, n, (float *) matrix_D, ldc, blas::Layout::ColMajor, blas::Op::NoTrans,
                                      blas::Uplo::General);

        gemm<float>(alpha, dense_tile_A, dense_tile_B, beta, dense_tile_C);
        float *output = dense_tile_C.GetTileSubMatrix(0).get().GetData();

        for (int i = 0; i < m; i++) {
            std::cout << "{ ";
            for (int j = 0; j < n; j++) {
                int index = i * m + j;
                REQUIRE(output[index] == matrix_C[i][j]);
                std::cout << output[index] << ", ";
            }
            std::cout << "} \n";
        }

    }
}

TEST_CASE("GemmTest", "[GEMM]") {
    TEST_GEMM();
}
