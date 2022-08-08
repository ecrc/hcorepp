#include <libraries/catch/catch.hpp>
#include <iostream>

#include <hcorepp/api/hcorepp.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/data-units/DataHolder.hpp>
#include <hcorepp/test-helpers/testHelpers.hpp>
#include <cstring>

using namespace std;
using namespace hcorepp::dataunits;
using namespace hcorepp::operators;
using namespace hcorepp::api;
using namespace hcorepp::test_helpers;


void TEST_Compressed() {

    SECTION("Compressed Tile Creation") {
        std::cout << "Compressed tile Creation functionality-\n =========================== \n";

        float matrix_AU[5][4] = {{162, 348, 534, 720},
                                 {174, 378, 582, 786},
                                 {186, 408, 630, 852},
                                 {198, 438, 678, 918},
                                 {210, 468, 726, 984}};

        float matrix_AV[4][4] = {{2,  4,  6,  8},
                                 {10, 12, 14, 16},
                                 {18, 20, 22, 24},
                                 {26, 28, 30, 32}};


        // AU num of rows
        int64_t au_m = 5;
        // AU num of cols
        int64_t au_n = 4;
        // assuming that AU and AV are COl major.
        int64_t ldaU = au_m;
        // AV num of rows
        int64_t av_m = 4;
        // AV num of cols
        int64_t av_n = 4;
        // assuming that AU and AV are COl major.
        int64_t ldaV = av_m;

        int64_t a_m = au_m;
        int64_t a_n = av_n;
        int64_t lda = a_m;
        int64_t arank = au_n;


        size_t au_size = au_m * au_n;
        size_t av_size = av_m * av_n;

        float *au_input = new float[au_size];
        float *av_input = new float[av_size];

        rowMajorToColumnMajor<float>((float *) matrix_AU, au_n, au_m, au_input);
        rowMajorToColumnMajor<float>((float *) matrix_AV, av_n, av_m, av_input);

        float *a_input = new float[au_size + av_size];
        memcpy((void *) a_input, au_input, au_size * sizeof(float));
        memcpy((void *) &a_input[au_size], av_input, av_size * sizeof(float));

        CompressedTile<float> compressed_tile_A(a_m, a_n, (float *) a_input, lda, arank,
                                                std::numeric_limits<blas::real_type<float>>::epsilon());

        REQUIRE(compressed_tile_A.GetNumberOfMatrices() == 2);
        REQUIRE(compressed_tile_A.GetTileStride(0) == au_m);
        REQUIRE(compressed_tile_A.GetTileStride(1) == av_m);
        REQUIRE(compressed_tile_A.GetTileSubMatrix(0).get().GetNumOfRows() == au_m);
        REQUIRE(compressed_tile_A.GetTileSubMatrix(0).get().GetNumOfCols() == au_n);
        REQUIRE(compressed_tile_A.GetTileSubMatrix(0).get().GetLeadingDim() == ldaU);
        REQUIRE(compressed_tile_A.GetTileSubMatrix(1).get().GetNumOfRows() == av_m);
        REQUIRE(compressed_tile_A.GetTileSubMatrix(1).get().GetNumOfCols() == av_n);
        REQUIRE(compressed_tile_A.GetTileSubMatrix(1).get().GetLeadingDim() == ldaV);
        REQUIRE(compressed_tile_A.layout() == blas::Layout::ColMajor);

        float *au_output = new float[au_size];
        float *av_output = new float[av_size];

        columnMajorToRowMajor<float>(compressed_tile_A.GetTileSubMatrix(0).get().GetData(),
                                     compressed_tile_A.GetTileSubMatrix(0).get().GetNumOfCols(),
                                     compressed_tile_A.GetTileSubMatrix(0).get().GetNumOfRows(),
                                     (float *) au_output);
        columnMajorToRowMajor<float>(compressed_tile_A.GetTileSubMatrix(1).get().GetData(),
                                     compressed_tile_A.GetTileSubMatrix(1).get().GetNumOfCols(),
                                     compressed_tile_A.GetTileSubMatrix(1).get().GetNumOfRows(),
                                     (float *) av_output);

        std::cout << "AUDATA \n";
        int index = 0;
        for (int i = 0; i < au_m; i++) {
            std::cout << "{ ";
            for (int j = 0; j < au_n; j++) {
                REQUIRE(au_output[index] == Approx(matrix_AU[i][j]).epsilon(1e-2));
                std::cout << au_output[index] << ", ";
                index++;
            }
            std::cout << "} \n";
        }

        std::cout << "AVDATA \n";
        index = 0;
        for (int i = 0; i < av_m; i++) {
            std::cout << "{ ";
            for (int j = 0; j < av_n; j++) {
                REQUIRE(av_output[index] == Approx(matrix_AV[i][j]).epsilon(1e-2));
                std::cout << av_output[index] << ", ";
                index++;
            }
            std::cout << "} \n";
        }

        delete[] a_input;
    }


    SECTION("Dense Tile Gemm") {
        std::cout << "Dense tile Gemm functionality-\n =========================== \n";

        float matrix_A[3][3] = {{1, 4, 7},
                                {2, 5, 8},
                                {3, 6, 9}};

        float matrix_B[3][2] = {{2, 8},
                                {4, 10},
                                {6, 12}};

        float matrix_C[3][2] = {{60, 132},
                                {72, 162},
                                {84, 192}};

        float matrix_CU[3][2] = {{0.467057, 0.784341},
                                 {0.57107,  0.0849248},
                                 {0.675083, -0.614489}};

        float matrix_CV[2][2] = {{125.847, 283.781},
                                 {1.55803, -0.690935}};

        float matrix_C_Input[3][2] = {{0, 0},
                                      {0, 0},
                                      {0, 0}};

        float alpha = 1;
        float beta = 1;

        // A num of rows
        int64_t a_m = 3;
        // A num of cols
        int64_t a_n = 3;
        // B num of rows
        int64_t b_m = 3;
        // B num of cols
        int64_t b_n = 2;
        // C num of rows
        int64_t c_m = 3;
        // C num of cols
        int64_t c_n = 2;

        int64_t cu_m = 3;
        int64_t cu_n = 1;
        int64_t cv_m = 1;
        int64_t cv_n = 2;

        // assuming that A, B , and C are COl major.
        int64_t lda = a_m;
        int64_t ldb = b_m;
        int64_t ldc = c_m;
        int64_t ldcu = cu_m;
        int64_t ldcv = cv_m;

        int64_t c_rank = 1;

        size_t a_size = a_m * a_n;
        size_t b_size = b_m * b_n;
        size_t c_size = c_m * c_n;
        size_t cu_size = cu_m * cu_n;
        size_t cv_size = cv_m * cv_n;

        float *a_input = new float[a_size];
        float *b_input = new float[b_size];
        float *c_input = new float[cu_size + cv_size];

        rowMajorToColumnMajor<float>((float *) matrix_A, a_n, a_m, a_input);
        rowMajorToColumnMajor<float>((float *) matrix_B, b_n, b_m, b_input);
        rowMajorToColumnMajor<float>((float *) matrix_C_Input, c_n, c_m, c_input);

        DenseTile<float> dense_tile_A(a_m, a_n, (float *) a_input, lda);
        DenseTile<float> dense_tile_B(b_m, b_n, (float *) b_input, ldb);
        CompressedTile<float> compressed_tile_C(c_m, c_n, (float *) c_input, ldc, c_rank,
                                                std::numeric_limits<blas::real_type<float>>::epsilon());

        REQUIRE(compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfRows() == cu_m);
        REQUIRE(compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfCols() == cu_n);
        REQUIRE(compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfRows() == cv_m);
        REQUIRE(compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfCols() == cv_n);

        hcorepp::helpers::SvdHelpers helpers;
        compressed_tile_C.Gemm(alpha, dense_tile_A.GetTileSubMatrix(0).get(), dense_tile_A.operation(),
                               dense_tile_B.GetTileSubMatrix(0).get(), dense_tile_B.operation(), beta,
                               dense_tile_A.GetTileSubMatrix(0).get().GetLeadingDim(),
                               dense_tile_A.GetTileSubMatrix(0).get().GetNumOfCols(), helpers);

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

        std::cout << "CUDATA \n";
        int index = 0;
        for (int i = 0; i < cu_m; i++) {
            std::cout << "{ ";
            for (int j = 0; j < cu_n; j++) {
                REQUIRE(cu_output_row[index] == Approx(matrix_CU[i][j]).epsilon(1e-2));
                std::cout << cu_output_row[index] << ", ";
                index++;
            }
            std::cout << "} \n";
        }

        std::cout << "CVDATA \n";
        index = 0;
        for (int i = 0; i < cv_m; i++) {
            std::cout << "{ ";
            for (int j = 0; j < cv_n; j++) {
                REQUIRE(cv_output_row[index] == Approx(matrix_CV[i][j]).epsilon(1e-2));
                std::cout << cv_output_row[index] << ", ";
                index++;
            }
            std::cout << "} \n";
        }

        delete[] a_input;
        delete[] b_input;
        delete[] c_input;
        delete[] cu_output_row;
        delete[] cv_output_row;
    }

}

TEST_CASE("CompressedTileTest", "[Compressed]") {
    TEST_Compressed();
}
