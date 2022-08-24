#include <libraries/catch/catch.hpp>
#include <iostream>

#include <hcorepp/api/hcorepp.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/data-units/DataHolder.hpp>
#include <hcorepp/test-helpers/testHelpers.hpp>
#include <cstring>

using namespace std;
using namespace hcorepp::dataunits;
using namespace hcorepp::operators;
using namespace hcorepp::api;
using namespace hcorepp::test_helpers;

template<typename T>
void TEST_DENSE() {

    SECTION("Dense Tile Creation") {
        std::cout << "Dense tile Creation functionality-\n =========================== \n";

        T matrix_A[3][4] = {{1, 5, 9,  13},
                                {2, 6, 10, 14},
                                {3, 7, 11, 15}};
        // A num of rows
        int64_t a_m = 3;
        // A num of cols
        int64_t a_n = 4;
        // assuming that A, B , and C are COl major.
        int64_t lda = a_m;

        size_t a_size = a_m * a_n;

        T *a_input = new T[a_size];

        rowMajorToColumnMajor<T>((T *) matrix_A, a_n, a_m, a_input);

        DenseTile<T> dense_tile_A(a_m, a_n, (T *) a_input, lda);

        REQUIRE(dense_tile_A.GetNumberOfMatrices() == 1);
        REQUIRE(dense_tile_A.GetTileStride(0) == a_m);
        REQUIRE(dense_tile_A.GetTileSubMatrix(0).get().GetNumOfRows() == a_m);
        REQUIRE(dense_tile_A.GetTileSubMatrix(0).get().GetNumOfCols() == a_n);
        REQUIRE(dense_tile_A.GetTileSubMatrix(0).get().GetLeadingDim() == lda);
        REQUIRE(dense_tile_A.layout() == blas::Layout::ColMajor);

        memset(a_input, 0, a_size * sizeof(T));

        columnMajorToRowMajor<T>(dense_tile_A.GetTileSubMatrix(0).get().GetData(),
                                     dense_tile_A.GetTileSubMatrix(0).get().GetNumOfCols(),
                                     dense_tile_A.GetTileSubMatrix(0).get().GetNumOfRows(),
                                     (T *) a_input);

//        std::cout << "CDATA \n";
        validateOutput(a_input, a_m, a_n, (T*) matrix_A);
//        printMatrix(a_input, a_m, a_n);


        delete[] a_input;
    }

    SECTION("Dense Tile Gemm") {
        std::cout << "Dense tile Gemm functionality-\n =========================== \n";

        T matrix_A[3][4] = {{1, 5, 9,  13},
                                {2, 6, 10, 14},
                                {3, 7, 11, 15}};

        T matrix_B[4][5] = {{2, 10, 18, 26, 34},
                                {4, 12, 20, 28, 36},
                                {6, 14, 22, 30, 38},
                                {8, 16, 24, 32, 40}};

        T matrix_C[3][5] = {{180, 404, 628, 852,  1076},
                                {200, 456, 712, 968,  1224},
                                {220, 508, 796, 1084, 1372}};

        T matrix_C_Row_Major[3][5] = {{0, 0, 0, 0, 0},
                                          {0, 0, 0, 0, 0},
                                          {0, 0, 0, 0, 0}};

        T alpha = 1;
        T beta = 1;

        // A num of rows
        int64_t a_m = 3;
        // A num of cols
        int64_t a_n = 4;
        // B num of rows
        int64_t b_m = 4;
        // B num of cols
        int64_t b_n = 5;
        // C num of rows
        int64_t c_m = 3;
        // C num of cols
        int64_t c_n = 5;

        // assuming that A, B , and C are COl major.
        int64_t lda = a_m;
        int64_t ldb = b_m;
        int64_t ldc = c_m;

        size_t a_size = a_m * a_n;
        size_t b_size = b_m * b_n;
        size_t c_size = c_m * c_n;

        T *a_input = new T[a_size];
        T *b_input = new T[b_size];
        T *c_input = new T[c_size];

        rowMajorToColumnMajor<T>((T *) matrix_A, a_n, a_m, a_input);
        rowMajorToColumnMajor<T>((T *) matrix_B, b_n, b_m, b_input);
        rowMajorToColumnMajor<T>((T *) matrix_C_Row_Major, c_n, c_m, c_input);

        DenseTile<T> dense_tile_A(a_m, a_n, (T *) a_input, lda);
        DenseTile<T> dense_tile_B(b_m, b_n, (T *) b_input, ldb);
        DenseTile<T> dense_tile_C(c_m, c_n, (T *) c_input, ldc);

        hcorepp::helpers::SvdHelpers helpers;
        int64_t ark = 1;

        dense_tile_C.Gemm(alpha, dense_tile_A.GetTileSubMatrix(0).get(), dense_tile_A.operation(),
                          dense_tile_B.GetTileSubMatrix(0).get(), dense_tile_B.operation(), beta,
                          lda, ark, helpers);

        T *c_output = dense_tile_C.GetTileSubMatrix(0).get().GetData();
        REQUIRE(dense_tile_C.GetTileSubMatrix(0).get().GetNumOfRows() == c_m);
        REQUIRE(dense_tile_C.GetTileSubMatrix(0).get().GetNumOfCols() == c_n);

        columnMajorToRowMajor<T>(c_output, c_n, c_m, (T *) matrix_C_Row_Major);

        auto c_pointer = (T *) matrix_C_Row_Major;
//        std::cout << "CDATA \n";
        validateOutput(c_pointer, c_m, c_n, (T*) matrix_C);
//        printMatrix(c_pointer, c_m, c_n);

        delete[] a_input;
        delete[] b_input;
        delete[] c_input;
    }
}

TEMPLATE_TEST_CASE("DenseTileTest", "[Dense]",float, double) {
    TEST_DENSE<TestType>();
}
