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


template<typename T>
void TEST_Compressed() {

    SECTION("Compressed Tile Creation") {
        std::cout << "Compressed tile Creation functionality-\n =========================== \n";

        T matrix_AU[5][4] = {{162, 348, 534, 720},
                             {174, 378, 582, 786},
                             {186, 408, 630, 852},
                             {198, 438, 678, 918},
                             {210, 468, 726, 984}};

        T matrix_AV[4][4] = {{2,  4,  6,  8},
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

        T *au_input = new T[au_size];
        T *av_input = new T[av_size];

        rowMajorToColumnMajor<T>((T *) matrix_AU, au_n, au_m, au_input);
        rowMajorToColumnMajor<T>((T *) matrix_AV, av_n, av_m, av_input);

        T *a_input = new T[au_size + av_size];
        memcpy((void *) a_input, au_input, au_size * sizeof(T));
        memcpy((void *) &a_input[au_size], av_input, av_size * sizeof(T));

        CompressedTile<T> compressed_tile_A(a_m, a_n, (T *) a_input, lda, arank,
                                            std::numeric_limits<blas::real_type<T>>::epsilon());

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

        T *host_data_array_au = new T[au_size];
        T *host_data_array_av = new T[av_size];

#ifdef USE_CUDA
        cudaMemcpy((void *) host_data_array_au, (void *) compressed_tile_A.GetTileSubMatrix(0).get().GetData(),
                   au_size * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy((void *) host_data_array_av, (void *) compressed_tile_A.GetTileSubMatrix(1).get().GetData(),
                   av_size * sizeof(T), cudaMemcpyDeviceToHost);
#else
        memcpy(host_data_array_au, compressed_tile_A.GetTileSubMatrix(0).get().GetData(),au_size * sizeof(T));
        memcpy(host_data_array_av, compressed_tile_A.GetTileSubMatrix(1).get().GetData(),av_size * sizeof(T));
#endif

        T *au_output = new T[au_size];
        T *av_output = new T[av_size];

        columnMajorToRowMajor<T>(host_data_array_au,
                                 compressed_tile_A.GetTileSubMatrix(0).get().GetNumOfCols(),
                                 compressed_tile_A.GetTileSubMatrix(0).get().GetNumOfRows(),
                                 (T *) au_output);
        columnMajorToRowMajor<T>(host_data_array_av,
                                 compressed_tile_A.GetTileSubMatrix(1).get().GetNumOfCols(),
                                 compressed_tile_A.GetTileSubMatrix(1).get().GetNumOfRows(),
                                 (T *) av_output);

        //        std::cout << "AU Output \n";
        validateOutput(au_output, au_m, au_n, (T *) matrix_AU);
//        printMatrix(au_output, au_m, au_n);


        //        std::cout << "AV Output \n";
        validateOutput(av_output, av_m, av_n, (T *) matrix_AV);
//        printMatrix(av_output, av_m, av_n);

        delete[] a_input;
        delete[] au_input;
        delete[] av_input;
        delete[] au_output;
        delete[] av_output;
        delete[] host_data_array_au;
        delete[] host_data_array_av;
    }


//    SECTION("Compressed Tile Gemm") {
//        std::cout << "Compressed tile Gemm functionality-\n =========================== \n";
//
//        T matrix_A[3][3] = {{1, 4, 7},
//                            {2, 5, 8},
//                            {3, 6, 9}};
//
//        T matrix_B[3][2] = {{2, 8},
//                            {4, 10},
//                            {6, 12}};
//
//        T matrix_C[3][2] = {{60, 132},
//                            {72, 162},
//                            {84, 192}};
//
//        T matrix_CU[3][2] = {{0.467057, 0.784341},
//                             {0.57107,  0.0849248},
//                             {0.675083, -0.614489}};
//
//        T matrix_CV[2][2] = {{125.847, 283.781},
//                             {1.55803, -0.690935}};
//
//        T output_matrix[3][2] = {{60, 132},
//                                 {72, 162},
//                                 {84, 192}};
//
//        T matrix_C_Input[3][2] = {{0, 0},
//                                  {0, 0},
//                                  {0, 0}};
//
//        T alpha = 1;
//        T beta = 1;
//
//        // A num of rows
//        int64_t a_m = 3;
//        // A num of cols
//        int64_t a_n = 3;
//        // B num of rows
//        int64_t b_m = 3;
//        // B num of cols
//        int64_t b_n = 2;
//        // C num of rows
//        int64_t c_m = 3;
//        // C num of cols
//        int64_t c_n = 2;
//
//        int64_t cu_m = 3;
//        int64_t cu_n = 1;
//        int64_t cv_m = 1;
//        int64_t cv_n = 2;
//
//        // assuming that A, B , and C are COl major.
//        int64_t lda = a_m;
//        int64_t ldb = b_m;
//        int64_t ldc = c_m;
//        int64_t ldcu = cu_m;
//        int64_t ldcv = cv_m;
//
//        int64_t c_rank = 1;
//
//        size_t a_size = a_m * a_n;
//        size_t b_size = b_m * b_n;
//        size_t c_size = c_m * c_n;
//        size_t cu_size = cu_m * cu_n;
//        size_t cv_size = cv_m * cv_n;
//
//        T *a_input = new T[a_size];
//        T *b_input = new T[b_size];
//        T *c_input = new T[cu_size + cv_size];
//
//
//        rowMajorToColumnMajor<T>((T *) matrix_A, a_n, a_m, a_input);
//        rowMajorToColumnMajor<T>((T *) matrix_B, b_n, b_m, b_input);
//
//        T *cu_input_R = (T *) malloc((cu_size) * sizeof(T));
//        T *cv_input_R = (T *) malloc((cv_size) * sizeof(T));
//
//        T *cu_input_C = (T *) malloc((cu_size) * sizeof(T));
//        T *cv_input_C = (T *) malloc((cv_size) * sizeof(T));
//
//        memset(cu_input_R, 0, (cu_size) * sizeof(T));
//        memset(cv_input_R, 0, (cv_size) * sizeof(T));
//
//        rowMajorToColumnMajor<T>((T *) cu_input_R, cu_n, cu_m, cu_input_C);
//        rowMajorToColumnMajor<T>((T *) cv_input_R, cv_n, cv_m, cv_input_C);
//
//        memcpy(c_input, cu_input_C, cu_size * sizeof(T));
//        memcpy(&c_input[cu_size], cv_input_C, cv_size * sizeof(T));
//
//        DenseTile<T> dense_tile_A(a_m, a_n, (T *) a_input, lda);
//        DenseTile<T> dense_tile_B(b_m, b_n, (T *) b_input, ldb);
//        CompressedTile<T> compressed_tile_C(c_m, c_n, (T *) c_input, ldc, c_rank,
//                                            std::numeric_limits<blas::real_type<T>>::epsilon());
//
//        REQUIRE(compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfRows() == cu_m);
//        REQUIRE(compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfCols() == cu_n);
//        REQUIRE(compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfRows() == cv_m);
//        REQUIRE(compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfCols() == cv_n);
//
//        hcorepp::helpers::SvdHelpers helpers;
//        compressed_tile_C.Gemm(alpha, dense_tile_A.GetTileSubMatrix(0).get(), dense_tile_A.operation(),
//                               dense_tile_B.GetTileSubMatrix(0).get(), dense_tile_B.operation(), beta,
//                               dense_tile_A.GetTileSubMatrix(0).get().GetLeadingDim(),
//                               dense_tile_A.GetTileSubMatrix(0).get().GetNumOfCols(), helpers);
//
//        T *cu_output = compressed_tile_C.GetTileSubMatrix(0).get().GetData();
//        T *cv_output = compressed_tile_C.GetTileSubMatrix(1).get().GetData();
//
//        cu_m = compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfRows();
//        cu_n = compressed_tile_C.GetTileSubMatrix(0).get().GetNumOfCols();
//        cv_m = compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfRows();
//        cv_n = compressed_tile_C.GetTileSubMatrix(1).get().GetNumOfCols();
//
//        T *cu_output_row = new T[cu_n * cu_m];
//        T *cv_output_row = new T[cv_n * cv_m];
//
//        columnMajorToRowMajor<T>(cu_output, cu_n, cu_m, (T *) cu_output_row);
//        columnMajorToRowMajor<T>(cv_output, cv_n, cv_m, (T *) cv_output_row);
//
////        std::cout << "CU Output \n";
////        validateCompressedOutput(cu_output_row, cu_m, cu_n, (T *) matrix_CU,
////                                 cv_output_row, cv_m, cv_n, (T *) matrix_CV);
////        printMatrix(cu_output_row, cu_m, cu_n);
//
////        std::cout << "CV Output \n";
////        validateCompressedOutput(cv_output_row, cv_m, cv_n, (T *) matrix_CV);
////        printMatrix(cv_output_row, cv_m, cv_n);
//
//        T *cu_cv = new T[c_m * c_n];
//
//        blas::gemm(blas::Layout::RowMajor, blas::Op::NoTrans, blas::Op::NoTrans,
//                   c_m, c_n, cu_n, alpha, cu_output_row, cu_n, cv_output_row, cv_n,
//                   0, cu_cv, c_n);
//
////        std::cout << " C Output ==== \n";
//        validateOutput(cu_cv, c_m, c_n, (T *) output_matrix);
////        printMatrix(cu_cv, c_m, c_n);
//
//        delete[] a_input;
//        delete[] b_input;
//        delete[] c_input;
//        delete[] cu_output_row;
//        delete[] cv_output_row;
//        delete[] cu_cv;
//        free(cu_input_R);
//        free(cv_input_R);
//        free(cu_input_C);
//        free(cv_input_C);
//
//    }

}

TEMPLATE_TEST_CASE("CompressedTileTest", "[Compressed]", float, double) {
    TEST_Compressed<TestType>();
}