/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <iostream>
#include <cstring>

#include <catch2/catch_all.hpp>

#include <hcorepp/test-helpers/testHelpers.hpp>
#include <hcorepp/kernels/kernels.hpp>
#include <hcorepp/kernels/ContextManager.hpp>


using namespace std;

using namespace blas;

using namespace hcorepp::common;
using namespace hcorepp::kernels;
using namespace hcorepp::memory;
using namespace hcorepp::operators;
using namespace hcorepp::test_helpers;

template<typename T>
void TEST_KERNELS() {

    hcorepp::kernels::RunContext &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();
    SECTION("Gemm Kernel Test") {

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

        auto a_dev = AllocateArray<T>(a_size * sizeof(T), context);
        auto b_dev = AllocateArray<T>(b_size * sizeof(T), context);;
        auto c_dev = AllocateArray<T>(c_size * sizeof(T), context);;

        auto a_input = new T[a_size];
        auto b_input = new T[b_size];
        auto c_input = new T[c_size];


        rowMajorToColumnMajor<T>((T *) matrix_A, a_n, a_m, a_input);
        rowMajorToColumnMajor<T>((T *) matrix_B, b_n, b_m, b_input);
        rowMajorToColumnMajor<T>((T *) matrix_C_Row_Major, c_n, c_m, c_input);

        Memcpy<T>(a_dev, a_input, a_size, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(b_dev, b_input, b_size, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(c_dev, c_input, c_size, context, MemoryTransfer::HOST_TO_DEVICE);
        context.Sync();


        CompressionParameters helpers(1e-9);

        HCoreKernels<T>::Gemm(Layout::ColMajor, Op::NoTrans, Op::NoTrans,
                              a_m, b_n, a_n, alpha, a_dev, lda, b_dev, ldb, beta, c_dev, ldc,
                              context);


        T *host_data_array = new T[c_m * c_n];
        Memcpy<T>(host_data_array, c_dev,
                  c_m * c_n, context,
                  MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();


        columnMajorToRowMajor<T>(host_data_array, c_n, c_m, (T *) matrix_C_Row_Major);

        auto c_pointer = (T *) matrix_C_Row_Major;

        validateOutput(c_pointer, c_m, c_n, (T *) matrix_C);

        DestroyArray(a_dev, context);
        DestroyArray(b_dev, context);
        DestroyArray(c_dev, context);
        delete[] host_data_array;
        delete[] a_input;
        delete[] b_input;
        delete[] c_input;
    }

    SECTION("MultiplyByAlpha Kernel") {

        T matrix_A[3][4] = {{1, 5, 9,  13},
                            {2, 6, 10, 14},
                            {3, 7, 11, 15}};
        T matrix_C[3][4] = {{2, 10, 18, 26},
                            {4, 12, 20, 28},
                            {6, 14, 22, 30}};

        T alpha = 2;

        // A num of rows
        int64_t a_m = 3;
        // A num of cols
        int64_t a_n = 4;


        CompressionParameters helpers(1e-9);
        auto a_pointer = AllocateArray<T>(a_m * a_n * sizeof(T), context);
        Memcpy(a_pointer, (T *) matrix_A, a_m * a_n, context, MemoryTransfer::HOST_TO_DEVICE);

        HCoreKernels<T>::MultiplyByAlpha(a_pointer, a_m, a_n, 0, 1, alpha, context);
        context.Sync();

        auto c_pointer = AllocateArray<T>(a_m * a_n * sizeof(T), context);
        Memcpy(c_pointer, (T *) matrix_C, a_m * a_n, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy((T *) matrix_A, (T *) a_pointer, a_m * a_n, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        validateOutput((T *) matrix_A, a_n, a_m, (T *) matrix_C);

        DestroyArray(a_pointer, context);
        DestroyArray(c_pointer, context);
    }

    SECTION("ProcessVpointer Kernel") {

        int64_t n_cv = 2;
        int64_t rank_c = 2;
        int64_t rank_a = 3;
        T beta = 1;
        bool ungqr = false;

        T cv[2 * 2] = {1, 2, 3, 4};
        T b_data[2 * 3] = {5, 6, 7, 8, 9, 10};
        T v[2 * 2 + 2 * 3] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        T expected[2 * 2 + 2 * 3] = {1, 3, 2, 4, 5, 8, 6, 9, 7, 10};

        auto cv_input = AllocateArray<T>(n_cv * rank_c, context);
        auto b_data_input = AllocateArray<T>(rank_c * rank_a, context);
        auto v_input = AllocateArray<T>(rank_c * rank_a + n_cv * rank_c, context);

        Memcpy(cv_input, cv, n_cv * rank_c, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy(b_data_input, b_data, rank_a * rank_c, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy(v_input, v, rank_c * rank_a + n_cv * rank_c, context, MemoryTransfer::HOST_TO_DEVICE);

        HCoreKernels<T>::ProcessVpointer(n_cv, rank_c, ungqr, n_cv, beta, cv_input, n_cv, v_input, rank_a, b_data_input,
                                         context);
        context.Sync();

        Memcpy(v, v_input, rank_c * rank_a + n_cv * rank_c, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        for (int i = 0; i < 10; i++) {
            REQUIRE(v[i] == expected[i]);
        }

        DestroyArray(cv_input, context);
        DestroyArray(b_data_input, context);
        DestroyArray(v_input, context);
    }

    SECTION("CalculateNewRank kernel aTruncatedSVD = false") {

        size_t new_rank = 0;
        bool truncated_SVD = false;
        int64_t size_sigma = 5;
        real_type<T> accuracy = 10;

        real_type<T> sigma[5] = {accuracy + 1, accuracy + 2, accuracy, accuracy - 1, accuracy - 2};
        auto sigma_input = AllocateArray<T>(size_sigma, context);
        Memcpy(sigma_input, sigma, size_sigma, context, MemoryTransfer::HOST_TO_DEVICE);

        HCoreKernels<T>::CalculateNewRank(new_rank, truncated_SVD, sigma_input, size_sigma, accuracy, context);
        context.Sync();
        size_t result = 0;
        Memcpy(&result, &new_rank, 1, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();
        REQUIRE(result == 3);


        DestroyArray(sigma_input, context);
    }

    SECTION("CalculateNewRank kernel aTruncatedSVD = true") {

        size_t new_rank = 0;
        bool truncated_SVD = true;
        int64_t size_sigma = 5;
        real_type<T> accuracy = 10;
        real_type<T> sigma_0 = 2;

        real_type<T> sigma[5] = {sigma_0, (accuracy * sigma_0) + 2, (accuracy * sigma_0), (accuracy * sigma_0) - 1,
                                 (accuracy * sigma_0) - 2};
        auto sigma_input = AllocateArray<T>(size_sigma, context);
        Memcpy(sigma_input, sigma, size_sigma, context, MemoryTransfer::HOST_TO_DEVICE);
        context.Sync();
        HCoreKernels<T>::CalculateNewRank(new_rank, truncated_SVD, sigma_input, size_sigma, accuracy, context);
        context.Sync();
        size_t result = 0;
        Memcpy(&result, &new_rank, 1, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();
        REQUIRE(result == 3);

        DestroyArray(sigma_input, context);
    }

    SECTION("CalculateUVptr kernel") {
        ::int64_t vm = 2;
        ::int64_t rank = 3;

        T v_new[] = {1, 2, 3, 4, 5, 6};
        T uv[] = {0, 0, 0, 0, 0, 0,};
        T expected[] = {1, 3, 5, 2, 4, 6};

        auto v_new_input = AllocateArray<T>(vm * rank, context);
        auto uv_input = AllocateArray<T>(vm * rank, context);

        Memcpy(v_new_input, v_new, vm * rank, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy(uv_input, uv, vm * rank, context, MemoryTransfer::HOST_TO_DEVICE);

        HCoreKernels<T>::CalculateUVptr(rank, vm, uv_input, v_new_input, context);
        context.Sync();

        Memcpy(uv, uv_input, vm * rank, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        for (int i = 0; i < vm * rank; i++) {
            REQUIRE(uv[i] == expected[i]);
        }

        DestroyArray(uv_input, context);
        DestroyArray(v_new_input, context);
    }

    SECTION("CalculateVTnew kernel aUngqr = false") {

        bool ungqr = false;
        ::int64_t vm = 4;
        ::int64_t vn = 3;
        int64_t size_sigma = 3;
        ::int64_t new_rank = 3;

        T vt[3 * 4] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3};
        T vt_new[3 * 4] = {2, 6, 0, 8, 15, 0, 14, 24, 0, 2, 6, 0};
        real_type<T> sigma[3] = {2, 3, 0};

        auto vt_input = AllocateArray<T>(vm * vn, context);
        auto sigma_input = AllocateArray<T>(size_sigma, context);

        Memcpy(vt_input, vt, vm * vn, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy(sigma_input, sigma, size_sigma, context, MemoryTransfer::HOST_TO_DEVICE);

        HCoreKernels<T>::CalculateVTnew(new_rank, ungqr, min(vm, vn), sigma_input, vt_input, size_sigma, vm, context);
        context.Sync();

        Memcpy(vt, vt_input, vm * vn, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        for (int i = 0; i < vm * vn; i++) {
            REQUIRE(vt[i] == vt_new[i]);
        }

        DestroyArray(vt_input, context);
        DestroyArray(sigma_input, context);
    }

    SECTION("CalculateVTnew kernel aUngqr = true") {
        bool ungqr = true;
        int64_t vm = 4;
        int64_t vn = 3;
        int64_t size_sigma = min(vm, vn);
        real_type<T> sigma[3] = {2, 3, 0};
        int64_t new_rank = 3;

        T vt[3 * 4] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3};
        T vt_new[3 * 4] = {2, 6, 0, 8, 15, 0, 14, 24, 0, 1, 2, 3};

        auto vt_input = AllocateArray<T>(vm * vn, context);
        auto sigma_input = AllocateArray<T>(size_sigma, context);

        Memcpy(vt_input, vt, vm * vn, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy(sigma_input, sigma, size_sigma, context, MemoryTransfer::HOST_TO_DEVICE);
        context.Sync();

        HCoreKernels<T>::CalculateVTnew(new_rank, ungqr, min(vm, vn), sigma_input, vt_input, size_sigma, vm, context);
        context.Sync();

        Memcpy(vt, vt_input, vm * vn, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        for (int i = 0; i < vm * vn; i++) {
            REQUIRE(vt[i] == vt_new[i]);
        }

        DestroyArray(vt_input, context);
        DestroyArray(sigma_input, context);
    }

    SECTION("CalculateUVptrConj kernel") {

        int64_t rank = 2;
        int64_t vm = 3;

        T uv[] = {1, 2, 3, 4, 5, 6};
        T expected[] = {1, 2, 3, 4, 5, 6};

        auto uv_input = AllocateArray<T>(vm * rank, context);
        Memcpy(uv_input, uv, rank * vm, context, MemoryTransfer::HOST_TO_DEVICE);

        HCoreKernels<T>::CalculateUVptrConj(rank, vm, uv_input, context);
        context.Sync();
        Memcpy(uv, uv_input, rank * vm, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        for (int i = 0; i < rank * vm; i++) {
            REQUIRE(uv[i] == expected[i]);
        }

        DestroyArray(uv_input, context);
    }

    SECTION("FillIdentityMatrix Kernel") {

        int dim = 3;

        T matrix_A[3][3] = {{0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0}};


        T matrix_output[3][3] = {{1, 0, 0},
                                 {0, 1, 0},
                                 {0, 0, 1}};

        CompressionParameters helpers(1e-9);

        auto a_input = AllocateArray<T>(dim * dim, context);
        Memcpy(a_input, (T *) matrix_A, dim * dim, context, MemoryTransfer::HOST_TO_DEVICE);

        HCoreKernels<T>::FillIdentityMatrix(dim, a_input, context);
        context.Sync();

        Memcpy((T *) matrix_A, (T *) a_input, dim * dim, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        validateOutput((T *) matrix_A, dim, dim, (T *) matrix_output);

        DestroyArray(a_input, context);
    }

    SECTION("LaCpy kernel general type") {

        MatrixType type = MatrixType::General;
        int64_t m = 4;
        int64_t n = 4;

        T matrix_a[4 * 4] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        T matrix_b[4 * 4];

        auto a_input = AllocateArray<T>(m * n, context);
        auto b_input = AllocateArray<T>(m * n, context);

        Memcpy(a_input, matrix_a, m * n, context, MemoryTransfer::HOST_TO_DEVICE);

        HCoreKernels<T>::LaCpy(type, m, n, a_input, m, b_input, m, context);
        context.Sync();

        Memcpy(matrix_a, a_input, m * n, context, MemoryTransfer::DEVICE_TO_HOST);
        Memcpy(matrix_b, b_input, m * n, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        for (int i = 0; i < m * n; i++) {
            REQUIRE(matrix_b[i] == matrix_a[i]);
        }

        DestroyArray(a_input, context);
        DestroyArray(b_input, context);

    }

    SECTION("LaCpy kernel upper type") {

        MatrixType type = MatrixType::Upper;
        int64_t m = 4;
        int64_t n = 4;

        T matrix_a[4 * 4] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        T matrix_b[4 * 4] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        T matrix_expected[4 * 4] = {1, 0, 0, 0, 5, 6, 0, 0, 9, 10, 11, 0, 13, 14, 15, 16};

        auto a_input = AllocateArray<T>(m * n, context);
        auto b_input = AllocateArray<T>(m * n, context);

        Memcpy(a_input, matrix_a, m * n, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy(b_input, matrix_b, m * n, context, MemoryTransfer::HOST_TO_DEVICE);

        HCoreKernels<T>::LaCpy(type, m, n, a_input, m, b_input, m, context);
        context.Sync();

        Memcpy(matrix_b, b_input, m * n, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();


        for (int i = 0; i < m * n; i++) {
            REQUIRE(matrix_b[i] == matrix_expected[i]);
        }

        DestroyArray(a_input, context);
        DestroyArray(b_input, context);
    }
//
    SECTION("LaCpy kernel lower type") {
        MatrixType type = MatrixType::Lower;
        int64_t m = 4;
        int64_t n = 4;

        T matrix_a[4 * 4] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        T matrix_b[4 * 4] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        T matrix_expected[4 * 4] = {1, 2, 3, 4, 0, 6, 7, 8, 0, 0, 11, 12, 0, 0, 0, 16};

        auto a_input = AllocateArray<T>(4 * 4, context);
        auto b_input = AllocateArray<T>(4 * 4, context);

        Memcpy(a_input, matrix_a, m * n, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy(b_input, matrix_b, m * n, context, MemoryTransfer::HOST_TO_DEVICE);
        context.Sync();

        HCoreKernels<T>::LaCpy(type, m, n, a_input, m, b_input, m, context);
        context.Sync();

        Memcpy(matrix_b, b_input, m * n, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        DestroyArray(a_input, context);
        DestroyArray(b_input, context);
    }

    SECTION("geqrf kernel") {
        T *workspace = nullptr;

        ::int64_t a_m = 2;
        ::int64_t a_n = 6;

        ::int64_t tau_m = 2;
        ::int64_t tau_n = 1;

        T matrix_A[2][6] = {{0, 2,  2,  0,   2,  2},
                            {2, -1, -1, 1.5, -1, -1}};

        T tau[2][1] = {{0},
                       {0}};

        T a_expected[2][6] = {{-4, 0.5, 0.5,      0,         0.5,      0.5},
                              {2,  2.5, 0.285714, -0.428571, 0.285714, 0.285714}};

        T tau_expected[2][1] = {{1},
                                {1.4}};

        auto a_input = AllocateArray<T>(a_m * a_n, context);
        auto tau_input = AllocateArray<T>(tau_m * tau_n, context);

        Memcpy(a_input, (T *) matrix_A, a_m * a_n, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy(tau_input, (T *) tau, tau_m * tau_n, context, MemoryTransfer::HOST_TO_DEVICE);
        context.Sync();

        //since it is column major pass dimensions of A.T
        HCoreKernels<T>::Geqrf(a_n, a_m, a_input, a_n, tau_input, workspace, 0, 0, context);
        context.Sync();

        Memcpy((T *) matrix_A, (T *) a_input, a_m * a_n, context, MemoryTransfer::DEVICE_TO_HOST);
        Memcpy((T *) tau, (T *) tau_input, tau_m * tau_n, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        validateOutput((T *) matrix_A, a_m, a_n, (T *) a_expected);
        validateOutput((T *) tau, tau_m, tau_n, (T *) tau_expected);

        DestroyArray(a_input, context);
        DestroyArray(tau_input, context);
    }

    SECTION("Laset kernel general type") {

        MatrixType type = MatrixType::General;
        int64_t m = 4;
        int64_t n = 4;
        T off_diag = 1;
        T diag = 2;

        T matrix_a[4 * 4] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        auto a_input = AllocateArray<T>(4 * 4, context);
        Memcpy(a_input, matrix_a, m * n, context, MemoryTransfer::HOST_TO_DEVICE);
        context.Sync();

        T matrix_expected[4 * 4] = {2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2};

        HCoreKernels<T>::Laset(type, m, n, off_diag, diag, a_input, m, context);
        context.Sync();

        auto host_a_input = new T[4 * 4];
        Memcpy<T>(host_a_input, a_input, 4 * 4, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();


        for (int i = 0; i < m * n; i++) {
            REQUIRE(host_a_input[i] == matrix_expected[i]);
        }

        DestroyArray(a_input, context);
        delete[] host_a_input;
    }

    SECTION("Laset kernel upper type") {
        MatrixType type = MatrixType::Upper;
        int64_t m = 4;
        int64_t n = 4;
        T off_diag = 1;
        T diag = 2;

        T matrix_a[4 * 4] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        auto array_a = AllocateArray<T>(4 * 4, context);
        Memcpy(array_a, matrix_a, m * n, context, MemoryTransfer::HOST_TO_DEVICE);
        context.Sync();

        T matrix_expected[4 * 4] = {2, 0, 0, 0, 1, 2, 0, 0, 1, 1, 2, 0, 1, 1, 1, 2};

        HCoreKernels<T>::Laset(type, m, n, off_diag, diag, array_a, m, context);
        context.Sync();


        auto host_a_input = new T[4 * 4];
        Memcpy<T>(host_a_input, array_a, 4 * 4, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();


        for (int i = 0; i < m * n; i++) {
            REQUIRE(host_a_input[i] == matrix_expected[i]);
        }

        DestroyArray(array_a, context);
        delete[] host_a_input;
    }

    SECTION("Laset kernel lower type") {

        MatrixType type = MatrixType::Lower;
        int64_t m = 4;
        int64_t n = 4;
        T off_diag = 1;
        T diag = 2;

        T matrix_a[4 * 4] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        auto a_input = AllocateArray<T>(4 * 4, context);
        Memcpy(a_input, matrix_a, m * n, context, MemoryTransfer::HOST_TO_DEVICE);

        context.Sync();
        T matrix_expected[4 * 4] = {2, 1, 1, 1, 0, 2, 1, 1, 0, 0, 2, 1, 0, 0, 0, 2};

        HCoreKernels<T>::Laset(type, m, n, off_diag, diag, a_input, m, context);
        context.Sync();

        auto host_a_input = new T[4 * 4];
        Memcpy<T>(host_a_input, a_input, 4 * 4, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();


        for (int i = 0; i < m * n; i++) {
            REQUIRE(host_a_input[i] == matrix_expected[i]);
        }

        DestroyArray(a_input, context);
        delete[] host_a_input;
    }

    SECTION("Trmm kernel") {

        T matrix_A[4][4] = {{1, 0, 0,  0},
                            {2, 6, 0,  0},
                            {3, 7, 11, 0},
                            {4, 8, 12, 16}};

        T matrix_B[4][4] = {{0, 8,  16, 24},
                            {2, 10, 18, 26},
                            {4, 12, 20, 28},
                            {6, 14, 22, 30}};

        T matrix_output[4][4] = {{0,   8,   16,  24},
                                 {12,  76,  140, 204},
                                 {58,  226, 394, 564},
                                 {160, 480, 800, 1120}};

        T alpha = 1;

        CompressionParameters helpers(1e-9);

        auto a_input = AllocateArray<T>(4 * 4, context);
        auto b_input = AllocateArray<T>(4 * 4, context);

        Memcpy(a_input, (T *) matrix_A, 4 * 4, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy(b_input, (T *) matrix_B, 4 * 4, context, MemoryTransfer::HOST_TO_DEVICE);
        context.Sync();


        HCoreKernels<T>::Trmm(Layout::ColMajor, Side::Right, Uplo::Upper,
                              Op::NoTrans, Diag::NonUnit, 4, 4, alpha, a_input, 4, b_input, 4,
                              context);
        context.Sync();

        auto host_b_input = new T[4 * 4];
        Memcpy<T>(host_b_input, b_input, 4 * 4, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        validateOutput(host_b_input, 4, 4, (T *) matrix_output);

        DestroyArray(a_input, context);
        DestroyArray(b_input, context);

        delete[] host_b_input;
    }

    SECTION("Trmm Unit kernel") {
        T matrix_A[4][4] = {{1, 0, 0,  0},
                            {2, 1, 0,  0},
                            {3, 7, 1,  0},
                            {4, 8, 12, 1}};

        T matrix_B[4][4] = {{0, 8,  16, 24},
                            {2, 10, 18, 26},
                            {4, 12, 20, 28},
                            {6, 14, 22, 30}};

        T matrix_output[4][4] = {{0,  8,   16,  24},
                                 {2,  26,  50,  74},
                                 {18, 106, 194, 282},
                                 {70, 270, 470, 670}};

        T alpha = 1;

        CompressionParameters helpers(1e-9);
        auto a_input = AllocateArray<T>(4 * 4, context);
        auto b_input = AllocateArray<T>(4 * 4, context);

        Memcpy(a_input, (T *) matrix_A, 4 * 4, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy(b_input, (T *) matrix_B, 4 * 4, context, MemoryTransfer::HOST_TO_DEVICE);
        context.Sync();


        HCoreKernels<T>::Trmm(Layout::ColMajor, Side::Right, Uplo::Upper,
                              Op::NoTrans, Diag::Unit, 4, 4, alpha, a_input, 4, b_input, 4,
                              context);
        context.Sync();

        auto host_b_input = new T[4 * 4];
        Memcpy<T>(host_b_input, b_input, 4 * 4, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();


        validateOutput(host_b_input, 4, 4, (T *) matrix_output);

        DestroyArray(a_input, context);
        DestroyArray(b_input, context);
    }

//         Verify A = U D V.T
    SECTION("SVD CompressionType LAPACK_GESVD  Kernel") {
        T matrix_A[3][3] = {{1, 2, 3},
                            {4, 5, 6},
                            {7, 8, 9}};

        T matrix_U[3][3] = {{0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0}};

        T matrix_V[3][3] = {{0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0}};
        //size of matrix_S = Number of non-singular values of matrix A (Rank of matrix A)
        //Rank of matrix A = 2 (size of the largest non-zero determent)
        T matrix_S[3] = {0, 0};

        int64_t m = 3;
        int64_t n = 3;

        int64_t lda = 3;
        int64_t ldu = 3;
        int64_t ldv = 3;

        int size_a = m * n;
        int size_b = size_a;
        int size_c = size_a;


        CompressionParameters helpers(1e-9);
        T *workspace;
        size_t workspace_size = 0;

        auto a_dev = AllocateArray<T>(size_a * sizeof(T), context);
        auto u_dev = AllocateArray<T>(size_b * sizeof(T), context);
        auto v_dev = AllocateArray<T>(size_c * sizeof(T), context);
        auto s_dev = AllocateArray<T>(3 * sizeof(T), context);

        auto a_pointer = new T[size_a];
        auto u_pointer = new T[size_b];
        auto v_pointer = new T[size_c];
        auto s_pointer = new T[3];

        columnMajorToRowMajor<T>((T *) matrix_A, n, m, a_pointer);
        columnMajorToRowMajor<T>((T *) matrix_U, n, m, u_pointer);
        columnMajorToRowMajor<T>((T *) matrix_V, n, m, v_pointer);
        columnMajorToRowMajor<T>((T *) matrix_S, 3, 1, s_pointer);

        Memcpy<T>(a_dev, a_pointer, size_a, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(u_dev, u_pointer, size_b, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(v_dev, v_pointer, size_c, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(s_dev, s_pointer, 3, context, MemoryTransfer::HOST_TO_DEVICE);

        context.Sync();

        HCoreKernels<T>::SVD(Job::AllVec, Job::AllVec, m, n,
                             a_dev, lda, s_dev, u_dev, ldu, v_dev, ldv,
                             CompressionType::LAPACK_GESVD, workspace, workspace_size, workspace_size, context);
        context.Sync();

        auto u_pointer_row = new T[size_b];
        auto v_pointer_trans = new T[size_c];

        auto host_u_pointer = new T[size_b];
        auto host_s_pointer = new T[3];
        Memcpy<T>(host_u_pointer, u_dev, size_b, context, MemoryTransfer::DEVICE_TO_HOST);
        Memcpy<T>(host_s_pointer, s_dev, 3, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        //U is row major
        columnMajorToRowMajor<T>(host_u_pointer, n, m, u_pointer_row);

        //V is transposed

        auto host_v_pointer = new T[size_c];

        Memcpy<T>(host_v_pointer, v_dev, size_c, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();
        rowMajorToColumnMajor<T>(host_v_pointer, n, m, v_pointer_trans);

        context.Sync();


        //singular values in s_pointer are sorted desc
        //First singular value
        T sigma_1 = host_s_pointer[0];

        //Second singular value
        T sigma_2 = host_s_pointer[1];

        //Diagonal matrix D with singular values on the diagonal
        T matrix_sigma[3][3] = {{sigma_1, 0,       0},
                                {0,       sigma_2, 0},
                                {0,       0,       0}};

        T *sigma_pointer = (T *) matrix_sigma;

        //Store result of U * D
        T matrix_U_Sigma[3][3] = {{0, 0, 0},
                                  {0, 0, 0},
                                  {0, 0, 0}};

        //Store result of U * D * V.T
        T result[3][3] = {{0, 0, 0},
                          {0, 0, 0},
                          {0, 0, 0}};

        auto U_sigma_pointer = (T *) matrix_U_Sigma;
        auto result_pointer = (T *) result;
        T alpha = 1;
        T beta = 1;


        auto u_pointer_row_dev = AllocateArray<T>(m * n * sizeof(T), context);
        auto sigma_pointer_dev = AllocateArray<T>(m * n * sizeof(T), context);
        auto U_sigma_pointer_dev = AllocateArray<T>(m * n * sizeof(T), context);

        Memcpy<T>(u_pointer_row_dev, u_pointer_row, m * n, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(sigma_pointer_dev, (T *) matrix_sigma, m * n, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(U_sigma_pointer_dev, (T *) matrix_U_Sigma, m * n, context, MemoryTransfer::HOST_TO_DEVICE);

        context.Sync();

        // Perform U * D
        HCoreKernels<T>::Gemm(Layout::RowMajor, Op::NoTrans, Op::NoTrans,
                              3, 3, 3, alpha, u_pointer_row_dev, 3, sigma_pointer_dev, 3, beta,
                              U_sigma_pointer_dev, 3,
                              context);
        context.Sync();

        auto U_sigma_host = new T[9];
        Memcpy<T>(U_sigma_host, U_sigma_pointer_dev, m * n, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();


        auto v_pointer_trans_dev = AllocateArray<T>(m * n * sizeof(T), context);
        auto result_dev = AllocateArray<T>(m * n * sizeof(T), context);

        Memcpy<T>(v_pointer_trans_dev, v_pointer_trans, size_c, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(result_dev, result_pointer, m * n, context, MemoryTransfer::HOST_TO_DEVICE);


        auto U_sigma_device_new = new T[m * n];
        Memcpy<T>(U_sigma_device_new, U_sigma_host, m * n, context, MemoryTransfer::HOST_TO_DEVICE);

        context.Sync();
//
        //Perform U * D * V.T
        HCoreKernels<T>::Gemm(Layout::RowMajor, Op::NoTrans, Op::NoTrans,
                              3, 3, 3, alpha, U_sigma_pointer_dev, 3, v_pointer_trans_dev, 3, beta,
                              result_dev, 3,
                              context);
        context.Sync();

        auto host_result = new T[m * n];
        Memcpy<T>(host_result, result_dev, m * n, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        validateOutput(host_result, m, n, (T *) matrix_A);

        DestroyArray(result_dev, context);
        DestroyArray(s_dev, context);
        DestroyArray(v_dev, context);
        DestroyArray(sigma_pointer_dev, context);
        DestroyArray(a_dev, context);
        DestroyArray(v_pointer_trans_dev, context);
        DestroyArray(u_dev, context);
        DestroyArray(u_pointer_row_dev, context);

        delete[] a_pointer;
        delete[] v_pointer;
        delete[] u_pointer;
        delete[] s_pointer;
        delete[] u_pointer_row;
        delete[] v_pointer_trans;
    }

    SECTION("SVD CompressionType LAPACK_GESSD") {
        T matrix_A[3][3] = {{1, 2, 3},
                            {4, 5, 6},
                            {7, 8, 9}};

        T matrix_U[3][3] = {{0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0}};

        T matrix_V[3][3] = {{0, 0, 0},
                            {0, 0, 0},
                            {0, 0, 0}};

        //size of matrix_S = Number of non-singular values of matrix A (Rank of matrix A)
        //Rank of matrix A = 2 (size of the largest non-zero determent)
        T matrix_S[3] = {0, 0};

        int64_t m = 3;
        int64_t n = 3;

        int64_t lda = 3;
        int64_t ldu = 3;
        int64_t ldv = 3;

        int size_a = m * n;
        int size_b = size_a;
        int size_c = size_a;

        CompressionParameters helpers(1e-9);
        T *workspace;
        size_t workspace_size = 0;

        auto a_pointer = new T[size_a];
        auto u_pointer = new T[size_b];
        auto v_pointer = new T[size_c];
        auto s_pointer = new T[3];

        columnMajorToRowMajor<T>((T *) matrix_A, n, m, a_pointer);
        columnMajorToRowMajor<T>((T *) matrix_U, n, m, u_pointer);
        columnMajorToRowMajor<T>((T *) matrix_V, n, m, v_pointer);
        columnMajorToRowMajor<T>((T *) matrix_S, lda, 1, s_pointer);

        auto a_dev = AllocateArray<T>(size_a * sizeof(T), context);
        auto u_dev = AllocateArray<T>(size_b * sizeof(T), context);
        auto v_dev = AllocateArray<T>(size_c * sizeof(T), context);
        auto s_dev = AllocateArray<T>(lda * sizeof(T), context);

        Memcpy<T>(a_dev, a_pointer, size_a, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(u_dev, u_pointer, size_b, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(v_dev, v_pointer, size_c, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(s_dev, s_pointer, lda, context, MemoryTransfer::HOST_TO_DEVICE);

        context.Sync();

        HCoreKernels<T>::SVD(Job::AllVec, Job::AllVec, m, n,
                             a_dev, lda, s_dev, u_dev, ldu, v_dev, ldv,
                             CompressionType::LAPACK_GESDD, workspace, workspace_size, workspace_size, context);
        context.Sync();


        auto u_pointer_row = new T[size_b];
        auto v_pointer_trans = new T[size_c];

        auto host_u_pointer = new T[size_b];
        auto host_s_pointer = new T[lda];
        Memcpy<T>(host_u_pointer, u_dev, size_b, context, MemoryTransfer::DEVICE_TO_HOST);
        Memcpy<T>(host_s_pointer, s_dev, lda, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        //U is row major
        columnMajorToRowMajor<T>(host_u_pointer, n, m, u_pointer_row);

        //V is transposed

        auto host_v_pointer = new T[size_c];
        Memcpy<T>(host_v_pointer, v_dev, size_c, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();
        rowMajorToColumnMajor<T>(host_v_pointer, n, m, v_pointer_trans);

        context.Sync();


        //singular values in s_pointer are sorted desc
        //First singular value
        T sigma_1 = host_s_pointer[0];

        //Second singular value
        T sigma_2 = host_s_pointer[1];

        //Diagonal matrix D with singular values on the diagonal
        T matrix_sigma[3][3] = {{sigma_1, 0,       0},
                                {0,       sigma_2, 0},
                                {0,       0,       0}};

        T *sigma_pointer = (T *) matrix_sigma;

        //Store result of U * D
        T matrix_U_Sigma[3][3] = {{0, 0, 0},
                                  {0, 0, 0},
                                  {0, 0, 0}};

        //Store result of U * D * V.T
        T result[3][3] = {{0, 0, 0},
                          {0, 0, 0},
                          {0, 0, 0}};

        auto result_pointer = (T *) result;
        T alpha = 1;
        T beta = 1;


        auto u_pointer_row_dev = AllocateArray<T>(size_b * sizeof(T), context);
        auto sigma_pointer_dev = AllocateArray<T>(size_b * sizeof(T), context);
        auto U_sigma_pointer_dev = AllocateArray<T>(size_b * sizeof(T), context);

        Memcpy<T>(u_pointer_row_dev, u_pointer_row, size_b, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(sigma_pointer_dev, (T *) matrix_sigma, size_b, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(U_sigma_pointer_dev, (T *) matrix_U_Sigma, size_b, context, MemoryTransfer::HOST_TO_DEVICE);

        context.Sync();


        // Perform U * D
        HCoreKernels<T>::Gemm(Layout::RowMajor, Op::NoTrans, Op::NoTrans,
                              3, 3, 3, alpha, u_pointer_row_dev, 3, sigma_pointer_dev, 3, beta,
                              U_sigma_pointer_dev, 3,
                              context);
        context.Sync();

        auto U_sigma_host = new T[m * n];
        Memcpy<T>(U_sigma_host, U_sigma_pointer_dev, m * n, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();


        auto v_pointer_trans_dev = AllocateArray<T>(m * n * sizeof(T), context);
        auto result_dev = AllocateArray<T>(m * n * sizeof(T), context);

        Memcpy<T>(v_pointer_trans_dev, v_pointer_trans, size_c, context, MemoryTransfer::HOST_TO_DEVICE);
        Memcpy<T>(result_dev, result_pointer, m * n, context, MemoryTransfer::HOST_TO_DEVICE);
        context.Sync();


        auto U_sigma_device_new = new T[m * n];
        Memcpy<T>(U_sigma_device_new, U_sigma_host, m * n, context, MemoryTransfer::HOST_TO_DEVICE);

        context.Sync();

        //Perform U * D * V.T
        HCoreKernels<T>::Gemm(Layout::RowMajor, Op::NoTrans, Op::NoTrans,
                              3, 3, 3, alpha, U_sigma_pointer_dev, 3, v_pointer_trans_dev, 3, beta,
                              result_dev, 3,
                              context);
        context.Sync();

        auto host_result = new T[m * n];
        Memcpy<T>(host_result, result_dev, m * n, context, MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();

        validateOutput(host_result, m, n, (T *) matrix_A);

        DestroyArray(result_dev, context);
        DestroyArray(s_dev, context);
        DestroyArray(v_dev, context);
        DestroyArray(sigma_pointer_dev, context);
        DestroyArray(a_dev, context);
        DestroyArray(v_pointer_trans_dev, context);
        DestroyArray(u_dev, context);
        DestroyArray(u_pointer_row_dev, context);

        delete[] a_pointer;
        delete[] v_pointer;
        delete[] u_pointer;
        delete[] s_pointer;
        delete[] u_pointer_row;
        delete[] v_pointer_trans;
    }

    SECTION("Unmqr kernel") {
        ///PENDING ON COMPLEX SUPPORT
    }

    SECTION("Allocate sigma kernel") {
        real_type<T> sigma[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < 8; i++) {

            real_type<T> b = i;
            sigma[i] = b;
        }
        for (int i = 0; i < 8; i++) {

            REQUIRE(sigma[i] == real_type<T>(i));
        }
    }

    SECTION("Ungqr kernel") {
        ///PENDING ON COMPLEX SUPPORT
    }

    SECTION("Potrf") {

        T matrix_A[3][3] = {{25, 15, 5},
                            {15, 13, 11},
                            {5,  11, 21}};

        T matrix_Output[3][3] = {{5, 0, 0},
                                 {3, 2, 0},
                                 {1, 4, 2}};

        int64_t m = 3;
        int64_t n = 3;

        int64_t lda = 3;

        int size_a = m * n;

        T *workspace;
        size_t workspace_size = 0;

        auto a_pointer = new T[size_a];
        rowMajorToColumnMajor<T>((T *) matrix_A, n, m, a_pointer);

        auto a_dev = AllocateArray<T>(size_a, context);

        Memcpy<T>(a_dev, a_pointer, size_a, context, MemoryTransfer::HOST_TO_DEVICE);

        context.Sync();

        HCoreKernels<T>::potrf(Uplo::Lower, workspace, workspace_size, workspace_size, m, a_dev, lda,
                               blas::Layout::ColMajor, context);

        context.Sync();
        auto *output = new T[size_a];
        Memcpy<T>(output, a_dev, size_a, context, MemoryTransfer::DEVICE_TO_HOST);

        auto output_pointer = new T[size_a];
        rowMajorToColumnMajor<T>((T *) matrix_Output, n, m, output_pointer);

        context.Sync();

        for (int i = 0; i < size_a; i++) {
            std::cout << " Output :: " << output[i] << "\n";
        }


//        validateOutput(output, m, n, output_pointer);

        DestroyArray(a_dev, context);

        delete[]output;
        delete[]a_pointer;
        delete[]output_pointer;
    }


    SECTION("Trsm") {
        T matrix_A[3][3] = {{1, 2, 3},
                            {0, 1, 1},
                            {0, 0, 2}};
        T matrix_B[3][1] = {{8},
                            {4},
                            {2}};

        T matrix_output[3][1] = {{-1},
                                 {3},
                                 {1}};

        int64_t m = 3;
        int64_t n = 1;

        int64_t lda = 3;

        int size_a = m * m;
        int size_b = m * n;

        T *workspace;
        size_t workspace_size = 0;

        auto a_pointer = new T[size_a];
        rowMajorToColumnMajor<T>((T *) matrix_A, m, m, a_pointer);

        auto b_pointer = new T[size_b];
        rowMajorToColumnMajor<T>((T *) matrix_B, n, m, b_pointer);

        auto a_dev = AllocateArray<T>(size_a, context);
        Memcpy<T>(a_dev, a_pointer, size_a, context, MemoryTransfer::HOST_TO_DEVICE);

        auto b_dev = AllocateArray<T>(size_b, context);
        Memcpy<T>(b_dev, b_pointer, size_b, context, MemoryTransfer::HOST_TO_DEVICE);

        context.Sync();
        HCoreKernels<T>::trsm(blas::Layout::ColMajor, blas::Side::Left, blas::Uplo::Upper, blas::Op::NoTrans,
                              blas::Diag::NonUnit, m, n, 1, a_dev, lda, b_dev, lda, context);

        context.Sync();
        auto output = new T[size_b];
        Memcpy<T>(output, b_dev, size_b, context, MemoryTransfer::DEVICE_TO_HOST);

        auto output_pointer = new T[size_b];
        rowMajorToColumnMajor<T>((T *) matrix_output, n, m, output_pointer);

        context.Sync();

        validateOutput(output, m, n, output_pointer);

        DestroyArray(a_dev, context);
        DestroyArray(b_dev, context);

        delete[]output;
        delete[]a_pointer;
        delete[]b_pointer;
        delete[]output_pointer;
    }

    SECTION("CrossProduct") {

        T matrix_A[4][4] = {{3.12393,   -1.16854, -0.304408, -2.15901},
                            {-1.16854,  1.86968,  1.04094,   1.35925},
                            {-0.304408, 1.04094,  4.43374,   1.21072},
                            {-2.15901,  1.35925,  1.21072,   5.57265}};

        T matrix_B[4][4] = {{3.12393,   -1.16854, -0.304408, -2.15901},
                            {-1.16854,  1.86968,  1.04094,   1.35925},
                            {-0.304408, 1.04094,  4.43374,   1.21072},
                            {-2.15901,  1.35925,  1.21072,   5.57265}};

        T matrix_output[4][4] = {{15.878412787064, -1.16854,      -0.304408,       -2.15901},
                                 {-9.08673783542,  7.7923056801,  1.04094,         1.35925},
                                 {-6.13095182416,  8.56286609912, 22.300113620064, 1.21072},
                                 {-20.73289403456, 13.8991634747, 14.18705411188,  39.0291556835}};

        int64_t m = 4;
        int64_t n = 4;

        int64_t lda = 4;

        int size_a = m * n;
        int size_b = m * n;

        T *workspace;
        size_t workspace_size = 0;

        auto a_pointer = new T[size_a];
        rowMajorToColumnMajor<T>((T *) matrix_A, n, m, a_pointer);

        auto b_pointer = new T[size_b];
        rowMajorToColumnMajor<T>((T *) matrix_B, n, m, b_pointer);

        auto a_dev = AllocateArray<T>(size_a, context);
        Memcpy<T>(a_dev, a_pointer, size_a, context, MemoryTransfer::HOST_TO_DEVICE);

        auto b_dev = AllocateArray<T>(size_b, context);
        Memcpy<T>(b_dev, b_pointer, size_b, context, MemoryTransfer::HOST_TO_DEVICE);

        context.Sync();

        HCoreKernels<T>::syrk(blas::Layout::ColMajor, blas::Uplo::Lower, blas::Op::NoTrans, m, n, 1, a_dev, lda, 0,
                              b_dev, lda, context);

        context.Sync();
        auto output = new T[size_b];
        Memcpy<T>(output, b_dev, size_b, context, MemoryTransfer::DEVICE_TO_HOST);

        auto output_pointer = new T[size_b];
        rowMajorToColumnMajor<T>((T *) matrix_output, n, m, output_pointer);

        context.Sync();

        validateOutput(output, m, n, output_pointer);

        DestroyArray(a_dev, context);
        DestroyArray(b_dev, context);

        delete[]output;
        delete[]a_pointer;
        delete[]b_pointer;
        delete[]output_pointer;
    }

}

TEMPLATE_TEST_CASE("Test Kernels", "", float, double) {
    TEST_KERNELS<TestType>();
    ContextManager::DestroyInstance();
}
