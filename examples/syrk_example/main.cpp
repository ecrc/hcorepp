/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include "blas/flops.hh"
#include <hcorepp/api/HCore.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/helpers/MatrixHelpers.hpp>
#include <hcorepp/kernels/memory.hpp>
#include <iostream>

using namespace std::chrono;
using namespace hcorepp::operators;
using namespace hcorepp::helpers::matrixhelpers;

int main(int argc, char *argv[]) {
    int64_t rows = 5;
    int64_t cols = 5;
    int64_t k = 4;
    int ld_C = rows;
    int ld_AU = rows;
    int ld_AV = rows;
    blas::Op trans_a = blas::Op::NoTrans;
    blas::Op trans_b = blas::Op::Trans;

    double alpha = 1;
    double beta = 1;

    auto &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();
    hcorepp::dataunits::MemoryHandler<double> &memoryHandler = hcorepp::dataunits::MemoryHandler<double>::GetInstance();

    double *C_matrix = hcorepp::memory::AllocateArray<double>(rows * rows, context);
    hcorepp::memory::Memset(C_matrix, 0, rows * rows, context);

    double *A_dense_matrix = hcorepp::memory::AllocateArray<double>(rows * rows, context);
    hcorepp::memory::Memset(A_dense_matrix, 0, rows * rows, context);

    double *AU_matrix = hcorepp::memory::AllocateArray<double>(rows * k, context);
    hcorepp::memory::Memset(AU_matrix, 0, rows * k, context);
    double *AV_matrix = hcorepp::memory::AllocateArray<double>(rows * k, context);
    hcorepp::memory::Memset(AV_matrix, 0, rows * k, context);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j <= i; j++) {
            C_matrix[i + j * ld_C] = rand() / (double) RAND_MAX;
        }
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < k; j++) {
            AU_matrix[i + j * ld_AU] = rand() / (double) RAND_MAX;
            AV_matrix[i + j * ld_AV] = rand() / (double) RAND_MAX;
        }
    }
    double *AUV_matrix = hcorepp::memory::AllocateArray<double>(rows * k * 2, context);

    hcorepp::memory::Memcpy(AUV_matrix, AU_matrix, rows * k, context, hcorepp::memory::MemoryTransfer::HOST_TO_HOST);
    hcorepp::memory::Memcpy(&AUV_matrix[rows * k], AV_matrix, rows * k, context,
                            hcorepp::memory::MemoryTransfer::HOST_TO_HOST);

    blas::gemm(blas::Layout::ColMajor, trans_a, trans_b, rows,
               rows, k, alpha, AU_matrix, rows, AV_matrix, rows, beta, A_dense_matrix, rows);

    DenseTile<double> tile_a_dense = DenseTile<double>(rows, rows, A_dense_matrix, rows, blas::Layout::ColMajor,
                                                       context);
    CompressedTile<double> tile_a_compressed = CompressedTile<double>(rows, rows, AUV_matrix, rows, k,
                                                                      blas::Layout::ColMajor, context);

    DenseTile<double> tile_c = DenseTile<double>(rows, rows, C_matrix, rows, blas::Layout::ColMajor, context);

    size_t flops = 0;

    hcorepp::api::HCore<double>::Syrk(alpha, tile_a_dense, trans_a, blas::Uplo::Lower, beta, tile_c, context,
                                      flops, memoryHandler.GetMemoryUnit());

    auto *output = tile_c.GetTileSubMatrix(0);

    printf("==== Printing Final Reference C Matrix ==== \n ============================= \n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            printf(" CRef[%d] = %lf", i + j * ld_C, output[i + j * ld_C]);
        }
        printf("\n");
    }
    printf("==== DONE! Printing Final Reference C Matrix ==== \n ============================= \n");

    hcorepp::memory::DestroyArray(C_matrix, context);
    hcorepp::memory::DestroyArray(A_dense_matrix, context);
    hcorepp::memory::DestroyArray(AU_matrix, context);
    hcorepp::memory::DestroyArray(AV_matrix, context);
    hcorepp::memory::DestroyArray(AUV_matrix, context);


}
