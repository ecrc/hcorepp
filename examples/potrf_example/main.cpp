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
    int64_t rows = 3;
    int64_t cols = 3;
    int ld_C = rows;

    auto &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();
    hcorepp::dataunits::MemoryHandler<double> &memoryHandler = hcorepp::dataunits::MemoryHandler<double>::GetInstance();

    double *C_matrix = hcorepp::memory::AllocateArray<double>(rows * cols, context);
    hcorepp::memory::Memset(C_matrix, 0, rows * rows, context);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j <= i; j++) {
            C_matrix[i + j * ld_C] = rand() % 10;
            C_matrix[i * ld_C + j] = C_matrix[i + j * ld_C];
        }
    }
//    C_matrix[0] = 25;
//    C_matrix[1] = 15;
//    C_matrix[2] = 5;
//    C_matrix[3] = 15;
//    C_matrix[4] = 13;
//    C_matrix[5] = 11;
//    C_matrix[6] = 5;
//    C_matrix[7] = 11;
//    C_matrix[8] = 21;

    printf("==== Printing C Matrix before POTRF ==== \n ============================= \n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            printf(" C_in[%d] = %lf", i + j * ld_C, C_matrix[i + j * ld_C]);
        }
        printf("\n");
    }
    printf("==== DONE! Printing C Matrix before POTRF ==== \n ============================= \n");

    DenseTile<double> tile_c = DenseTile<double>(rows, rows, C_matrix, rows, blas::Layout::ColMajor, context);

    size_t flops = 0;

    hcorepp::api::HCore<double>::Potrf(tile_c, blas::Uplo::Lower, context, flops, memoryHandler.GetMemoryUnit());

    auto *output = tile_c.GetDataHolder().get().GetData();

    printf("==== Printing Final C Matrix after POTRF ==== \n ============================= \n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++) {
            printf(" C_out[%d] = %lf", i + j * ld_C, output[i + j * ld_C]);
        }
        printf("\n");
    }
    printf("==== DONE! Printing Final C Matrix after POTRF ==== \n ============================= \n");

    hcorepp::memory::DestroyArray(C_matrix, context);
}
