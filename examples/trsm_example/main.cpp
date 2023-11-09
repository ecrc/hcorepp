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
    int64_t rows_a = 3;
    int64_t cols_a = 3;
    int64_t rows_b = 3;
    int64_t cols_b = 1;
    int64_t k = 1;


    int ld_a = rows_a;
    int ld_b = rows_b;

    auto &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();
    hcorepp::dataunits::MemoryHandler<double> &memoryHandler = hcorepp::dataunits::MemoryHandler<double>::GetInstance();

    double *A_matrix = hcorepp::memory::AllocateArray<double>(rows_a * cols_a, context);
    hcorepp::memory::Memset(A_matrix, 0, rows_a * cols_a, context);

    double *B_matrix = hcorepp::memory::AllocateArray<double>(rows_b * cols_b, context);
    hcorepp::memory::Memset(B_matrix, 0, rows_b * cols_b, context);
    A_matrix[0] = 1;
    A_matrix[1] = 0;
    A_matrix[2] = 0;
    A_matrix[3] = 2;
    A_matrix[4] = 1;
    A_matrix[5] = 0;
    A_matrix[6] = 3;
    A_matrix[7] = 1;
    A_matrix[8] = 2;

    B_matrix[0] = 8;
    B_matrix[1] = 4;
    B_matrix[2] = 2;

    printf("==== Printing A Matrix before TRSM ==== \n ============================= \n");
    for (int i = 0; i < rows_a; i++) {
        for (int j = 0; j < cols_a; j++) {
            printf(" A_in[%d] = %lf", i + j * ld_a, A_matrix[i + j * ld_a]);
        }
        printf("\n");
    }
    printf("==== DONE! Printing A Matrix before TRSM ==== \n ============================= \n");

    printf("==== Printing B Matrix before TRSM ==== \n ============================= \n");
    for (int i = 0; i < rows_b; i++) {
        for (int j = 0; j < cols_b; j++) {
            printf(" B_in[%d] = %lf", i + j * ld_b, B_matrix[i + j * ld_b]);
        }
        printf("\n");
    }
    printf("==== DONE! Printing B Matrix before TRSM ==== \n ============================= \n");

    DenseTile<double> tile_a = DenseTile<double>(rows_a, cols_a, A_matrix, ld_a, blas::Layout::ColMajor, context);
}
