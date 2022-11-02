/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */
#include <limits>
#include <hcorepp/api/HCore.hpp>
#include <hcorepp/helpers/RawMatrix.hpp>
#include <hcorepp/helpers/TileMatrix.hpp>
#include <hcorepp/helpers/Timer.hpp>

using namespace std::chrono;
using namespace hcorepp::operators;
using namespace hcorepp::helpers;

int main(int argc, char *argv[]) {
    // single tile dimensions.
    int tile_size = 512;
    // parameters needed for matrix multiplication driver to operate correctly.
    double alpha = 1;
    double beta = 1;
    blas::Op trans_a = blas::Op::NoTrans;
    blas::Op trans_b = blas::Op::NoTrans;
    // parameters for matrix generation.
    int64_t mode = 0;
    blas::real_type<double> cond = std::numeric_limits<double>::epsilon();
    // Target accuracy.
    std::vector<double> accuracy_list = {1e-4};
    // Assuming square matrix, default tile matrix is 2 x 2 tiles.
    int matrix_tiles = 2;
    // Parse optional arguments from command line.
    if (argc > 1) {
        matrix_tiles = atoi(argv[1]);
        if (argc > 2) {
            accuracy_list.clear();
            std::string acc_str = argv[2];
            std::stringstream ss(acc_str);
            for (double i; ss >> i;) {
                accuracy_list.push_back(i);
                if (ss.peek() == ',')
                    ss.ignore();
            }
            if (argc > 3) {
                tile_size = atoi(argv[3]);
            }
        }
    }
    // Check for verbosity
    bool print_header = false;
    {
        const char *val = std::getenv("HCOREPP_VERBOSE");
        if (val != nullptr) { // invalid to assign nullptr to std::string
            std::string value = val;
            if (value == "ON") {
                print_header = true;
            }
        }
    }
    // matrix dimensions (number of tiles)
    int a_mt = matrix_tiles;
    int a_nt = matrix_tiles;
    int b_mt = a_nt;
    int b_nt = matrix_tiles;
    int c_mt = a_mt;
    int c_nt = b_nt;
    int row_tile_size = tile_size;
    int column_tile_size = tile_size;

    int64_t iseed[4] = {0, 0, 0, 1};
    // Create full matrices with automatic generation.
    Timer timer;
    RawMatrix<double> full_a(a_mt * row_tile_size, a_nt * column_tile_size, iseed, mode, cond);
    RawMatrix<double> full_b(b_mt * row_tile_size, b_nt * column_tile_size, iseed, mode, cond);
    RawMatrix<double> full_c(c_mt * row_tile_size, c_nt * column_tile_size, iseed, mode, cond);
    auto initial_c = full_c.Clone();
    timer.Snapshot("generation");
    // Solve reference solution
    blas::gemm(blas::Layout::ColMajor, trans_a, trans_b, full_c.GetM(),
               full_c.GetN(), full_a.GetN(), alpha, full_a.GetData(),
               full_a.GetM(), full_b.GetData(),
               full_b.GetM(), beta, full_c.GetData(), full_c.GetM());
    timer.Snapshot("ref_gemm");
    // Get memory footprint in KB
    size_t ref_memory_footprint = (full_a.GetMemoryFootprint() + full_b.GetMemoryFootprint()
                                   + full_c.GetMemoryFootprint()) / 1024;
    // Norm for error calculations
    blas::real_type<double> a_norm = full_a.Norm();
    blas::real_type<double> b_norm = full_b.Norm();
    blas::real_type<double> c_norm = full_c.Norm();
    size_t dense_memory_footprint;
    double dense_error;
    // Dense Flow
    {
        timer.StartSnapshot();
        // Create dense tile matrix
        TileMatrix<double> a_dense(full_a, row_tile_size, column_tile_size);
        TileMatrix<double> b_dense(full_b, row_tile_size, column_tile_size);
        TileMatrix<double> c_dense(initial_c, row_tile_size, column_tile_size);
        timer.Snapshot("dense_creation");
        // Do matrix multiplication.
        for (int i = 0; i < c_nt; i++) {
            for (int j = 0; j < c_mt; j++) {
                auto dense_c_tile = c_dense.GetTile(j, i);
                for (int k = 0; k < b_mt; k++) {
                    auto dense_a_tile = a_dense.GetTile(j, k);
                    auto dense_b_tile = b_dense.GetTile(k, i);
                    // Pure Dense HCORE multiplication.
                    timer.StartSnapshot();
                    hcorepp::api::HCore<double>::Gemm(alpha, *dense_a_tile, trans_a, *dense_b_tile,
                                                      trans_b, beta, *dense_c_tile);
                    timer.Snapshot("dense_gemm");
                }
            }
        }
        // Retrieve results back from tile format for verification.
        timer.StartSnapshot();
        auto full_dense_c = c_dense.ToRawMatrix();
        full_dense_c.ReferenceDifference(full_c);
        dense_error = full_dense_c.Norm() /
                      ((a_norm + b_norm + c_norm) * std::max(full_dense_c.GetM(), full_dense_c.GetN()) *
                       std::numeric_limits<double>::epsilon());
        timer.Snapshot("dense_error_calc");
        // Error checking.
        if (dense_error >= 10) {
            std::cout << "Example didn't pass, dense HCORE++ error > 10 " << std::endl;
        }
        // Get memory footprint in KB
        dense_memory_footprint = (a_dense.GetMemoryFootprint() + b_dense.GetMemoryFootprint()
                                  + c_dense.GetMemoryFootprint()) / 1024;
    }
    // Compressed flow
    bool first_print = true;
    for (auto &accuracy : accuracy_list) {
        //Reset all compression timers
        timer.ResetSnapshot("comp_creation");
        timer.ResetSnapshot("comp_gemm");
        timer.ResetSnapshot("comp_error_calc");
        // Create compressed tiles matrix
        TileMatrix<double> a_comp(full_a, row_tile_size, column_tile_size, accuracy);
        TileMatrix<double> b_comp(full_b, row_tile_size, column_tile_size, accuracy);
        TileMatrix<double> c_comp(initial_c, row_tile_size, column_tile_size, accuracy);
        timer.Snapshot("comp_creation");
        // Do matrix multiplication.
        SVDParameters svd_parameters(accuracy);
        for (int i = 0; i < c_nt; i++) {
            for (int j = 0; j < c_mt; j++) {
                auto comp_c_tile = c_comp.GetTile(j, i);
                for (int k = 0; k < b_mt; k++) {
                    auto comp_a_tile = a_comp.GetTile(j, k);
                    auto comp_b_tile = b_comp.GetTile(k, i);
                    timer.StartSnapshot();
                    // Pure Compressed HCORE multiplication.
                    hcorepp::api::HCore<double>::Gemm(alpha, *comp_a_tile, trans_a, *comp_b_tile,
                                                      trans_b, beta, *comp_c_tile, svd_parameters);
                    timer.Snapshot("comp_gemm");
                }
            }
        }
        // Retrieve results back from tile format for verification.
        timer.StartSnapshot();
        auto full_approximate_c = c_comp.ToRawMatrix();
        // Calculate compressed tile matrix reference error
        full_approximate_c.ReferenceDifference(full_c);
        double comp_error = full_approximate_c.Norm() /
                            ((a_norm + b_norm + c_norm) *
                             std::max(full_approximate_c.GetM(), full_approximate_c.GetN()) *
                             accuracy);
        timer.Snapshot("comp_error_calc");
        // Error checking.
        if (comp_error >= 10) {
            std::cout << "Example didn't pass, compressed HCORE++ error > 10 " << std::endl;
        }
        // Get memory footprint in KB
        size_t compressed_memory_footprint = (a_comp.GetMemoryFootprint() + b_comp.GetMemoryFootprint()
                                              + c_comp.GetMemoryFootprint()) / 1024;
        // Print results
        if (first_print) {
            if (print_header) {
                printf("tile_count, tile_size, matrix_size, type, error, memory(KB), creation(ms), gemm_time(ms)\n");
                print_header = false;
            }
            printf("%d, %d, %d, ref, 0, %zu, %f, %f\n",
                   matrix_tiles, tile_size, matrix_tiles * tile_size,
                   ref_memory_footprint, timer.GetSnapshot("generation"),
                   timer.GetSnapshot("ref_gemm"));
            printf("%d, %d, %d, dense, %e, %zu, %f, %f\n",
                   matrix_tiles, tile_size, matrix_tiles * tile_size, dense_error,
                   ref_memory_footprint, timer.GetSnapshot("dense_creation"),
                   timer.GetSnapshot("dense_gemm"));
            first_print = false;
        }
        printf("%d, %d, %d, %2.1e, %e, %zu, %f, %f\n",
               matrix_tiles, tile_size, matrix_tiles * tile_size, accuracy, comp_error,
               compressed_memory_footprint, timer.GetSnapshot("comp_creation"),
               timer.GetSnapshot("comp_gemm"));
    }
    return 0;
}
