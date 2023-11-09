/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */
#include <limits>
#include <hcorepp/api/HCore.hpp>
#include <hcorepp/helpers/RawMatrix.hpp>
#include <hcorepp/helpers/TileMatrix.hpp>
#include <hcorepp/helpers/Timer.hpp>
#include <hcorepp/helpers/generators/concrete/LatmsGenerator.hpp>
#include <hcorepp/helpers/generators/concrete/TileLatmsGenerator.hpp>
#include <hcorepp/kernels/memory.hpp>
#include <hcorepp/kernels/kernels.hpp>
#include <hcorepp/helpers/DebuggingTimer.hpp>
#include <fstream>

using namespace std::chrono;
using namespace hcorepp::operators;
using namespace hcorepp::helpers;


/**
 * @brief
 * Do a full tile-matrix multiplication using HCore++ Gemm
 *
 * @tparam T
 * The datatype of each element.
 *
 * @param[in] aMatrixA
 * First Input matrix
 *
 * @param[in] aAOp
 * The operation to apply on A(whether to transpose or not)
 *
 * @param[in] aMatrixB
 * Second Input matrix
 *
 * @param[in] aBOp
 * The operation to apply on B(whether to transpose or not)
 *
 * @param[out] aMatrixC
 * The output matrix: C = A * B
 *
 * @param[in] aTimer
 * The timer object in case of snapshot.
 *
 * @param[in] aAlpha
 * Alpha parameter
 *
 * @param[in] aBeta
 * Beta parameter
 *
 * @param[in] aSnapshotName
 * The snapshot name to use.
 *
 * @param[in] aContext
 * The run context to execute the operations in.
 *
 * @param[in] aParameters
 * The SVD parameters utilized.
 */
template<typename T>
size_t tile_matrix_multiplication(TileMatrix<T> &aMatrixA,
                                  const blas::Op &aAOp,
                                  TileMatrix<T> &aMatrixB,
                                  const blas::Op &aBOp,
                                  TileMatrix<T> &aMatrixC,
                                  Timer &aTimer,
                                  T aAlpha, T aBeta,
                                  const std::string &aSnapshotName,
                                  hcorepp::kernels::RunContext &aContext,
                                  const CompressionParameters &aParameters = {1e-8}) {
    auto c_nt = aMatrixC.GetColTileCount();
    auto c_mt = aMatrixC.GetRowTileCount();
    auto b_mt = aMatrixB.GetRowTileCount();

    DebuggingTimer *timer = DebuggingTimer::GetDebuggingTimer();

    if (!aSnapshotName.empty()) {
        aTimer.StartSnapshot();
#ifdef USING_TIMER
        if (timer != nullptr) {
            timer->ResetAllSnapshots();
        }
#endif
    }
    timer->StartSnapshot("ParFixedRank::TileMatrixMul::AllocatingTempBuffers");
    hcorepp::dataunits::MemoryHandler<T>& memoryHandler = hcorepp::dataunits::MemoryHandler<T>::GetInstance();
    timer->Snapshot("ParFixedRank::TileMatrixMul::AllocatingTempBuffers");
    size_t flops = 0;
    for (int i = 0; i < c_nt; i++) {
        for (int j = 0; j < c_mt; j++) {
            auto c_tile = aMatrixC.GetTile(j, i);
            for (int k = 0; k < b_mt; k++) {
                auto a_tile = aMatrixA.GetTile(j, k);
                auto b_tile = aMatrixB.GetTile(k, i);
                hcorepp::api::HCore<T>::Gemm(aAlpha, *a_tile, aAOp, *b_tile,
                                             aBOp, aBeta, *c_tile, aContext,
                                             flops, memoryHandler.GetMemoryUnit(), aParameters);
                aContext.Sync();
            }
        }
    }

    if (!aSnapshotName.empty()) {
        aTimer.Snapshot(aSnapshotName);
#ifdef USING_TIMER
        std::stringstream ss;
        ss << aParameters.GetAccuracy();
        std::ofstream time_file(aSnapshotName + "_" + std::to_string(aMatrixC.GetN()) + "_" + ss.str() + "_" +
                                std::to_string(aParameters.GetFixedRank()) + ".time");

        timer->PrintAllSnapshots(time_file);
#endif
    }

    timer->StartSnapshot("ParFixedRank::TileMatrixMul::DestroyingTempBuffers");
    memoryHandler.FreeAllocations();
    timer->Snapshot("ParFixedRank::TileMatrixMul::DestroyingTempBuffers");

    return flops;
}

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
    blas::real_type<double> cond = 1;
    // Target accuracy.
    std::vector<double> accuracy_list = {1e-1, 1e-4, 1e-8};
    // Assuming square matrix, default tile matrix is 2 x 2 tiles.
    int matrix_tiles = 2;
    int per_tile_generation = 0;
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
                if (argc > 4) {
                    per_tile_generation = atoi(argv[4]);;
                }
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
#ifdef USE_SYCL
    std::string sycl_device_name;
    {
        const char *val = std::getenv("HCOREPP_DEVICE");
        if (val != nullptr) { // invalid to assign nullptr to std::string
            sycl_device_name = val;
        }
    }
    hcorepp::kernels::RunContext& context = hcorepp::kernels::ContextManager::GetInstance().GetContext();
    context.SetDevice(sycl_device_name);
    if (print_header) {
        context.Print();
    }
#else
    hcorepp::kernels::RunContext& context = hcorepp::kernels::ContextManager::GetInstance().GetContext();
#endif
    // matrix dimensions (number of tiles)

    int a_mt = matrix_tiles;
    int a_nt = matrix_tiles;
    int b_mt = a_nt;
    int b_nt = matrix_tiles;
    int c_mt = a_mt;
    int c_nt = b_nt;
    int row_tile_size = tile_size;
    int column_tile_size = tile_size;
    size_t ref_flops;
    size_t dense_flops;

    int64_t iseed[4] = {0, 0, 0, 1};
    // Create full matrices with automatic generation.
    Timer timer;
    generators::Generator<double> *generator;
    if (per_tile_generation > 0) {
        generator = new generators::TileLatmsGenerator<double>(iseed, mode, cond, tile_size);
    } else {
        generator = new generators::LatmsGenerator<double>(iseed, mode, cond);
    }
    RawMatrix<double> full_a(a_mt * row_tile_size, a_nt * column_tile_size, *generator);
    RawMatrix<double> full_b(b_mt * row_tile_size, b_nt * column_tile_size, *generator);
    RawMatrix<double> full_c(c_mt * row_tile_size, c_nt * column_tile_size);
    delete generator;
    auto initial_c = full_c.Clone();
    timer.Snapshot("generation");
    {
        auto warm_a = full_a.Clone();
        auto warm_b = full_b.Clone();
        auto warm_c = full_c.Clone();
        blas::gemm(blas::Layout::ColMajor, trans_a, trans_b, warm_c.GetM(),
                   warm_c.GetN(), warm_a.GetN(), alpha, warm_a.GetData(),
                   warm_a.GetM(), warm_b.GetData(),
                   warm_b.GetM(), beta, warm_c.GetData(), warm_c.GetM());
    }
    // Solve reference solution
    {
        auto a_device = hcorepp::memory::AllocateArray<double>(full_a.GetM() * full_a.GetN(), context);
        auto b_device = hcorepp::memory::AllocateArray<double>(full_b.GetM() * full_b.GetN(), context);
        auto c_device = hcorepp::memory::AllocateArray<double>(full_c.GetM() * full_c.GetN(), context);
        hcorepp::memory::Memcpy<double>(a_device, full_a.GetData(),
                                        full_a.GetM() * full_a.GetN(), context,
                                        hcorepp::memory::MemoryTransfer::HOST_TO_DEVICE);
        hcorepp::memory::Memcpy<double>(b_device, full_b.GetData(), full_b.GetM() * full_b.GetN(), context,
                                        hcorepp::memory::MemoryTransfer::HOST_TO_DEVICE);
        hcorepp::memory::Memcpy<double>(c_device, full_c.GetData(), full_c.GetM() * full_c.GetN(), context,
                                        hcorepp::memory::MemoryTransfer::HOST_TO_DEVICE);
        context.Sync();
        timer.StartSnapshot();
        hcorepp::kernels::HCoreKernels<double>::Gemm(blas::Layout::ColMajor, trans_a, trans_b, full_c.GetM(),
                                                     full_c.GetN(), full_a.GetN(), alpha, a_device,
                                                     full_a.GetM(), b_device,
                                                     full_b.GetM(), beta, c_device, full_c.GetM(), context);
        context.Sync();
        timer.Snapshot("ref_gemm");
        ref_flops = 2 * full_c.GetM() * full_c.GetN() * full_a.GetN();
        hcorepp::memory::Memcpy<double>(full_c.GetData(), c_device, full_c.GetM() * full_c.GetN(),
                                        context,
                                        hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();
        hcorepp::memory::DestroyArray(a_device, context);
        hcorepp::memory::DestroyArray(b_device, context);
        hcorepp::memory::DestroyArray(c_device, context);
    }
    // Get memory footprint in KB
    size_t ref_memory_footprint = (full_a.GetMemoryFootprint() + full_b.GetMemoryFootprint()
                                   + full_c.GetMemoryFootprint()) / 1024;
    // Norm for error calculations
    blas::real_type<double> a_norm = full_a.Norm();
    blas::real_type<double> b_norm = full_b.Norm();
    blas::real_type<double> c_init_norm = initial_c.Norm();
    size_t dense_memory_footprint;
    double dense_error;
    double dense_error_normalized;
    // Dense Warmup
    {

        TileMatrix<double> a_dense(full_a, row_tile_size, column_tile_size, context);
        TileMatrix<double> b_dense(full_b, row_tile_size, column_tile_size, context);
        TileMatrix<double> c_dense(initial_c, row_tile_size, column_tile_size, context);
        context.Sync();
        tile_matrix_multiplication(a_dense, trans_a, b_dense, trans_b, c_dense, timer, alpha,
                                   beta, "", context);
    }
    // Dense Flow
    {

        timer.StartSnapshot();
        // Create dense tile matrix
        TileMatrix<double> a_dense(full_a, row_tile_size, column_tile_size, context);
        TileMatrix<double> b_dense(full_b, row_tile_size, column_tile_size, context);
        TileMatrix<double> c_dense(initial_c, row_tile_size, column_tile_size, context);
        context.Sync();
        timer.Snapshot("dense_creation");
        // Do matrix multiplication.
        dense_flops = tile_matrix_multiplication(a_dense, trans_a, b_dense, trans_b, c_dense, timer, alpha,
                                                 beta, "dense_gemm", context);
        // Retrieve results back from tile format for verification.
        timer.StartSnapshot();
        auto full_dense_c = c_dense.ToRawMatrix(context);
        full_dense_c.ReferenceDifference(full_c);
        dense_error = full_dense_c.Norm();
        dense_error_normalized = dense_error / ((a_norm + b_norm + c_init_norm) *
                                                std::numeric_limits<double>::epsilon() *
                                                std::min(initial_c.GetN(), initial_c.GetM()));
        timer.Snapshot("dense_error_calc");
        // Error checking.
        if (dense_error_normalized >= 10) {
            std::cout << "Example didn't pass, dense HCore++ error > 10 " << std::endl;
        }
        // Get memory footprint in KB
        dense_memory_footprint = (a_dense.GetMemoryFootprint() + b_dense.GetMemoryFootprint()
                                  + c_dense.GetMemoryFootprint()) / 1024;
    }

    // Compressed flow
    bool first_print = true;
    for (auto &accuracy: accuracy_list) {
        CompressionParameters svd_parameters(accuracy);
        // Compressed Warmup
        {

            TileMatrix<double> a_comp(full_a, row_tile_size, column_tile_size, svd_parameters, context);
            TileMatrix<double> b_comp(full_b, row_tile_size, column_tile_size, svd_parameters, context);
            TileMatrix<double> c_comp(initial_c, row_tile_size, column_tile_size, svd_parameters, context);
            context.Sync();
            tile_matrix_multiplication(a_comp, trans_a, b_comp, trans_b, c_comp, timer, alpha,
                                       beta, "", context, svd_parameters);
        }

        //Reset all compression timers
        timer.ResetSnapshot("comp_creation");
        timer.ResetSnapshot("comp_gemm");
        timer.ResetSnapshot("comp_error_calc");
        timer.StartSnapshot();
        // Create compressed tiles matrix
        TileMatrix<double> a_comp(full_a, row_tile_size, column_tile_size, svd_parameters, context);
        TileMatrix<double> b_comp(full_b, row_tile_size, column_tile_size, svd_parameters, context);
        TileMatrix<double> c_comp(initial_c, row_tile_size, column_tile_size, svd_parameters, context);
        context.Sync();
        timer.Snapshot("comp_creation");
        // Do matrix multiplication.
        auto comp_flops = tile_matrix_multiplication(a_comp, trans_a, b_comp, trans_b, c_comp, timer, alpha,
                                                     beta, "comp_gemm", context, svd_parameters);
        // Retrieve results back from tile format for verification.
        timer.StartSnapshot();
        auto full_approximate_c = c_comp.ToRawMatrix(context);
        // Calculate compressed tile matrix reference error
        full_approximate_c.ReferenceDifference(full_c);
        double comp_error = full_approximate_c.Norm();
        double comp_error_normalized = comp_error / ((a_norm + b_norm + c_init_norm) * accuracy *
                                                     std::min(initial_c.GetN(), initial_c.GetM()));
        timer.Snapshot("comp_error_calc");
        // Error checking.
        if (comp_error_normalized >= 10) {
            std::cout << "Example didn't pass, compressed HCore++ error > 10 " << std::endl;
        }
        // Get memory footprint in KB
        size_t compressed_memory_footprint = (a_comp.GetMemoryFootprint() + b_comp.GetMemoryFootprint()
                                              + c_comp.GetMemoryFootprint()) / 1024;
        // Print results
        if (first_print) {
            if (print_header) {
                printf("tile_count, tile_size, matrix_size, type, error, error_normalized, memory(KB), creation(ms), gemm_time(ms), flops\n");
                print_header = false;
            }
            printf("%d, %d, %d, ref, 0, 0, %zu, %f, %f, %zu\n",
                   matrix_tiles, tile_size, matrix_tiles * tile_size,
                   ref_memory_footprint, timer.GetSnapshot("generation"),
                   timer.GetSnapshot("ref_gemm"), ref_flops);
            printf("%d, %d, %d, dense, %e, %e, %zu, %f, %f, %zu\n",
                   matrix_tiles, tile_size, matrix_tiles * tile_size, dense_error, dense_error_normalized,
                   dense_memory_footprint, timer.GetSnapshot("dense_creation"),
                   timer.GetSnapshot("dense_gemm"), dense_flops);
            first_print = false;
        }
        printf("%d, %d, %d, %2.1e, %e, %e, %zu, %f, %f, %zu\n",
               matrix_tiles, tile_size, matrix_tiles * tile_size, accuracy, comp_error, comp_error_normalized,
               compressed_memory_footprint, timer.GetSnapshot("comp_creation"),
               timer.GetSnapshot("comp_gemm"), comp_flops);
    }
    return 0;
}

