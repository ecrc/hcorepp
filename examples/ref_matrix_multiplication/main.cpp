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

#ifdef BLAS_HAVE_MKL

#include <mkl.h>

#endif

#ifdef BLAS_HAVE_CUBLAS

#include <omp.h>

#endif

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
    if(print_header) {
        context.Print();
    }
#else
    hcorepp::kernels::RunContext context;
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
    bool first_print = true;
    
    if (first_print) {
        if (print_header) {
            printf("tile_count, tile_size, matrix_size, type, error, error_normalized, memory(KB), creation(ms), gemm_time(ms)\n");
            print_header = false;
        }
        printf("%d, %d, %d, ref, 0, 0, %zu, %f, %f\n",
               matrix_tiles, tile_size, matrix_tiles * tile_size,
               ref_memory_footprint, timer.GetSnapshot("generation"),
               timer.GetSnapshot("ref_gemm"));
        first_print = false;
    } 
    return 0;
}

