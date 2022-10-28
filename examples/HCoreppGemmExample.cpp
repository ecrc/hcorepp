#include <cstring>
#include <chrono>
#include "blas/flops.hh"
#include <hcorepp/api/hcorepp.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/helpers/MatrixHelpers.hpp>
#include <hcorepp/helpers/lapack_wrappers.hpp>
#include <hcorepp/kernels/memory.hpp>
#include <iostream>

using namespace std::chrono;
using namespace hcorepp::operators;
using namespace hcorepp::helpers::matrixhelpers;

template<typename T>
void run(blas::Op aTransA, blas::Op aTransB, blas::Op aTransC, T aAlpha, T aBeta, int64_t aM, int64_t aN, int64_t aK,
         T aTol, T aAcc, int64_t aMode, int64_t aAlignment, TILE_COMBINATION aCombination) {
    blas::Op transA = aTransA;
    blas::Op transB = aTransB;
    blas::Op transC = aTransC;
    T alpha = aAlpha;
    T beta = aBeta;

    int64_t m = aM;
    int64_t n = aN;
    int64_t k = aK;

    blas::real_type<T> tol = 3;
    blas::real_type<T> cond = 1;
    blas::real_type<T> accuracy = 0.0001;
    int64_t mode = aMode;
    int64_t align = aAlignment;

    int64_t Am = transA == blas::Op::NoTrans ? m : k;
    int64_t An = transA == blas::Op::NoTrans ? k : m;
    int64_t Bm = transB == blas::Op::NoTrans ? k : n;
    int64_t Bn = transB == blas::Op::NoTrans ? n : k;
    int64_t Cm = transC == blas::Op::NoTrans ? m : n;
    int64_t Cn = transC == blas::Op::NoTrans ? n : m;

    int64_t lda = ((Am + align - 1) / align) * align;
    int64_t ldb = ((Bm + align - 1) / align) * align;
    int64_t ldc = ((Cm + align - 1) / align) * align;

    T *Adata = new T[lda * An];
    T *Bdata = new T[ldb * Bn];
    T *Cdata = new T[ldc * Cn];
    int64_t iseed[4] = {0, 0, 0, 1};

    generate_dense_matrix(Am, An, Adata, lda, iseed, mode, cond);

    generate_dense_matrix(Bm, Bn, Bdata, ldb, iseed, mode, cond);

    generate_dense_matrix(Cm, Cn, Cdata, ldc, iseed, mode, cond);

    hcorepp::helpers::Norm norm = hcorepp::helpers::Norm::INF; // todo: variable norm type
    blas::real_type<T> Anorm = lapack_lange(norm, Am, An, Adata, lda);
    blas::real_type<T> Bnorm = lapack_lange(norm, Bm, Bn, Bdata, ldb);
    blas::real_type<T> Cnorm = lapack_lange(norm, Cm, Cn, Cdata, ldc);

    DenseTile<T> A(Am, An, Adata, lda, blas::Layout::ColMajor, transA, blas::Uplo::General);
    DenseTile<T> B(Bm, Bn, Bdata, ldb, blas::Layout::ColMajor, transB, blas::Uplo::General);
    DenseTile<T> C(Cm, Cn, Cdata, ldc, blas::Layout::ColMajor, transC, blas::Uplo::General);

    int64_t ldcref = ((m + align - 1) / align) * align;

    T *Cref = new T[ldcref * n];

    hcorepp::memory::Memcpy<T>(Cref, C.GetTileSubMatrix(0).get().GetData(),
                               Cm * Cn, hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);

    int64_t Ark, Brk, Crk;
    Ark = 0;
    Brk = 0;
    Crk = 0;

    T *AUVdata;
    T *BUVdata;
    T *CUVdata;

    CompressedTile<T> *AUV;
    CompressedTile<T> *BUV;
    CompressedTile<T> *CUV;
    if (aCombination == CDD || aCombination == CDC || aCombination == CCD || aCombination == CCC) {
        compress_dense_matrix(Am, An, Adata, lda, &AUVdata, Ark, accuracy);
        AUV = new CompressedTile<T>(Am, An, AUVdata, lda, Ark, accuracy, blas::Layout::ColMajor, transA,
                                    blas::Uplo::General);
        free(AUVdata);
    }
    if (aCombination == DCD || aCombination == DCC || aCombination == CCD || aCombination == CCC) {
        compress_dense_matrix(Bm, Bn, Bdata, ldb, &BUVdata, Brk, accuracy);
        BUV = new CompressedTile<T>(Bm, Bn, BUVdata, ldb, Brk, accuracy, blas::Layout::ColMajor, transB,
                                    blas::Uplo::General);
        free(BUVdata);
    }
    if (aCombination == DDC || aCombination == DCC || aCombination == CDC || aCombination == CCC) {
        compress_dense_matrix(Cm, Cn, Cdata, ldc, &CUVdata, Crk, accuracy);
        CUV = new CompressedTile<T>(Cm, Cn, CUVdata, ldc, Crk, accuracy, blas::Layout::ColMajor, transC,
                                    blas::Uplo::General);
        free(CUVdata);
    }

    double gflops = 0.0;
    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();

    hcorepp::helpers::SvdHelpers helpers;
    switch (aCombination) {
        case DDD:
            hcorepp::api::gemm(alpha, A, transA, B, transB, beta, C, transC, helpers);
            gflops = blas::Gflop<T>::gemm(m, n, k);
            break;
        case DDC:
            hcorepp::api::gemm(alpha, A, transA, B, transB, beta, *CUV, transC, helpers);
            gflops = blas::Gflop<T>::gemm(Cm, Cn, An) + blas::Gflop<T>::gemm(Cm, Cn, Crk);
            break;
        case DCD:
            hcorepp::api::gemm(alpha, A, transA, *BUV, transB, beta, C, transC, helpers);
            gflops = blas::Gflop<T>::gemm(Cm, Brk, An) + blas::Gflop<T>::gemm(Cm, Cn, Brk);
            break;
        case DCC:
            hcorepp::api::gemm(alpha, A, transA, *BUV, transB, beta, *CUV, transC, helpers);
            // todo
            // gflops = blas::Gflop<T>::gemm(Cm, Brk, An) +
            //          internal::gemm;
            break;
        case CDD:
            hcorepp::api::gemm(alpha, *AUV, transA, B, transB, beta, C, transC, helpers);
            gflops = blas::Gflop<T>::gemm(Ark, Cn, An) + blas::Gflop<T>::gemm(Cm, Cn, Ark);
            break;
        case CDC:
            hcorepp::api::gemm(alpha, *AUV, transA, B, transB, beta, *CUV, transC, helpers);
            // todo
            // gflops = blas::Gflop<T>::gemm(Ark, Cn, An) +
            //          internal::gemm;
            break;
        case CCD:
            hcorepp::api::gemm(alpha, *AUV, transA, *BUV, transB, beta, C, transC, helpers);
            gflops = blas::Gflop<T>::gemm(Ark, Brk, An) +
                     (Ark <= Brk ? blas::Gflop<T>::gemm(Ark, Cn, Brk) +
                                   blas::Gflop<T>::gemm(Cm, Cn, Ark)
                                 : blas::Gflop<T>::gemm(Cm, Brk, Ark) +
                                   blas::Gflop<T>::gemm(Cm, Cn, Brk));
            break;
        case CCC:
            hcorepp::api::gemm(alpha, *AUV, transA, *BUV, transB, beta, *CUV, transC, helpers);
            // todo: for now use PASC paper, which assumes square matrices
            int64_t max_Ark_Brk_Crk = std::max({Ark, Brk, Crk});
            int64_t max_m_n_k = std::max({m, n, k});
            gflops = (1e-9 * ((blas::is_complex<T>::value ? 3 : 1)
                              * 36 * max_m_n_k * (max_Ark_Brk_Crk
                                                  * max_Ark_Brk_Crk) + 157 * (max_Ark_Brk_Crk
                                                                              * max_Ark_Brk_Crk * max_Ark_Brk_Crk)));
            // todo
            // gflops = blas::Gflop<T>::gemm(Ark, Brk, An) +
            //          (Ark <= Brk ? blas::Gflop<T>::gemm(Ark, Cn, Brk) +
            //                        internal::gemm
            //                      : blas::Gflop<T>::gemm(Cm, Brk, Ark) +
            //                        internal::gemm;
            break;
    }

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;
    auto elapsed_time = elapsed_seconds.count();
    auto flops = gflops / elapsed_time;

    double ref_elapsed_time;
    double ref_flops;
    double error = 0;
    bool pass = false;
    T *C_output;

    {
        std::chrono::time_point<std::chrono::system_clock> ref_time_start = std::chrono::system_clock::now();
        {

            blas::gemm(blas::Layout::ColMajor, transA, transB, m, n, k, alpha, Adata, lda, Bdata, ldb, beta, Cref,
                       ldcref);

        }
        std::chrono::time_point<std::chrono::system_clock> ref_time_end = std::chrono::system_clock::now();

        std::chrono::duration<double> ref_elapsed_seconds = ref_time_end - ref_time_start;
        ref_elapsed_time = ref_elapsed_seconds.count();
        double ref_gflops = blas::Gflop<T>::gemm(m, n, k);

        ref_flops = ref_gflops / ref_elapsed_time;

        if (aCombination == DCC || aCombination == CDC || aCombination == CCC) {
            size_t cu_size = CUV->GetTileSubMatrix(0).get().GetNumOfRows()
                             * CUV->GetTileSubMatrix(0).get().GetNumOfCols();
            size_t cv_size = CUV->GetTileSubMatrix(1).get().GetNumOfRows()
                             * CUV->GetTileSubMatrix(1).get().GetNumOfCols();
            T *cu = new T[cu_size];
            T *cv = new T[cv_size];
            hcorepp::memory::Memcpy<T>(cu, CUV->GetTileSubMatrix(0).get().GetData(),
                                       cu_size, hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
            hcorepp::memory::Memcpy<T>(cv, CUV->GetTileSubMatrix(1).get().GetData(),
                                       cv_size, hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);

            blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                       Cm, Cn, CUV->GetTileRank(), 1.0, cu,
                       CUV->GetTileSubMatrix(0).get().GetLeadingDim(), cv,
                       CUV->GetTileSubMatrix(1).get().GetLeadingDim(), 0.0, Cdata, ldc);
            delete[] cu;
            delete[] cv;
            C_output = (T *) malloc(Cm * Cn * sizeof(T));

            memcpy((void *) C_output, (void *) Cdata, Cm * Cn * sizeof(T));
            diff(Cref, ldcref, C_output, Cm, Cn, ldc);

        } else if (aCombination == DDC) {
            auto cu_m_new = CUV->GetTileSubMatrix(0).get().GetNumOfRows();
            auto cu_n_new = CUV->GetTileSubMatrix(0).get().GetNumOfCols();
            auto cv_m_new = CUV->GetTileSubMatrix(1).get().GetNumOfRows();
            auto cv_n_new = CUV->GetTileSubMatrix(1).get().GetNumOfCols();

            C_output = (T *) malloc((cu_m_new * cu_n_new + cv_m_new * cv_n_new) * sizeof(T));
            hcorepp::memory::Memcpy<T>(C_output, CUV->GetTileSubMatrix(0).get().GetData(),
                                       cu_m_new * cu_n_new, hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);

            hcorepp::memory::Memcpy<T>(&C_output[cu_m_new * cu_n_new], CUV->GetTileSubMatrix(1).get().GetData(),
                                       cv_m_new * cv_n_new, hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);

            diff(Cref, ldcref, C_output, Cm, Cn, ldc);

        } else {
            C_output = (T *) malloc(Cm * Cn * sizeof(T));
            hcorepp::memory::Memcpy<T>(C_output, C.GetTileSubMatrix(0).get().GetData(),
                                       Cm * Cn, hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
            diff(Cref, ldcref, C_output, Cm, Cn, ldc);

        }

        error = lapack_lange(norm, m, n, Cref, ldcref)
                / (sqrt(blas::real_type<T>(k) + 2) * std::abs(alpha) *
                   Anorm * Bnorm + 2 * std::abs(beta) * Cnorm);

        if (blas::is_complex<T>::value) {
            error /= 2 * sqrt(2);
        }
        pass = (error < tol * accuracy);

    }

    delete[]Cref;
    delete[]Adata;
    delete[]Bdata;
    delete[]Cdata;
    if (aCombination == CDD || aCombination == CDC || aCombination == CCD || aCombination == CCC) {
        delete AUV;
    }
    if (aCombination == DCD || aCombination == DCC || aCombination == CCD || aCombination == CCC) {
        delete BUV;
    }
    if (aCombination == DDC || aCombination == DCC || aCombination == CDC || aCombination == CCC) {
        delete CUV;
    }
    free(C_output);

    printf("%s\n", std::string(196, '=').c_str());
    printf("|%-5s|%-10s|%-10s|%-10s|%-10s|%-5s|%-5s|%-5s|%-8s|%-8s|%-15s|%-15s|%-15s|%-15s|%-15s|%-5s|%-5s|%-5s|%-10s|\n",
           "Gemm", "Datatype", "opA", "opB", "opC", "m", "n", "k", "alpha",
           "beta", "time(s)", "gflops", "ref_time(s)", "ref_gflops", "error", "Ark", "Brk", "Crk", "status");
    printf("%s\n", std::string(196, '=').c_str());
    printf("|%-5s|%-10s|%-10s|%-10s|%-10s|%-5ld|%-5ld|%-5ld|%-8.3f|%-8.3f|%-15f|%-15f|%-15f|%-15f|%-15e|%-5ld|%-5ld|%-5ld|%-10s|\n",
           tile_combination_strings[aCombination], typeid(T).name(),
           op2str(transA), op2str(transB), op2str(transC), m, n, k, alpha,
           beta, elapsed_time, gflops, ref_elapsed_time, ref_flops, error, Ark, Brk, Crk, ((pass) ? "Pass" : "Fail"));

}

int main(int argc, char *argv[]) {
    int64_t m = 500;
    int64_t n = 500;
    int64_t k = 500;
    double alpha = 3.5;
    double beta = 2.5;
    blas::real_type<double> accuracy = 0.0001;
    blas::Op transA = blas::Op::NoTrans;
    blas::Op transB = blas::Op::NoTrans;
    blas::Op transC = blas::Op::NoTrans;
    int64_t mode = 0;
    int64_t align = 1;
    blas::real_type<double> cond = 1;
    blas::real_type<double> tol = 3;

    TILE_COMBINATION combination = CCC;

    run(transA, transB, transC, alpha, beta, m, n, k, tol,
        accuracy, mode, align, combination);
}