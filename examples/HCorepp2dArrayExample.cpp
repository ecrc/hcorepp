#include <cstring>
#include <chrono>
#include <limits>
#include "blas/flops.hh"
#include <hcorepp/api/hcorepp.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/helpers/MatrixHelpers.hpp>
#include <iostream>
#include <hcorepp/kernels/memory.hpp>
#include <hcorepp/helpers/lapack_wrappers.hpp>

using namespace std::chrono;
using namespace hcorepp::operators;
using namespace hcorepp::helpers::matrixhelpers;

static double total_time = 0;


template<typename T>
T *copy_output(const hcorepp::dataunits::DataHolder<T> &apData) {
    size_t num_elements = apData.GetNumOfCols() * apData.GetNumOfRows();
    T *arr = new T[num_elements];
    hcorepp::memory::Memcpy<T>(arr, apData.GetData(), num_elements, hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
    return arr;
}

template<typename T>
void CalculateExact(blas::Layout aLayout, blas::Op aTransA, blas::Op aTransB, int64_t aLdA, int64_t aLdB, int64_t aLdC,
                    T *aAData, T *aBData, T *aCData, int64_t aM, int64_t aN, int64_t aK, T aAlpha, T aBeta) {

    blas::gemm(aLayout, aTransA, aTransB, aM, aN, aK, aAlpha, aAData, aLdA, aBData, aLdB, aBeta, aCData, aLdC);
}

template<typename T>
void CalculateApprox(Tile<T> &A, blas::Op aTransA, Tile<T> &B, blas::Op aTransB, Tile<T> &C,
                     blas::Op aTransC, int64_t aLdC, T aAlpha, T aBeta, T **aCOutput,
                     hcorepp::helpers::SvdHelpers aHelpers, TILE_COMBINATION aCombination) {

    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();

    hcorepp::api::gemm(aAlpha, A, aTransA, B, aTransB, aBeta, C, aTransC, aHelpers);

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    total_time += elapsed_seconds.count();

    if (aCombination == DCC || aCombination == CDC || aCombination == CCC) {
        auto Cm = C.GetTileSubMatrix(0).get().GetNumOfRows();
        auto Cn = C.GetTileSubMatrix(1).get().GetNumOfCols();
        auto rank = C.GetTileSubMatrix(0).get().GetNumOfCols();
        auto cu = copy_output(C.GetTileSubMatrix(0).get());
        auto cv = copy_output(C.GetTileSubMatrix(1).get());
//        *aCOutput = (T *) malloc(Cm * Cn * sizeof(T));
        memset(*aCOutput, 0, Cm * Cn * sizeof(T));

        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                   Cm, Cn, rank, 1.0, cu,
                   C.GetTileSubMatrix(0).get().GetLeadingDim(), cv,
                   C.GetTileSubMatrix(1).get().GetLeadingDim(), 0.0, *aCOutput, aLdC);
        delete[] cu;
        delete[] cv;
    } else if (aCombination == DDC) {
        auto cu_m_new = C.GetTileSubMatrix(0).get().GetNumOfRows();
        auto cu_n_new = C.GetTileSubMatrix(0).get().GetNumOfCols();

        hcorepp::memory::Memcpy<T>(*aCOutput, C.GetTileSubMatrix(0).get().GetData(), cu_m_new * cu_n_new,
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);

    } else {
        auto Cm = C.GetTileSubMatrix(0).get().GetNumOfRows();
        auto Cn = C.GetTileSubMatrix(0).get().GetNumOfCols();

        hcorepp::memory::Memcpy<T>(*aCOutput, C.GetTileSubMatrix(0).get().GetData(), Cm * Cn,
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
    }
}

template<typename T>
void
CalculateApproxAndExactTilesGemm(blas::Op aTransA, blas::Op aTransB, blas::Op aTransC, T aAlpha, T aBeta,
                                 DenseTile<T> &dense_tileA, DenseTile<T> &dense_tileB, DenseTile<T> &dense_tileC,
                                 CompressedTile<T> &comp_tileA, CompressedTile<T> &comp_tileB,
                                 CompressedTile<T> &comp_tileC, T **Approx_C_output,
                                 T **Exact_C_Output) {

    hcorepp::memory::Memcpy<T>(*Exact_C_Output, dense_tileC.GetTileSubMatrix(0).get().GetData(),
                               dense_tileC.GetTileSubMatrix(0).get().GetNumOfRows() *
                               dense_tileC.GetTileSubMatrix(0).get().GetNumOfCols(),
                               hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
    auto a = copy_output(dense_tileA.GetTileSubMatrix(0).get());
    auto b = copy_output(dense_tileB.GetTileSubMatrix(0).get());
    CalculateExact(blas::Layout::ColMajor, aTransA, aTransB, dense_tileA.GetTileSubMatrix(0).get().GetLeadingDim(),
                   dense_tileB.GetTileSubMatrix(0).get().GetLeadingDim(),
                   dense_tileC.GetTileSubMatrix(0).get().GetNumOfRows(),
                   a, b,
                   *Exact_C_Output,
                   dense_tileC.GetTileSubMatrix(0).get().GetNumOfRows(),
                   dense_tileC.GetTileSubMatrix(0).get().GetNumOfCols(),
                   dense_tileA.GetTileSubMatrix(0).get().GetNumOfCols(),
                   aAlpha, aBeta);
    delete[] a;
    delete[] b;
    hcorepp::memory::Memcpy<T>(dense_tileC.GetTileSubMatrix(0).get().GetData(), *Exact_C_Output,
                               dense_tileC.GetTileSubMatrix(0).get().GetNumOfRows() *
                               dense_tileC.GetTileSubMatrix(0).get().GetNumOfCols(),
                               hcorepp::memory::MemoryTransfer::HOST_TO_DEVICE);

    hcorepp::helpers::SvdHelpers helpers;
    CalculateApprox(comp_tileA, aTransA, comp_tileB, aTransB, comp_tileC, aTransC,
                    dense_tileC.GetTileSubMatrix(0).get().GetLeadingDim(), aAlpha, aBeta,
                    Approx_C_output, helpers, CCC);

}

template<typename T>
void
GenerateTiles(blas::Op aTransA, int64_t aM, int64_t aN, int64_t aMode, blas::real_type<T> aCond, T aAcc,
              int64_t aAlignment, int64_t *iseed, DenseTile<T> **aDenseTile, CompressedTile<T> **aCompressedTile) {

    int64_t align = aAlignment;

    /// Dense TIle shouldn't be affected by K passed.!!
    /// no need for aTransA since no matrix operation is applied.
    int64_t Am = aM;
    int64_t An = aN;

    int64_t lda = ((Am + align - 1) / align) * align;

    T *Adata = new T[lda * An];

    generate_dense_matrix(Am, An, Adata, lda, iseed, aMode, aCond);
    *aDenseTile = new DenseTile<T>(Am, An, Adata, lda, blas::Layout::ColMajor, aTransA, blas::Uplo::General);

    int64_t Ark;
    T *AUVdata;

    compress_dense_matrix(Am, An, Adata, lda, &AUVdata, Ark, aAcc);
    *aCompressedTile = new CompressedTile<T>(Am, An, AUVdata, lda, Ark, aAcc, blas::Layout::ColMajor, aTransA,
                                             blas::Uplo::General);
    free(AUVdata);
    delete[] Adata;
}

template<typename T>
void
GenerateDenseAndCompressedTiles(int64_t aRows, int64_t aCols, blas::Op aTransA, int64_t aM, int64_t aN,
                                int64_t aMode, blas::real_type<T> aCond, T aAcc, int64_t aAlignment, int64_t *iseed,
                                std::vector<std::vector<DenseTile<T> *>> &ADense,
                                std::vector<std::vector<CompressedTile<T> *>> &AComp) {
    /// column major
    ADense.resize(aCols);
    AComp.resize(aCols);
    for (int i = 0; i < aCols; i++) {
        ADense[i].resize(aRows);
        AComp[i].resize(aRows);
        for (int j = 0; j < aRows; j++) {
            GenerateTiles(aTransA, aM, aN, aMode, aCond, aAcc, aAlignment, iseed,
                          &ADense[i][j], &AComp[i][j]);
        }
    }
}

/**
 * Merge tiles in single array
 *
 * @tparam T
 * @param aTileSize number of elements in a tile, total_tile_size = aTileSize*aTileSize.
 * @param aMatrixSize number of tiles in a matrix, total_matrix_size = (aMatrixSize*aTileSize)*(aMatrixSize*aTileSize).
 * @param apTiles vector of vector of tile pointers. (2d matrix of tiles)
 * @param apOutArray Pointer to output array to hold the tile pointers,
 */
template<typename T>
void MergeTilesInSingleArray(int64_t aTileSize, int64_t aMatrixRows, int64_t aMatrixCols,
                             std::vector<std::vector<T *>> &apTiles, T *&apOutArray) {

    size_t full_array_index = 0;
    size_t tile_index_r = 0;
    size_t tile_index_c = 0;
    size_t index_in_tile = 0;

    for (int64_t cols = 0; cols < aMatrixCols; cols += aTileSize) {
        for (int64_t rows = 0; rows < aMatrixRows; rows += aTileSize) {

            int64_t tile_rows = std::min(aTileSize, aMatrixRows - rows);
            int64_t tile_cols = std::min(aTileSize, aMatrixCols - cols);

            for (int i = 0; i < tile_cols; i++) {
                for (int j = 0; j < tile_rows; j++) {
                    tile_index_r = rows / aTileSize;
                    tile_index_c = cols / aTileSize;
                    index_in_tile = i * tile_rows + j;

                    full_array_index = (tile_index_r * aTileSize) + j + ((tile_index_c * aTileSize + i) * aMatrixRows);

                    apOutArray[full_array_index] = (apTiles[tile_index_c][tile_index_r])[index_in_tile];
                }
            }
        }
    }

}

template<typename T>
void DeleteDenseAndCompressedTiles(int64_t aRows, int64_t aCols, std::vector<std::vector<DenseTile<T> *>> &ADense,
                                   std::vector<std::vector<CompressedTile<T> *>> &AComp) {
    for (int i = 0; i < aCols; i++) {
        for (int j = 0; j < aRows; j++) {
            delete ADense[i][j];
            delete AComp[i][j];
        }
    }
}

template<typename T>
void GenerateFullDenseMatrixElements(int64_t aMatrixRows, int64_t aMatrixCols, int64_t *iseed, T *&apDenseArray,
                                     int64_t aMode, blas::real_type<T> aCond, int64_t align) {
    auto lda = ((aMatrixRows + align - 1) / align) * align;
    generate_dense_matrix(aMatrixRows, aMatrixCols, apDenseArray, lda, iseed, aMode, aCond);
}

template<typename T>
void
GenerateDenseAndCompressedTilesFromDense(int64_t aTileSize, int64_t aMatrixRowElements, int64_t aMatrixColElements,
                                         T *&apDenseArray, std::vector<std::vector<DenseTile<T> *>> &ADense,
                                         std::vector<std::vector<CompressedTile<T> *>> &AComp, T aAcc) {

    auto mt = (aMatrixRowElements / aTileSize);
    if (aMatrixRowElements % aTileSize > 0) {
        mt += 1;
    }
    auto nt = (aMatrixColElements / aTileSize);
    if (aMatrixColElements % aTileSize > 0) {
        nt += 1;
    }

    int64_t processed_rows = 0;
    int64_t processed_cols = 0;
    /// column major
    ADense.resize(nt);
    AComp.resize(nt);
    int64_t tile_cols;
    int64_t tile_rows;

    processed_cols = 0;
    for (int i = 0; i < nt; i++) {
        processed_rows = 0;
        ADense[i].resize(mt);
        AComp[i].resize(mt);
        tile_cols = std::min(aTileSize, aMatrixColElements - processed_cols);

        for (int j = 0; j < mt; j++) {
            tile_rows = std::min(aTileSize, aMatrixRowElements - processed_rows);

            T *Adata = new T[tile_rows * tile_cols];

            auto st_idx = i * aTileSize * aMatrixRowElements + j * aTileSize;
            memcpy(Adata, &apDenseArray[st_idx], tile_rows * tile_cols * sizeof(T));
            ADense[i][j] = new DenseTile<T>(tile_rows, tile_cols, Adata, tile_rows, blas::Layout::ColMajor,
                                            blas::Op::NoTrans, blas::Uplo::General);

            int64_t Ark;
            T *AUVdata;

            compress_dense_matrix(tile_rows, tile_cols, Adata, tile_rows, &AUVdata, Ark, aAcc);
            AComp[i][j] = new CompressedTile<T>(tile_rows, tile_cols, AUVdata, tile_rows, Ark, aAcc,
                                                blas::Layout::ColMajor, blas::Op::NoTrans, blas::Uplo::General);

            processed_rows += tile_rows;
            delete[] Adata;
            free(AUVdata);
        }
        processed_cols += tile_cols;
    }
}

int main(int argc, char *argv[]) {

    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();

    /// single tile dimensions.
    int64_t m = 500;
    int64_t n = 500;

    double alpha = 1;
    double beta = 1;
    double accuracy = 1e-4;
//    blas::real_type<double> accuracy = 1e-4;
    blas::Op transA = blas::Op::NoTrans;
    blas::Op transB = blas::Op::NoTrans;
    blas::Op transC = blas::Op::NoTrans;
    int64_t mode = 0;
    int64_t align = 1;
    blas::real_type<double> cond = std::numeric_limits<double>::epsilon();

    // assuming squared matrices
    int matrix_tiles = 2;
    if (argc > 1) {
        matrix_tiles = atoi(argv[1]);
        accuracy = atof(argv[2]);
    }
    std::cout << " matrix tiles = " << matrix_tiles << " \t accuracy = " << accuracy << "\n";
    /// matrix dimensions (number of tiles)
    int a_mt = matrix_tiles;
    int a_nt = matrix_tiles;
    int b_mt = a_nt;
    int b_nt = matrix_tiles;
    int c_mt = a_mt;
    int c_nt = b_nt;

    int64_t iseed[4] = {0, 0, 0, 1};

    auto *full_dense_a = (double *) malloc(a_mt * m * a_nt * n * sizeof(double));
    auto *full_dense_b = (double *) malloc(b_mt * m * b_nt * n * sizeof(double));
    auto *full_dense_c = (double *) malloc(c_mt * m * c_nt * n * sizeof(double));

    GenerateFullDenseMatrixElements(a_mt * m, a_nt * n, iseed, full_dense_a, mode, cond, align);
    GenerateFullDenseMatrixElements(b_mt * m, b_nt * n, iseed, full_dense_b, mode, cond, align);
    GenerateFullDenseMatrixElements(c_mt * m, c_nt * n, iseed, full_dense_c, mode, cond, align);

    ///Generate Matrix A Dense and compressed tiles.
    std::vector<std::vector<DenseTile<double> *>> ADense;
    std::vector<std::vector<CompressedTile<double> *>> AComp;
    GenerateDenseAndCompressedTilesFromDense(m, a_mt * m, a_nt * n, full_dense_a, ADense, AComp, accuracy);

    ///Generate Matrix B Dense and compressed tiles.
    std::vector<std::vector<DenseTile<double> *>> BDense;
    std::vector<std::vector<CompressedTile<double> *>> BComp;
    GenerateDenseAndCompressedTilesFromDense(m, b_mt * m, b_nt * n, full_dense_b, BDense, BComp, accuracy);

    ///Generate Matrix C Dense and compressed tiles.
    std::vector<std::vector<DenseTile<double> *>> CDense;
    std::vector<std::vector<CompressedTile<double> *>> CComp;
    GenerateDenseAndCompressedTilesFromDense(m, c_mt * m, c_nt * n, full_dense_c, CDense, CComp, accuracy);

    std::vector<std::vector<double *>> Exact_a;
    std::vector<std::vector<double *>> Exact_b;
    std::vector<std::vector<double *>> Exact_c;
    std::vector<std::vector<double *>> Approx_c;

    Exact_a.resize(a_nt);
    Exact_b.resize(b_nt);
    Exact_c.resize(c_nt);
    Approx_c.resize(c_nt);
    for (int i = 0; i < a_nt; i++) {
        Exact_a[i].resize(a_mt);
    }
    for (int i = 0; i < b_nt; i++) {
        Exact_b[i].resize(b_mt);
    }

    for (int i = 0; i < c_nt; i++) {
        Exact_c[i].resize(c_mt);
        Approx_c[i].resize(c_mt);
        for (int j = 0; j < c_mt; j++) {

            Approx_c[i][j] = (double *) malloc(m * n * sizeof(double));

            Exact_c[i][j] = (double *) malloc(m * n * sizeof(double));

            auto denseC = CDense[i][j];
            auto compC = CComp[i][j];

            for (int k = 0; k < b_mt; k++) {

                Exact_a[k][j] = copy_output(ADense[k][j]->GetTileSubMatrix(0).get());
                Exact_b[i][k] = copy_output(BDense[i][k]->GetTileSubMatrix(0).get());

                auto denseA = ADense[k][j];
                auto denseB = BDense[i][k];
                auto compA = AComp[k][j];
                auto compB = BComp[i][k];

                CalculateApproxAndExactTilesGemm(transA, transB, transC, alpha, beta, *denseA, *denseB, *denseC, *compA,
                                                 *compB, *compC, &Approx_c[i][j], &Exact_c[i][j]);
            }
        }
    }

    ///Delete Matrix C Dense and compressed tiles.
    DeleteDenseAndCompressedTiles(c_mt, c_nt, CDense, CComp);

    auto *full_approx_c = (double *) malloc(c_mt * c_nt * (m * n * sizeof(double)));
    auto *full_exact_c = (double *) malloc(c_mt * c_nt * (m * n * sizeof(double)));
    MergeTilesInSingleArray(m, c_mt * m, c_nt * n, Approx_c, full_approx_c);
    MergeTilesInSingleArray(m, c_mt * m, c_nt * n, Exact_c, full_exact_c);

    for (int j = 0; j < b_nt; j++) {
        for (int i = 0; i < a_mt; i++) {
            free(Exact_c[j][i]);
            free(Approx_c[j][i]);
        }
    }

    auto *full_exact_a = (double *) malloc(a_mt * a_nt * (m * n * sizeof(double)));
    MergeTilesInSingleArray(m, a_mt * m, a_nt * n, Exact_a, full_exact_a);
    ///Delete Matrix A Dense and compressed tiles.
    DeleteDenseAndCompressedTiles(a_mt, a_nt, ADense, AComp);

    auto *full_exact_b = (double *) malloc(b_mt * b_nt * (m * n * sizeof(double)));
    MergeTilesInSingleArray(m, b_mt * m, b_nt * n, Exact_b, full_exact_b);
    ///Delete Matrix B Dense and compressed tiles.
    DeleteDenseAndCompressedTiles(b_mt, b_nt, BDense, BComp);


    hcorepp::helpers::Norm norm = hcorepp::helpers::Norm::INF;

    blas::real_type<double> Anorm = lapack_lange(norm, a_mt * m, a_nt * n, full_exact_a, a_mt * m);
    blas::real_type<double> Bnorm = lapack_lange(norm, b_mt * m, b_nt * n, full_exact_b, b_mt * m);
    blas::real_type<double> Cnorm = lapack_lange(norm, c_mt * m, c_nt * n, full_dense_c, c_mt * m);

    diff(full_exact_c, c_mt * m, full_approx_c, c_mt * m, c_nt * n, c_mt * m);

    double error = lapack_lange(norm, c_mt * m, c_nt * n, full_exact_c, c_mt * m) /
                   ((Anorm + Bnorm + Cnorm) * std::max(c_mt * m, c_nt * n) * accuracy);

    std::cout << "FINAL ERROR VALUE : " << error << " expected accuracy:  " << 10 << "\n";
    bool pass = (error < 10);

    if (pass) {
        std::cout << "Example passed " << std::endl;
    } else {
        std::cout << "Example didn't pass, error > 10 " << std::endl;
    }

    for (int i = 0; i < b_mt; i++) {
        for (int j = 0; j < c_mt; j++) {
            delete[] Exact_a[i][j];
        }
    }
    for (int i = 0; i < c_nt; i++) {
        for (int j = 0; j < b_mt; j++) {
            delete[] Exact_b[i][j];
        }
    }

    free(full_exact_c);
    free(full_approx_c);
    free(full_exact_a);
    free(full_exact_b);
    free(full_dense_a);
    free(full_dense_b);
    free(full_dense_c);

    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << " Matrix tiles =  " << matrix_tiles << " , HCorepp Gemm total execution time = " << total_time
              << std::endl;
    std::cout << " Matrix tiles =  " << matrix_tiles << " , Complete Program Total execution time  = "
              << elapsed_seconds.count() << std::endl;

}
