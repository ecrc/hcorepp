/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <catch2/catch_all.hpp>
#include <iostream>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>
#include <hcorepp/operators/interface/TilePacker.hpp>
#include <hcorepp/test-helpers/testHelpers.hpp>
#include <hcorepp/kernels/memory.hpp>
#include <cstring>

using namespace std;
using namespace hcorepp::dataunits;
using namespace hcorepp::operators;
using namespace hcorepp::test_helpers;

template<typename T>
void TEST_TILE_PACKER() {
    hcorepp::kernels::RunContext &context = hcorepp::kernels::ContextManager::GetInstance().GetContext();

    SECTION("Dense Tile Packing_Unpacking") {
        std::cout << "Dense tile Packing_Unpacking functionality-\n =========================== \n";

        T matrix_A[3][4] = {{1, 5, 9,  13},
                            {2, 6, 10, 14},
                            {3, 7, 11, 15}};

        T matrix_A_row_major[3][4] = {{1, 5, 9,  13},
                                      {2, 6, 10, 14},
                                      {3, 7, 11, 15}};

        // A num of rows
        int64_t a_m = 3;
        // A num of cols
        int64_t a_n = 4;

        // assuming that A, B , and C are COl major.
        int64_t lda = a_m;
        size_t a_size = a_m * a_n;

        T *a_input = new T[a_size];

        rowMajorToColumnMajor<T>((T *) matrix_A, a_n, a_m, a_input);

        DenseTile<T> dense_tile_A(a_m, a_n, (T *) a_input, lda, context);

        /// unpack the tile.
        auto metadata_data = hcorepp::operators::TilePacker<T>::UnPackTile(dense_tile_A,context);

        auto* metadata = metadata_data.first;
        REQUIRE(metadata->mNumOfRows == a_m);
        REQUIRE(metadata->mNumOfCols == a_n);
        REQUIRE(metadata->mMatrixRank == 0);
        REQUIRE(metadata->mLeadingDimension == lda);
        REQUIRE(metadata->mLayout == blas::Layout::ColMajor);
        REQUIRE(metadata->mType == DENSE);

        auto data_arrays = metadata_data.second;

        T *host_data_array = new T[a_m * a_n];
        hcorepp::memory::Memcpy<T>(host_data_array, data_arrays,
                                   a_m * a_n, context,
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();
        columnMajorToRowMajor<T>(host_data_array, a_n, a_m, (T *) matrix_A_row_major);

        auto a_pointer = (T *) matrix_A_row_major;
        validateOutput(a_pointer, a_m, a_n, (T *) matrix_A);

        /// pack the tile again.
        auto *dense_packed_tile = hcorepp::operators::TilePacker<T>::PackTile(*metadata, data_arrays, context);

        REQUIRE(dense_packed_tile->isDense() == 1);
        REQUIRE(dense_packed_tile->GetTileStride(0) == a_m);
        REQUIRE(dense_packed_tile->GetNumOfRows() == a_m);
        REQUIRE(dense_packed_tile->GetNumOfCols() == a_n);
        REQUIRE(dense_packed_tile->GetLeadingDim() == lda);
        REQUIRE(dense_packed_tile->GetLayout() == blas::Layout::ColMajor);
        T *packed_data_array = new T[a_size];
        hcorepp::memory::Memcpy<T>(packed_data_array, dense_packed_tile->GetTileSubMatrix(0),
                                   a_size, context, hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();
        T *a_output = new T[a_size];

        columnMajorToRowMajor<T>(packed_data_array,
                                 dense_packed_tile->GetNumOfCols(),
                                 dense_packed_tile->GetNumOfRows(),
                                 (T *) a_output);

        validateOutput(a_output, a_m, a_n, (T *) matrix_A);

        delete dense_packed_tile;

        delete[] a_input;
        delete[] a_output;
        delete[] host_data_array;
        delete[] packed_data_array;
    }

    SECTION("Compressed Tile Packing_Unpacking") {
        std::cout << "Compressed tile Packing_Unpacking functionality-\n =========================== \n";

        T matrix_AU[5][4] = {{162},
                             {174},
                             {186},
                             {198},
                             {210}};

        T matrix_AV[4][4] = {{2,  4,  6,  8}};

        // AU num of rows
        int64_t au_m = 5;
        // AU num of cols
        int64_t au_n = 1;
        // assuming that AU and AV are COl major.
        int64_t ldaU = au_m;
        // AV num of rows
        int64_t av_m = 1;
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

        CompressedTile<T> compressed_tile_A(a_m, a_n, (T *) a_input, lda, arank, context);

        auto metadata_data = hcorepp::operators::TilePacker<T>::UnPackTile(compressed_tile_A,context);

        TileMetadata* metadata = metadata_data.first;
        REQUIRE(metadata->mNumOfRows == a_m);
        REQUIRE(metadata->mNumOfCols == a_n);
        REQUIRE(metadata->mMatrixRank == arank);
        REQUIRE(metadata->mLeadingDimension == lda);
        REQUIRE(metadata->mLayout == blas::Layout::ColMajor);
        REQUIRE(metadata->mType == COMPRESSED);

        auto data_arrays = metadata_data.second;

        T *host_data_unpacked_au = new T[au_size];
        T *host_data_unpacked_av = new T[av_size];

        auto max_rank = (int64_t)((double) std::min(metadata->mNumOfRows, metadata->mNumOfCols) / MAX_RANK_RATIO);

        hcorepp::memory::Memcpy<T>(host_data_unpacked_au, data_arrays, au_size, context,
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
        hcorepp::memory::Memcpy<T>(host_data_unpacked_av, &data_arrays[metadata->mNumOfRows * max_rank], av_size, context,
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);

        context.Sync();
        T *au_output_row_unpacked = new T[au_size];
        T *av_output_row_unpacked = new T[av_size];

        columnMajorToRowMajor<T>(host_data_unpacked_au,
                                 compressed_tile_A.GetTileRank(),
                                 compressed_tile_A.GetNumOfRows(),
                                 (T *) au_output_row_unpacked);
        columnMajorToRowMajor<T>(host_data_unpacked_av,
                                 compressed_tile_A.GetNumOfCols(),
                                 compressed_tile_A.GetTileRank(),
                                 (T *) av_output_row_unpacked);

        validateOutput(au_output_row_unpacked, au_m, au_n, (T *) matrix_AU);
        validateOutput(av_output_row_unpacked, av_m, av_n, (T *) matrix_AV);

        /// pack the tile again..

        auto compressed_tile_packed = hcorepp::operators::TilePacker<T>::PackTile(*metadata, data_arrays, context);

        REQUIRE(compressed_tile_A.isCompressed());
        REQUIRE(compressed_tile_A.GetTileStride(0) == au_m);
        REQUIRE(compressed_tile_A.GetTileStride(1) == av_m);
        REQUIRE(compressed_tile_A.GetNumOfRows() == au_m);
        REQUIRE(compressed_tile_A.GetTileRank() == au_n);
        REQUIRE(compressed_tile_A.GetULeadingDim() == ldaU);
        REQUIRE(compressed_tile_A.GetTileRank() == av_m);
        REQUIRE(compressed_tile_A.GetNumOfCols()== av_n);
        REQUIRE(compressed_tile_A.GetVLeadingDim() == ldaV);
        REQUIRE(compressed_tile_A.GetLayout() == blas::Layout::ColMajor);

        T *host_data_array_au = new T[au_size];
        T *host_data_array_av = new T[av_size];

        hcorepp::memory::Memcpy<T>(host_data_array_au,
                                   compressed_tile_A.GetUMatrix(),
                                   au_size, context,
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
        hcorepp::memory::Memcpy<T>(host_data_array_av,
                                   compressed_tile_A.GetVMatrix(),
                                   av_size, context,
                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
        context.Sync();
        T *au_output = new T[au_size];
        T *av_output = new T[av_size];

        columnMajorToRowMajor<T>(host_data_array_au,
                                 compressed_tile_A.GetTileRank(),
                                 compressed_tile_A.GetNumOfRows(),
                                 (T *) au_output);
        columnMajorToRowMajor<T>(host_data_array_av,
                                 compressed_tile_A.GetNumOfCols(),
                                 compressed_tile_A.GetTileRank(),
                                 (T *) av_output);

        validateOutput(au_output, au_m, au_n, (T *) matrix_AU);


        validateOutput(av_output, av_m, av_n, (T *) matrix_AV);

        delete compressed_tile_packed;

        delete[] a_input;
        delete[] au_input;
        delete[] av_input;
        delete[] au_output;
        delete[] av_output;
        delete[] au_output_row_unpacked;
        delete[] av_output_row_unpacked;
        delete[] host_data_unpacked_au;
        delete[] host_data_unpacked_av;
        delete[] host_data_array_au;
        delete[] host_data_array_av;

    }

}

TEMPLATE_TEST_CASE("TilePackerTest", "[Dense]", float, double) {
    TEST_TILE_PACKER<TestType>();
    hcorepp::kernels::ContextManager::DestroyInstance();
    hcorepp::dataunits::MemoryHandler<TestType>::DestroyInstance();
}
