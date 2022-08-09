#include <hcorepp/api/hcorepp.hpp>
#include <iostream>
#include <functional>

using namespace hcorepp::operators;
using namespace hcorepp::dataunits;

namespace hcorepp {
    namespace api {

        template<typename T>
        void
        gemm(T alpha, Tile<T> const &A, Tile<T> const &B, T beta, Tile<T> &C) {
            int tile_a_size = A.GetNumberOfMatrices();
            int tile_b_size = B.GetNumberOfMatrices();
            int tile_c_size = C.GetNumberOfMatrices();

            /// calculate total number of input tiles to determine how many intermediate Gemm functions needs to be done.
            int total_input_tiles = tile_a_size + tile_b_size;

            /// check if its Dense Dense compressed case, needs special handling only at the last iteration.
            bool dense_dense_comp = (total_input_tiles == 2 && tile_c_size == 2);

            int iterations = 0;
            if (total_input_tiles == 3 || dense_dense_comp) {
                iterations = 1;
            } else if (total_input_tiles == 4) {
                iterations = 2;
            } else {
                iterations = 0;
            }

            /// dump A dataHolders/buffers in vector.
            blas::Op a_operation = A.operation();
            std::vector<std::reference_wrapper<DataHolder<T>>> a_data_holders;
            a_data_holders.reserve(tile_a_size);
            for (int j = 0; j < tile_a_size; j++) {
                a_data_holders[j] = A.GetTileSubMatrix(j);
            }

            /// dump B dataHolders/buffers in vector.
            blas::Op b_operation = B.operation();
            std::vector<std::reference_wrapper<DataHolder<T>>> b_data_holders;
            b_data_holders.reserve(tile_b_size);
            for (int j = 0; j < tile_b_size; j++) {
                b_data_holders[j] = B.GetTileSubMatrix(j);
            }

            T alpha_local = 1;
            T beta_local = 0;
            if (dense_dense_comp) {
                alpha_local = alpha;
            }

            helpers::SvdHelpers helpers;
            int a_idx = tile_a_size - 1;
            int b_idx = 0;

            /// Resolve intermediate calls to gemm functionality through creating a temporary Dense tile for each iteration..
            std::vector<DenseTile<T> *> temp_tiles;
            while (iterations > 0) {
                auto a_data = a_data_holders[a_idx];
                auto a_op = a_operation;
                auto b_data = b_data_holders[b_idx];
                auto b_op = b_operation;

                temp_tiles.emplace_back(new DenseTile<T>(a_data.get().GetNumOfRows(), b_data.get().GetNumOfCols(),
                                                         nullptr, a_data.get().GetLeadingDim()));

                auto tile = temp_tiles.back();

                tile->Gemm(alpha_local, a_data, a_op, b_data, b_op, beta_local, a_data.get().GetLeadingDim(),
                           std::min(b_data.get().GetNumOfRows(), b_data.get().GetNumOfCols()), helpers);

                if (iterations == 2) {
                    auto a_data_holder = (tile->GetTileSubMatrix(0));
                    a_operation = tile->operation();

                    a_data_holders[a_idx] = a_data_holder;
                    b_idx++;
                } else if (iterations == 1) {
                    if (a_idx == 0) {
                        auto a_tile = (tile->GetTileSubMatrix(0));
                        a_operation = tile->operation();

                        a_data_holders[a_idx] = a_tile;
                        b_idx++;
                    } else {
                        auto b_tile = tile->GetTileSubMatrix(0);
                        b_operation = tile->operation();

                        b_data_holders[b_idx] = b_tile;
                        a_idx--;
                    }
                }
                iterations--;
            }


            /// the last iteration needs to be resolved according to the Tile type either Compressed or Dense.
            if (iterations == 0) {
                ///Dense dense compressed case special
                if (dense_dense_comp) {
                    auto target = temp_tiles[0];
                    auto a_data = C.GetTileSubMatrix(0);
                    auto a_op = C.operation();
                    auto b_data = C.GetTileSubMatrix(1);
                    auto b_op = C.operation();
                    alpha_local = 1;

                    target->Gemm(beta, a_data, a_op, b_data, b_op, alpha_local, a_data.get().GetLeadingDim(),
                                 std::min(b_data.get().GetNumOfRows(), b_data.get().GetNumOfCols()), helpers);

                    int num_of_rows = target->GetTileSubMatrix(0).get().GetNumOfRows();
                    int num_of_cols = target->GetTileSubMatrix(0).get().GetNumOfCols();

                    int64_t c_rank = -1;

                    if (c_rank == std::min(C.GetTileSubMatrix(0).get().GetNumOfRows(),
                                           C.GetTileSubMatrix(1).get().GetNumOfCols())) {
                        c_rank = -1;
                    }

                    C.ReadjustTile(num_of_rows, num_of_cols, target->GetTileSubMatrix(0).get().GetData(), num_of_rows,
                                   c_rank);

                } else {
                    auto a_data = a_data_holders[a_idx];
                    auto b_data = b_data_holders[b_idx];

                    int64_t c_rank = a_data.get().GetNumOfCols();

                    C.Gemm(alpha, a_data, a_operation, b_data, b_operation, beta, a_data.get().GetLeadingDim(),
                           c_rank, helpers);
                }
            }

            /// free the allocated temporary tiles
            for (auto tile: temp_tiles) {
                delete tile;
            }
        }

        template void
        gemm(float, hcorepp::operators::Tile<float> const &, hcorepp::operators::Tile<float> const &, float,
             hcorepp::operators::Tile<float> &);

        template void
        gemm(double, hcorepp::operators::Tile<double> const &, hcorepp::operators::Tile<double> const &, double,
             hcorepp::operators::Tile<double> &);

    }
}
