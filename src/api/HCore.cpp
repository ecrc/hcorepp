/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#include <hcorepp/api/HCore.hpp>
#include <functional>

using namespace hcorepp::operators;
using namespace hcorepp::dataunits;

namespace hcorepp {
    namespace api {

        template<typename T>
        void HCore<T>::Gemm(T aAlpha, const Tile<T> &aA, const blas::Op &aAOp, const Tile<T> &aB, const blas::Op &aBOp,
                            T aBeta, Tile<T> &aC, const CompressionParameters &aCompressionParameters) {
            int tile_a_size = aA.GetNumberOfMatrices();
            int tile_b_size = aB.GetNumberOfMatrices();
            int tile_c_size = aC.GetNumberOfMatrices();

            /// calculate total number of input tiles to determine how many intermediate Gemm functions needs to be done.
            int total_input_tiles = tile_a_size + tile_b_size;

            /// check if its Dense Dense compressed case, needs special handling only at the last iteration.
            bool dense_dense_comp = (total_input_tiles == 2 && tile_c_size == 2);

            /// controls the number of temporary gemms that will be done.
            int number_of_temporary_dense_gemms = 0;
            if (total_input_tiles == 3 || dense_dense_comp) {
                number_of_temporary_dense_gemms = 1;
            } else if (total_input_tiles == 4) {
                number_of_temporary_dense_gemms = 2;
            } else {
                number_of_temporary_dense_gemms = 0;
            }

            /// dump A dataHolders/buffers in vector.
            blas::Op a_operation = aAOp;
            std::vector<std::reference_wrapper<DataHolder<T>>> a_data_holders;
            a_data_holders.reserve(tile_a_size);
            for (int j = 0; j < tile_a_size; j++) {
                a_data_holders[j] = aA.GetTileSubMatrix(j);
            }

            /// dump B dataHolders/buffers in vector.
            blas::Op b_operation = aBOp;
            std::vector<std::reference_wrapper<DataHolder<T>>> b_data_holders;
            b_data_holders.reserve(tile_b_size);
            for (int j = 0; j < tile_b_size; j++) {
                b_data_holders[j] = aB.GetTileSubMatrix(j);
            }

            T alpha_local = 1;
            T beta_local = 0;
            if (dense_dense_comp) {
                alpha_local = aAlpha;
            }

            int a_idx = tile_a_size - 1;
            int b_idx = 0;

            /// Resolve intermediate calls to gemm functionality through creating a temporary Dense tile for each iteration..
            std::vector<DenseTile<T> *> temp_tiles;
            while (number_of_temporary_dense_gemms > 0) {
                auto a_data = a_data_holders[a_idx];
                auto a_op = a_operation;
                auto b_data = b_data_holders[b_idx];
                auto b_op = b_operation;

                temp_tiles.emplace_back(new DenseTile<T>(a_data.get().GetNumOfRows(), b_data.get().GetNumOfCols(),
                                                         nullptr, a_data.get().GetLeadingDim()));

                auto tile = temp_tiles.back();

                tile->Gemm(alpha_local, a_data, a_op, b_data, b_op, beta_local, a_data.get().GetLeadingDim(),
                           std::min(b_data.get().GetNumOfRows(), b_data.get().GetNumOfCols()), aCompressionParameters);

                if (number_of_temporary_dense_gemms == 2) {
                    auto a_data_holder = (tile->GetTileSubMatrix(0));
                    a_operation = blas::Op::NoTrans;

                    a_data_holders[a_idx] = a_data_holder;
                    b_idx++;
                } else if (number_of_temporary_dense_gemms == 1) {
                    if (a_idx == 0) {
                        auto a_tile = (tile->GetTileSubMatrix(0));
                        a_operation = blas::Op::NoTrans;

                        a_data_holders[a_idx] = a_tile;
                        b_idx++;
                    } else {
                        auto b_tile = tile->GetTileSubMatrix(0);
                        b_operation = blas::Op::NoTrans;

                        b_data_holders[b_idx] = b_tile;
                        a_idx--;
                    }
                }
                number_of_temporary_dense_gemms--;
            }


            /// the last iteration needs to be resolved according to the Tile type either Compressed or Dense.
            if (number_of_temporary_dense_gemms == 0) {
                ///Dense dense compressed case special
                if (dense_dense_comp) {
                    auto target = temp_tiles[0];
                    auto a_data = aC.GetTileSubMatrix(0);
                    auto a_op = blas::Op::NoTrans;
                    auto b_data = aC.GetTileSubMatrix(1);
                    auto b_op = blas::Op::NoTrans;
                    alpha_local = 1;

                    // W += beta * Cu*Cv;

                    target->Gemm(aBeta, a_data, a_op, b_data, b_op, alpha_local, a_data.get().GetLeadingDim(),
                                 std::min(b_data.get().GetNumOfRows(), b_data.get().GetNumOfCols()), aCompressionParameters);

                    int num_of_rows = target->GetTileSubMatrix(0).get().GetNumOfRows();
                    int num_of_cols = target->GetTileSubMatrix(0).get().GetNumOfCols();

                    int64_t c_rank = -1;

                    if (c_rank == std::min(aC.GetTileSubMatrix(0).get().GetNumOfRows(),
                                           aC.GetTileSubMatrix(1).get().GetNumOfCols())) {
                        c_rank = -1;
                    }

                    // todo :: Revisit the Handling of DDC case.
                    aC.ReadjustTile(num_of_rows, num_of_cols, target->GetTileSubMatrix(0).get().GetData(), num_of_rows,
                                    c_rank);

                } else {
                    auto a_data = a_data_holders[a_idx];
                    auto b_data = b_data_holders[b_idx];

                    int64_t c_rank = a_data.get().GetNumOfCols();

                    aC.Gemm(aAlpha, a_data, a_operation, b_data, b_operation, aBeta, a_data.get().GetLeadingDim(),
                            c_rank, aCompressionParameters);
                }
            }

            /// free the allocated temporary tiles
            for (auto tile: temp_tiles) {
                delete tile;
            }
        }

        HCOREPP_INSTANTIATE_CLASS(HCore)

    }
}
