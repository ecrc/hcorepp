
#include <hcorepp/api/hcorepp.hpp>
#include <iostream>
#include <functional>

using namespace hcorepp::operators;
using namespace hcorepp::dataunits;

namespace hcorepp {
    namespace api {

        template<typename T>
        void set_parameters(api::INPUT_TILES input, Tile<T> const &A, Tile<T> const &B, Tile<T> &C,
                            int &num_of_rows, int &num_of_cols, int64_t &leading_dim, int64_t &rank, int iteration) {
            switch (input) {
                case api::DENSE_DENSE_DENSE:
                    num_of_rows = C.GetTileSubMatrix(0).get().GetNumOfRows();
                    num_of_cols = C.GetTileSubMatrix(0).get().GetNumOfCols();
                    leading_dim = C.GetTileSubMatrix(0).get().GetLeadingDim();
                    rank = leading_dim;
                    break;
                case api::DENSE_DENSE_COMPRESSED:
                    num_of_rows = C.GetTileSubMatrix(0).get().GetNumOfRows();
                    num_of_cols = C.GetTileSubMatrix(0).get().GetNumOfCols();
                    leading_dim = C.GetTileSubMatrix(0).get().GetLeadingDim();
                    rank = leading_dim;
                    break;
                case api::COMPRESSED_DENSE_DENSE:
                    num_of_rows = A.GetTileSubMatrix(0).get().GetNumOfRows();
                    num_of_cols = C.GetTileSubMatrix(0).get().GetNumOfCols();
                    leading_dim = num_of_rows;
                    rank = num_of_rows;
                    break;
                case api::COMPRESSED_DENSE_COMPRESSED:
                    num_of_rows = A.GetTileSubMatrix(0).get().GetNumOfRows();
                    num_of_cols = C.GetTileSubMatrix(0).get().GetNumOfCols();
                    leading_dim = num_of_rows;
                    rank = num_of_rows;
                    break;
                case api::COMPRESSED_COMPRESSED_DENSE:
                    if (iteration == 0) {
                        num_of_rows = A.GetTileSubMatrix(0).get().GetNumOfRows();
                        num_of_cols = B.GetTileSubMatrix(0).get().GetNumOfCols();
                        leading_dim = num_of_rows;
                        rank = num_of_rows;
                    } else if (iteration == 1) {
                        num_of_rows = A.GetTileSubMatrix(1).get().GetNumOfRows();
                        num_of_cols = C.GetTileSubMatrix(0).get().GetNumOfCols();
                        leading_dim = num_of_rows;
                        rank = num_of_rows;
                    } else {
                        num_of_rows = C.GetTileSubMatrix(0).get().GetNumOfRows();
                        num_of_cols = C.GetTileSubMatrix(0).get().GetNumOfCols();
                        leading_dim = C.GetTileSubMatrix(0).get().GetLeadingDim();
                        rank = leading_dim;
                    }
                    break;
                case api::COMPRESSED_COMPRESSED_COMPRESSED:
                    if (iteration == 0) {
                        num_of_rows = A.GetTileSubMatrix(0).get().GetNumOfRows();
                        num_of_cols = B.GetTileSubMatrix(0).get().GetNumOfCols();
                        leading_dim = num_of_rows;
                        rank = num_of_rows;
                    } else if (iteration == 1) {
                        num_of_rows = A.GetTileSubMatrix(1).get().GetNumOfRows();
                        num_of_cols = C.GetTileSubMatrix(1).get().GetNumOfCols();
                        leading_dim = num_of_rows;
                        rank = num_of_rows;
                    } else {
                        num_of_rows = C.GetTileSubMatrix(0).get().GetNumOfRows();
                        num_of_cols = C.GetTileSubMatrix(0).get().GetNumOfCols();
                        leading_dim = C.GetTileSubMatrix(0).get().GetLeadingDim();
                        rank = leading_dim;
                    }
                    break;
                case api::DENSE_COMPRESSED_DENSE:
                    num_of_rows = C.GetTileSubMatrix(0).get().GetNumOfRows();
                    num_of_cols = B.GetTileSubMatrix(0).get().GetNumOfCols();
                    leading_dim = num_of_rows;
                    rank = num_of_rows;
                    break;
                case api::DENSE_COMPRESSED_COMPRESSED:
                    num_of_rows = C.GetTileSubMatrix(0).get().GetNumOfRows();
                    num_of_cols = B.GetTileSubMatrix(0).get().GetNumOfRows();
                    leading_dim = num_of_rows;
                    rank = num_of_rows;
                    break;
                default:
                    break;
            }


        }

        template void set_parameters(api::INPUT_TILES, Tile<float> const &, Tile<float> const &, Tile<float> &,
                                     int &, int &, int64_t &, int64_t &, int);

        template void set_parameters(api::INPUT_TILES, Tile<double> const &, Tile<double> const &, Tile<double> &,
                                     int &, int &, int64_t &, int64_t &, int);

        template<typename T>
        void
        gemm(T alpha, Tile<T> const &A, Tile<T> const &B, T beta, Tile<T> &C) {
            int tile_a_size = A.GetNumberOfMatrices();
            int tile_b_size = B.GetNumberOfMatrices();
            int tile_c_size = C.GetNumberOfMatrices();

            api::INPUT_TILES input;

            if (tile_a_size == 1) {
                if (tile_b_size == 1) {
                    if (tile_c_size == 1) {
                        input = api::DENSE_DENSE_DENSE;
                    } else {
                        input = api::DENSE_DENSE_COMPRESSED;
                    }
                } else {
                    if (tile_c_size == 1) {
                        input = api::DENSE_COMPRESSED_DENSE;
                    } else {
                        input = api::DENSE_COMPRESSED_COMPRESSED;
                    }
                }
            } else {
                if (tile_b_size == 1) {
                    if (tile_c_size == 1) {
                        input = api::COMPRESSED_DENSE_DENSE;
                    } else {
                        input = api::COMPRESSED_DENSE_COMPRESSED;
                    }
                } else {
                    if (tile_c_size == 1) {
                        input = api::COMPRESSED_COMPRESSED_DENSE;
                    } else {
                        input = api::COMPRESSED_COMPRESSED_COMPRESSED;
                    }
                }
            }

            int total_input_tiles = tile_a_size + tile_b_size;
            int iterations = 0;
            if (total_input_tiles == 3 || input == api::DENSE_DENSE_COMPRESSED) {
                iterations = 1;
            } else if (total_input_tiles == 4) {
                iterations = 2;
            } else {
                iterations = 0;
            }

            int tile_a_st_idx = tile_a_size - 1;
            int tile_b_st_idx = 0;

            int num_of_rows = 0;
            int num_of_cols = 0;
            int64_t leading_dim = 0;
            int64_t rank = 0;

            std::vector<DenseTile<T> *> temp_tiles;
            temp_tiles.resize(iterations);

            helpers::SvdHelpers helpers;
            int i = 0;

            std::vector<std::reference_wrapper<DataHolder<T>>> a_pairs;
            blas::Op a_operation = A.operation();

            a_pairs.reserve(tile_a_size);
            for (int j = 0; j < tile_a_size; j++) {
                auto a_tile = (A.GetTileSubMatrix(j));
                a_pairs[j]= a_tile;
            }

            std::vector<std::reference_wrapper<DataHolder<T>>> b_pairs;
            blas::Op b_operation = B.operation();

            b_pairs.reserve(tile_b_size);
            for (int j = 0; j < tile_b_size; j++) {
                auto b_tile = (B.GetTileSubMatrix(j));
                b_pairs[j]= b_tile;
            }

            T alpha_local = 1;
            T beta_local = 0;
            if (api::DENSE_DENSE_COMPRESSED) {
                alpha_local = alpha;
            }


            while (iterations > 0) {
                auto a_data = a_pairs[tile_a_st_idx];
                auto a_op = a_operation;
                auto b_data = b_pairs[tile_b_st_idx];
                auto b_op = b_operation;

                api::set_parameters(input, A, B, C, num_of_rows, num_of_cols, leading_dim, rank, i);

                temp_tiles[i] = new DenseTile<T>(num_of_rows, num_of_cols, nullptr, leading_dim);

                temp_tiles[i]->Gemm(alpha_local, a_data, a_op, b_data, b_op, beta_local, a_data.get().GetLeadingDim(),
                                    rank, helpers);

                if (iterations == 2) {
                    auto a_tile = (temp_tiles[i]->GetTileSubMatrix(0));
                    a_operation = temp_tiles[i]->operation();

                    a_pairs[tile_a_st_idx] = a_tile;
                    tile_b_st_idx++;
                } else if (iterations == 1) {
                    if (tile_a_st_idx == 0) {
                        auto a_tile = (temp_tiles[i]->GetTileSubMatrix(0));
                        a_operation = temp_tiles[i]->operation();

                        a_pairs[tile_a_st_idx] = a_tile;
                        tile_b_st_idx++;
                    } else {
                        auto b_tile = temp_tiles[i]->GetTileSubMatrix(0);
                        a_operation = temp_tiles[i]->operation();

                        b_pairs[tile_b_st_idx] = b_tile;
                        tile_a_st_idx--;
                    }
                }
                iterations--;
                i++;
            }


            if (iterations == 0) {
                if (input == api::DENSE_DENSE_COMPRESSED) {
                    auto target = temp_tiles[0];
                    auto a_data = C.GetTileSubMatrix(0);
                    auto a_op = C.operation();
                    auto b_data = C.GetTileSubMatrix(1);
                    auto b_op = C.operation();
                    alpha_local = 1;

                    target->Gemm(beta, a_data, a_op, b_data, b_op, alpha_local, a_data.get().GetLeadingDim(), rank, helpers);

                } else {
                    auto a_data = a_pairs[tile_a_st_idx];
                    auto b_data = b_pairs[tile_b_st_idx];

                    C.Gemm(alpha, a_data, a_operation, b_data, b_operation, beta, a_data.get().GetLeadingDim(), rank,
                           helpers);
                }
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
