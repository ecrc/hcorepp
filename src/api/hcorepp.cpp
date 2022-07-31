
#include <hcorepp/api/hcorepp.hpp>
#include <iostream>
#include <functional>
#include <cstring>

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

            int total_input_tiles = tile_a_size + tile_b_size;
            bool dense_dense_comp = (total_input_tiles == 2 && tile_c_size == 2);

            int iterations = 0;
            if (total_input_tiles == 3 || dense_dense_comp) {
                iterations = 1;
            } else if (total_input_tiles == 4) {
                iterations = 2;
            } else {
                iterations = 0;
            }

            blas::Op a_operation = A.operation();
            std::vector<std::reference_wrapper<DataHolder<T>>> a_pairs;
            a_pairs.reserve(tile_a_size);
            for (int j = 0; j < tile_a_size; j++) {
                a_pairs[j] = A.GetTileSubMatrix(j);
            }

            blas::Op b_operation = B.operation();
            std::vector<std::reference_wrapper<DataHolder<T>>> b_pairs;
            b_pairs.reserve(tile_b_size);
            for (int j = 0; j < tile_b_size; j++) {
                b_pairs[j] = B.GetTileSubMatrix(j);
            }

            T alpha_local = 1;
            T beta_local = 0;
            if (dense_dense_comp) {
                alpha_local = alpha;
            }

            std::vector<DenseTile<T> *> temp_tiles;
//            temp_tiles.reserve(iterations);
            helpers::SvdHelpers helpers;
            int tile_a_st_idx = tile_a_size - 1;
            int tile_b_st_idx = 0;

            while (iterations > 0) {
                std::cout << " iteration  " << iterations << "\n";
                auto a_data = a_pairs[tile_a_st_idx];
                auto a_op = a_operation;
                auto b_data = b_pairs[tile_b_st_idx];
                auto b_op = b_operation;

//                std::cout << " INput A \n";
//
//                int m = a_data.get().GetNumOfRows();
//                int n = a_data.get().GetNumOfCols();
//                T *output = a_data.get().GetData();

//                for (int i = 0; i < m; i++) {
//                    std::cout << "{ ";
//                    for (int j = 0; j < n; j++) {
//                        int index = i * n + j;
////                REQUIRE(output[index] == matrix_C[i][j]);
//                        std::cout << output[index] << ", ";
//                    }
//                    std::cout << "} \n";
//                }

//                std::cout << " INput B \n";
//
//                m = b_data.get().GetNumOfRows();
//                n = b_data.get().GetNumOfCols();
//                output = b_data.get().GetData();
//
//                for (int i = 0; i < m; i++) {
//                    std::cout << "{ ";
//                    for (int j = 0; j < n; j++) {
//                        int index = i * n + j;
////                REQUIRE(output[index] == matrix_C[i][j]);
//
//                        std::cout << output[index] << ", ";
//                    }
//                    std::cout << "} \n";
//                }

//                auto temp_data_holder = new DataHolder<T>(a_data.get().GetNumOfRows(), b_data.get().GetNumOfCols(),
//                                                          a_data.get().GetNumOfRows(), nullptr);
//
//                blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
//                           a_data.get().GetNumOfRows(), b_data.get().GetNumOfCols(), a_data.get().GetNumOfCols(),
//                           1, (const T *) a_data.get().GetData(), a_data.get().GetLeadingDim(),
//                           (const T *) b_data.get().GetData(), b_data.get().GetLeadingDim(),
//                           0, temp_data_holder->GetData(), temp_data_holder->GetLeadingDim());
//
//                output = temp_data_holder->GetData();
//
//                m = temp_data_holder->GetNumOfRows();
//                n = temp_data_holder->GetNumOfCols();
//
//                std::cout << " Output \n";
//
//                for (int i = 0; i < m; i++) {
//                    std::cout << "{ ";
//                    for (int j = 0; j < n; j++) {
//                        int index = i * m + j;
////                REQUIRE(output[index] == matrix_C[i][j]);
//                        std::cout << output[index] << ", ";
//                    }
//                    std::cout << "} \n";
//                }
//
//                delete temp_data_holder;
//
//                return;
//
//                std::cout << " Temp tile properties : " << " num of rows :  " << a_data.get().GetNumOfRows()
//                          << " Num of cols :" << a_data.get().GetNumOfCols() << " LEading dim "
//                          << a_data.get().GetLeadingDim() << "\n";
//                std::cout << " Tile B properties : " << " num of rows :  " << b_data.get().GetNumOfRows()
//                          << " Num of cols :" << b_data.get().GetNumOfCols() << " LEading dim "
//                          << b_data.get().GetLeadingDim() << "\n";

                temp_tiles.emplace_back(new DenseTile<T>(a_data.get().GetNumOfRows(), b_data.get().GetNumOfCols(),
                                                         nullptr, a_data.get().GetLeadingDim()));


                auto tile = temp_tiles.back();


                tile->Gemm(alpha_local, a_data, a_op, b_data, b_op, beta_local, a_data.get().GetLeadingDim(),
                           std::min(b_data.get().GetNumOfRows(), b_data.get().GetNumOfCols()), helpers);

//                output = tile->GetTileSubMatrix(0).get().GetData();
//
//                m = tile->GetTileSubMatrix(0).get().GetNumOfRows();
//                n = tile->GetTileSubMatrix(0).get().GetNumOfCols();
//
//                std::cout << " Output \n";
//
//                for (int i = 0; i < m; i++) {
//                    std::cout << "{ ";
//                    for (int j = 0; j < n; j++) {
//                        int index = i * n + j;
////                REQUIRE(output[index] == matrix_C[i][j]);
//                        std::cout << output[index] << ", ";
//                    }
//                    std::cout << "} \n";
//                }

                if (iterations == 2) {
                    auto a_data_holder = (tile->GetTileSubMatrix(0));
                    a_operation = tile->operation();

                    a_pairs[tile_a_st_idx] = a_data_holder;
                    tile_b_st_idx++;
                } else if (iterations == 1) {
                    if (tile_a_st_idx == 0) {
                        auto a_tile = (tile->GetTileSubMatrix(0));
                        a_operation = tile->operation();

                        a_pairs[tile_a_st_idx] = a_tile;
                        tile_b_st_idx++;
                    } else {
                        auto b_tile = tile->GetTileSubMatrix(0);
                        b_operation = tile->operation();

                        b_pairs[tile_b_st_idx] = b_tile;
                        tile_a_st_idx--;
                    }
                }
                iterations--;
            }


            if (iterations == 0) {

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


                    int64_t c_rank;
                    c_rank = C.GetTileSubMatrix(0).get().GetNumOfCols();

                    if (c_rank == std::min(C.GetTileSubMatrix(0).get().GetNumOfRows(),
                                           C.GetTileSubMatrix(1).get().GetNumOfCols())) {
                        c_rank = -1;
                    }

                    C.ReadjustTile(target->GetTileSubMatrix(0).get().GetNumOfRows(),
                                   target->GetTileSubMatrix(0).get().GetNumOfCols(),
                                   target->GetTileSubMatrix(0).get().GetData(),num_of_rows, c_rank);

//                    C.GetTileSubMatrix(0).get().Resize(num_of_rows, c_rank, num_of_rows);
//                    C.GetTileSubMatrix(1).get().Resize(c_rank, num_of_cols, c_rank);
//
//                    C.GetTileSubMatrix(0).get().CopyDataArray(0, target->GetTileSubMatrix(0).get().GetData(),
//                                                              num_of_rows * c_rank);
//                    C.GetTileSubMatrix(1).get().CopyDataArray(0,
//                                                              &target->GetTileSubMatrix(0).get().GetData()[num_of_rows *
//                                                                                                           c_rank],
//                                                              c_rank * num_of_cols);
                } else {
                    auto a_data = a_pairs[tile_a_st_idx];
                    auto b_data = b_pairs[tile_b_st_idx];

                    int64_t c_rank;
                    c_rank = a_data.get().GetNumOfCols();

                    C.Gemm(alpha, a_data, a_operation, b_data, b_operation, beta, a_data.get().GetLeadingDim(),
                           c_rank, helpers);
                }
            }

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
