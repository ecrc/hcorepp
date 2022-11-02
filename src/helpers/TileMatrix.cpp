/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#include <cstring>
#include <hcorepp/helpers/TileMatrix.hpp>
#include <hcorepp/common/Definitions.hpp>
#include <hcorepp/helpers/LapackWrappers.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>
#include <hcorepp/kernels/memory.hpp>

namespace hcorepp {
    namespace helpers {

        template<typename T>
        TileMatrix<T>::TileMatrix(const RawMatrix<T> &aRawMatrix, int64_t aRowTileSize, int64_t aColumnTileSize) :
                mColTileSize(aColumnTileSize), mRowTileSize(aRowTileSize), mRowTileCount(0), mColTileCount(0),
                mM(aRawMatrix.GetM()), mN(aRawMatrix.GetN()) {
            // Get number of tiles in first-direction.
            auto mt = (aRawMatrix.GetM() / aRowTileSize);
            if (aRawMatrix.GetM() % aRowTileSize > 0) {
                mt += 1;
            }
            // Get number of tiles in second-direction.
            auto nt = (aRawMatrix.GetN() / aColumnTileSize);
            if (aRawMatrix.GetN() % aColumnTileSize > 0) {
                nt += 1;
            }
            mRowTileCount = mt;
            mColTileCount = nt;
            mMemory = 0;
            this->mMatrixTiles.resize(nt, std::vector<operators::Tile<T> *>(mt, nullptr));
            int64_t tile_cols;
            int64_t tile_rows;
            for (int i = 0; i < nt; i++) {
                tile_cols = std::min(aColumnTileSize, aRawMatrix.GetN() - i * aColumnTileSize);
                for (int j = 0; j < mt; j++) {
                    tile_rows = std::min(aRowTileSize, aRawMatrix.GetM() - j * aRowTileSize);
                    auto tile_data = new T[tile_rows * tile_cols];
                    auto st_idx = i * aColumnTileSize * aRawMatrix.GetM() + j * aRowTileSize;
                    auto org_data = &aRawMatrix.GetData()[st_idx];
                    for (int jj = 0; jj < tile_cols; jj++) {
                        for (int ii = 0; ii < tile_rows; ii++) {
                            tile_data[ii + jj * tile_rows] = org_data[ii + jj * aRawMatrix.GetM()];
                        }
                    }
                    mMemory += (tile_rows * tile_cols) * sizeof(T);
                    this->mMatrixTiles[i][j] = new operators::DenseTile<T>(tile_rows, tile_cols,
                                                                           tile_data, tile_rows,
                                                                           blas::Layout::ColMajor);

                    delete[] tile_data;
                }
            }
        }

        template<typename T>
        TileMatrix<T>::TileMatrix(const RawMatrix<T> &aRawMatrix, int64_t aRowTileSize, int64_t aColumnTileSize,
                                  double aAccuracy) :
                mColTileSize(aColumnTileSize), mRowTileSize(aRowTileSize), mRowTileCount(0), mColTileCount(0),
                mM(aRawMatrix.GetM()), mN(aRawMatrix.GetN()) {
            // Get number of tiles in first-direction.
            auto mt = (aRawMatrix.GetM() / aRowTileSize);
            if (aRawMatrix.GetM() % aRowTileSize > 0) {
                mt += 1;
            }
            // Get number of tiles in second-direction.
            auto nt = (aRawMatrix.GetN() / aColumnTileSize);
            if (aRawMatrix.GetN() % aColumnTileSize > 0) {
                nt += 1;
            }
            mRowTileCount = mt;
            mColTileCount = nt;
            this->mMatrixTiles.resize(nt, std::vector<operators::Tile<T> *>(mt, nullptr));
            int64_t tile_cols;
            int64_t tile_rows;
            mMemory = 0;
            for (int i = 0; i < nt; i++) {
                tile_cols = std::min(aColumnTileSize, aRawMatrix.GetN() - i * aColumnTileSize);
                for (int j = 0; j < mt; j++) {
                    tile_rows = std::min(aRowTileSize, aRawMatrix.GetM() - j * aRowTileSize);
                    auto tile_data = new T[tile_rows * tile_cols];
                    auto st_idx = i * aColumnTileSize * aRawMatrix.GetM() + j * aRowTileSize;
                    auto org_data = &aRawMatrix.GetData()[st_idx];
                    for (int jj = 0; jj < tile_cols; jj++) {
                        for (int ii = 0; ii < tile_rows; ii++) {
                            tile_data[ii + jj * tile_rows] = org_data[ii + jj * aRawMatrix.GetM()];
                        }
                    }

                    int64_t rk;
                    T *auv;
                    int64_t min_m_n = std::min(tile_rows, tile_cols);
                    auto sigma = (blas::real_type<T> *) malloc(min_m_n * sizeof(blas::real_type<T>));
                    auto u = (T *) malloc(tile_rows * min_m_n * sizeof(T));
                    auto vt = (T *) malloc(min_m_n * tile_cols * sizeof(T));

                    auto a_temp = (T *) malloc(tile_rows * tile_cols * sizeof(T));
                    memcpy((void *) a_temp, (void *) tile_data, tile_rows * tile_cols * sizeof(T));
                    lapack_gesvd(common::Job::SomeVec, common::Job::SomeVec,
                                 tile_rows, tile_cols, a_temp, tile_rows, sigma, u, tile_rows, vt,
                                 min_m_n);
                    rk = 0;
                    while (sigma[rk] >= aAccuracy && rk < min_m_n) {
                        rk++;
                        if (rk < min_m_n) {
                            continue;
                        } else {
                            break;
                        }

                    }

                    // todo: more conservative max rank assumption, e.g., min_m_n / 3.
                    int64_t max_rk = min_m_n / 2;
                    if (rk > max_rk) {
                        rk = max_rk;
                    }
                    // Ensure at least rank is 1.
                    rk = std::max(rk, 1L);

                    // VT eats Sigma.
                    // todo: we may need to have uplo parameter:
                    //       scale VT, if Lower, or scale U otherwise.
                    for (int64_t i_r = 0; i_r < rk; ++i_r) {
                        for (int j_r = 0; j_r < tile_cols; j_r++) {
                            vt[i_r + j_r * min_m_n] *= sigma[i_r];
                        }
                    }

                    auv = (T *) malloc((tile_rows + tile_cols) * rk * sizeof(T));

                    memcpy((void *) auv, (void *) u, (tile_rows * rk) * sizeof(T));

                    lapack_lacpy(common::MatrixType::General, rk, tile_cols, vt, min_m_n,
                                 &auv[tile_rows * rk], rk);

                    free(u);
                    free(vt);
                    free(sigma);
                    free(a_temp);
                    mMemory += (tile_rows + tile_cols) * rk * sizeof(T);
                    this->mMatrixTiles[i][j] = new operators::CompressedTile<T>(tile_rows, tile_cols, auv,
                                                                                tile_rows, rk, blas::Layout::ColMajor);

                    delete[] tile_data;
                    free(auv);
                }
            }
        }

        template<typename T>
        RawMatrix<T> TileMatrix<T>::ToRawMatrix() {
            size_t full_array_index;
            size_t tile_index_r;
            size_t tile_index_c;
            size_t index_in_tile;
            RawMatrix<T> ret(this->mM, this->mN);
            auto data_ptr = ret.GetData();
            for (int64_t cols = 0; cols < mN; cols += mColTileSize) {
                for (int64_t rows = 0; rows < mM; rows += mRowTileSize) {
                    int64_t tile_rows = std::min(mRowTileSize, mM - rows);
                    int64_t tile_cols = std::min(mColTileSize, mN - cols);
                    tile_index_r = rows / mRowTileSize;
                    tile_index_c = cols / mColTileSize;
                    auto tile = this->mMatrixTiles[tile_index_c][tile_index_r];
                    T *temp;
                    if (tile->GetNumberOfMatrices() == 1) {
                        auto &sub_matrix = tile->GetTileSubMatrix(0).get();
                        int n = sub_matrix.GetNumOfCols();
                        int m = sub_matrix.GetNumOfRows();
                        temp = new T[n * m];
                        hcorepp::memory::Memcpy<T>(temp, sub_matrix.GetData(), n * m,
                                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                    } else {
                        auto m = tile->GetTileSubMatrix(0).get().GetNumOfRows();
                        auto n = tile->GetTileSubMatrix(1).get().GetNumOfCols();
                        auto rank = tile->GetTileSubMatrix(0).get().GetNumOfCols();
                        auto &sub0 = tile->GetTileSubMatrix(0).get();
                        auto &sub1 = tile->GetTileSubMatrix(1).get();
                        size_t num_elements = sub0.GetNumOfCols() * sub0.GetNumOfRows();
                        T *cu = new T[num_elements];
                        hcorepp::memory::Memcpy<T>(cu, sub0.GetData(), num_elements,
                                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                        num_elements = sub1.GetNumOfCols() * sub1.GetNumOfRows();
                        T *cv = new T[num_elements];
                        hcorepp::memory::Memcpy<T>(cv, sub1.GetData(), num_elements,
                                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                        temp = new T[n * m];
                        memset(temp, 0, m * n * sizeof(T));

                        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                                   m, n, rank, 1.0, cu,
                                   tile->GetTileSubMatrix(0).get().GetLeadingDim(), cv,
                                   tile->GetTileSubMatrix(1).get().GetLeadingDim(), 0.0, temp, m);
                        delete[] cu;
                        delete[] cv;
                    }
                    for (int i = 0; i < tile_cols; i++) {
                        for (int j = 0; j < tile_rows; j++) {
                            index_in_tile = i * tile_rows + j;
                            full_array_index = rows + j + ((cols + i) * mM);
                            data_ptr[full_array_index] = temp[index_in_tile];
                        }
                    }
                    delete[] temp;
                }
            }
            return ret;
        }

        template<typename T>
        size_t TileMatrix<T>::GetMemoryFootprint() {
            return this->mMemory;
        }

        template<typename T>
        TileMatrix<T>::~TileMatrix() {
            for (auto &sub_vector : this->mMatrixTiles) {
                for (auto &tile : sub_vector) {
                    delete tile;
                }
            }
        }

        HCOREPP_INSTANTIATE_CLASS(TileMatrix)

    }//namespace helpers
}//namespace hcorepp