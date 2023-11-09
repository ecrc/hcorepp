/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
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
        TileMatrix<T>::TileMatrix(const RawMatrix<T> &aRawMatrix, size_t aRowTileSize, size_t aColumnTileSize,
                                  kernels::RunContext &aContext) :
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
            std::vector<T *> to_delete;
            for (size_t i = 0; i < nt; i++) {
                for (size_t j = 0; j < mt; j++) {
                    auto tile_cols = std::min(aColumnTileSize, aRawMatrix.GetN() - i * aColumnTileSize);
                    auto tile_rows = std::min(aRowTileSize, aRawMatrix.GetM() - j * aRowTileSize);
                    auto tile_data = new T[tile_rows * tile_cols];
                    to_delete.push_back(tile_data);
                }
            }
            if(aContext.SupportsOMP()) {
#pragma omp parallel default(none) shared(nt, mt, aColumnTileSize, aRowTileSize, aRawMatrix, to_delete, aContext)
                {
#pragma omp  for collapse(2)
                    for (size_t i = 0; i < nt; i++) {
                        for (size_t j = 0; j < mt; j++) {
                            auto tile_cols = std::min(aColumnTileSize, aRawMatrix.GetN() - i * aColumnTileSize);
                            auto tile_rows = std::min(aRowTileSize, aRawMatrix.GetM() - j * aRowTileSize);
                            auto st_idx = i * aColumnTileSize * aRawMatrix.GetM() + j * aRowTileSize;
                            auto org_data = &aRawMatrix.GetData()[st_idx];
                            auto tile_data = to_delete[i * mt + j];
                            for (size_t jj = 0; jj < tile_cols; jj++) {
                                for (size_t ii = 0; ii < tile_rows; ii++) {
                                    tile_data[ii + jj * tile_rows] = org_data[ii + jj * aRawMatrix.GetM()];
                                }
                            }
                            this->mMatrixTiles[i][j] = new operators::DenseTile<T>(tile_rows, tile_cols,
                                                                                   tile_data, tile_rows,
                                                                                   blas::Layout::ColMajor,
                                                                                   aContext);
                        }
                    }
                }
            }
            else {
                for (size_t i = 0; i < nt; i++) {
                    for (size_t j = 0; j < mt; j++) {
                        auto tile_cols = std::min(aColumnTileSize, aRawMatrix.GetN() - i * aColumnTileSize);
                        auto tile_rows = std::min(aRowTileSize, aRawMatrix.GetM() - j * aRowTileSize);
                        auto st_idx = i * aColumnTileSize * aRawMatrix.GetM() + j * aRowTileSize;
                        auto org_data = &aRawMatrix.GetData()[st_idx];
                        auto tile_data = to_delete[i * mt + j];
                        for (size_t jj = 0; jj < tile_cols; jj++) {
                            for (size_t ii = 0; ii < tile_rows; ii++) {
                                tile_data[ii + jj * tile_rows] = org_data[ii + jj * aRawMatrix.GetM()];
                            }
                        }
                        this->mMatrixTiles[i][j] = new operators::DenseTile<T>(tile_rows, tile_cols,
                                                                               tile_data, tile_rows,
                                                                               blas::Layout::ColMajor,
                                                                               aContext);
                    }
                }
            }
            aContext.Sync();
            for (auto ptr : to_delete) {
                delete[] ptr;
            }
            mMemory = aRawMatrix.GetM() * aRawMatrix.GetN() * sizeof(T);
        }

        template<typename T>
        TileMatrix<T>::TileMatrix(const RawMatrix<T> &aRawMatrix, size_t aRowTileSize, size_t aColumnTileSize,
                                  const operators::CompressionParameters &aParameters,
                                  kernels::RunContext &aContext) :
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
            std::vector<T *> to_delete;
            for (size_t i = 0; i < nt; i++) {
                for (size_t j = 0; j < mt; j++) {
                    auto tile_cols = std::min(aColumnTileSize, aRawMatrix.GetN() - i * aColumnTileSize);
                    auto tile_rows = std::min(aRowTileSize, aRawMatrix.GetM() - j * aRowTileSize);
                    auto tile_data = new T[tile_rows * tile_cols];
                    to_delete.push_back(tile_data);
                }
            }
            if(aContext.SupportsOMP()) {
#pragma omp parallel default(none) shared(nt, mt, aColumnTileSize, aRowTileSize, aRawMatrix, aParameters, to_delete, aContext)
                {
#pragma omp for  collapse(2)
                    for (size_t i = 0; i < nt; i++) {
                        for (size_t j = 0; j < mt; j++) {
                            auto tile_cols = std::min(aColumnTileSize, aRawMatrix.GetN() - i * aColumnTileSize);
                            auto tile_rows = std::min(aRowTileSize, aRawMatrix.GetM() - j * aRowTileSize);
                            auto tile_data = to_delete[i * mt + j];
                            auto st_idx = i * aColumnTileSize * aRawMatrix.GetM() + j * aRowTileSize;
                            auto org_data = &aRawMatrix.GetData()[st_idx];
                            for (size_t jj = 0; jj < tile_cols; jj++) {
                                for (size_t ii = 0; ii < tile_rows; ii++) {
                                    tile_data[ii + jj * tile_rows] = org_data[ii + jj * aRawMatrix.GetM()];
                                }
                            }

                            this->mMatrixTiles[i][j] = new operators::CompressedTile<T>(tile_rows, tile_cols,
                                                                                        tile_data, tile_rows,
                                                                                        aParameters,
                                                                                        blas::Layout::ColMajor,
                                                                                        aContext);
                        }
                    }
                }
            }
            else {
                for (size_t i = 0; i < nt; i++) {
                    for (size_t j = 0; j < mt; j++) {
                        auto tile_cols = std::min(aColumnTileSize, aRawMatrix.GetN() - i * aColumnTileSize);
                        auto tile_rows = std::min(aRowTileSize, aRawMatrix.GetM() - j * aRowTileSize);
                        auto tile_data = to_delete[i * mt + j];
                        auto st_idx = i * aColumnTileSize * aRawMatrix.GetM() + j * aRowTileSize;
                        auto org_data = &aRawMatrix.GetData()[st_idx];
                        for (size_t jj = 0; jj < tile_cols; jj++) {
                            for (size_t ii = 0; ii < tile_rows; ii++) {
                                tile_data[ii + jj * tile_rows] = org_data[ii + jj * aRawMatrix.GetM()];
                            }
                        }

                        this->mMatrixTiles[i][j] = new operators::CompressedTile<T>(tile_rows, tile_cols,
                                                                                    tile_data, tile_rows,
                                                                                    aParameters,
                                                                                    blas::Layout::ColMajor,
                                                                                    aContext);
                    }
                }
            }
            aContext.Sync();
            for (auto ptr : to_delete) {
                delete[] ptr;
            }
            mMemory = 0;
            for (size_t i = 0; i < nt; i++) {
                for (size_t j = 0; j < mt; j++) {
                    auto tile_cols = std::min(aColumnTileSize, aRawMatrix.GetN() - i * aColumnTileSize);
                    auto tile_rows = std::min(aRowTileSize, aRawMatrix.GetM() - j * aRowTileSize);
                    mMemory += ((tile_rows + tile_cols) *
                            ((operators::CompressedTile<T>*)(this->mMatrixTiles[i][j]))->GetTileRank() * sizeof(T));
                }
            }
        }

        template<typename T>
        RawMatrix<T> TileMatrix<T>::ToRawMatrix(kernels::RunContext &aContext) {
            size_t full_array_index;
            size_t tile_index_r;
            size_t tile_index_c;
            size_t index_in_tile;
            RawMatrix<T> ret(this->mM, this->mN);
            auto data_ptr = ret.GetData();
            for (size_t cols = 0; cols < mN; cols += mColTileSize) {
                for (size_t rows = 0; rows < mM; rows += mRowTileSize) {
                    size_t tile_rows = std::min(mRowTileSize, mM - rows);
                    size_t tile_cols = std::min(mColTileSize, mN - cols);
                    tile_index_r = rows / mRowTileSize;
                    tile_index_c = cols / mColTileSize;
                    auto &tile = this->mMatrixTiles[tile_index_c][tile_index_r];
                    T *temp;
                    if (tile->isDense()) {
                        dataunits::DataHolder<T>& sub_matrix = tile->GetDataHolder();
                        size_t n = sub_matrix.GetNumOfCols();
                        size_t m = sub_matrix.GetNumOfRows();
                        temp = new T[n * m];
                        hcorepp::memory::Memcpy<T>(temp, sub_matrix.GetData(), n * m,
                                                   aContext,
                                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                        aContext.Sync();
                    } else {
                        auto m = tile->GetNumOfRows();
                        auto n = tile->GetNumOfCols();
                        auto rank = ((operators::CompressedTile<T>*)tile)->GetTileRank();
                        auto *u = ((operators::CompressedTile<T>*)tile)->GetUMatrix();
                        auto *v = ((operators::CompressedTile<T>*)tile)->GetVMatrix();
                        auto uld = ((operators::CompressedTile<T>*)tile)->GetULeadingDim();
                        auto vld = ((operators::CompressedTile<T>*)tile)->GetVLeadingDim();
                        size_t num_elements = m * rank;
                        T *cu = new T[num_elements];
                        hcorepp::memory::Memcpy<T>(cu, u, num_elements,
                                                   aContext,
                                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                        num_elements = rank * n;
                        T *cv = new T[num_elements];
                        hcorepp::memory::Memcpy<T>(cv, v, num_elements,
                                                   aContext,
                                                   hcorepp::memory::MemoryTransfer::DEVICE_TO_HOST);
                        aContext.Sync();
                        temp = new T[n * m];
                        memset(temp, 0, m * n * sizeof(T));

                        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                                   m, n, rank, 1.0, cu,
                                   uld, cv,
                                   vld, 0.0, temp, m);
                        delete[] cu;
                        delete[] cv;
                    }
                    for (size_t i = 0; i < tile_cols; i++) {
                        for (size_t j = 0; j < tile_rows; j++) {
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
