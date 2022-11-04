/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCOREPP_HELPERS_TILE_MATRIX_HPP
#define HCOREPP_HELPERS_TILE_MATRIX_HPP

#include "RawMatrix.hpp"
#include <hcorepp/operators/interface/Tile.hpp>

namespace hcorepp {
    namespace helpers {
        /**
         * @brief
         * Class encapsulating a tile-based matrix.
         *
         * @tparam T
         * The type of each element inside the matrix
         */
        template<typename T>
        class TileMatrix {
        public:
            /**
             * @brief
             * Constructor for a fully dense tile matrix.
             *
             * @param[in] aRawMatrix
             * The raw matrix to break into tiles.
             *
             * @param[in] aRowTileSize
             * The number of rows in each tile.
             *
             * @param[in] aColumnTileSize
             * The number of columns in each tile.
             */
            TileMatrix(const RawMatrix <T> &aRawMatrix, int64_t aRowTileSize, int64_t aColumnTileSize);

            /**
             * @brief
             * Constructor for a fully compressed tile matrix.
             *
             * @param[in] aRawMatrix
             * The raw matrix to break into tiles.
             *
             * @param[in] aRowTileSize
             * The number of rows in each tile.
             *
             * @param[in] aColumnTileSize
             * The number of columns in each tile.
             *
             * @param[in] aParameters
             * The parameters used for the compression.
             */
            TileMatrix(const RawMatrix <T> &aRawMatrix, int64_t aRowTileSize, int64_t aColumnTileSize,
                       const operators::CompressionParameters &aParameters);

            /**
             * @brief
             * Getter for a tile from inside the matrix given its index.
             *
             * @param[in] aRowIndex
             * The row index of the tile.
             *
             * @param[in] aColIndex
             * The column index of the tile.
             *
             * @return
             * A reference to the corresponding tile object.
             */
            operators::Tile <T> *GetTile(int64_t aRowIndex, int64_t aColIndex) {
                return this->mMatrixTiles[aColIndex][aRowIndex];
            }

            /**
             * @brief
             * Getter for the number of tiles in the row direction.
             *
             * @return
             * The count of tiles in the row direction.
             */
            int64_t GetRowTileCount() const {
                return this->mRowTileCount;
            }

            /**
             * @brief
             * Getter for the number of tiles in the column direction.
             *
             * @return
             * The count of tiles in the column direction.
             */
            int64_t GetColTileCount() const {
                return this->mColTileCount;
            }

            /**
             * @brief
             * Getter for the tile size in the row direction.
             *
             * @return
             * The size of each tile in the row direction.
             */
            int64_t GetRowTileSize() const {
                return this->mRowTileSize;
            }

            /**
             * @brief
             * Getter for the tile size in the column direction.
             *
             * @return
             * The size of each tile in the column direction.
             */
            int64_t GetColTileSize() const {
                return this->mColTileSize;
            }

            /**
             * @brief
             * Getter for the matrix total size in the row direction.
             *
             * @return
             * The total number of elements of the matrix in row direction.
             */
            int64_t GetM() const {
                return this->mM;
            }

            /**
             * @brief
             * Getter for the matrix total size in the column direction.
             *
             * @return
             * The total number of elements of the matrix in column direction.
             */
            int64_t GetN() const {
                return this->mN;
            }

            /**
             * @brief
             * Convert data back to raw matrix from tile format.
             *
             * @return
             * A raw matrix containing all the data inside the tile matrix.
             */
            RawMatrix <T> ToRawMatrix();

            /**
             * @brief
             * Returns the memory currently occupied by the matrix data.
             *
             * @return
             * The number of bytes currently occupied by the matrix data.
             */
            size_t GetMemoryFootprint();

            /**
             * @brief
             * Default destructor.
             */
            ~TileMatrix();

        private:
            /// 2D vector of all the different tiles composing the matrix
            std::vector<std::vector<operators::Tile < T> *>> mMatrixTiles;
            /// Number of tiles inside the matrix in the row direction.
            int64_t mRowTileCount;
            /// Number of tiles inside the matrix in the column direction.
            int64_t mColTileCount;
            /// Size of each tile in the row direction.
            int64_t mRowTileSize;
            /// Size of each tile in the column direction.
            int64_t mColTileSize;
            /// Total size in row direction.
            int64_t mM;
            /// Total size in column direction.
            int64_t mN;
            /// The memory in bytes occupied by the tiles in the tile matrix.
            size_t mMemory;
        };
    }//namespace helpers
}//namespace hcorepp

#endif //HCOREPP_HELPERS_TILE_MATRIX_HPP
