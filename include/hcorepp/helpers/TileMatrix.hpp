/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
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
             *
             * @param[in] aContext
             * The run context to utilize for the tiles.
             */
            TileMatrix(const RawMatrix <T> &aRawMatrix, size_t aRowTileSize, size_t aColumnTileSize,
                       kernels::RunContext &aContext);

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
             *
             * @param[in] aContext
             * The run context to utilize for the tiles.
             */
            TileMatrix(const RawMatrix <T> &aRawMatrix, size_t aRowTileSize, size_t aColumnTileSize,
                       const operators::CompressionParameters &aParameters, kernels::RunContext &aContext);

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
            operators::Tile <T> *GetTile(size_t aRowIndex, size_t aColIndex) {
                return this->mMatrixTiles[aColIndex][aRowIndex];
            }

            /**
             * @brief
             * Getter for the number of tiles in the row direction.
             *
             * @return
             * The count of tiles in the row direction.
             */
            size_t GetRowTileCount() const {
                return this->mRowTileCount;
            }

            /**
             * @brief
             * Getter for the number of tiles in the column direction.
             *
             * @return
             * The count of tiles in the column direction.
             */
            size_t GetColTileCount() const {
                return this->mColTileCount;
            }

            /**
             * @brief
             * Getter for the tile size in the row direction.
             *
             * @return
             * The size of each tile in the row direction.
             */
            size_t GetRowTileSize() const {
                return this->mRowTileSize;
            }

            /**
             * @brief
             * Getter for the tile size in the column direction.
             *
             * @return
             * The size of each tile in the column direction.
             */
            size_t GetColTileSize() const {
                return this->mColTileSize;
            }

            /**
             * @brief
             * Getter for the matrix total size in the row direction.
             *
             * @return
             * The total number of elements of the matrix in row direction.
             */
            size_t GetM() const {
                return this->mM;
            }

            /**
             * @brief
             * Getter for the matrix total size in the column direction.
             *
             * @return
             * The total number of elements of the matrix in column direction.
             */
            size_t GetN() const {
                return this->mN;
            }

            /**
             * @brief
             * Convert data back to raw matrix from tile format.
             *
             * @return
             * A raw matrix containing all the data inside the tile matrix.
             */
            RawMatrix <T> ToRawMatrix(kernels::RunContext &aContext);

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
            size_t mRowTileCount;
            /// Number of tiles inside the matrix in the column direction.
            size_t mColTileCount;
            /// Size of each tile in the row direction.
            size_t mRowTileSize;
            /// Size of each tile in the column direction.
            size_t mColTileSize;
            /// Total size in row direction.
            size_t mM;
            /// Total size in column direction.
            size_t mN;
            /// The memory in bytes occupied by the tiles in the tile matrix.
            size_t mMemory;
        };
    }//namespace helpers
}//namespace hcorepp

#endif //HCOREPP_HELPERS_TILE_MATRIX_HPP
