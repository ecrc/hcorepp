/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_OPERATORS_INTERFACE_TILE_PACKER_HPP
#define HCOREPP_OPERATORS_INTERFACE_TILE_PACKER_HPP

#include <vector>
#include <hcorepp/operators/interface/Tile.hpp>
#include <hcorepp/operators/concrete/Dense.hpp>
#include <hcorepp/operators/concrete/Compressed.hpp>

namespace hcorepp {
    namespace operators {

        template<typename T>
        class TilePacker {
        public:

            /**
             * @brief
             * Virtual destructor to allow correct destruction of concrete Tile classes.
             */
            virtual ~TilePacker() = default;

            /**
             * @brief
             * Unpack tile function extracting the tile metadata and the actual data arrays.
             * The data arrays are allocated on heap and are the responsibility of the user to delete them.
             *@param[in] aTile
             * Tile to be unpacked
             *
             * @param[in] aContext
             * The runtime context
             *
             * @return
             * pair of tile metadata and data array.
             */
            static std::pair<TileMetadata *, T *>
            UnPackTile(Tile<T> &aTile, const hcorepp::kernels::RunContext &aContext);

            /**
             * @brief
             * Pack tile through constructing dense or compressed tile based on the data arrays and metadata sent.
             *
             * @param[in] aMetadata
             * Tile metadata
             * @param[in] aDataArrays
             * data arrays
             * @param aContext
             * The runtime context.
             *
             * @return
             * Tile created - The tile is allocated on heap so its the responsibility of the user to delete it.
             */
            static Tile<T> *
            PackTile(TileMetadata aMetadata, T * aDataArrays, const kernels::RunContext &aContext);

        private:

            TilePacker() = default;
        };

    }
}

#endif //HCOREPP_OPERATORS_INTERFACE_TILE_HPP
