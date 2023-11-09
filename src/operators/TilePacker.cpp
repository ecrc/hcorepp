#include "hcorepp/operators/interface/TilePacker.hpp"


template<typename T>
std::pair<hcorepp::operators::TileMetadata *, T *>

hcorepp::operators::TilePacker<T>::UnPackTile(hcorepp::operators::Tile<T> &aTile,
                                              const hcorepp::kernels::RunContext &aContext) {
    return aTile.UnPackTile(aContext);
}

template<typename T>
hcorepp::operators::Tile<T> *
hcorepp::operators::TilePacker<T>::PackTile(hcorepp::operators::TileMetadata aMetadata, T * apDataArray,
                                            const hcorepp::kernels::RunContext &aContext) {
    Tile<T> *tile;

    if (aMetadata.mType == DENSE) {
        tile = new DenseTile<T>();
    } else {
        tile = new CompressedTile<T>();
    }
    tile->PackTile(aMetadata, apDataArray, aContext);

    return tile;
}

HCOREPP_INSTANTIATE_CLASS(hcorepp::operators::TilePacker)
