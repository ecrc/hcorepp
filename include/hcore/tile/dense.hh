// Copyright (c) 2017-2021, King Abdullah University of Science and Technology
// (KAUST). All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TILE_DENSE_HH
#define HCORE_TILE_DENSE_HH

#include "hcore/tile/tile.hh"

#include "blas.hh"

#include <cstdint>

namespace hcore {

/// Dense tile class.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
template <typename T>
class DenseTile : public Tile<T> {
public:

    // =========================================================================
    //
    /// Empty Dense tile.
    DenseTile() : Tile<T>()
        {}

    // =========================================================================
    //
    /// Dense tile that wraps existing (preallocated) memory buffer.
    /// @param[in] m
    ///     Number of rows of the tile. m >= 0.
    /// @param[in] n
    ///     Number of columns of the tile. b >= 0.
    /// @param[in,out] A
    ///     The m-by-n matrix tile, stored in an array data buffer of size:
    ///     ld-by-n: if layout = blas::Layout::ColMajor, or
    ///     ld-by-m: if layout = blas::Layout::RowMajor.
    /// @param[in] ld
    ///     Leading dimension of the data array buffer.
    ///     ld >= m: if layout = blas::Layout::ColMajor, or
    ///     ld >= n: if layout = blas::Layout::RowMajor.
    /// @param[in] layout
    ///     The physical ordering of matrix elements in the data array buffer.
    ///     blas::Layout::ColMajor: column elements are 1-strided (default), or
    ///     blas::Layout::RowMajor: row elements are 1-strided.
    DenseTile(int64_t m, int64_t n, T* A, int64_t ld,
        blas::Layout layout=blas::Layout::ColMajor) : Tile<T>(m, n, A, ld,
        layout)
        {}

    // =========================================================================
    //
    /// [explicit]
    /// Conversion from compressed tile, which creates a shallow copy view of
    /// the base tile.
    /// @param[in,out] tile
    ///     Base tile.
    explicit DenseTile(Tile<T> const& tile) : Tile<T>(tile)
        {}

    /// @return const pointer to array data buffer of this tile.
    T const* data() const
        { return this->data_; }

    /// @return pointer to array data buffer of this tile.
    T* data()
        { return this->data_; }

    /// @return column/row stride of this tile.
    int64_t ld() const
        { return this->ld_; }

}; // class DenseTile
}  // namespace hcore

#endif // HCORE_TILE_DENSE_HH
