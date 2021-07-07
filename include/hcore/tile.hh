// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_DENSE_TILE_HH
#define HCORE_DENSE_TILE_HH

#include "hcore/base_tile.hh"

#include "blas.hh"

#include <cstdint>

namespace hcore {

// =============================================================================
//
/// Dense tile class.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
template <typename T>
class Tile : public BaseTile<T> {
public:

    // =========================================================================
    //
    /// Empty Dense tile.
    Tile() : BaseTile<T>()
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
    Tile(int64_t m, int64_t n, T* A, int64_t ld,
        blas::Layout layout=blas::Layout::ColMajor) : BaseTile<T>(m, n, A, ld,
        layout)
        {}

    // =========================================================================
    //
    /// [explicit]
    /// Conversion from compressed tile, which creates a shallow copy view of
    /// the base tile.
    /// @param[in,out] tile
    ///     Base tile.
    explicit Tile(BaseTile<T> const& tile) : BaseTile<T>(tile)
        {}

    // =========================================================================
    //
    /// @return const pointer to array data buffer of this tile.
    T const* data() const
        { return this->data_; }

    // =========================================================================
    //
    /// @return pointer to array data buffer of this tile.
    T* data()
        { return this->data_; }

    // =========================================================================
    //
    /// @return column/row stride of this tile.
    int64_t ld() const
        { return this->ld_; }

    // =========================================================================
    //
    /// @return the number of locations in memory between beginnings of
    /// successive array elements of a row, accounting for row/column major
    /// data layout and transposed tiles.
    int64_t row_stride() const {
        if ((this->op() == blas::Op::NoTrans) ==
            (this->layout() == blas::Layout::ColMajor)) {
            return this->ld_;
        }
        else {
            return 1;
        }
    }

    // =========================================================================
    //
    /// @return the number of locations in memory between beginnings of
    /// successive array elements of a column), accounting for row/column major
    /// data layout and transposed tiles.
    int64_t column_stride() const {
        if ((this->op() == blas::Op::NoTrans) ==
            (this->layout() == blas::Layout::ColMajor)) {
            return 1;
        }
        else {
            return this->ld_;
        }
    }

}; // class Tile
}  // namespace hcore

#endif // HCORE_DENSE_TILE_HH
