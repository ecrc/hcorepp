// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TILE_HH
#define HCORE_TILE_HH

#include <cstdint>

#include "blas.hh"

#include "hcore/compressed_tile.hh"
#include "hcore/exception.hh"
#include "hcore/base_tile.hh"

namespace hcore {

template <typename T>
class Tile : public BaseTile<T> {
public:
    Tile() : BaseTile<T>(), ld_(0) {}

    /// Tile that wraps existing (preallocated) memory buffer.
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
         blas::Layout layout = blas::Layout::ColMajor)
        : BaseTile<T>(m, n, A, layout), ld_(ld) {
        hcore_error_if(layout == blas::Layout::ColMajor && ld < m);
        hcore_error_if(layout == blas::Layout::RowMajor && ld < n);
    }

    /// [explicit]
    /// Conversion from compressed tile, creating a shallow copy view of
    /// the base tile.
    /// @param[in,out] tile
    ///     Base tile.
    explicit Tile(CompressedTile<T> const& tile) : BaseTile<T>(tile), ld_(tile.ldu()) {}

    /// @return const pointer to array data buffer.
    T const* data() const { return this->data_; }

    /// @return pointer to array data buffer.
    T* data() { return this->data_; }

    /// @return column (row-major) or row (column-major) stride.
    int64_t ld() const { return ld_; }

    /// Set column (row-major) or row (column-major) stride.
    /// @param[in] ld
    ///     Leading dimension of the data array buffer.
    ///     ld >= m: if layout = blas::Layout::ColMajor, or
    ///     ld >= n: if layout = blas::Layout::RowMajor.
    void ld(int64_t ld) {
        hcore_error_if(
            this->layout_ == blas::Layout::ColMajor && ld < this->m_);
        hcore_error_if(
            this->layout_ == blas::Layout::RowMajor && ld < this->n_);

        ld_ = ld;
    }

    /// @return the number of locations in memory between beginnings of
    /// successive array elements of a row.
    int64_t row_stride() const {
        if ((this->op_ == blas::Op::NoTrans ) ==
            (this->layout_ == blas::Layout::ColMajor)) {
            return ld_;
        }
        else {
            return 1;
        }
    }

    /// @return the number of locations in memory between beginnings of
    /// successive array elements of a column.
    int64_t col_stride() const {
        if ((this->op_ == blas::Op::NoTrans) ==
            (this->layout_ == blas::Layout::ColMajor)) {
            return 1;
        }
        else {
            return ld_;
        }
    }

    /// @return element {i, j} of this tile. The actual value is returned, not a
    /// reference. If op() == blas::Op::ConjTrans then data is conjugated,
    /// taking the layout into account.
    /// @param[in] i
    ///     Row index. 0 <= i < m.
    /// @param[in] j
    ///     Column index. 0 <= j < n.
    T operator()(int64_t i, int64_t j) const {
        hcore_error_if(0 > i || i >= this->m());
        hcore_error_if(0 > j || j >= this->n());

        using blas::conj;

        if (this->op_ == blas::Op::ConjTrans) {
            if (this->layout_ == blas::Layout::ColMajor) {
                return conj(this->data_[j + i*ld_]);
            }
            else {
                return conj(this->data_[i + j*ld_]);
            }
        }
        else if ((this->op_ == blas::Op::NoTrans) ==
                 (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_[i + j*ld_];
        }
        else {
            return this->data_[j + i*ld_];
        }
    }

    /// @return a const reference to element {i, j} of this tile.
    /// If op() == blas::Op::ConjTrans then data isn't conjugated, because a
    /// reference is returned, taking the layout into account.
    /// @param[in] i
    ///     Row index. 0 <= i < m.
    /// @param[in] j
    ///     Column index. 0 <= j < n.
    T const& at(int64_t i, int64_t j) const {
        hcore_error_if(0 > i || i >= this->m());
        hcore_error_if(0 > j || j >= this->n());

        if ((this->op_ == blas::Op::NoTrans) ==
            (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_[i + j*ld_];
        }
        else {
            return this->data_[j + i*ld_];
        }
    }

    /// @return a reference to element {i, j} of this tile.
    /// If op() == blas::Op::ConjTrans then data isn't conjugated, because a
    /// reference is returned, taking the layout into account.
    /// @param[in] i
    ///     Row index. 0 <= i < m.
    /// @param[in] j
    ///     Column index. 0 <= j < n.
    T& at(int64_t i, int64_t j) {
        return const_cast<T&>(static_cast<const Tile>(*this).at(i, j));
    }

private:
    int64_t ld_; ///> Leading dimension.

}; // class Tile
}  // namespace hcore

#endif // HCORE_TILE_HH
