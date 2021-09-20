// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TILE_HH
#define HCORE_TILE_HH

#include <algorithm>
#include <cstdint>

#include "blas.hh"

#include "hcore/compressed_tile.hh"
#include "hcore/base_tile.hh"
#include "hcore/exception.hh"

namespace hcore {

// forward declaration
template <typename T>
class CompressedTile;

//==============================================================================
//
template <typename T>
class Tile : public BaseTile<T>
{
public:
    //--------------------------------------------------------------------------
    /// Tile empty class.
    Tile() : BaseTile<T>(), stride_(0) {}

    //--------------------------------------------------------------------------
    /// Tile class that wraps existing (preallocated) memory buffer.
    ///
    /// @param[in] mb
    ///     Number of rows. mb >= 0.
    /// @param[in] nb
    ///     Number of columns. nb >= 0.
    /// @param[in,out] A
    ///     The mb-by-nb tile, stored in a data buffer of size:
    ///     lda-by-nb: if layout = blas::Layout::ColMajor, or
    ///     lda-by-mb: if layout = blas::Layout::RowMajor.
    /// @param[in] lda
    ///     Leading dimension of the data buffer.
    ///     lda >= mb: if layout = blas::Layout::ColMajor, or
    ///     lda >= nb: if layout = blas::Layout::RowMajor.
    /// @param[in] layout
    ///     The physical ordering of matrix elements in the data array buffer.
    ///     blas::Layout::ColMajor: column elements are 1-strided (default), or
    ///     blas::Layout::RowMajor: row elements are 1-strided.
    Tile(int64_t mb, int64_t nb, T* A, int64_t lda,
         blas::Layout layout = blas::Layout::ColMajor)
        : BaseTile<T>(mb, nb, A, layout), stride_(lda)
    {
        hcore_assert((layout == blas::Layout::ColMajor && lda >= mb)
                     || (layout == blas::Layout::RowMajor && lda >= nb));
    }

    //--------------------------------------------------------------------------
    /// Conversion from CompressedTile
    /// Creates shallow copy view of the original CompressedTile.
    ///
    /// @param[in] orig
    ///     Original CompressedTile of which to make a Tile.
    Tile(CompressedTile<T>& orig)
        : BaseTile<T>(orig),
          stride_(this->layout_ == blas::Layout::ColMajor ? orig.Ustride()
                                                          : orig.Vstride())
    {
        hcore_assert(orig.rk() == std::min(orig.mb(), orig.nb())); // full rank
    }

    //--------------------------------------------------------------------------
    /// Conversion from CompressedTile
    /// Creates shallow copy view with a stride of the original CompressedTile.
    ///
    /// @param[in] stride
    ///     Leading dimension of the data buffer.
    ///     stride >= mb: if layout = blas::Layout::ColMajor, or
    ///     stride >= nb: if layout = blas::Layout::RowMajor.
    /// @param[in] orig
    ///     Original CompressedTile of which to make a Tile.
    Tile(int64_t stride, CompressedTile<T>& orig)
        : BaseTile<T>(orig), stride_(stride)
    {
        hcore_assert((this->layout_ == blas::Layout::ColMajor
                     && stride >= this->mb_)
                     || (this->layout_ == blas::Layout::RowMajor
                     && stride >= this->nb_));
        hcore_assert(orig.rk() == std::min(orig.mb(), orig.nb())); // full rank
    }

    //--------------------------------------------------------------------------
    /// @return const pointer to array data buffer.
    T const* data() const { return this->data_; }

    //--------------------------------------------------------------------------
    /// @return pointer to array data buffer.
    T* data() { return this->data_; }

    //--------------------------------------------------------------------------
    /// @return column (row-major) or row (column-major) stride.
    int64_t stride() const { return stride_; }

    //--------------------------------------------------------------------------
    /// @return the number of locations in memory between beginnings of
    /// successive array elements of a row.
    int64_t rowIncrement() const
    {
        if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return stride_;
        }
        else {
            return 1;
        }
    }

    //--------------------------------------------------------------------------
    /// @return the number of locations in memory between beginnings of
    /// successive array elements of a column.
    int64_t colIncrement() const
    {
        if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return 1;
        }
        else {
            return stride_;
        }
    }

    //--------------------------------------------------------------------------
    /// @return element {i, j} of this tile. The actual value is returned, not a
    /// reference. If op() == blas::Op::ConjTrans then data is conjugated,
    /// taking the layout into account.
    ///
    /// @param[in] i
    ///     Row index. 0 <= i < m.
    /// @param[in] j
    ///     Column index. 0 <= j < n.
    T operator()(int64_t i, int64_t j) const
    {
        hcore_assert(0 <= i && i < this->mb());
        hcore_assert(0 <= j && j < this->nb());

        using blas::conj;

        if (this->op_ == blas::Op::ConjTrans) {
            if (this->layout_ == blas::Layout::ColMajor)
                return conj(this->data_[j + i*stride_]);
            else
                return conj(this->data_[i + j*stride_]);
        }
        else if ((this->op_ == blas::Op::NoTrans)
                 == (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_[i + j*stride_];
        }
        else {
            return this->data_[j + i*stride_];
        }
    }

    //--------------------------------------------------------------------------
    /// @return a const reference to element {i, j} of this tile.
    /// If op() == blas::Op::ConjTrans then data isn't conjugated, because a
    /// reference is returned, taking the layout into account.
    ///
    /// @param[in] i
    ///     Row index. 0 <= i < m.
    /// @param[in] j
    ///     Column index. 0 <= j < n.
    T const& at(int64_t i, int64_t j) const
    {
        hcore_assert(0 <= i && i < this->mb());
        hcore_assert(0 <= j && j < this->nb());

        if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_[i + j*stride_];
        }
        else {
            return this->data_[j + i*stride_];
        }
    }

    //--------------------------------------------------------------------------
    /// @return a reference to element {i, j} of this tile.
    /// If op() == blas::Op::ConjTrans then data isn't conjugated, because a
    /// reference is returned, taking the layout into account.
    ///
    /// @param[in] i
    ///     Row index. 0 <= i < m.
    /// @param[in] j
    ///     Column index. 0 <= j < n.
    T& at(int64_t i, int64_t j)
    {
        return const_cast<T&>(static_cast<const Tile>(*this).at(i, j));
    }
protected:
    int64_t stride_; ///> Leading dimension.

    //--------------------------------------------------------------------------
    /// Set column (row-major) or row (column-major) stride.
    ///
    /// @param[in] new_stride
    ///     Leading dimension of the data array buffer.
    ///     new_stride >= mb: if layout = blas::Layout::ColMajor, or
    ///     new_stride >= nb: if layout = blas::Layout::RowMajor.
    void stride(int64_t new_stride)
    {
        hcore_assert((this->layout_ == blas::Layout::ColMajor
                     && new_stride >= this->mb_)
                     || (this->layout_ == blas::Layout::RowMajor
                     && new_stride >= this->nb_));
        stride_ = new_stride;
    }

}; // class Tile
}  // namespace hcore

#endif // HCORE_TILE_HH
