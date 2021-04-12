// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TILE_TILE_HH
#define HCORE_TILE_TILE_HH

#include "hcore/exception.hh"

#include "blas.hh"

#include <cstdint>

namespace hcore {

/// Tile C++ superclass.
/// A tile is an m-by-n matrix, with a leading dimension.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
template <typename T>
class Tile
{
protected:
    /// Empty tile.
    Tile() : m_(0), n_(0), data_(nullptr), ld_(0), op_(blas::Op::NoTrans),
        uplo_(blas::Uplo::General), layout_(blas::Layout::ColMajor)
    {
    }

    /// Tile that wraps existing (preallocated) memory buffer.
    /// @param[in] m
    ///     Number of rows of the tile. m >= 0.
    /// @param[in] n
    ///     Number of columns of the tile. b >= 0.
    /// @param[in,out] A
    ///     The m-by-n matrix, stored in an array data buffer of size:
    ///     ld-by-n: if layout = blas::Layout::ColMajor, or
    ///     ld-by-m: if layout = blas::Layout::RowMajor.
    /// @param[in] ld
    ///     Leading dimension of the data array buffer.
    ///     ld >= m: if layout = blas::Layout::ColMajor, or
    ///     ld >= n: if layout = blas::Layout::RowMajor.
    /// @param[in] layout
    ///     The physical ordering of matrix elements in the data array buffer.
    ///     blas::Layout::ColMajor: column elements are 1-strided, or
    ///     blas::Layout::RowMajor: row elements are 1-strided.
    Tile(int64_t m, int64_t n, T* A, int64_t ld, blas::Layout layout) : m_(m),
        n_(n), data_(A), ld_(ld), op_(blas::Op::NoTrans),
        uplo_(blas::Uplo::General), layout_(layout)
    {
        hcore_error_if(m < 0);
        hcore_error_if(n < 0);
        hcore_error_if(A == nullptr);
        hcore_error_if(layout == blas::Layout::ColMajor && ld < m);
        hcore_error_if(layout == blas::Layout::RowMajor && ld < n);
    }

public:
    /// @return number of rows of this tile.
    int64_t m() const
    {
        return (op_ == blas::Op::NoTrans ? m_ : n_);
    }

    /// @return number of columns of this tile.
    int64_t n() const
    {
        return (op_ == blas::Op::NoTrans ? n_ : m_);
    }

    /// @return transposition operation of this tile.
    blas::Op op() const
    {
        return op_;
    }

    /// Set transposition operation of this tile.
    /// @param[in] op
    ///     - Trans: this tile has a transposed structure.
    ///     - ConjTrans: this tile has a conjugate-transposed structure.
    ///     - NoTrans: this tile has no transposed structure.
    void op(blas::Op op)
    {
        hcore_error_if(
            op != blas::Op::Trans   &&
            op != blas::Op::NoTrans &&
            op != blas::Op::ConjTrans);
        op_ = op;
    }

    /// @param[in] logical
    ///     If true (default), @return the logical packed storage type of this
    ///     tile; see @uplo_logical, otherwise @return the physical packed
    ///     storage type of this tile; see @uplo_physical.
    blas::Uplo uplo(bool logical = true) const
    {
        if (logical)
            return uplo_logical();
        else
            return uplo_physical();
    }

    /// Set the physical packed storage type of this tile.
    /// @param[in] uplo
    ///     - General: both triangles of this tile are stored.
    ///     - Upper: upper triangle of this tile is stored.
    ///     - Lower: lower triangle of this tile is stored.
    void uplo(blas::Uplo uplo)
    {
        hcore_error_if(
            uplo != blas::Uplo::General &&
            uplo != blas::Uplo::Lower   &&
            uplo != blas::Uplo::Upper);
        uplo_ = uplo;
    }

    /// @return the physical packed storage type of this tile.
    blas::Uplo uplo_physical() const
    {
        return uplo_;
    }

    /// @return the logical packed storage type of this tile.
    blas::Uplo uplo_logical() const
    {
        if (uplo_ == blas::Uplo::General)
            return blas::Uplo::General;
        else if ((uplo_ == blas::Uplo::Lower) == (op_ == blas::Op::NoTrans))
            return blas::Uplo::Lower;
        else
            return blas::Uplo::Upper;
    }

    /// @return the physical ordering of the matrix elements in the data array
    /// buffer of this tile.
    blas::Layout layout() const
    {
        return layout_;
    }

    /// @return element {i, j} of this tile.
    /// The actual value is returned, not a reference.
    /// If op() == blas::Op::ConjTrans then data is conjugated; it takes
    /// the layout into account.
    /// @param[in] i
    ///     Row index. 0 <= i < m.
    /// @param[in] j
    ///     Column index. 0 <= j < n.
    T operator()(int64_t i, int64_t j) const
    {
        hcore_error_if(0 > i || i >= m());
        hcore_error_if(0 > j || j >= n());

        using blas::conj;

        if (op_ == blas::Op::ConjTrans) {
            if (layout_ == blas::Layout::ColMajor)
                return conj(data_[j + i * ld_]);
            else
                return conj(data_[i + j * ld_]);
        }
        else if (
            (op_ == blas::Op::NoTrans) == (layout_ == blas::Layout::ColMajor)) {
            return data_[i + j * ld_];
        }
        else {
            return data_[j + i * ld_];
        }
    }

    /// @return a const reference to element {i, j} of this tile.
    /// If op() == blas::Op::ConjTrans then data isn't conjugated, because a
    /// reference is returned; it takes the layout into account.
    /// @param[in] i
    ///     Row index. 0 <= i < m.
    /// @param[in] j
    ///     Column index. 0 <= j < n.
    T const& at(int64_t i, int64_t j) const
    {
        hcore_error_if(0 > i || i >= m());
        hcore_error_if(0 > j || j >= n());

        if ((op_ == blas::Op::NoTrans) == (layout_ == blas::Layout::ColMajor)) {
            return data_[i + j * ld_];
        }
        else {
            return data_[j + i * ld_];
        }
    }

    /// @return a reference to element {i, j} of this tile.
    /// If op() == blas::Op::ConjTrans then data isn't conjugated, because a
    /// reference is returned; it takes the layout into account.
    /// @param[in] i
    ///     Row index. 0 <= i < m.
    /// @param[in] j
    ///     Column index. 0 <= j < n.
    T& at(int64_t i, int64_t j)
    {
        // forward to const at() version
        return const_cast<T&>(static_cast<const Tile>(*this).at(i, j));
    }

private:
    int64_t m_; ///> Number of rows.
    int64_t n_; ///> Number of columns.

protected:
    T* data_; ///> Data array buffer.
    int64_t ld_; ///> Leading dimension.

private:
    blas::Op op_; ///> Transposition operation.
    blas::Uplo uplo_; ///> Physical packed storage type.
    blas::Layout layout_; ///> Physical ordering of the matrix elements.

}; // class Tile
} // namespace hcore

#endif // HCORE_TILE_TILE_HH
