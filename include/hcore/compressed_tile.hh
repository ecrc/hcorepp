// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TILE_COMPRESSED_HH
#define HCORE_TILE_COMPRESSED_HH

#include "hcore/exception.hh"
#include "hcore/base_tile.hh"

#include "blas.hh"

#include <cstdint>
#include <algorithm>

namespace hcore {

// =============================================================================
//
/// Compressed tile class.
/// @tparam T
///     Data type: float, double, std::complex<float>, or std::complex<double>.
template <typename T>
class CompressedTile : public BaseTile<T>
{
    using real_t = blas::real_type<T>;

private:
    static const int64_t FULL_RANK_ = -1; ///> Constant representing full rank.

public:
    // =========================================================================
    //
    /// Empty compressed tile.
    CompressedTile() : BaseTile<T>(), rk_(0), accuracy_(0)
        {}

    // =========================================================================
    //
    /// Compressed tile that wraps existing (preallocated) memory buffer.
    /// @param[in] m
    ///     Number of rows of the tile. m >= 0.
    /// @param[in] n
    ///     Number of columns of the tile. n >= 0.
    /// @param[in,out] UV
    ///     The m-by-n matrix compressed tile (A=UV): A is m-by-n, U is m-by-rk,
    ///     and V is rk-by-n. If layout = blas::Layout::ColMajor, the data array
    ///     of A is stored in an ld-by-n array buffer; the data array of U is
    ///     stored in an ld-by-rk array buffer; and data array of V is stored in
    ///     an rk-by-n array buffer. However, if layout=blas::Layout::RowMajor:
    ///     the data array of A is stored in an m-by-ld array buffer, the data
    ///     array of U is stored in an m-by-rk array buffer, and data array of V
    ///     is stored in an rk-by-ld array buffer.
    /// @param[in] ld
    ///     Leading dimension of the data array buffer of U/V.
    ///     ldu >= m: if layout = blas::Layout::ColMajor, or
    ///     ldv >= n: if layout = blas::Layout::RowMajor.
    /// @param[in] rk
    ///     Linear algebra rank of the tile. rk >= 0.
    /// @param[in] accuracy
    ///     Numerical error threshold. accuracy >= 0.
    /// @param[in] layout
    ///     The physical ordering of matrix elements in the data array buffer.
    ///     blas::Layout::ColMajor: column elements are 1-strided (default), or
    ///     blas::Layout::RowMajor: row elements are 1-strided.
    CompressedTile(int64_t m, int64_t n, T* UV, int64_t ld, int64_t rk,
        real_t accuracy, blas::Layout layout=blas::Layout::ColMajor)
        : BaseTile<T>(m, n, UV, ld, layout), rk_(rk), accuracy_(accuracy)
    {
        hcore_error_if(rk < 0);
        hcore_error_if(accuracy < 0);
    }

    /// @return const pointer to array data buffer of U.
    T const* Udata() const
        { return this->data_; }

    /// @return pointer to array data buffer of U.
    T* Udata()
        { return this->data_; }

    /// @return column/row stride of U.
    int64_t ldu() const
        { return (this->layout() == blas::Layout::ColMajor ? this->ld_ : rk_); }

    /// @return const pointer to array data buffer of V.
    T const* Vdata() const
    {
        return (
            this->layout() == blas::Layout::ColMajor
            ? (this->data_ + this->ld_ * rk_)
            : (this->data_ + this->m() * rk_)
            );
    }

    /// @return pointer to array data buffer of V.
    T* Vdata()
    {
        return (this->layout() == blas::Layout::ColMajor
            ? (this->data_ + this->ld_ * rk_)
            : (this->data_ + this->m() * rk_));
    }

    /// Update pointer to array data buffer of U and V.
    void UVdata(T* UV)
    {
        hcore_error_if(UV == nullptr);
        this->data_ = UV;
    }

    /// @return column/row stride of V.
    int64_t ldv() const
        { return (this->layout() == blas::Layout::ColMajor ? rk_ : this->ld_); }

    /// @return linear algebra rank of this tile.
    int64_t rk() const
        { return (rk_ == FULL_RANK_ ? std::min(this->m(), this->n()) : rk_); }

    /// Update linear algebra rank of this tile.
    void rk(int64_t rk)
    {
        hcore_error_if(rk < 0 && rk != FULL_RANK_);
        rk_ = (rk == std::min(this->m(), this->n()) ? FULL_RANK_ : rk);
    }

    /// Update linear algebra rank of this tile to full rank.
    void to_full_rk()
        { rk(FULL_RANK_); }

    /// @return whether the linear algebra rank of this tile is full or not.
    bool is_full_rk() const
        { return (rk_ == FULL_RANK_ ? true : false); }

    /// @return numerical error threshold of this tile.
    real_t accuracy() const
        { return accuracy_; }

private:
    int64_t rk_; ///> Linear algebra matrix rank.
    real_t accuracy_; ///> Numerical error threshold.

}; // class CompressedTile
}  // namespace hcore

#endif // HCORE_TILE_COMPRESSED_HH
