// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TILE_COMPRESSED_HH
#define HCORE_TILE_COMPRESSED_HH

#include <algorithm>
#include <cstdint>
#include <utility>

#include "blas.hh"

#include "hcore/base_tile.hh"
#include "hcore/exception.hh"
#include "hcore/tile.hh"

namespace hcore {

//==============================================================================
//
template <typename T>
class CompressedTile : public BaseTile<T>
{
public:
    //--------------------------------------------------------------------------
    /// CompressedTile empty class.
    CompressedTile() : BaseTile<T>(), stride_({0, 0}), rk_(0), tol_(0)
    {}

    //--------------------------------------------------------------------------
    /// CompressedTile class that wraps existing (preallocated) memory buffer.
    ///
    /// @param[in] mb
    ///     Number of rows. mb >= 0.
    /// @param[in] nb
    ///     Number of columns. nb >= 0.
    /// @param[in,out] UV
    ///     The mb-by-nb compressed tile, stored in a data buffer of size:
    ///     UV: lduv1-by-rk + lduv2-by-nb, if layout=blas::Layout::ColMajor, or
    ///     VU: lduv1-by-rk + lduv2-by-mb, if layout=blas::Layout::RowMajor.
    /// @param[in] lduv1
    ///     Leading dimension of the U or V array of the data buffer.
    ///     lduv1 >= mb: if layout = blas::Layout::ColMajor (stride of U), or
    ///     lduv1 >= nb: if layout = blas::Layout::RowMajor (stride of V).
    /// @param[in] lduv2
    ///     Leading dimension of the V or U array of the data buffer.
    ///     lduv2 >= rk.
    ///     if layout = blas::Layout::ColMajor (stride of V), or
    ///     if layout = blas::Layout::RowMajor (stride of U).
    /// @param[in] rk
    ///     Linear algebra matrix rank. rk >= 0.
    /// @param[in] tol
    ///     Scalar numerical error threshold. tol >= 0.
    /// @param[in] layout
    ///     The physical ordering of matrix elements in the data array buffer.
    ///     blas::Layout::ColMajor: column elements are 1-strided, or
    ///     blas::Layout::RowMajor: row elements are 1-strided.
    CompressedTile(int64_t mb, int64_t nb, T* UV, int64_t lduv1, int64_t lduv2,
                   int64_t rk, blas::real_type<T> tol,
                   blas::Layout layout = blas::Layout::ColMajor)
        : BaseTile<T>(mb, nb, UV, layout),
          stride_({lduv1, lduv2}),
          rk_(rk),
          tol_(tol)
    {
        hcore_assert((layout == blas::Layout::ColMajor && lduv1 >= mb)
                     || (layout == blas::Layout::RowMajor && lduv1 >= nb));
        hcore_assert(lduv2 >= rk);
        hcore_assert(rk >= 0);
        hcore_assert(tol >= 0);
    }

    //--------------------------------------------------------------------------
    /// @return const pointer to array data buffer of U.
    T const* Udata() const
    {
        // (blas::Op::NoTrans && blas::Layout::ColMajor)
        //  || (blas::Op::Trans && blas::Layout::RowMajor)
        if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_;                 // UV: mb-by-rk + rk-by-nb
        }
        else {
            return this->data_ + Vstride()*rk_; // VU: nb-y-rk + rk-by-mb
        }
    }

    //--------------------------------------------------------------------------
    /// @return pointer to array data buffer of U.
    T* Udata()
    {
        // (blas::Op::NoTrans && blas::Layout::ColMajor)
        //  || (blas::Op::Trans && blas::Layout::RowMajor)
        if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_;                 // UV: mb-by-rk + rk-by-nb
        }
        else {
            return this->data_ + Vstride()*rk_; // VU: nb-y-rk + rk-by-mb
        }
    }

    //--------------------------------------------------------------------------
    /// @return const pointer to array data buffer of V.
    T const* Vdata() const
    {
        // (blas::Op::NoTrans && blas::Layout::ColMajor)
        //  || (blas::Op::Trans && blas::Layout::RowMajor)
        if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_ + Ustride()*rk_; // UV: mb-by-rk + rk-by-nb
        }
        else {
            return this->data_;                 // VU: nb-y-rk + rk-by-mb
        }
    }

    //--------------------------------------------------------------------------
    /// @return pointer to array data buffer of V.
    T* Vdata()
    {
        // (blas::Op::NoTrans && blas::Layout::ColMajor)
        //  || (blas::Op::Trans && blas::Layout::RowMajor)
        if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_ + Ustride()*rk_; // UV: mb-by-rk + rk-by-nb
        }
        else {
            return this->data_;                 // VU: nb-y-rk + rk-by-mb
        }
    }

    //--------------------------------------------------------------------------
    /// @return column/row stride of U.
    int64_t Ustride() const
    {
        // (blas::Op::NoTrans && blas::Layout::ColMajor)
        //  || (blas::Op::Trans && blas::Layout::RowMajor)
        if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return stride_.first;  // UV: mb-by-rk + rk-by-nb
        }
        else {
            return stride_.second; // VU: nb-y-rk + rk-by-mb
        }
    }
    //--------------------------------------------------------------------------
    /// @return column/row stride of V.
    int64_t Vstride() const
    {
        // (blas::Op::NoTrans && blas::Layout::ColMajor)
        //  || (blas::Op::Trans && blas::Layout::RowMajor)
        if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return stride_.second; // UV: mb-by-rk + rk-by-nb
        }
        else {
            return stride_.first;  // VU: nb-y-rk + rk-by-mb
        }
    }

    //--------------------------------------------------------------------------
    /// @return linear algebra matrix rank.
    int64_t rk() const { return rk_; }

    //--------------------------------------------------------------------------
    /// @return scalar numerical error threshold.
    blas::real_type<T> tol() const { return tol_; }

    //--------------------------------------------------------------------------
    /// Resizes the container so that it contains n elements.
    ///
    /// @param[in,out] UV
    ///     The mb-by-nb compressed tile, stored in a data buffer of size:
    ///     UV: lduv1-by-rk + lduv2-by-nb, if layout=blas::Layout::ColMajor, or
    ///     VU: lduv1-by-rk + lduv2-by-mb, if layout=blas::Layout::RowMajor.
    /// @param[in] lduv1
    ///     Leading dimension of the U or V array of the data buffer.
    ///     lduv1 >= mb: if layout = blas::Layout::ColMajor (stride of U), or
    ///     lduv1 >= nb: if layout = blas::Layout::RowMajor (stride of V).
    /// @param[in] lduv2
    ///     Leading dimension of the V or U array of the data buffer.
    ///     lduv2 >= rk.
    ///     if layout = blas::Layout::ColMajor (stride of V), or
    ///     if layout = blas::Layout::RowMajor (stride of U).
    /// @param[in] new_rk
    ///     Linear algebra matrix rank. rk >= 0.
    void resize(T* UV, int64_t lduv1, int64_t lduv2, int64_t new_rk)
    {
        hcore_assert(UV != nullptr);
        this->data_ = UV;
        rk(new_rk);
        stride(lduv1, lduv2);
    }

    //--------------------------------------------------------------------------
    /// Removes all elements from the data array buffer (which are destroyed),
    /// leaving it with a size of 0.
    void clear() { delete [] this->data_; }

protected:
    std::pair<int64_t, int64_t> stride_; ///> Leading dimension

    int64_t rk_; ///> Linear algebra matrix rank.

    blas::real_type<T> tol_; ///> Numerical error threshold.

    //--------------------------------------------------------------------------
    /// Set column (row-major) or row (column-major) stride of U.
    ///
    /// @param[in] lduv1
    ///     Leading dimension of the U or V array of the data buffer.
    ///     lduv1 >= mb: if layout = blas::Layout::ColMajor (stride of U), or
    ///     lduv1 >= nb: if layout = blas::Layout::RowMajor (stride of V).
    /// @param[in] lduv2
    ///     Leading dimension of the V or U array of the data buffer.
    ///     lduv2 >= rk.
    ///     if layout = blas::Layout::ColMajor (stride of V), or
    ///     if layout = blas::Layout::RowMajor (stride of U).
    void stride(int64_t lduv1, int64_t lduv2)
    {
        hcore_assert((this->layout_ == blas::Layout::ColMajor
                     && lduv1 >= this->mb_)
                     || (this->layout_ == blas::Layout::RowMajor
                     && lduv1 >= this->nb_));
        hcore_assert(lduv2 >= rk_);
        stride_ = {lduv1, lduv2};
    }

    //--------------------------------------------------------------------------
    /// Set linear algebra matrix rank.
    ///
    /// @param[in] new_rknew_rk
    ///     Linear algebra matrix rank. new_rk >= 0.
    void rk(int64_t new_rk)
    {
        hcore_assert(new_rk >= 0);
        rk_ = new_rk;
    }

    //--------------------------------------------------------------------------
    /// Set scalar numerical error threshold. tol >= 0.
    ///
    /// @param[in] new_tol
    ///     New accuracy. new_tol >= 0.
    void tol(blas::real_type<T> new_tol)
    {
        hcore_assert(new_tol >= 0);
        tol_ = new_tol;
    }

}; // class CompressedTile
}  // namespace hcore

#endif // HCORE_TILE_COMPRESSED_HH
