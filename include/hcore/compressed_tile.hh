// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TILE_COMPRESSED_HH
#define HCORE_TILE_COMPRESSED_HH

#include <algorithm>
#include <cstdint>

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
    CompressedTile() : BaseTile<T>(), Ustride_(0), Vstride_(0), rk_(0), tol_(0)
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
    ///     ldu-by-rk + ldv-by-nb: if layout = blas::Layout::ColMajor, or
    ///     ldu-by-rk + ldv-by-mb: if layout = blas::Layout::RowMajor.
    /// @param[in] ldu
    ///     Leading dimension of the U array of the data buffer.
    ///     ldu >= mb: if layout = blas::Layout::ColMajor, or
    ///     ldu >= nb: if layout = blas::Layout::RowMajor.
    /// @param[in] ldv
    ///     Leading dimension of the V array of the data buffer. ldv >= rk.
    /// @param[in] rk
    ///     Linear algebra matrix rank. rk >= 0.
    /// @param[in] tol
    ///     Scalar numerical error threshold. tol >= 0.
    /// @param[in] layout
    ///     The physical ordering of matrix elements in the data array buffer.
    ///     blas::Layout::ColMajor: column elements are 1-strided, or
    ///     blas::Layout::RowMajor: row elements are 1-strided.
    CompressedTile(int64_t mb, int64_t nb, T* UV, int64_t ldu, int64_t ldv,
                   int64_t rk, blas::real_type<T> tol,
                   blas::Layout layout = blas::Layout::ColMajor)
        : BaseTile<T>(mb, nb, UV, layout),
          Ustride_(ldu),
          Vstride_(ldv),
          rk_(rk),
          tol_(tol)
    {
        hcore_assert((layout == blas::Layout::ColMajor && ldu >= mb)
                     || (layout == blas::Layout::RowMajor && ldu >= rk));
        hcore_assert((layout == blas::Layout::ColMajor && ldv >= rk)
                     || (layout == blas::Layout::RowMajor && ldv >= nb));
        hcore_assert(rk >= 0);
        hcore_assert(tol >= 0);
    }

    //--------------------------------------------------------------------------
    /// @return const pointer to array data buffer of U.
    T const* Udata() const
    {
       if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_;
        }
        else {
            return this->data_ + Vstride()*rk_;
        }
    }

    //--------------------------------------------------------------------------
    /// @return pointer to array data buffer of U.
    T* Udata()
    {
       if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_;
        }
        else {
            return this->data_ + Vstride()*rk_;
        }
    }

    //--------------------------------------------------------------------------
    /// @return const pointer to array data buffer of V.
    T const* Vdata() const
    {
       if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_ + Ustride()*rk_;
        }
        else {
            return this->data_;
        }
    }

    //--------------------------------------------------------------------------
    /// @return pointer to array data buffer of V.
    T* Vdata()
    {
       if ((this->op_ == blas::Op::NoTrans)
            == (this->layout_ == blas::Layout::ColMajor)) {
            return this->data_ + Ustride()*rk_;
        }
        else {
            return this->data_;
        }
    }

    //--------------------------------------------------------------------------
    /// @return column/row stride of U.
    int64_t Ustride() const
    {
        return (this->op_ == blas::Op::NoTrans ? Ustride_ : Vstride_);
    }
    //--------------------------------------------------------------------------
    /// @return column/row stride of V.
    int64_t Vstride() const
    {
        return (this->op_ == blas::Op::NoTrans ? Vstride_ : Ustride_);
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
    ///     ldu-by-rk + ldv-by-nb: if layout = blas::Layout::ColMajor, or
    ///     ldu-by-rk + ldv-by-mb: if layout = blas::Layout::RowMajor.
    /// @param[in] ldu
    ///     Leading dimension of the U array of the data buffer.
    ///     ldu >= mb: if layout = blas::Layout::ColMajor, or
    ///     ldu >= nb: if layout = blas::Layout::RowMajor.
    /// @param[in] ldv
    ///     Leading dimension of the V array of the data buffer. ldv >= rk.
    /// @param[in] rk
    ///     Linear algebra matrix rank. rk >= 0.
    void resize(T* UV, int64_t ldu, int64_t ldv, int64_t new_rk)
    {
        hcore_assert(UV != nullptr);
        this->data_ = UV;
        rk(new_rk);
        Ustride(ldu);
        Vstride(ldv);
    }

    //--------------------------------------------------------------------------
    /// Removes all elements from the data array buffer (which are destroyed),
    /// leaving it with a size of 0.
    void clear() { delete [] this->data_; }

protected:
    int64_t Ustride_; ///> Leading dimension of U.
    int64_t Vstride_; ///> Leading dimension of V.

    int64_t rk_; ///> Linear algebra matrix rank.

    blas::real_type<T> tol_; ///> Numerical error threshold.

    //--------------------------------------------------------------------------
    /// Set column (row-major) or row (column-major) stride of U.
    ///
    /// @param[in] new_Ustride
    ///     Leading dimension of the U data array buffer.
    ///     new_Ustride >= mb: if layout = blas::Layout::ColMajor, or
    ///     new_Ustride >= nb: if layout = blas::Layout::RowMajor.
    void Ustride(int64_t new_Ustride)
    {
        hcore_assert((this->layout_ == blas::Layout::ColMajor
                     && new_Ustride >= this->mb_)
                     || (this->layout_ == blas::Layout::RowMajor
                     && new_Ustride >= rk_));
        Ustride_ = new_Ustride;
    }

    //--------------------------------------------------------------------------
    /// Set column (row-major) or row (column-major) stride of V.
    ///
    /// @param[in] new_Vstride
    ///     Leading dimension of the V array of the data buffer.
    ///     new_Vstride >= rk.
    void Vstride(int64_t new_Vstride)
    {
        hcore_assert((this->layout_ == blas::Layout::ColMajor
                     && new_Vstride >= rk_)
                     || (this->layout_ == blas::Layout::RowMajor
                     && new_Vstride >= this->nb_));

        Vstride_ = new_Vstride;
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
