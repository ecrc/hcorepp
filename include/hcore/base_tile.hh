// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_BASE_TILE_HH
#define HCORE_BASE_TILE_HH

#include <cstdint>

#include "blas.hh"

#include "hcore/exception.hh"

namespace hcore {

//------------------------------------------------------------------------------
/// Transpose a tile; changing op flag from NoTrans to Trans, or from Trans to
/// NoTrans. @return shallow copy with updated op flag.
///
/// @param[in] A
///     Tile to be transposed.
///
/// @ingroup util
template <typename T>
T transpose(T& A)
{
    T AT = A;

    if (AT.op_ == blas::Op::NoTrans)
        AT.op_ = blas::Op::Trans;
    else if (AT.op_ == blas::Op::Trans || A.is_real)
        AT.op_ = blas::Op::NoTrans;
    else
        throw Error("Unsupported operation: conjugate-no-transpose");

    return AT;
}

//------------------------------------------------------------------------------
/// Transpose a tile; changing op flag from NoTrans to Trans, or from Trans to
/// NoTrans. @return shallow copy with updated op flag.
/// Convert rvalue refs to lvalue refs.
///
/// @param[in] A
///     Tile to be transposed.
///
/// @ingroup util
template <typename T>
T transpose(T&& A) { return transpose(A); }

//------------------------------------------------------------------------------
/// Conjugate-transpose a tile; changing op flag from NoTrans to Trans, or from
/// Trans to NoTrans. @return shallow copy with updated op flag.
///
/// @param[in] A
///     Tile to be conjugate-transposed.
///
/// @ingroup util
template <typename T>
T conjugate_transpose(T& A)
{
    T AT = A;

    if (AT.op_ == blas::Op::NoTrans)
        AT.op_ = blas::Op::ConjTrans;
    else if (AT.op_ == blas::Op::ConjTrans || A.is_real)
        AT.op_ = blas::Op::NoTrans;
    else
        throw Error("Unsupported operation: conjugate-no-transpose");

    return AT;
}

//------------------------------------------------------------------------------
/// Conjugate-transpose a tile; changing op flag from NoTrans to Trans, or from
/// Trans to NoTrans. @return shallow copy with updated op flag.
/// Convert rvalue refs to lvalue refs.
///
/// @param[in] A
///     Tile to be conjugate-transposed.
///
/// @ingroup util
template <typename T>
T conjugate_transpose(T&& A) { return conjugate_transpose(A); }

//==============================================================================
//
template <typename T>
class BaseTile
{
protected:
    //--------------------------------------------------------------------------
    /// BaseTile parent empty class.
    BaseTile()
        : mb_(0),
          nb_(0),
          data_(nullptr),
          op_(blas::Op::NoTrans),
          uplo_(blas::Uplo::General),
          layout_(blas::Layout::ColMajor)
    {}

    //--------------------------------------------------------------------------
    /// BaseTile parent class that wraps existing (preallocated) memory buffer.
    ///
    /// @param[in] mb
    ///     Number of rows. mb >= 0.
    /// @param[in] nb
    ///     Number of columns. nb >= 0.
    /// @param[in,out] A
    ///     The mb-by-nb tile, stored in a data buffer.
    /// @param[in] layout
    ///     The physical ordering of matrix elements in the data array buffer.
    ///     blas::Layout::ColMajor: column elements are 1-strided, or
    ///     blas::Layout::RowMajor: row elements are 1-strided.
    BaseTile(int64_t mb, int64_t nb, T* A, blas::Layout layout)
        : mb_(mb),
          nb_(nb),
          data_(A),
          op_(blas::Op::NoTrans),
          uplo_(blas::Uplo::General),
          layout_(layout)
    {
        hcore_assert(mb >= 0);
        hcore_assert(nb >= 0);
        hcore_assert(A != nullptr);
    }

    //--------------------------------------------------------------------------
    /// Sets number of rows.
    ///
    /// @param[in] new_mb
    ///     Number of rows.
    void mb(int64_t new_mb)
    {
        hcore_assert(0 <= new_mb && new_mb <= mb());

        if (op_ == blas::Op::NoTrans)
            mb_ = new_mb;
        else
            nb_ = new_mb;
    }

    //--------------------------------------------------------------------------
    /// Sets number of columns.
    ///
    /// @param[in] new_nb
    ///     Number of columns.
    void nb(int64_t new_nb)
    {
        hcore_assert(0 <= new_nb && new_nb <= nb());

        if (op_ == blas::Op::NoTrans)
            nb_ = new_nb;
        else
            mb_ = new_nb;
    }

    //--------------------------------------------------------------------------
    /// Set the physical ordering of the matrix elements in the data array.
    ///
    /// @param[in] new_layout
    ///     blas::Layout::ColMajor: column elements are 1-strided.
    ///     blas::Layout::RowMajor: row elements are 1-strided.
    void layout(blas::Layout new_layout) { return layout_ = new_layout; }

    //--------------------------------------------------------------------------
    /// Set transposition operation.
    ///
    /// @param[in] new_op
    ///     - blas::Op::NoTrans: tile has no transposed structure.
    ///     - blas::Op::Trans: tile has a transposed structure.
    ///     - blas::Op::ConjTrans: tile has a conjugate-transposed structure.
    void op(blas::Op new_op) { op_ = new_op; }

public:
    //--------------------------------------------------------------------------
    /// True if class template identifier is complex, and false otherwise.
    static constexpr bool is_complex = blas::is_complex<T>::value;

    //--------------------------------------------------------------------------
    /// True if class template identifier isn't complex, and false otherwise.
    static constexpr bool is_real = !is_complex;

    //--------------------------------------------------------------------------
    /// Transpose and @return a shallow copy with updated op flag.
    /// @param[in] A
    ///     Tile to be transposed.
    template <typename T1>
    friend T1 transpose(T1& A);

    //--------------------------------------------------------------------------
    /// Transpose and @return a shallow copy with updated op flag.
    /// Convert rvalue refs to lvalue refs.
    /// @param[in] A
    ///     Tile to be transposed.
    template <typename T1>
    friend T1 transpose(T1&& A);

    //--------------------------------------------------------------------------
    /// Conjugate-transpose and @return a shallow copy with updated op flag.
    /// @param[in] A
    ///     Tile to be conjugate-transposed.
    template <typename T1>
    friend T1 conjugate_transpose(T1& A);

    //--------------------------------------------------------------------------
    /// Conjugate-transpose and @return a shallow copy with updated op flag.
    /// Convert rvalue refs to lvalue refs.
    /// @param[in] A
    ///     Tile to be conjugate-transposed.
    template <typename T1>
    friend T1 conjugate_transpose(T1&& A);

    //--------------------------------------------------------------------------
    /// @return number of rows.
    int64_t mb() const { return (op_ == blas::Op::NoTrans ? mb_ : nb_); }

    //--------------------------------------------------------------------------
    /// @return number of columns.
    int64_t nb() const { return (op_ == blas::Op::NoTrans ? nb_ : mb_); }

    //--------------------------------------------------------------------------
    /// @return transposition operation.
    blas::Op op() const { return op_; }

    //--------------------------------------------------------------------------
    /// @return the logical packed storage type.
    blas::Uplo uplo() const { return uploLogical(); }

    //--------------------------------------------------------------------------
    /// @return the logical packed storage type.
    blas::Uplo uploLogical() const
    {
        if (uplo_ == blas::Uplo::General)
            return blas::Uplo::General;
        else if ((uplo_ == blas::Uplo::Lower) == (op_ == blas::Op::NoTrans))
            return blas::Uplo::Lower;
        else
            return blas::Uplo::Upper;
    }

    //--------------------------------------------------------------------------
    /// @return the physical packed storage type.
    blas::Uplo uploPhysical() const { return uplo_; }

    //--------------------------------------------------------------------------
    /// @return the physical ordering of the matrix elements in the data array.
    blas::Layout layout() const { return layout_; }

    //--------------------------------------------------------------------------
    /// Set the physical packed storage type.
    /// @param[in] uplo
    ///     - General: both triangles are stored.
    ///     - Upper: upper triangle is stored.
    ///     - Lower: lower triangle is stored.
    void uplo(blas::Uplo uplo) { uplo_ = uplo; } // note: public (not protected)

protected:
    int64_t mb_; ///> Number of rows.
    int64_t nb_; ///> Number of columns.

    T* data_; ///> Data array buffer.

    blas::Op op_; ///> Transposition operation.
    blas::Uplo uplo_; ///> Physical packed storage type.
    blas::Layout layout_; ///> Physical ordering of the matrix elements.

}; // class BaseTile
}  // namespace hcore

#endif // HCORE_BASE_TILE_HH
