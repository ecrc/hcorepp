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

/// Transpose a tile; changing op flag from NoTrans to Trans, or from Trans to
/// NoTrans. @return shallow copy with updated op flag.
/// @param[in] A
///     tile to be transposed.
template <typename Tt>
Tt transpose(Tt& A) {
    Tt AT = A;

    if (AT.op_ == blas::Op::NoTrans)
        AT.op_ = blas::Op::Trans;
    else if (AT.op_ == blas::Op::Trans || A.is_real)
        AT.op_ = blas::Op::NoTrans;
    else
        throw Error("Unsupported operation: conjugate-no-transpose");

    return AT;
}

/// Transpose a tile; changing op flag from NoTrans to Trans, or from Trans to
/// NoTrans. @return shallow copy with updated op flag.
/// Convert rvalue refs to lvalue refs.
/// @param[in] A
///     tile to be transposed.
template <typename Tt>
Tt transpose(Tt&& A) { return transpose(A); }

/// Conjugate-transpose a tile; changing op flag from NoTrans to Trans, or from
/// Trans to NoTrans. @return shallow copy with updated op flag.
/// @param[in] A
///     tile to be conjugate-transposed.
template <typename Tt>
Tt conjugate_transpose(Tt& A) {
    Tt AT = A;

    if (AT.op_ == blas::Op::NoTrans)
        AT.op_ = blas::Op::ConjTrans;
    else if (AT.op_ == blas::Op::ConjTrans || A.is_real)
        AT.op_ = blas::Op::NoTrans;
    else
        throw Error("Unsupported operation: conjugate-no-transpose");

    return AT;
}
/// Conjugate-transpose a tile; changing op flag from NoTrans to Trans, or from
/// Trans to NoTrans. @return shallow copy with updated op flag.
/// Convert rvalue refs to lvalue refs.
/// @param[in] A
///     tile to be conjugate-transposed.
template <typename Tt>
Tt conjugate_transpose(Tt&& A) { return conjugate_transpose(A); }

template <typename T>
class BaseTile {
protected:
    BaseTile()
        : m_(0),
          n_(0),
          data_(nullptr),
          op_(blas::Op::NoTrans),
          uplo_(blas::Uplo::General),
          layout_(blas::Layout::ColMajor) {

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
    /// @param[in] layout
    ///     The physical ordering of matrix elements in the data array buffer.
    ///     blas::Layout::ColMajor: column elements are 1-strided, or
    ///     blas::Layout::RowMajor: row elements are 1-strided.
    BaseTile(int64_t m, int64_t n, T* A, blas::Layout layout)
        : m_(m),
          n_(n),
          data_(A),
          op_(blas::Op::NoTrans),
          uplo_(blas::Uplo::General),
          layout_(layout) {
        hcore_error_if(m < 0);
        hcore_error_if(n < 0);
        hcore_error_if(A == nullptr);
    }

    /// Set transposition operation.
    /// @param[in] new_op
    ///     - blas::Op::NoTrans: tile has no transposed structure.
    ///     - blas::Op::Trans: tile has a transposed structure.
    ///     - blas::Op::ConjTrans: tile has a conjugate-transposed structure.
    void op(blas::Op new_op) {
        hcore_error_if(new_op != blas::Op::NoTrans &&
                       new_op != blas::Op::Trans   &&
                       new_op != blas::Op::ConjTrans);
        op_ = new_op;
    }

public:
    /// @return true if the class template identifier is complex, and false
    /// otherwise.
    static constexpr bool is_complex = blas::is_complex<T>::value;
    /// @return true if the class template identifier isn't complex, and false
    /// otherwise.
    static constexpr bool is_real = !is_complex;

    /// Transpose and @return a shallow copy with updated op flag.
    /// @param[in] A
    ///     tile to be transposed.
    template <typename Tt>
    friend Tt transpose(Tt& A);

    /// Transpose and @return a shallow copy with updated op flag.
    /// Convert rvalue refs to lvalue refs.
    /// @param[in] A
    ///     tile to be transposed.
    template <typename Tt>
    friend Tt transpose(Tt&& A);

    /// Conjugate-transpose and @return a shallow copy with updated op flag.
    /// @param[in] A
    ///     tile to be conjugate-transposed.
    template <typename Tt>
    friend Tt conjugate_transpose(Tt& A);

    /// Conjugate-transpose and @return a shallow copy with updated op flag.
    /// Convert rvalue refs to lvalue refs.
    /// @param[in] A
    ///     tile to be conjugate-transposed.
    template <typename Tt>
    friend Tt conjugate_transpose(Tt&& A);

    /// @return number of rows.
    int64_t m() const { return (op_ == blas::Op::NoTrans ? m_ : n_); }

    /// Sets number of rows.
    /// @param[in] m
    ///     Number of rows.
    void m(int64_t m) {
        hcore_error_if(0 > m || m >= this->m());

        if (op_ == blas::Op::NoTrans)
            m_ = m;
        else
            n_ = m;
    }

    /// @return number of columns.
    int64_t n() const { return (op_ == blas::Op::NoTrans ? n_ : m_); }

    /// Sets number of columns.
    /// @param[in] n
    ///     Number of columns.
    void n(int64_t n) {
        hcore_error_if(0 > n || n >= this->n());

        if (op_ == blas::Op::NoTrans)
            n_ = n;
        else
            m_ = n;
    }

    /// @return transposition operation.
    blas::Op op() const { return op_; }

    /// @return the logical packed storage type.
    blas::Uplo uplo() const { return uplo_logical(); }

    /// Set the physical packed storage type.
    /// @param[in] uplo
    ///     - General: both triangles are stored.
    ///     - Upper:   upper triangle is stored.
    ///     - Lower:   lower triangle is stored.
    void uplo(blas::Uplo uplo) {
        hcore_error_if(uplo != blas::Uplo::General &&
                       uplo != blas::Uplo::Lower   &&
                       uplo != blas::Uplo::Upper);
        uplo_ = uplo;
    }

    /// @return the physical packed storage type.
    blas::Uplo uplo_physical() const { return uplo_; }

    /// @return the logical packed storage type.
    blas::Uplo uplo_logical() const {
        if (uplo_ == blas::Uplo::General)
            return blas::Uplo::General;
        else if ((uplo_ == blas::Uplo::Lower) == (op_ == blas::Op::NoTrans))
            return blas::Uplo::Lower;
        else
            return blas::Uplo::Upper;
    }

    /// @return the physical ordering of the matrix elements in the data array.
    blas::Layout layout() const { return layout_; }

    /// Set the physical ordering of the matrix elements in the data array.
    void layout(blas::Layout layout) const {
        hcore_error_if(layout != blas::Layout::ColMajor &&
                       layout != blas::Layout::RowMajor);

        return layout_ = layout;
    }

protected:
    int64_t m_; ///> Number of rows.
    int64_t n_; ///> Number of columns.

    T* data_; ///> Data array buffer.

    blas::Op op_;         ///> Transposition operation.
    blas::Uplo uplo_;     ///> Physical packed storage type.
    blas::Layout layout_; ///> Physical ordering of the matrix elements.

}; // class BaseTile
}  // namespace hcore

#endif // HCORE_BASE_TILE_HH
