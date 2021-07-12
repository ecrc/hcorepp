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

public:
    /// @return true if the class template identifier is complex, and false
    /// otherwise.
    static constexpr bool is_complex = blas::is_complex<T>::value;

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

    /// Set transposition operation.
    /// @param[in] op
    ///     - Trans: this tile has a transposed structure.
    ///     - ConjTrans: this tile has a conjugate-transposed structure.
    ///     - NoTrans: this tile has no transposed structure.
    void op(blas::Op op) {
        hcore_error_if(op != blas::Op::Trans   &&
                       op != blas::Op::NoTrans &&
                       op != blas::Op::ConjTrans);
        op_ = op;
    }

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
