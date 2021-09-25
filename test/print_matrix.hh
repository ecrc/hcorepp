// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_PRINT_MATRIX_HH
#define HCORE_TEST_PRINT_MATRIX_HH

#include <cstdint>
#include <string>
#include <cstdio>

#include "blas.hh"

#include "hcore/compressed_tile.hh"
#include "hcore/tile.hh"

namespace hcore {
namespace test {

template <typename T>
void print_matrix(int64_t m, int64_t n, T const* A, int64_t lda,
                  char const* label, const char* format = "%9.4f")
{
    #define A(i_, j_) A[(i_) + size_t(lda)*(j_)]

    assert(m >= 0);
    assert(n >= 0);
    assert(lda >= m);

    char format2[32];
    if (blas::is_complex<T>::value) {
        snprintf(format2, sizeof(format2), " %s + %si", format, format);
    }
    else {
        snprintf(format2, sizeof(format2), " %s", format);
    }

    using blas::real;
    using blas::imag;

    printf("%s = [\n", label);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            if (blas::is_complex<T>::value) {
                printf(format2, real(A(i, j)), imag(A(i, j)));
            }
            else {
                printf(format2, A(i, j));
            }
        }
        printf("\n");
    }
    printf("];\n");

    #undef A
}

template <typename T>
void print_matrix(Tile<T> A, char const* label, const char* format="%9.4f")
{
    int64_t m = A.layout() == blas::Layout::ColMajor ? A.mb() : A.nb();
    int64_t n = A.layout() == blas::Layout::ColMajor ? A.nb() : A.mb();

    print_matrix(m, n, A.data(), A.stride(), label, format);
}

template <typename T>
void print_matrix(CompressedTile<T> A, char const* label,
                  const char* format="%9.4f")
{
    int64_t m = A.layout() == blas::Layout::ColMajor ? A.mb() : A.nb();
    int64_t n = A.layout() == blas::Layout::ColMajor ? A.nb() : A.mb();

    T const* U = A.layout() == blas::Layout::ColMajor ? A.Udata() : A.Vdata();
    T const* V = A.layout() == blas::Layout::ColMajor ? A.Vdata() : A.Udata();

    int64_t ldu = A.layout() == blas::Layout::ColMajor ? A.Ustride()
                                                       : A.Vstride();
    int64_t ldv = A.layout() == blas::Layout::ColMajor ? A.Vstride()
                                                       : A.Ustride();
    int64_t rk = A.rk();

    std::string const& Ulabel = std::string(label) + "U";
    std::string const& Vlabel = std::string(label) + "V";

    print_matrix(m, rk, U, ldu, Ulabel.c_str(), format);
    print_matrix(rk, n, V, ldv, Vlabel.c_str(), format);
}

} // namespace test
} // namespace hcore

#endif // HCORE_TEST_PRINT_MATRIX_HH