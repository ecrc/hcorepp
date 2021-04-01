// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_TILE_UTILS_HH
#define HCORE_TEST_TILE_UTILS_HH

#include "hcore/tile/tile.hh"
#include "hcore/tile/dense.hh"
#include "hcore/tile/compressed.hh"

#include "blas.hh"

#include <string>
#include <cstdio>
#include <cstdint>

template <typename T>
void diff(T* Aref, int64_t lda_ref, hcore::Tile<T> const& A)
{
    for (int64_t j = 0; j < A.n(); ++j) {
        for (int64_t i = 0; i < A.m(); ++i) {
            Aref[i + j * lda_ref] -= A(i, j);
        }
    }
}

template <typename T>
void copy(T* Aref, int64_t lda_ref, hcore::Tile<T> const& A)
{
    for (int64_t j = 0; j < A.n(); ++j) {
        for (int64_t i = 0; i < A.m(); ++i) {
            Aref[i + j * lda_ref] = A(i, j);
        }
    }
}

template <typename T>
void pretty_print(
    int64_t m, int64_t n, T const* A, int64_t lda,
    char const* label, int width=12, int precision=9)
{
    using blas::real;
    using blas::imag;

    char buffer[1024];
    std::string output;

    printf("%s = [\n", label);
    for (int64_t i = 0; i < m; ++i) {
        output = "";
        for (int64_t j = 0; j < n; ++j) {
            const T a = A[i+j*lda];
            if (a == T(int64_t(real(a)))) {
                if (blas::is_complex<T>::value) {
                    std::snprintf(buffer, sizeof(buffer), " %#*.0f%*s   %*s ",
                        width - precision, real(a), precision, "", width, "");
                }
                else {
                    std::snprintf(buffer, sizeof(buffer), " %#*.0f%*s",
                        width - precision, real(a), precision, "");
                }
            }
            else {
                if (blas::is_complex<T>::value) {
                    std::snprintf(buffer, sizeof(buffer), " %*.*f + %*.*fi",
                        width, precision, real(a), width, precision, imag(a));
                }
                else {
                    std::snprintf(buffer, sizeof(buffer), " %*.*f",
                        width, precision, real(a));
                }
            }
            output += buffer;
        }
        printf("%s\n", output.c_str());
    }
    printf("];\n");
}

template <typename T>
void pretty_print(
    hcore::DenseTile<T> const& A,
    char const* label, int width=12, int precision=9)
{
    // forward
    pretty_print(A.m(), A.n(), A.data(), A.ld(), label, width, precision);
}

template <typename T>
void pretty_print(
    hcore::CompressedTile<T> const& A,
    char const* label, int width=12, int precision=9)
{
    std::string const& Ulabel = (A.is_full_rk() ? std::string(label)
                                                : std::string(label) + "U");
    // forward
    pretty_print(
        A.m(), A.rk(), A.Udata(), A.ldu(), Ulabel.c_str(), width, precision);
    if (!A.is_full_rk()) {
        std::string const& Vlabel = std::string(label) + "V";
        pretty_print(
            A.rk(), A.n(), A.Vdata(), A.ldv(), Vlabel.c_str(),
            width, precision);
    }
}

#endif // HCORE_TEST_TILE_UTILS_HH
