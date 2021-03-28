// Copyright (c) 2017,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_PRETTY_PRINT_HH
#define HCORE_TEST_PRETTY_PRINT_HH

#include "hcore/tile/dense.hh"
#include "hcore/tile/compressed.hh"

#include "blas.hh"

#include <string>
#include <cstdint>
#include <stdio.h>

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
                    snprintf(buffer, sizeof(buffer), " %#*.0f%*s   %*s ",
                        width - precision, real(a), precision, "", width, "");
                }
                else {
                    snprintf(buffer, sizeof(buffer), " %#*.0f%*s",
                        width - precision, real(a), precision, "");
                }
            }
            else {
                if (blas::is_complex<T>::value) {
                    snprintf(buffer, sizeof(buffer), " %*.*f + %*.*fi",
                        width, precision, real(a), width, precision, imag(a));
                }
                else {
                    snprintf(buffer, sizeof(buffer), " %*.*f",
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
    std::string const& Ulabel = std::string(label) + "U";
    std::string const& Vlabel = std::string(label) + "V";

    // forward
    pretty_print(
        A.m(), A.rk(), A.Udata(), A.ldu(), Ulabel.c_str(), width, precision);
    pretty_print(
        A.rk(), A.n(), A.Vdata(), A.ldv(), Vlabel.c_str(), width, precision);
}

#endif // HCORE_TEST_PRETTY_PRINT_HH
