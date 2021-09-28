// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST).
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include <algorithm> // std::min
#include <stdint.h>  // int64_t
#include <stdio.h>   // printf
#include <vector>    // std::vector

#include "hcore/hcore.hh"

template <typename T>
void execute(int64_t m, int64_t n, int64_t k)
{
    int64_t ldau = m;
    int64_t ldbu = k;
    int64_t ldcu = m;

    int64_t Ark = std::min(m, k) / 2;
    int64_t Brk = std::min(k, n) / 2;
    int64_t Crk = std::min(m, n) / 2;

    int64_t ldav = Ark;
    int64_t ldbv = Brk;
    int64_t ldcv = Crk;

    blas::real_type<T> tol = 1e-4;

    std::vector<T> A_data(ldau*Ark + ldav*Ark, 1.0);
    std::vector<T> B_data(ldbu*Brk + ldbv*Brk, 2.0);
    std::vector<T> C_data(ldcu*Crk + ldcv*Crk, 3.0);

    hcore::CompressedTile<T> A(m, k, &A_data[0], ldau, ldav, Ark, tol);
    hcore::CompressedTile<T> B(k, n, &B_data[0], ldbu, ldbv, Brk, tol);
    hcore::CompressedTile<T> C(m, n, &C_data[0], ldcu, ldcv, Crk, tol);

    // C = -1.0 * A * B + 1.0 * C
    hcore::gemm<T>(-1.0, A, B, 1.0, C);

    // Note: The matrices can be transposed or conjugate-transposed beforehand.
    // For example:
    //
    //     auto AT = hcore::transpose(A);
    //     auto BT = hcore::conjugate_transpose(B);
    //     hcore::gemm<T>(alpha, AT, BT, beta, C);

    // This is need to avoid memory leak because the C data array is resized
    // inside gemm, when C is compressed
    C.clear();
}

// -----------------------------------------------------------------------------
// Performs tile low-rank Matrix-matrix multiplication:
// C = alpha * A * B + beta * C, where alpha and beta are scalars, and A, B, and
// C are compressed matrices, with A an m-by-k tile low-rank matrix
// (U: m-by-Ark, V: Ark-by-k), B a k-by-n tile low-rank matrix
// (U: k-by-Brk, V: Brk-by-n), and C an m-by-n tile low-rank matrix
// (U: m-by-Crk, V: Crk-by-n).
int main(int argc, char* argv[])
{
    int64_t m = 100, n = 200, k = 50;

    printf("execute<float>(%lld, %lld, %lld)\n", m, n, k);
    execute<float>(m, n, k);

    printf("execute<double>(%lld, %lld, %lld)\n", m, n, k);
    execute<double>(m, n, k);

    printf("execute<complex<float>>(%lld, %lld, %lld)\n", m, n, k);
    execute<std::complex<float>>(m, n, k);

    printf("execute<complex<double>>(%lld, %lld, %lld)\n", m, n, k);
    execute<std::complex<double>>(m, n, k);

    return 0;
}
