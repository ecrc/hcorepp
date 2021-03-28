// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "hcore.hh"

#include <vector>
#include <cstdint> // int64_t
#include <stdio.h> // printf
#include <algorithm> // std::min

template <typename T>
void execute(int64_t m, int64_t n, int64_t k)
{
    int64_t lda = m;
    int64_t ldb = n;
    int64_t ldc = m;

    int64_t Ark = std::min(m, k) / 2;
    int64_t Brk = std::min(k, n) / 2;
    int64_t Crk = std::min(m, n) / 2;

    blas::real_type<T> accuracy = 1e-4;

    // m-by-k (U: m-by-Ark; V: Ark-by-k)
    std::vector<T> AUVdata((lda + k) * Ark, 1.0);
    hcore::CompressedTile<T> A(m, k, &AUVdata[0], lda, Ark, accuracy);

    // k-by-n (U: k-by-Brk; V: Brk-by-n)
    std::vector<T> BUVdata((ldb + n) * Brk, 2.0);
    hcore::CompressedTile<T> B(k, n, &BUVdata[0], ldb, Brk, accuracy);

    // m-by-n (U: m-by-Crk; V: Crk-by-n)
    std::vector<T> CUVdata((ldc + n) * Crk, 3.0);
    hcore::CompressedTile<T> C(m, n, &CUVdata[0], ldc, Crk, accuracy);

    // C = -1.0 * A * B + 1.0 * C
    hcore::gemm<T>(-1.0, A, B, 1.0, C);
}

int main(int argc, char* argv[])
{
    int64_t m = 100, n = 200, k = 50;

    printf("execute<float>(%d, %d, %d)\n", m, n, k);
    execute<float>(m, n, k);

    printf("execute<double>(%d, %d, %d)\n", m, n, k);
    execute<double>(m, n, k);

    printf("execute<complex<float>>(%d, %d, %d)\n", m, n, k);
    execute<std::complex<float>>(m, n, k);

    printf("execute<complex<double>>(%d, %d, %d)\n", m, n, k);
    execute<std::complex<double>>(m, n, k);

    return 0;
}
