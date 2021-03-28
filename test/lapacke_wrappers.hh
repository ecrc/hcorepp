// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_LAPACKE_WRAPPERS_HH
#define HCORE_TEST_LAPACKE_WRAPPERS_HH

#include "lapack/util.hh"
#include "lapack/config.h"
#ifdef HCORE_WITH_MKL
    #define MKL_Complex8  lapack_complex_float
    #define MKL_Complex16 lapack_complex_double
    #include <mkl_lapacke.h>
#else
    #include <lapacke.h>
#endif
#include "lapack/mangling.h"

#include <string>
#include <complex>
#include <stdexcept>

inline void lapacke_latms(
    lapack_int m, lapack_int n, char dist, lapack_int* iseed, char sym,
    float* d, lapack_int mode, float cond, float dmax, lapack_int kl,
    lapack_int ku, char pack, float* A, lapack_int lda)
{
    // todo: shouldn't assume column-major
    lapack_int info = LAPACKE_slatms(LAPACK_COL_MAJOR,
        m, n, dist, iseed, sym, d, mode, cond, dmax, kl, ku, pack,
        A, lda);
    if (info != 0) {
        const std::string& what_arg =
            "LAPACKE_slatms error: " + std::to_string(info) + std::string(".");
        throw std::runtime_error(what_arg);
    }
}

inline void lapacke_latms(
    lapack_int m, lapack_int n, char dist, lapack_int* iseed, char sym,
    double* d, lapack_int mode, double cond, double dmax, lapack_int kl,
    lapack_int ku, char pack, double* A, lapack_int lda)
{
    // todo: shouldn't assume column-major
    lapack_int info = LAPACKE_dlatms(LAPACK_COL_MAJOR,
        m, n, dist, iseed, sym, d, mode, cond, dmax, kl, ku, pack,
        A, lda);
    if (info < 0) {
        const std::string& what_arg =
            "LAPACKE_dlatms error: " + std::to_string(info) + std::string(".");
        throw std::runtime_error(what_arg);
    }
}

inline void lapacke_latms(
    lapack_int m, lapack_int n, char dist, lapack_int* iseed, char sym,
    float* d, lapack_int mode, float cond, float dmax, lapack_int kl,
    lapack_int ku, char pack, std::complex<float>* A, lapack_int lda)
{
    // todo: shouldn't assume column-major
    lapack_int info = LAPACKE_clatms(LAPACK_COL_MAJOR,
        m, n, dist, iseed, sym, d, mode, cond, dmax, kl, ku, pack,
        (lapack_complex_float*)A, lda);
    if (info < 0) {
        const std::string& what_arg =
            "LAPACKE_clatms error: " + std::to_string(info) + std::string(".");
        throw std::runtime_error(what_arg);
    }
}

inline void lapacke_latms(
    lapack_int m, lapack_int n, char dist, lapack_int* iseed, char sym,
    double* d, lapack_int mode, double cond, double dmax, lapack_int kl,
    lapack_int ku, char pack, std::complex<double>* A, lapack_int lda)
{
    // todo: shouldn't assume column-major
    lapack_int info = LAPACKE_zlatms(LAPACK_COL_MAJOR,
        m, n, dist, iseed, sym, d, mode, cond, dmax, kl, ku, pack,
        (lapack_complex_double*)A, lda);
    if (info < 0) {
        const std::string& what_arg =
            "LAPACKE_zlatms error: " + std::to_string(info) + std::string(".");
        throw std::runtime_error(what_arg);
    }
}

#endif // HCORE_TEST_LAPACKE_WRAPPERS_HH
