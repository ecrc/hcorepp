// Copyright (c) 2017-2021, King Abdullah University of Science and Technology
// (KAUST). All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_LAPACK_WRAPPERS_HH
#define HCORE_TEST_LAPACK_WRAPPERS_HH

#include "lapack/fortran.h"

#include <vector>
#include <complex>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

inline void lapack_latms(
    int64_t m, int64_t n, char dist, int64_t* iseed, char sym,
    float* d, int64_t mode, float cond, float dmax, int64_t kl, int64_t ku,
    char pack, float* A, int64_t lda)
{
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        assert( std::abs( m    ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( n    ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( kl   ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( ku   ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( lda  ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( mode ) <= std::numeric_limits<lapack_int>::max() );
    }

    lapack_int    m_ = (lapack_int)    m;
    lapack_int    n_ = (lapack_int)    n;
    lapack_int   kl_ = (lapack_int)   kl;
    lapack_int   ku_ = (lapack_int)   ku;
    lapack_int  lda_ = (lapack_int)  lda;
    lapack_int mode_ = (lapack_int) mode;

    std::vector<float> work(3 * std::max(m, n));

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        // 32-bit copy
        std::vector<lapack_int> iseed_(&iseed[0], &iseed[(4)]);
        lapack_int* iseed_ptr = &iseed_[0];
    #else
        lapack_int* iseed_ptr = iseed;
    #endif

    lapack_int info_ = 0;

    LAPACK_slatms(
        &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
        &pack, A, &lda_,
        &work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );

    if (info_ != 0)
        throw std::runtime_error("lapack_latms error " + std::to_string(info_));

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        std::copy(iseed_.begin(), iseed_.end(), iseed);
    #endif
}

inline void lapack_latms(
    int64_t m, int64_t n, char dist, int64_t* iseed, char sym,
    double* d, int64_t mode, double cond, double dmax, int64_t kl, int64_t ku,
    char pack, double* A, int64_t lda)
{
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        assert( std::abs( m    ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( n    ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( kl   ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( ku   ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( lda  ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( mode ) <= std::numeric_limits<lapack_int>::max() );
    }

    lapack_int    m_ = (lapack_int)    m;
    lapack_int    n_ = (lapack_int)    n;
    lapack_int   kl_ = (lapack_int)   kl;
    lapack_int   ku_ = (lapack_int)   ku;
    lapack_int  lda_ = (lapack_int)  lda;
    lapack_int mode_ = (lapack_int) mode;

    std::vector<double> work(3 * std::max(m, n));

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        // 32-bit copy
        std::vector<lapack_int> iseed_(&iseed[0], &iseed[(4)]);
        lapack_int* iseed_ptr = &iseed_[0];
    #else
        lapack_int* iseed_ptr = iseed;
    #endif

    lapack_int info_ = 0;

    LAPACK_dlatms(
        &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
        &pack, A, &lda_,
        &work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );

    if (info_ != 0)
        throw std::runtime_error("lapack_latms error " + std::to_string(info_));

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        std::copy(iseed_.begin(), iseed_.end(), iseed);
    #endif
}

inline void lapack_latms(
    int64_t m, int64_t n, char dist, int64_t* iseed, char sym,
    float* d, int64_t mode, float cond, float dmax, int64_t kl, int64_t ku,
    char pack, std::complex<float>* A, int64_t lda)
{
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        assert( std::abs( m    ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( n    ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( kl   ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( ku   ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( lda  ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( mode ) <= std::numeric_limits<lapack_int>::max() );
    }

    lapack_int    m_ = (lapack_int)    m;
    lapack_int    n_ = (lapack_int)    n;
    lapack_int   kl_ = (lapack_int)   kl;
    lapack_int   ku_ = (lapack_int)   ku;
    lapack_int  lda_ = (lapack_int)  lda;
    lapack_int mode_ = (lapack_int) mode;

    std::vector<std::complex<float>> work(3 * std::max(m, n));

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        // 32-bit copy
        std::vector<lapack_int> iseed_(&iseed[0], &iseed[(4)]);
        lapack_int* iseed_ptr = &iseed_[0];
    #else
        lapack_int* iseed_ptr = iseed;
    #endif

    lapack_int info_ = 0;

    LAPACK_clatms(
        &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
        &pack, (lapack_complex_float*)A, &lda_,
        (lapack_complex_float*)&work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );

    if (info_ != 0)
        throw std::runtime_error("lapack_latms error " + std::to_string(info_));

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        std::copy(iseed_.begin(), iseed_.end(), iseed);
    #endif
}

inline void lapack_latms(
    int64_t m, int64_t n, char dist, int64_t* iseed, char sym,
    double* d, int64_t mode, double cond, double dmax, int64_t kl, int64_t ku,
    char pack, std::complex<double>* A, int64_t lda)
{
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        assert( std::abs( m    ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( n    ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( kl   ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( ku   ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( lda  ) <= std::numeric_limits<lapack_int>::max() );
        assert( std::abs( mode ) <= std::numeric_limits<lapack_int>::max() );
    }

    lapack_int    m_ = (lapack_int)    m;
    lapack_int    n_ = (lapack_int)    n;
    lapack_int   kl_ = (lapack_int)   kl;
    lapack_int   ku_ = (lapack_int)   ku;
    lapack_int  lda_ = (lapack_int)  lda;
    lapack_int mode_ = (lapack_int) mode;

    std::vector<std::complex<double>> work(3 * std::max(m, n));

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        // 32-bit copy
        std::vector<lapack_int> iseed_(&iseed[0], &iseed[(4)]);
        lapack_int* iseed_ptr = &iseed_[0];
    #else
        lapack_int* iseed_ptr = iseed;
    #endif

    lapack_int info_ = 0;

    LAPACK_zlatms(
        &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
        &pack, (lapack_complex_double*)A, &lda_,
        (lapack_complex_double*)&work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );

    if (info_ != 0)
        throw std::runtime_error("lapack_latms error " + std::to_string(info_));

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        std::copy(iseed_.begin(), iseed_.end(), iseed);
    #endif
}

#endif // HCORE_TEST_LAPACK_WRAPPERS_HH
