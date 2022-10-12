// Copyright (c) 2017-2021, King Abdullah University of Science and Technology
// (KAUST). All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_LAPACK_WRAPPERS_HH
#define HCORE_TEST_LAPACK_WRAPPERS_HH

#include <vector>

#include <complex>
#include <cassert>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <lapack/fortran.h>

template<typename T>
struct is_complex_t : public std::false_type {
};

template<typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {
};

template<typename T>
constexpr bool is_complex() {
    return is_complex_t<T>::value;
}

template<typename T>
struct is_double_t : public std::false_type {
};

template<>
struct is_double_t<double> : public std::true_type {
};

template<typename T>
constexpr bool is_double() {
    return is_double_t<T>::value;
}

template<typename T, typename C>
void lapack_latms_core(int64_t m, int64_t n, char dist, int64_t *iseed, char sym,
                       T *d, int64_t mode, T cond, T dmax, int64_t kl, int64_t ku,
                       char pack, C *A, int64_t lda) {
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        assert(std::abs(m) <= std::numeric_limits<lapack_int>::max());
        assert(std::abs(n) <= std::numeric_limits<lapack_int>::max());
        assert(std::abs(kl) <= std::numeric_limits<lapack_int>::max());
        assert(std::abs(ku) <= std::numeric_limits<lapack_int>::max());
        assert(std::abs(lda) <= std::numeric_limits<lapack_int>::max());
        assert(std::abs(mode) <= std::numeric_limits<lapack_int>::max());
    }

    lapack_int m_ = (lapack_int) m;
    lapack_int n_ = (lapack_int) n;
    lapack_int kl_ = (lapack_int) kl;
    lapack_int ku_ = (lapack_int) ku;
    lapack_int lda_ = (lapack_int) lda;
    lapack_int mode_ = (lapack_int) mode;

    std::vector<C> work(3 * std::max(m, n));

#ifndef HCORE_WITH_LAPACK_ILP64
    // 32-bit copy
    std::vector<lapack_int> iseed_(&iseed[0], &iseed[(4)]);
    lapack_int *iseed_ptr = &iseed_[0];
#else
    lapack_int* iseed_ptr = iseed;
#endif

    lapack_int info_ = 0;

    if constexpr(is_complex<C>()) {
        if constexpr(is_double<T>()) {
            LAPACK_zlatms(
                    &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
                    &pack, (lapack_complex_double *) A, &lda_,
                    (lapack_complex_double *) &work[0], &info_
#ifdef LAPACK_FORTRAN_STRLEN_END
                    , 1, 1, 1
#endif
            );
        } else {
            LAPACK_clatms(
                    &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
                    &pack, (lapack_complex_float*)A, &lda_,
                    (lapack_complex_float*)&work[0], &info_
#ifdef LAPACK_FORTRAN_STRLEN_END
                    , 1, 1, 1
#endif
            );
        }
    } else {
        if constexpr (is_double<T>()) {
            LAPACK_dlatms(
                    &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
                    &pack, A, &lda_,
                    &work[0], &info_
#ifdef LAPACK_FORTRAN_STRLEN_END
                    , 1, 1, 1
#endif
            );
        } else {
            LAPACK_slatms(
                    &m_, &n_, &dist, iseed_ptr, &sym, d, &mode_, &cond, &dmax, &kl_, &ku_,
                    &pack, A, &lda_,
                    &work[0], &info_
#ifdef LAPACK_FORTRAN_STRLEN_END
                    , 1, 1, 1
#endif
            );
        }
    }
    if (info_ != 0) {
        throw std::runtime_error("lapack_latms error " + std::to_string(info_));
    }

#ifndef HCORE_WITH_LAPACK_ILP64
    std::copy(iseed_.begin(), iseed_.end(), iseed);
#endif
}

template<typename T>
void lapack_latms(int64_t m, int64_t n, char dist, int64_t *iseed, char sym,
                  T *d, int64_t mode, T cond, T dmax, int64_t kl, int64_t ku,
                  char pack, T *A, int64_t lda) {
    lapack_latms_core<T, T>(m, n, dist, iseed, sym, d, mode, cond, dmax, kl, ku, pack, A, lda);
}

template<typename T>
void lapack_latms(int64_t m, int64_t n, char dist, int64_t *iseed, char sym,
                  T *d, int64_t mode, T cond, T dmax, int64_t kl, int64_t ku,
                  char pack, std::complex<T> *A, int64_t lda) {
    lapack_latms_core<T, std::complex<T>>(m, n, dist, iseed, sym, d, mode, cond, dmax, kl, ku, pack, A, lda);
}

#endif // HCORE_TEST_LAPACK_WRAPPERS_HH
