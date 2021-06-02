// Copyright (c) 2017-2021, King Abdullah University of Science and Technology
// (KAUST). All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_LAPACK_WRAPPERS_HH
#define HCORE_TEST_LAPACK_WRAPPERS_HH

#include "lapack.hh"
#include "lapack/fortran.h"

#include <cmath>     // std::abs
#include <cctype>    // std::toupper
#include <limits>    // std::numeric_limits<T>::max
#include <vector>    // std::vector
#include <complex>   // std::complex
#include <cstdint>   // int64_t
#include <algorithm> // std::copy, std::max

namespace lapack {

// =============================================================================
//
/// Specifies the type of distribution to be used to generate the random
/// eigenvalues or singular values.
enum class Dist {
    Normal           = 'N', ///> Normal(0, 1).
    Uniform          = 'U', ///> Uniform(0, 1).
    SymmetricUniform = 'S', ///> Uniform for symmetric(-1, 1).
};

inline char dist2char(lapack::Dist dist)
    { return char(dist); }

inline lapack::Dist char2dist(char dist)
{
    dist = char(std::toupper(dist));

    lapack_error_if(dist != 'N' && dist != 'U' && dist != 'S');

    return lapack::Dist(dist);
}

inline const char* dist2str(lapack::Dist dist)
{
    switch (dist) {
        case Dist::Normal:           return "normal";
        case Dist::Uniform:          return "uniform";
        case Dist::SymmetricUniform: return "symmetric uniform";
    }
    return "?";
}

/// Generated matrix types.
enum class Sym {
    /// Symmetric matrix with positive, negative, or zero eigenvalues.
    Symmetric         = 'S', Hermitian = 'H',
    /// Nonsymmetric matrix with positive eigenvalues.
    Nonsymmetric      = 'N',
    /// Symmetric matrix with positive eigenvalues.
    PositiveSymmetric = 'P',
}; // enum class Sym

inline char sym2char(lapack::Sym sym)
    { return char(sym); }

inline lapack::Sym char2sym(char sym)
{
    sym = char(std::toupper(sym));

    lapack_error_if(sym != 'S' && sym != 'H' && sym != 'N' && sym != 'P');

    return lapack::Sym(sym);
}

inline const char* sym2str(lapack::Sym sym)
{
    switch (sym) {
        case Sym::Symmetric:         return "symmetric";
        case Sym::Hermitian:         return "hermitian";
        case Sym::Nonsymmetric:      return "nonsymmetric";
        case Sym::PositiveSymmetric: return "positive symmetric";
    }
    return "?";
}

/// Specifies packing of matrix.
enum class Pack {
    /// No packing.
    NoPacking         = 'N',
    /// Zero out all subdiagonal entries (if symmetric).
    ZeroSubdiagonal   = 'U',
    /// Zero out all superdiagonal entries (if symmetric).
    ZeroSuperdiagonal = 'L',
    /// Store the upper triangle (if symmetric or upper triangular).
    UpperTriangle     = 'C',
    /// Store the lower triangle (if symmetric or lower triangular).
    LowerTriangle     = 'R',
    /// Store the banded lower triangle (if symmetric or lower triangular).
    LowerTriangleBand = 'B',
    /// Store the banded upper triangle (if symmetric or upper triangular).
    UpperTriangleBand = 'Q',
    /// Store the entire banded matrix.
    EntireMatrixBand  = 'Z',
}; // enum class Pack

inline char pack2char(lapack::Pack pack)
    { return char(pack); }

inline lapack::Pack char2pack(char pack)
{
    pack = char(std::toupper(pack));

    lapack_error_if(pack != 'N' && pack != 'U' && pack != 'L' && pack != 'C' &&
                    pack != 'R' && pack != 'B' && pack != 'Q' && pack != 'Z');

    return lapack::Pack(pack);
}

inline const char* pack2str(lapack::Pack pack)
{
    switch (pack) {
        case Pack::NoPacking:         return "no packing";
        case Pack::ZeroSubdiagonal:   return "zero subdiagonal";
        case Pack::ZeroSuperdiagonal: return "zero superdiagonal";
        case Pack::UpperTriangle:     return "store upper triangle";
        case Pack::LowerTriangle:     return "store lower triangle";
        case Pack::LowerTriangleBand: return "store banded lower triangle";
        case Pack::UpperTriangleBand: return "store banded upper triangle";
        case Pack::EntireMatrixBand:  return "store entire banded matrix";
    }
    return "?";
}

// =============================================================================
//
inline void latms(
    int64_t m, int64_t n, Dist dist, int64_t* iseed, Sym sym, float* d,
    int64_t mode, float cond, float dmax, int64_t kl, int64_t ku, Pack pack,
    float* A, int64_t lda)
{
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if(std::abs(m   )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(n   )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(kl  )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(ku  )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(lda )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(mode)>std::numeric_limits<lapack_int>::max());
    }

    char dist_ = dist2char ( dist );
    char sym_  = sym2char  ( sym  );
    char pack_ = pack2char ( pack );

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
        &m_, &n_, &dist_, iseed_ptr, &sym_, d, &mode_, &cond, &dmax, &kl_, &ku_,
        &pack_, A, &lda_, &work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );

    if (info_ < 0) {
        throw lapack::Error();
    }

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        std::copy(iseed_.begin(), iseed_.end(), iseed);
    #endif
}

inline void latms(
    int64_t m, int64_t n, Dist dist, int64_t* iseed, Sym sym, double* d,
    int64_t mode, double cond, double dmax, int64_t kl, int64_t ku, Pack pack,
    double* A, int64_t lda)
{
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if(std::abs(m   )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(n   )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(kl  )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(ku  )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(lda )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(mode)>std::numeric_limits<lapack_int>::max());
    }

    char dist_ = dist2char ( dist );
    char sym_  = sym2char  ( sym  );
    char pack_ = pack2char ( pack );

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
        &m_, &n_, &dist_, iseed_ptr, &sym_, d, &mode_, &cond, &dmax, &kl_, &ku_,
        &pack_, A, &lda_, &work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );

    if (info_ < 0) {
        throw lapack::Error();
    }

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        std::copy(iseed_.begin(), iseed_.end(), iseed);
    #endif
}

inline void latms(
    int64_t m, int64_t n, Dist dist, int64_t* iseed, Sym sym, float* d,
    int64_t mode, float cond, float dmax, int64_t kl, int64_t ku, Pack pack,
    std::complex<float>* A, int64_t lda)
{
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if(std::abs(m   )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(n   )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(kl  )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(ku  )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(lda )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(mode)>std::numeric_limits<lapack_int>::max());
    }

    char dist_ = dist2char ( dist );
    char sym_  = sym2char  ( sym  );
    char pack_ = pack2char ( pack );

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
        &m_, &n_, &dist_, iseed_ptr, &sym_, d, &mode_, &cond, &dmax, &kl_, &ku_,
        &pack_, (lapack_complex_float*)A, &lda_,
        (lapack_complex_float*)&work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );

    if (info_ < 0) {
        throw lapack::Error();
    }

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        std::copy(iseed_.begin(), iseed_.end(), iseed);
    #endif
}

inline void latms(
    int64_t m, int64_t n, Dist dist, int64_t* iseed, Sym sym, double* d,
    int64_t mode, double cond, double dmax, int64_t kl, int64_t ku, Pack pack,
    std::complex<double>* A, int64_t lda)
{
    if (sizeof(int64_t) > sizeof(lapack_int)) {
        lapack_error_if(std::abs(m   )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(n   )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(kl  )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(ku  )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(lda )>std::numeric_limits<lapack_int>::max());
        lapack_error_if(std::abs(mode)>std::numeric_limits<lapack_int>::max());
    }

    char dist_ = dist2char ( dist );
    char sym_  = sym2char  ( sym  );
    char pack_ = pack2char ( pack );

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
        &m_, &n_, &dist_, iseed_ptr, &sym_, d, &mode_, &cond, &dmax, &kl_, &ku_,
        &pack_, (lapack_complex_double*)A, &lda_,
        (lapack_complex_double*)&work[0], &info_
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );

    if (info_ < 0) {
        throw lapack::Error();
    }

    #if ! defined(BLAS_ILP64) || ! defined(LAPACK_ILP64)
        std::copy(iseed_.begin(), iseed_.end(), iseed);
    #endif
}

} // namespace lapack

#endif // HCORE_TEST_LAPACK_WRAPPERS_HH
