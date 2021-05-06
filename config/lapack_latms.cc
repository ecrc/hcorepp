// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include <cstdio> // printf

#if defined( FORTRAN_UPPER )
    #define FORTRAN_NAME( lower, UPPER ) UPPER
#elif defined( FORTRAN_LOWER )
    #define FORTRAN_NAME( lower, UPPER ) lower
#else
    #define FORTRAN_NAME( lower, UPPER ) lower ## _
#endif

#if defined( BLAS_ILP64 ) || defined( LAPACK_ILP64 )
    #include <vector>    // std::vector
    #include <cstdint>   // int64_t
    #include <algorithm> // std::copy

    typedef int64_t blas_int;
    typedef int64_t lapack_int;
#else
    typedef int blas_int;
    typedef int lapack_int;
#endif

#define LAPACK_dlatms FORTRAN_NAME(dlatms, DLATMS)
#ifdef __cplusplus
extern "C"
#endif
void LAPACK_dlatms(
    lapack_int const* m, lapack_int const* n, char const* dist,
    lapack_int* iseed, char const* sym, double* d, lapack_int const* mode,
    double const* cond, double const* dmax, lapack_int const* kl,
    lapack_int const* ku, char const* pack, double* a, lapack_int const* lda,
    double* work, lapack_int* info
    #ifdef LAPACK_FORTRAN_STRLEN_END
    , unsigned dist_len, unsigned sym_len, unsigned pack_len
    #endif
    );

int main( int argc, char* argv[] )
{
    lapack_int m[]    = { 5, 5 };
    lapack_int klu[]  = { 4, 4 };
    lapack_int mode[] = { 0, 0 };

    lapack_int iseed[] = { 0, 0, 0, 1 };
    #if defined( BLAS_ILP64 ) || defined( LAPACK_ILP64 )
        // 32-bit copy
        std::vector<lapack_int> iseed_( &iseed[0], &iseed[(4)] );
        lapack_int* iseed_ptr = &iseed_[0];
    #else
        lapack_int* iseed_ptr = iseed;
    #endif

    double D[]    = { 1,0, 0.1, 0.01, 0.001, 0.0001 };
    double cond[] = { 1.0, 1.0 };
    double dmax[] = { -1.0, -1.0 };

    double    A[25];
    double work[15];

    lapack_int info = -1;
    LAPACK_dlatms(
        m, m, "U", iseed_ptr, "N", D, mode, cond, dmax, klu, klu, "N", A, m,
        work, &info
        #ifdef LAPACK_FORTRAN_STRLEN_END
        , 1, 1, 1
        #endif
    );

    #if defined( BLAS_ILP64 ) || defined( LAPACK_ILP64 )
        std::copy( iseed_.begin(), iseed_.end(), iseed );
    #endif

    bool okay = (info == 0);
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
