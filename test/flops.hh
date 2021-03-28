// Copyright (c) 2017-2021,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#ifndef HCORE_TEST_FLOPS_HH
#define HCORE_TEST_FLOPS_HH

#include "hcore/exception.hh"

#include "blas/flops.hh"

#include <cstdint>
#include <algorithm>
#include <initializer_list>

namespace hcore {

// todo
// inline double fmuls_gemm(
//    double m, double n, double k, double Ark, double Brk, double Crk)
// {
//     return m*n*k;
// }
// 
// inline double fadds_gemm(
//    double m, double n, double k, double Ark, double Brk, double Crk)
// {
//     return m*n*k;
// }

template <typename T>
class Gflop : public blas::Gflop<T>
{
public:
    static double gemm(
        double m, double n, double k, double Ark, double Brk, double Crk)
    {
        // todo
        // return 1e-9 * (mul_ops*fmuls_gemm(m, n, k) +
        //                add_ops*fadds_gemm(m, n, k));
        // todo: relax this assumption (m = n = k (square matrices))
        // todo reference PASC papers.
        // hcore_assert(m == k);
        // hcore_assert(k == n);
        // hcore_assert(m == n);
        int64_t max_Ark_Brk_Crk = std::max({Ark, Brk, Crk});
        int64_t max_m_n_k = std::max({m, n, k});
        return (1e-9 * ((blas::is_complex<T>::value ? 3 : 1)
                     * 36 * max_m_n_k * (max_Ark_Brk_Crk
                     * max_Ark_Brk_Crk) + 157 * (max_Ark_Brk_Crk
                     * max_Ark_Brk_Crk * max_Ark_Brk_Crk)));
    }
}; // class Gflop

template <typename T>
class Gbyte : public blas::Gbyte<T>
{
public:
    static double gemm(
        double m, double n, double k, double Ark, double Brk, double Crk)
    {
        // return 1e-9 * ((m*k + k*n + 2*m*n) * sizeof(T));
        return 1e-9 * (
            ((     (m + k) * Ark) +
             (     (k + n) * Brk) +
             (2 * ((m + n) * Crk))) * sizeof(T));
    }
}; // class Gbyte
} // namespace hore

#endif // HCORE_TEST_FLOPS_HH
