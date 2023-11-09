/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_KERNELS_RUN_CONTEXT_H
#define HCOREPP_KERNELS_RUN_CONTEXT_H

#include <blas.hh>

/** TODO Add Thread Safety across all technologies for singleton Context **/

#ifdef USE_CUDA
#include "cuda/RunContext.hpp"
#elif defined(USE_SYCL)
#include "sycl/RunContext.hpp"
#else
#include "cpu/RunContext.hpp"
#endif

#endif //HCOREPP_KERNELS_RUN_CONTEXT_H
