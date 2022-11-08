/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCOREPP_KERNELS_RUN_CONTEXT_H
#define HCOREPP_KERNELS_RUN_CONTEXT_H

#include <blas.hh>

#ifdef BLAS_HAVE_CUBLAS
#include "cuda/RunContext.hpp"
#else
#include "cpu/RunContext.hpp"
#endif

#endif //HCOREPP_KERNELS_RUN_CONTEXT_H
