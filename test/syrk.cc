// Copyright (c) 2017,
// King Abdullah University of Science and Technology (KAUST)
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.

#include "test.hh"

#include <complex>

template <typename scalar_t>
void syrk_test_execute(Params& params, bool run)
{
    if (!run) return;
}

void syrk_test_dispatch(Params& params, bool run)
{
    switch(params.datatype()) {
        case testsweeper::DataType::Single:
            syrk_test_execute<float>(params, run);
            break;
        case testsweeper::DataType::Double:
            syrk_test_execute<double>(params, run);
            break;
        case testsweeper::DataType::SingleComplex:
            syrk_test_execute<std::complex<float>>(params, run);
            break;
        case testsweeper::DataType::DoubleComplex:
            syrk_test_execute<std::complex<double>>(params, run);
            break;
        default:
            throw std::invalid_argument("Unsupported data type.");
            break;
    }
}
