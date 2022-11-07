/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCOREPP_KERNELS_CUDA_ERROR_CHECKING_H
#define HCOREPP_KERNELS_CUDA_ERROR_CHECKING_H

#include <cuda_runtime.h>
#include <cstdio>

namespace hcorepp {

    /**
     * @brief
     * Useful macro wrapper for all cuda API calls to ensure correct returns,
     * and error throwing on failures.
     */
    #define GPU_ERROR_CHECK(ans) { hcorepp::AssertGPU((ans), __FILE__, __LINE__); }

    /**
     * @brief
     * Function to assert the return code of a CUDA API call, and ensure
     * it completed successfully.
     *
     * @param[in] aCode
     * The code returned from the CUDA API call.
     *
     * @param[in] aFile
     * The name of the file that the assertion was called from.
     *
     * @param[in] aLine
     * The line number in the file that the assertion that was called from.
     *
     * @param[in] aAbort
     * Boolean to indicate whether to exit on failure or not, by default is true and will exit.
     */
    inline void AssertGPU(cudaError_t aCode, const char *aFile, int aLine, bool aAbort=true)
    {
        if (aCode != cudaSuccess)
        {
            fprintf(stderr,"GPU Assert: %s %s %d\n", cudaGetErrorString(aCode), aFile, aLine);
            if (aAbort) exit(aCode);
        }
    }
}

#endif //HCOREPP_KERNELS_CUDA_ERROR_CHECKING_H
