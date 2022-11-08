/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing Research Center Property           *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCOREPP_KERNELS_CUDA_RUN_CONTEXT_HPP
#define HCOREPP_KERNELS_CUDA_RUN_CONTEXT_HPP

#include <blas.hh>
#include <cusolverDn.h>

namespace hcorepp {
    namespace kernels {
        class RunContext {
        public:
            RunContext();

            ~RunContext();

            blas::Queue &GetBLASQueue();

            cudaStream_t GetStream();

            cusolverDnHandle_t GetCusolverDnHandle();

            int *GetInfoPointer();

            void *RequestWorkBuffer(size_t aBufferSize);

            void Sync();

        private:
            size_t mWorkBufferSize;
            void *mpWorkBuffer;
            int *mpInfo;
            blas::Queue *mpQueue;
            cusolverDnHandle_t mCuSolverHandle;
        };
    }//namespace kernels
}//namespace hcorepp

#endif //HCOREPP_KERNELS_CUDA_RUN_CONTEXT_HPP
