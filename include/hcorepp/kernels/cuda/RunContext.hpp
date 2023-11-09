/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_KERNELS_CUDA_RUN_CONTEXT_HPP
#define HCOREPP_KERNELS_CUDA_RUN_CONTEXT_HPP

#include <blas.hh>
#include <memory>
#include <cusolverDn.h>

namespace hcorepp {
    namespace kernels {
        class RunContext {
        public:
            RunContext();

            RunContext(const RunContext &aContext);

            blas::Queue &GetBLASQueue() const;

            cudaStream_t GetStream() const;

            cusolverDnHandle_t GetCusolverDnHandle() const;

            int *GetInfoPointer() const;

            void *RequestWorkBuffer(size_t aBufferSize) const;

            void Sync() const;

            RunContext ForkChildContext();

            ~RunContext();

            /**
             * @brief
             * Check if Context supports OMP Parallelization
             */
            bool SupportsOMP() const {
                return false;
            }

        private:
            mutable size_t mWorkBufferSize;
            mutable void *mpWorkBuffer;
            int *mpInfo;
            std::shared_ptr<blas::Queue> mpQueue;
            cusolverDnHandle_t mCuSolverHandle;
            bool mCuSolverOwner;
            mutable bool mWorkSpaceOwner;
        };
    }//namespace kernels
}//namespace hcorepp

#endif //HCOREPP_KERNELS_CUDA_RUN_CONTEXT_HPP
