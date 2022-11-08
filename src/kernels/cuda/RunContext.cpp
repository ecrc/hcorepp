/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <hcorepp/kernels/cuda/RunContext.hpp>

namespace hcorepp {
    namespace kernels {

        RunContext::RunContext() {
            this->mpQueue = new blas::Queue(0, 2048);
            cusolverDnCreate(&mCuSolverHandle);
            cusolverDnSetStream(mCuSolverHandle, this->mpQueue->stream());
            cudaMalloc(&this->mpInfo, sizeof(int));
            this->mWorkBufferSize = 0;
            this->mpWorkBuffer = nullptr;
        }

        int *RunContext::GetInfoPointer() {
            return this->mpInfo;
        }

        void * RunContext::RequestWorkBuffer(size_t aBufferSize) {
            if (aBufferSize > this->mWorkBufferSize) {
                if (this->mpWorkBuffer != nullptr) {
                    cudaFree(this->mpWorkBuffer);
                }
                this->mWorkBufferSize = aBufferSize;
                cudaMalloc(&this->mpWorkBuffer, aBufferSize);
            }
            return this->mpWorkBuffer;
        }

        blas::Queue &RunContext::GetBLASQueue() {
            return *this->mpQueue;
        }

        cudaStream_t RunContext::GetStream() {
            return this->mpQueue->stream();
        }

        cusolverDnHandle_t RunContext::GetCusolverDnHandle() {
            return this->mCuSolverHandle;
        }

        void RunContext::Sync() {
            this->mpQueue->sync();
        }

        RunContext::~RunContext() {
            cusolverDnDestroy(mCuSolverHandle);
            this->mpQueue->sync();
            delete this->mpQueue;
            cudaFree(this->mpInfo);
            if (this->mpWorkBuffer != nullptr) {
                cudaFree(this->mpWorkBuffer);
            }
            this->mpQueue = nullptr;
        }

    }//namespace kernels
}//namespace hcorepp

