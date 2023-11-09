/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#include <hcorepp/kernels/cuda/RunContext.hpp>

namespace hcorepp {
    namespace kernels {

        RunContext::RunContext() {
            this->mpQueue = std::make_shared<blas::Queue>(0, 2048);
            cusolverDnCreate(&mCuSolverHandle);
            cusolverDnSetStream(mCuSolverHandle, this->mpQueue->stream());
            cudaMalloc(&this->mpInfo, sizeof(int));
            this->mWorkBufferSize = 0;
            this->mpWorkBuffer = nullptr;
            this->mCuSolverOwner = true;
            this->mWorkSpaceOwner = false;
        }

        RunContext::RunContext(const RunContext &aContext) {
            this->mpQueue = aContext.mpQueue;
            this->mCuSolverHandle = aContext.mCuSolverHandle;
            this->mpInfo = aContext.mpInfo;
            this->mWorkBufferSize = 0;
            this->mpWorkBuffer = nullptr;
            this->mCuSolverOwner = false;
            this->mWorkSpaceOwner = false;
        }

        int *RunContext::GetInfoPointer() const {
            return this->mpInfo;
        }

        void * RunContext::RequestWorkBuffer(size_t aBufferSize) const {
            if (aBufferSize > this->mWorkBufferSize) {
                if (this->mpWorkBuffer != nullptr) {
                    cudaFree(this->mpWorkBuffer);
                }
                this->mWorkBufferSize = aBufferSize;
                cudaMalloc(&this->mpWorkBuffer, aBufferSize);
                this->mWorkSpaceOwner = true;
            }
            return this->mpWorkBuffer;
        }

        blas::Queue &RunContext::GetBLASQueue() const {
            return *this->mpQueue;
        }

        cudaStream_t RunContext::GetStream() const {
            return this->mpQueue->stream();
        }

        cusolverDnHandle_t RunContext::GetCusolverDnHandle() const {
            return this->mCuSolverHandle;
        }

        RunContext RunContext::ForkChildContext() {
	    RunContext context(*this);
	    return context;
	}

        void RunContext::Sync() const {
            this->mpQueue->sync();
        }

        RunContext::~RunContext() {
            if(mCuSolverOwner) {
                cusolverDnDestroy(mCuSolverHandle);
                this->mpQueue->sync();
                cudaFree(this->mpInfo);
            }
            if(mWorkSpaceOwner) {
                if (this->mpWorkBuffer != nullptr) {
                    cudaFree(this->mpWorkBuffer);
                }
            }
        }

    }//namespace kernels
}//namespace hcorepp

