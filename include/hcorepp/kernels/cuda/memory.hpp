/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_KERNELS_CUDA_MEMORY_H
#define HCOREPP_KERNELS_CUDA_MEMORY_H

#include "error_checking.h"


namespace hcorepp {
    namespace memory {

        template<typename T>
        T *AllocateArray(size_t aNumElements, const hcorepp::kernels::RunContext &aContext) {
            T *array;
            GPU_ERROR_CHECK(cudaMalloc((void **) &array, aNumElements * sizeof(T)));
            return array;
        }

        template<typename T>
        void DestroyArray(T *apArray, const hcorepp::kernels::RunContext &aContext) {
            if (apArray != nullptr) {
                GPU_ERROR_CHECK(cudaFree(apArray));
            }
        }

        template<typename T>
        void Memcpy(T *apDestination, const T *apSrcDataArray, size_t aNumOfElements,
                    const hcorepp::kernels::RunContext &aContext, MemoryTransfer aTransferType, bool aBlocking) {
            GPU_ERROR_CHECK(cudaMemcpyAsync(apDestination, apSrcDataArray, aNumOfElements * sizeof(T),
                                            cudaMemcpyDefault, aContext.GetStream()));
            if (aBlocking) {
                cudaStreamSynchronize(aContext.GetStream());
            }
        }

        template<typename T>
        void Memset(T *apDestination, char aValue, size_t aNumOfElements,
                    const hcorepp::kernels::RunContext &aContext) {
            GPU_ERROR_CHECK(cudaMemsetAsync(apDestination, aValue, aNumOfElements * sizeof(T), aContext.GetStream()))
        }


    }//namespace memory
}//namespace hcorepp

#endif //HCOREPP_KERNELS_CUDA_MEMORY_H
