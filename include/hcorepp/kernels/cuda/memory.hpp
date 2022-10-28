#ifndef HCOREPP_KERNELS_CUDA_MEMORY_H
#define HCOREPP_KERNELS_CUDA_MEMORY_H

#include "error_checking.h"

namespace hcorepp {
    namespace memory {

        template<typename T>
        T *AllocateArray(int64_t aNumElements) {
            T *array;
            GPU_ERROR_CHECK(cudaMalloc((void **) &array, aNumElements * sizeof(T)));
            return array;
        }

        template<typename T>
        void DestroyArray(T *apArray) {
            if (apArray != nullptr) {
                GPU_ERROR_CHECK(cudaFree(apArray));
            }
        }

        template<typename T>
        void Memcpy(T *apDestination, const T *apSrcDataArray, int64_t aNumOfElements,
                    MemoryTransfer aTransferType) {
            GPU_ERROR_CHECK(cudaDeviceSynchronize());
            GPU_ERROR_CHECK(cudaMemcpy(apDestination, apSrcDataArray, aNumOfElements * sizeof(T),
                                       cudaMemcpyDefault));
        }

        template<typename T>
        void Memset(T *apDestination, char aValue, int64_t aNumOfElements) {
            GPU_ERROR_CHECK(cudaMemset(apDestination, aValue, aNumOfElements * sizeof(T)));
        }

    }//namespace memory
}//namespace hcorepp

#endif //HCOREPP_KERNELS_CUDA_MEMORY_H
