/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_KERNELS_CPU_MEMORY_H
#define HCOREPP_KERNELS_CPU_MEMORY_H

#include <cstring>

namespace hcorepp {
    namespace memory {

        template<typename T>
        T *AllocateArray(int64_t aNumElements) {
            T *apArray = (T *) malloc(aNumElements * sizeof(T));
            return apArray;
        }

        template<typename T>
        void DestroyArray(T *apArray) {
            if (apArray != nullptr) {
                free(apArray);
            }
        }

        template<typename T>
        void Memcpy(T *apDestination, const T *apSrcDataArray, int64_t aNumOfElements,
                    MemoryTransfer aTransferType) {
            memcpy(apDestination, apSrcDataArray,aNumOfElements * sizeof(T));
        }

        template<typename T>
        void Memcpy(T *apDestination, const T *apSrcDataArray, int64_t aNumOfElements,
                    hcorepp::kernels::RunContext &aContext, MemoryTransfer aTransferType) {
            memcpy(apDestination, apSrcDataArray,aNumOfElements * sizeof(T));
        }

        template<typename T>
        void Memset(T *apDestination, char aValue, int64_t aNumOfElements) {
            memset((void *) apDestination, aValue, aNumOfElements * sizeof(T));
        }

    }//namespace memory
}//namespace hcorepp

#endif //HCOREPP_KERNELS_CPU_MEMORY_H
