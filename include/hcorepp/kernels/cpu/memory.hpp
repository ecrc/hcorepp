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
        T *AllocateArray(size_t aNumElements, const hcorepp::kernels::RunContext &aContext) {
            T *apArray = (T *) malloc(aNumElements * sizeof(T));
            return apArray;
        }

        template<typename T>
        void DestroyArray(T *apArray, const hcorepp::kernels::RunContext &aContext) {
            if (apArray != nullptr) {
                free(apArray);
            }
        }

        template<typename T>
        void Memcpy(T *apDestination, const T *apSrcDataArray, size_t aNumOfElements,
                    const hcorepp::kernels::RunContext &aContext, MemoryTransfer aTransferType, bool aBlocking) {
            memcpy(apDestination, apSrcDataArray, aNumOfElements * sizeof(T));
        }

        template<typename T>
        void
        Memset(T *apDestination, char aValue, size_t aNumOfElements, const hcorepp::kernels::RunContext &aContext) {
            memset((void *) apDestination, aValue, aNumOfElements * sizeof(T));
        }

    }//namespace memory
}//namespace hcorepp

#endif //HCOREPP_KERNELS_CPU_MEMORY_H
