/**
 * @copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
 *                     All rights reserved.
 */

#ifndef HCOREPP_KERNELS_MEMORY_H
#define HCOREPP_KERNELS_MEMORY_H

#include <hcorepp/kernels/RunContext.hpp>

/**
 * @brief
 * File abstracting all memory operations from the underlying technology.
 */
namespace hcorepp {
    namespace memory {

        /**
         * @brief
         * The supported memory transfers types.
         */
        enum class MemoryTransfer {
            HOST_TO_DEVICE,
            DEVICE_TO_DEVICE,
            DEVICE_TO_HOST,
            HOST_TO_HOST,
            AUTOMATIC
        };

        /**
         * @brief
         * Allocates an array of elements on the target accelerator.
         *
         * @tparam T
         * The type of each element
         *
         * @param[in] aNumElements
         * The number of elements for the array.
         *
         * @param[in] aContext
         * The run context for the allocation operation.
         *
         * @return
         * A pointer to the allocated array.
         */
        template<typename T>
        T *AllocateArray(size_t aNumElements, const hcorepp::kernels::RunContext &aContext);

        /**
         * @brief
         * Deallocate a previously allocated array.
         *
         * @tparam T
         * The type of the elements the pointer is pointing to.
         *
         * @param[in] apArray
         * The pointer to deallocate.
         *
         * @param[in] aContext
         * The run context for the de-allocation operation.
         */
        template<typename T>
        void DestroyArray(T *apArray, const hcorepp::kernels::RunContext &aContext);

        /**
         * @brief
         * Copy memory from a source pointer to a target pointer according to the transfer type.
         *
         * @tparam T
         * The type of the element the pointer are pointing to.
         *
         * @param[in] apDestination
         * The destination pointer to copy data to.
         *
         * @param[in] apSrcDataArray
         * The source pointer to copy data from.
         *
         * @param[in] aNumOfElements
         * The number of elements to transfer between the two arrays.
         *
         * @param[in] aContext
         * The run context for the memcpy operation.
         *
         * @param[in] aTransferType
         * The transfer type telling the memcpy where each pointer resides(host or accelerator).
         *
         * @param[in] aBlocking
         * If true, will ensure the call will only return after it finishes, otherwise
         * no such guarantee will be provided.
         */
        template<typename T>
        void Memcpy(T *apDestination, const T *apSrcDataArray, size_t aNumOfElements,
                    const hcorepp::kernels::RunContext &aContext,
                    MemoryTransfer aTransferType = MemoryTransfer::DEVICE_TO_DEVICE,
                    bool aBlocking = false);

        /**
         * @brief
         * Memset mechanism for a device pointer.
         *
         * @tparam T
         * The type of the element the pointer are pointing to.
         *
         * @param[in] apDestination
         * The destination pointer.
         *
         * @param[in] aValue
         * The value to set each byte to.
         *
         * @param[in] aNumOfElements
         * The number of elements of the given pointer.
         *
         * @param[in] aContext
         * The run context for the memset operation.
         */
        template<typename T>
        void Memset(T *apDestination, char aValue, size_t aNumOfElements,
                    const hcorepp::kernels::RunContext &aContext);

        template<typename T>
        T *Realloc(T *apDataArray, size_t aNumOfElements, const hcorepp::kernels::RunContext &aContext);

    }//namespace memory
}//namespace hcorepp

#ifdef USE_CUDA
#include "cuda/memory.hpp"
#elif defined(USE_SYCL)

#include "sycl/memory.hpp"

#else

#include "cpu/memory.hpp"

#endif

#endif //HCOREPP_KERNELS_MEMORY_H
