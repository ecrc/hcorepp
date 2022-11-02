/**
 * Copyright (c) 2017-2022, King Abdullah University of Science and Technology
 * ***************************************************************************
 * *****      KAUST Extreme Computing and Research Center Property       *****
 * ***************************************************************************
 *
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
 */

#ifndef HCOREPP_KERNELS_MEMORY_H
#define HCOREPP_KERNELS_MEMORY_H

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
         * @return
         * A pointer to the allocated array.
         */
        template<typename T>
        T *AllocateArray(int64_t aNumElements);

        /**
         * @brief
         * Deallocate a previously allocated array.
         *
         * @tparam T
         * The type of the elements the pointer is pointing to.
         *
         * @param[in] apArray
         * The pointer to deallocate.
         */
        template<typename T>
        void DestroyArray(T *apArray);

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
         * @param[in] aTransferType
         * The transfer type telling the memcpy where each pointer resides(host or accelerator).
         */
        template<typename T>
        void Memcpy(T *apDestination, const T *apSrcDataArray, int64_t aNumOfElements,
                    MemoryTransfer aTransferType = MemoryTransfer::DEVICE_TO_DEVICE);

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
         */
        template<typename T>
        void Memset(T *apDestination, char aValue, int64_t aNumOfElements);

    }//namespace memory
}//namespace hcorepp

#ifdef USE_CUDA
#include "cuda/memory.hpp"
#else
#include "cpu/memory.hpp"
#endif

#endif //HCOREPP_KERNELS_MEMORY_H
